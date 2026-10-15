"""Train the RL head of the inspection planner with PPO in the kinematic engine on procedural scenes.

    python scripts/pose_inspection/rl_train.py [--config rl.yaml] [--name run1] [--set rl.envs=8 ...]

Each iteration: one fresh procedural scene (warehouse / yard alternating), ``rl.envs`` robots from different starts;
every time a robot needs a new target the policy samples one of the planner's candidates (or STOP). After the
episode the decisions get their reward (GT surface-coverage gain minus time) and the policy is updated with PPO
(semi-MDP discounting by elapsed time). Logs: <out>/<name>/log.csv, checkpoints <out>/<name>/ckpt_<it>.pt.
"""

import argparse
import csv
import os
import shutil
import sys
import time

import numpy as np
import torch

SIM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SIM)
from go2_lab.pose_inspection.config import load_config, sim_path  # noqa: E402
from go2_lab.pose_inspection.kinematic import KinematicEngine  # noqa: E402
from go2_lab.pose_inspection.navgrid import NavGrid  # noqa: E402
from go2_lab.pose_inspection.rl.collector import Collector  # noqa: E402
from go2_lab.pose_inspection.rl.features import to_batch  # noqa: E402
from go2_lab.pose_inspection.rl.policy import InspectionPolicy  # noqa: E402
from go2_lab.pose_inspection.rl.procgen import load_parts, make_scene  # noqa: E402
from go2_lab.pose_inspection.session import Session  # noqa: E402


def starts(cfg, meta, seeds):
    nav = NavGrid(os.path.join(meta["dir"], "grid.npz"), cfg.robot.radius).restrict_to_region(
        meta["robot_start"]["region"])
    return np.array([nav.sample_starts(meta["robot_start"]["region"], 1, s, meta["robot_start"]["min_clearance"])[0]
                     for s in seeds])


def episode_returns(dec, t_end, co_end, rl):
    """Rewards, discounts, GAE advantages and returns for one robot's decision list (in place)."""
    n = len(dec)
    for k in range(n):
        t1, c1 = (dec[k + 1]["t"], dec[k + 1]["co"]) if k + 1 < n else (t_end, co_end)
        dec[k]["r"] = rl.w_co * (c1 - dec[k]["co"]) - rl.w_time * (t1 - dec[k]["t"]) / 100.0
        dec[k]["g"] = rl.gamma ** ((t1 - dec[k]["t"]) / 10.0)
    adv = 0.0
    for k in reversed(range(n)):
        v_next = dec[k + 1]["v"] if k + 1 < n else 0.0
        delta = dec[k]["r"] + dec[k]["g"] * v_next - dec[k]["v"]
        adv = delta + dec[k]["g"] * rl.gae_lambda * adv
        dec[k]["adv"], dec[k]["ret"] = adv, adv + dec[k]["v"]


def ppo_update(net, opt, batch, rl, device):
    obs = [d["obs"] for d in batch]
    a = torch.tensor([d["a"] for d in batch], device=device)
    logp0 = torch.tensor([d["logp"] for d in batch], device=device)
    adv = torch.tensor([d["adv"] for d in batch], device=device, dtype=torch.float32)
    ret = torch.tensor([d["ret"] for d in batch], device=device, dtype=torch.float32)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    stats = []
    for _ in range(rl.epochs):
        perm = np.random.permutation(len(batch))
        for s in range(0, len(batch), rl.minibatch):
            idx = perm[s:s + rl.minibatch]
            cand, mask, glob = to_batch([obs[i] for i in idx], device)
            logits, v = net(cand, mask, glob)
            # an action == number of candidates of that observation is STOP, the last column of the padded row
            n_real = mask.sum(1)
            ai = a[idx]
            col = torch.where(ai >= n_real, torch.full_like(ai, logits.shape[1] - 1), ai)
            dist = torch.distributions.Categorical(logits=logits)
            logp = dist.log_prob(col)
            ratio = (logp - logp0[idx]).exp()
            s1, s2 = ratio * adv[idx], ratio.clamp(1 - rl.clip, 1 + rl.clip) * adv[idx]
            pl = -torch.min(s1, s2).mean()
            vl = ((v - ret[idx]) ** 2).mean()
            ent = dist.entropy().mean()
            loss = pl + rl.value_coef * vl - rl.entropy * ent
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), rl.max_grad_norm)
            opt.step()
            stats.append((pl.item(), vl.item(), ent.item(), ((ratio - 1).abs() > rl.clip).float().mean().item()))
    return np.mean(stats, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="rl.yaml")
    ap.add_argument("--name", default="run1")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--resume", default=None)
    ap.add_argument("--init", default=None, help="start from these policy weights (fresh optimizer, iteration 0)")
    args = ap.parse_args()
    import yaml

    over = {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in args.set)}
    cfg = load_config(args.config, over)
    rl = cfg.rl
    dev = "cuda:0"
    out = sim_path(os.path.join(rl.out, args.name))
    os.makedirs(out, exist_ok=True)
    torch.manual_seed(rl.seed)
    rng = np.random.default_rng(rl.seed)
    parts = load_parts(sim_path(rl.parts))
    net = InspectionPolicy(rl.d_model, rl.heads, rl.layers).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=rl.lr)
    it0 = 0
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location=dev)["net"])
    if args.resume:
        ck = torch.load(args.resume, map_location=dev)
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        it0 = ck["it"] + 1
    logf = open(os.path.join(out, "log.csv"), "a", newline="")
    log = csv.writer(logf)
    if it0 == 0:
        log.writerow(["it", "kind", "scene_seed", "machines", "wall_s", "Co", "t_eff", "auc", "ret", "decisions",
                      "stop_frac", "timeouts", "pl", "vl", "ent", "clipfrac"])
    scene_dir = os.path.join(rl.scene_dir, args.name)
    for it in range(it0, rl.iterations):
        t0 = time.time()
        kind = rl.scene_kinds[it % len(rl.scene_kinds)]
        sseed = int(rng.integers(1 << 30))
        shutil.rmtree(scene_dir, ignore_errors=True)
        meta = make_scene(np.random.default_rng(sseed), parts, kind, scene_dir, f"rl_{it}")
        if not any(o["target"] for o in meta["objects"]):
            continue
        seeds = [int(rng.integers(1 << 30)) for _ in range(rl.envs)]
        try:
            st = starts(cfg, meta, seeds)
        except Exception as ex:  # noqa: BLE001
            print(f"it {it}: no start ({ex}); skipped")
            continue
        col = Collector(net, dev)
        s = Session(cfg, meta, seeds, os.path.join(out, "episodes"), tag="rl", selector_factory=col.factory)
        col.session = s
        net.eval()
        eng = KinematicEngine(s, st)
        eng.run(log=lambda *a: None)
        batch, cos, tes, aucs, rets, nstop = [], [], [], [], [], 0
        for e in range(rl.envs):
            r = s.summary(e)
            dec = col.dec[e]
            if not dec:
                continue
            episode_returns(dec, r["t_eff"], r["Co"], rl)
            batch += dec
            cos.append(r["Co"])
            tes.append(r["t_eff"])
            curve = [(c[0], c[1]) for c in s.curves[e]]
            from go2_lab.pose_inspection.metrics import auc  # noqa: PLC0415

            aucs.append(auc(curve, 1200.0, r["t_eff"]))
            rets.append(sum(d["r"] for d in dec))
            nstop += dec[-1]["a"] >= dec[-1]["n"]
        net.train()
        pl, vl, ent, cf = ppo_update(net, opt, batch, rl, dev) if batch else (0, 0, 0, 0)
        row = [it, kind, sseed, sum(o["target"] for o in meta["objects"]), round(time.time() - t0, 1),
               round(np.mean(cos), 4), round(np.mean(tes), 1), round(np.mean(aucs), 4), round(np.mean(rets), 3),
               len(batch), round(nstop / max(1, len(cos)), 3), int(sum(s.term.timeout)), round(pl, 4), round(vl, 4),
               round(ent, 4), round(cf, 4)]
        log.writerow(row)
        logf.flush()
        print(" ".join(map(str, row)), flush=True)
        if (it + 1) % rl.checkpoint_every == 0 or it + 1 == rl.iterations:
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "it": it, "cfg": dict(rl)},
                       os.path.join(out, f"ckpt_{it + 1:04d}.pt"))


if __name__ == "__main__":
    main()
