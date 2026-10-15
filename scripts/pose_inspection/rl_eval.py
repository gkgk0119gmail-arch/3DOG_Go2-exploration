"""Evaluate the RL head (greedy) against the rule-based POSE planner, kinematic engine, same scenes and starts.

    python scripts/pose_inspection/rl_eval.py <ckpt.pt> [--val 20] [--handmade] [--envs 4] [--out results/...]

Scenes: the first ``--val`` validation scenes of the procedural pool (assets/pose_inspection/rl_scenes/val, made by
rl_make_scenes.py; never trained on) and, with ``--handmade``, the two hand-built scenes (held out, 6 seeds as in
stages 3-5). The POSE baseline of each scene is cached (<out>/pose_cache.json). Per scene T = the longest t_eff of
the two methods (AUCo with a common horizon, as in the stage tables). Writes <out>.json / .md.
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

SIM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SIM)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from go2_lab.pose_inspection.config import load_config  # noqa: E402
from go2_lab.pose_inspection.kinematic import KinematicEngine  # noqa: E402
from go2_lab.pose_inspection.metrics import auc  # noqa: E402
from go2_lab.pose_inspection.rl.policy import InspectionPolicy  # noqa: E402
from go2_lab.pose_inspection.scene_builder import load_scene  # noqa: E402
from go2_lab.pose_inspection.session import Session  # noqa: E402
from rl_train import Collector, starts  # noqa: E402


def run(cfg, meta, seeds, net=None):
    col = Collector(net, "cuda:0", greedy=True) if net is not None else None
    s = Session(cfg, meta, seeds, "/tmp/pose_rl_eval_unused", tag="eval",
                selector_factory=col.factory if col else None)
    if col:
        col.session = s
    KinematicEngine(s, starts(cfg, meta, seeds)).run(log=lambda *a: None)
    out = []
    for e in range(len(seeds)):
        r = s.summary(e)
        out.append({"Co": r["Co"], "t_eff": r["t_eff"], "L": r["L_eff"], "N_post": r["N_post"],
                    "timeout": r["timeout"], "curve": [(c[0], c[1]) for c in s.curves[e]],
                    "decisions": len(col.dec[e]) if col else None})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--config", default="rl.yaml")
    ap.add_argument("--val", type=int, default=20)
    ap.add_argument("--envs", type=int, default=4)
    ap.add_argument("--handmade", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    ck = torch.load(args.ckpt, map_location="cuda:0")
    net = InspectionPolicy(cfg.rl.d_model, cfg.rl.heads, cfg.rl.layers).to("cuda:0")
    net.load_state_dict(ck["net"])
    net.eval()
    out = args.out or os.path.splitext(args.ckpt)[0] + "_eval"
    cache_f = os.path.join(os.path.dirname(args.ckpt), "..", "pose_cache.json")
    cache = json.load(open(cache_f)) if os.path.exists(cache_f) else {}
    scenes = sorted(glob.glob(os.path.join(SIM, "assets/pose_inspection/rl_scenes/val/*")))[:args.val]
    jobs = [(d, list(range(args.envs))) for d in scenes]
    if args.handmade:
        jobs += [(os.path.join(SIM, "assets/pose_inspection", n), list(range(6))) for n in ("A_warehouse",
                                                                                           "B_construction")]
    res = {}
    for d, seeds in jobs:
        meta = load_scene(d)
        key = f"{meta['name']}|{seeds}"
        if key not in cache:
            cache[key] = run(cfg, meta, seeds)
            json.dump(cache, open(cache_f, "w"))
        pol = run(cfg, meta, seeds, net)
        T = max(max(r["t_eff"] for r in cache[key]), max(r["t_eff"] for r in pol))
        row = {}
        for name, rr in (("pose", cache[key]), ("rl", pol)):
            row[name] = {k: float(np.mean([r[k] for r in rr])) for k in ("Co", "t_eff", "L", "N_post")}
            row[name]["AUCo"] = float(np.mean([auc(r["curve"], T, r["t_eff"]) for r in rr]))
            row[name]["timeouts"] = int(sum(r["timeout"] for r in rr))
        # matched comparisons (same seed = same start): POSE coverage at the time RL stopped, and the time POSE
        # needs to reach RL's final coverage
        from go2_lab.pose_inspection.metrics import value_at  # noqa: PLC0415

        co_at, t_to = [], []
        for rp, rr_ in zip(cache[key], pol):
            co_at.append(value_at(rp["curve"], rr_["t_eff"]))
            reach = [c[0] for c in rp["curve"] if c[1] >= rr_["Co"] - 1e-9]
            t_to.append(reach[0] if reach else float("nan"))
        row["pose"]["Co_at_rl_t"] = float(np.mean(co_at))
        row["pose"]["t_to_rl_Co"] = float(np.nanmean(t_to)) if not all(np.isnan(t_to)) else float("nan")
        row["kind"] = meta.get("kind", "handmade")
        res[meta["name"]] = row
        print(f"{meta['name']:>16} ({row['kind']}):  POSE Co {row['pose']['Co']:.3f} t {row['pose']['t_eff']:5.0f} "
              f"AUCo {row['pose']['AUCo']:.3f}  |  RL Co {row['rl']['Co']:.3f} t {row['rl']['t_eff']:5.0f} "
              f"AUCo {row['rl']['AUCo']:.3f}  ||  POSE Co at RL stop {row['pose']['Co_at_rl_t']:.3f}, POSE time "
              f"to RL Co {row['pose']['t_to_rl_Co']:.0f}", flush=True)
    json.dump(res, open(out + ".json", "w"), indent=1)
    md = ["| scene | kind | POSE AUCo | RL AUCo | POSE Co | RL Co | POSE t_eff | RL t_eff | POSE Co at RL stop | "
          "POSE time to RL Co |", "|---|---|---|---|---|---|---|---|---|---|"]
    keys = [("pose", "AUCo", 3), ("rl", "AUCo", 3), ("pose", "Co", 3), ("rl", "Co", 3), ("pose", "t_eff", 0),
            ("rl", "t_eff", 0), ("pose", "Co_at_rl_t", 3), ("pose", "t_to_rl_Co", 0)]
    for n, r in res.items():
        md.append(f"| {n} | {r['kind']} | " + " | ".join(f"{r[a][b]:.{d}f}" for a, b, d in keys) + " |")
    for kind in sorted({r["kind"] for r in res.values()}):
        rr = [r for r in res.values() if r["kind"] == kind]
        md.append(f"| **mean ({kind}, {len(rr)})** | | " + " | ".join(
            f"{np.nanmean([r[a][b] for r in rr]):.{d}f}" for a, b, d in keys) + " |")
    open(out + ".md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
