"""SAC fine-tuning of ARiADNE for Go2 turning cost + 3D coverage (original driver.py without Ray).

    python driver3d.py [--episodes 4000] [--workers 20] [--resume]

Episodes run in worker processes (Env3D + Agent3D on CPU); the learner (GPU) keeps the replay buffer and does
8 SAC updates per finished episode, as in the original. Differences from the original:
  * warm start from the RA-L 2024 checkpoint, new input weights zero (nets.py)
  * critic-only warm-up (Q_WARMUP updates) so the old critic adapts to the new reward before the policy moves
  * 10x learning rate on the first (input) layer so the new features get used
  * masks are bit-packed for transfer and in the replay buffer
Logs: train/<FOLDER_NAME>/ (tensorboard + metrics.csv), checkpoints: model/<FOLDER_NAME>/checkpoint.pth
"""

import argparse
import csv
import multiprocessing as mp
import os
import random
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from parameter import (BATCH_SIZE, EMBEDDING_DIM, GAMMA, K_SIZE, LR, MINIMUM_BUFFER_SIZE, NODE_INPUT_DIM,
                       NUM_META_AGENT, PRETRAINED, REPLAY_SIZE, model_path, train_path)

Q_WARMUP = 1500  # critic-only SAC updates before the policy is updated
EVAL_EPISODE_START = 5600  # episodes (= maps) >= this are kept for evaluation (eval3d.py)
METRICS = ["travel_dist", "explored_rate", "success_rate", "episode_reward", "coverage_3d", "wall_cov", "ceiling_cov",
           "time_s", "turn_total_rad", "turn_time_s", "steps"]

# ---- worker side ------------------------------------------------------------------------------------
_net = None


def _init_worker():
    global _net
    torch.set_num_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    from model import PolicyNet

    _net = PolicyNet(NODE_INPUT_DIM, EMBEDDING_DIM)


def pack(buffer):
    """27 lists of per-step tensors -> 27 arrays [T, ...] (bool arrays bit-packed on the last axis)."""
    out = []
    for slot in buffer:
        a = torch.stack(slot, 0).cpu().numpy()  # same layout as the original driver's torch.stack(rollouts[i])
        if a.dtype == np.bool_:
            out.append(("b", a.shape, np.packbits(a, axis=-1)))
        else:
            out.append(("a", a.shape, a))
    return out


def run_job(weights, episode):
    from worker3d import Worker3D

    _net.load_state_dict(weights)
    w = Worker3D(0, _net, episode)
    w.run_episode()
    return pack(w.episode_buffer), w.perf_metrics, episode


# ---- learner side -----------------------------------------------------------------------------------
class Replay:
    def __init__(self, size):
        self.size, self.slots = size, None

    def add(self, packed):
        if self.slots is None:
            self.slots = [[] for _ in packed]
        n = packed[0][1][0]
        for s, (kind, shape, arr) in zip(self.slots, packed):
            for t in range(n):
                s.append((kind, shape[1:], arr[t]))
        if len(self.slots[0]) > self.size:
            cut = len(self.slots[0]) - self.size
            self.slots = [s[cut:] for s in self.slots]

    def __len__(self):
        return 0 if self.slots is None else len(self.slots[0])

    def sample(self, n, device):
        idx = random.sample(range(len(self)), n)
        out = []
        for s in self.slots:
            kind, shape, _ = s[idx[0]]
            a = np.stack([s[i][2] for i in idx], 0)
            if kind == "b":
                a = np.unpackbits(a, axis=-1, count=shape[-1]).astype(bool)
            out.append(torch.from_numpy(a).to(device))
        return out


def main():
    from nets import load_pretrained
    from model import QNet

    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=NUM_META_AGENT)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--q_warmup", type=int, default=Q_WARMUP)
    ap.add_argument("--min_buffer", type=int, default=MINIMUM_BUFFER_SIZE)
    ap.add_argument("--name", default=None, help="run name (default FOLDER_NAME): model/<name>, train/<name>")
    args = ap.parse_args()
    global model_path, train_path
    if args.name:
        model_path, train_path = f"model/{args.name}", f"train/{args.name}"
    os.makedirs(model_path, exist_ok=True)
    os.makedirs(train_path, exist_ok=True)
    device = torch.device("cuda")
    ck_path = os.path.join(model_path, "checkpoint.pth")
    policy, q1, q2, log_alpha, ck = load_pretrained(ck_path if args.resume else PRETRAINED, device)
    log_alpha.requires_grad = True
    tq1, tq2 = QNet(NODE_INPUT_DIM + 1, EMBEDDING_DIM).to(device), QNet(NODE_INPUT_DIM + 1, EMBEDDING_DIM).to(device)
    tq1.load_state_dict(q1.state_dict())
    tq2.load_state_dict(q2.state_dict())

    def opt(net):
        first = [p for n, p in net.named_parameters() if n.startswith("initial_embedding")]
        rest = [p for n, p in net.named_parameters() if not n.startswith("initial_embedding")]
        return optim.Adam([{"params": first, "lr": 10 * LR}, {"params": rest, "lr": LR}])

    p_opt, q1_opt, q2_opt = opt(policy), opt(q1), opt(q2)
    a_opt = optim.Adam([log_alpha], lr=1e-4)
    episode, updates = 0, 0
    if args.resume:
        p_opt.load_state_dict(ck["policy_optimizer"])
        q1_opt.load_state_dict(ck["q_net1_optimizer"])
        q2_opt.load_state_dict(ck["q_net2_optimizer"])
        a_opt = optim.Adam([log_alpha], lr=1e-4)
        episode, updates = ck["episode"], ck.get("updates", args.q_warmup)
    entropy_target = 0.05 * (-np.log(1 / K_SIZE))
    writer = SummaryWriter(train_path)
    csv_path = os.path.join(train_path, "metrics.csv")
    new_csv = not os.path.exists(csv_path) or not args.resume
    csv_f = open(csv_path, "w" if new_csv else "a", newline="")
    log = csv.writer(csv_f)
    if new_csv:
        log.writerow(["episode", "updates", "wall_s"] + METRICS)
    replay = Replay(REPLAY_SIZE)
    t_start = time.time()
    mse = nn.MSELoss()

    def weights():
        return {k: v.detach().cpu() for k, v in policy.state_dict().items()}

    ex = ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"), initializer=_init_worker)
    w = weights()
    jobs = set()
    next_ep = episode + 1
    for _ in range(args.workers):
        jobs.add(ex.submit(run_job, w, next_ep))
        next_ep += 1
    window = {m: [] for m in METRICS}
    try:
        while episode < args.episodes:
            done, jobs = wait(jobs, return_when=FIRST_COMPLETED)
            for fut in done:
                try:
                    packed, metrics, ep = fut.result()
                except Exception as e:  # noqa: BLE001  (a map the graph code chokes on: skip it)
                    print(f"[driver] episode failed: {e!r}", flush=True)
                    continue
                episode += 1
                replay.add(packed)
                log.writerow([ep, updates, round(time.time() - t_start)] + [metrics.get(m, np.nan) for m in METRICS])
                for m in METRICS:
                    window[m].append(metrics.get(m, np.nan))
            csv_f.flush()
            w = weights()
            while len(jobs) < args.workers and next_ep < EVAL_EPISODE_START:
                jobs.add(ex.submit(run_job, w, next_ep))
                next_ep += 1

            if len(replay) >= args.min_buffer:
                for _ in range(8 * len(done)):
                    b = replay.sample(BATCH_SIZE, device)
                    obs, act, rew, dn, nobs = b[0:6], b[6].long(), b[7].float(), b[8].float(), b[9:15]
                    cobs, cnobs = b[15:21], b[21:27]
                    with torch.no_grad():
                        q_now = torch.min(q1(*cobs), q2(*cobs))
                        next_logp = policy(*nobs)
                        nq = torch.min(tq1(*cnobs), tq2(*cnobs))
                        v_next = torch.sum(next_logp.unsqueeze(2).exp() * (nq - log_alpha.exp() * next_logp.unsqueeze(2)),
                                           dim=1).unsqueeze(1)
                        target = rew + GAMMA * (1 - dn) * v_next
                    logp = policy(*obs)
                    if updates >= args.q_warmup:
                        p_loss = torch.sum(logp.exp().unsqueeze(2) * (log_alpha.exp().detach() * logp.unsqueeze(2) - q_now),
                                           dim=1).mean()
                        p_opt.zero_grad()
                        p_loss.backward()
                        torch.nn.utils.clip_grad_norm_(policy.parameters(), 100)
                        p_opt.step()
                    else:
                        p_loss = torch.zeros(())
                    q1_loss = mse(torch.gather(q1(*cobs), 1, act), target)
                    q1_opt.zero_grad()
                    q1_loss.backward()
                    torch.nn.utils.clip_grad_norm_(q1.parameters(), 20000)
                    q1_opt.step()
                    q2_loss = mse(torch.gather(q2(*cobs), 1, act), target)
                    q2_opt.zero_grad()
                    q2_loss.backward()
                    torch.nn.utils.clip_grad_norm_(q2.parameters(), 20000)
                    q2_opt.step()
                    entropy = (logp.detach() * logp.detach().exp()).sum(dim=-1)
                    if updates >= args.q_warmup:
                        a_loss = -(log_alpha * (entropy + entropy_target)).mean()
                        a_opt.zero_grad()
                        a_loss.backward()
                        a_opt.step()
                    updates += 1
                    if updates % 64 == 0:
                        tq1.load_state_dict(q1.state_dict())
                        tq2.load_state_dict(q2.state_dict())
                writer.add_scalar("Losses/Policy", p_loss.item(), episode)
                writer.add_scalar("Losses/Q1", q1_loss.item(), episode)
                writer.add_scalar("Losses/Entropy", entropy.mean().item(), episode)
                writer.add_scalar("Losses/LogAlpha", log_alpha.item(), episode)
                writer.add_scalar("Losses/Value", v_next.mean().item(), episode)

            if len(window["steps"]) >= 32:
                for m in METRICS:
                    writer.add_scalar(f"Perf/{m}", float(np.nanmean(window[m])), episode)
                print(f"[driver] ep {episode} upd {updates} buf {len(replay)} | 2D {np.nanmean(window['explored_rate']):.3f} "
                      f"3D {np.nanmean(window['coverage_3d']):.3f} time {np.nanmean(window['time_s']):.0f}s "
                      f"turn {np.rad2deg(np.nanmean(window['turn_total_rad'])):.0f}deg R {np.nanmean(window['episode_reward']):.1f} "
                      f"| {time.time() - t_start:.0f}s", flush=True)
                window = {m: [] for m in METRICS}

            if episode // 32 != (episode - len(done)) // 32:
                torch.save({"policy_model": policy.state_dict(), "q_net1_model": q1.state_dict(),
                            "q_net2_model": q2.state_dict(), "log_alpha": log_alpha.detach(),
                            "policy_optimizer": p_opt.state_dict(), "q_net1_optimizer": q1_opt.state_dict(),
                            "q_net2_optimizer": q2_opt.state_dict(), "episode": episode, "updates": updates},
                           ck_path)
                if episode // 500 != (episode - len(done)) // 500:
                    torch.save({"policy_model": policy.state_dict(), "episode": episode},
                               os.path.join(model_path, f"policy_ep{episode // 500 * 500}.pth"))
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
        csv_f.close()


if __name__ == "__main__":
    main()
