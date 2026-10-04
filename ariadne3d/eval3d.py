"""Greedy evaluation on held-out maps (episodes >= 5600, never used in training).

    python eval3d.py pretrained/ariadne_ral2024.pth model/go2_turn_3d/checkpoint.pth [--n 60] [--tilt 15]
"""

import argparse
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

EVAL_START = 5600
KEYS = ["explored_rate", "done_2d", "time_2d_s", "cov3d_at_2d", "time_s", "coverage_3d", "wall_cov", "ceiling_cov",
        "turn_total_rad", "turn_time_s", "travel_dist", "steps"]


def _run(args):
    path, ep, tilt, world, greedy = args
    torch.set_num_threads(1)
    from nets import load_pretrained
    from worker3d import Worker3D

    policy, *_ = load_pretrained(path)
    policy.eval()
    if world == "proc":
        from procwarehouse import ProcWarehouseEnv3D

        w = Worker3D(0, policy, ep, greedy=greedy, tilt_deg=tilt, env_cls=ProcWarehouseEnv3D, record=False)
    elif world == "warehouse":
        from warehouse_env import WarehouseEnv3D

        w = Worker3D(0, policy, ep, greedy=greedy, tilt_deg=tilt, env_cls=WarehouseEnv3D,
                     record=False)
    else:
        w = Worker3D(0, policy, ep, greedy=greedy, tilt_deg=tilt, record=False)
    try:
        w.run_episode()
    except Exception as e:  # noqa: BLE001
        return path, ep, {"error": repr(e)}
    return path, ep, w.perf_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--tilt", type=float, default=15.0)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--out", default="eval")
    ap.add_argument("--world", default="maps", choices=["maps", "warehouse", "proc"],
                    help="maps: held-out training-style maps; warehouse: the Isaac warehouse, random starts")
    ap.add_argument("--node_res", type=float, default=None, help="graph node spacing [m] (training: 4)")
    ap.add_argument("--sample", action="store_true", help="sample actions (as in training) instead of argmax")
    ap.add_argument("--node_pad", type=int, default=None)
    ap.add_argument("--max_step", type=int, default=None)
    args = ap.parse_args()
    if args.node_res:
        os.environ["ARIADNE_NODE_RES"] = str(args.node_res)
        if args.node_res < 4:
            os.environ["ARIADNE_NODE_PAD"] = "500"  # warehouse at 2 m: <= ~360 nodes; attention cost ~ pad^2
    if args.node_pad:
        os.environ["ARIADNE_NODE_PAD"] = str(args.node_pad)
    if args.max_step:
        os.environ["ARIADNE_MAX_STEP"] = str(args.max_step)
    jobs = [(c, EVAL_START + i, args.tilt, args.world, not args.sample) for c in args.checkpoints for i in range(args.n)]
    res = {c: {} for c in args.checkpoints}
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as ex:
        for c, ep, m in ex.map(_run, jobs):
            res[c][ep] = m
    os.makedirs(args.out, exist_ok=True)
    summary = {}
    common = set.intersection(*[{e for e, m in r.items() if "error" not in m} for r in res.values()])
    print(f"{len(common)} maps evaluated by every checkpoint (tilt {args.tilt} deg, world {args.world}, "
          f"node spacing {os.environ.get('ARIADNE_NODE_RES', '4.0')} m, {'sampled' if args.sample else 'greedy'})")
    for c, r in res.items():
        rows = [r[e] for e in sorted(common)]
        s = {k: float(np.nanmean([x[k] for x in rows])) for k in KEYS}
        s["turn_total_deg"] = float(np.rad2deg(s.pop("turn_total_rad")))
        summary[c] = s
        print(f"\n{c}")
        print(f"  2D done {s['done_2d']:.0%} (explored {s['explored_rate']:.3f}) at {s['time_2d_s']:.0f} s, 3D then {s['cov3d_at_2d']:.3f}")
        print(f"  end: {s['time_s']:.0f} s, 3D {s['coverage_3d']:.3f} (wall {s['wall_cov']:.3f}, ceiling {s['ceiling_cov']:.3f})")
        print(f"  turning {s['turn_total_deg']:.0f} deg = {s['turn_time_s']:.0f} s, distance {s['travel_dist']:.0f} m, {s['steps']:.0f} steps")
    tag = f"{args.world}_tilt{args.tilt:g}" + (f"_res{args.node_res:g}" if args.node_res else "") + ("_sample" if args.sample else "")
    with open(os.path.join(args.out, f"eval_{tag}.json"), "w") as f:
        json.dump({"summary": summary, "per_map": {c: {str(k): v for k, v in r.items()} for c, r in res.items()}}, f, indent=1)


if __name__ == "__main__":
    main()
