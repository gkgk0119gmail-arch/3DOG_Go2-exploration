"""Run an inspection experiment with the kinematic engine (no Isaac; development / quick sweeps).

    python scripts/pose_inspection/run_kinematic.py --exp exp/pose.yaml --scene B_construction [--seeds 0 1 2 3 4 5]

Results: <experiment.results_dir>/kinematic/<scene>/<exp name>/seed<k>.json
"""

import argparse
import os
import sys
import time

import numpy as np

SIM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SIM)

from go2_lab.pose_inspection.config import load_config, sim_path  # noqa: E402
from go2_lab.pose_inspection.kinematic import KinematicEngine  # noqa: E402
from go2_lab.pose_inspection.navgrid import NavGrid  # noqa: E402
from go2_lab.pose_inspection.scene_builder import load_scene  # noqa: E402
from go2_lab.pose_inspection.session import Session  # noqa: E402


def starts_for(cfg, meta, seeds):
    nav = NavGrid(os.path.join(meta["dir"], "grid.npz"), cfg.robot.radius).restrict_to_region(
        meta["robot_start"]["region"])
    return np.array([nav.sample_starts(meta["robot_start"]["region"], 1, s, meta["robot_start"]["min_clearance"])[0]
                     for s in seeds])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", default="exp/pose.yaml")
    ap.add_argument("--scene", default="B_construction")
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--set", nargs="*", default=[], help="dotted overrides key=value (YAML values)")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    import yaml

    over = {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in args.set)}
    cfg = load_config(args.exp, over)
    meta = load_scene(f"assets/pose_inspection/{args.scene}")
    seeds = args.seeds if args.seeds is not None else list(range(cfg.experiment.seed_base,
                                                                 cfg.experiment.seed_base + cfg.experiment.n_seeds))
    name = args.name or os.path.splitext(os.path.basename(args.exp))[0]
    out = sim_path(os.path.join(cfg.experiment.results_dir, "kinematic", args.scene, name))
    s = Session(cfg, meta, seeds, out, tag=name)
    eng = KinematicEngine(s, starts_for(cfg, meta, seeds))
    t0 = time.time()
    print(f"[{name} / {args.scene}] seeds {seeds}")
    T = eng.run()
    s.save({"engine": "kinematic"})
    print(f"done: sim {T:.0f} s in {time.time() - t0:.0f} s wall")
    for e in range(len(seeds)):
        r = s.summary(e)
        print(f"  seed {r['seed']}: Co {r['Co']:.3f} (vs UB {r['Co_vs_ub']:.3f}) Cs {r['Cs']:.3f} t_eff {r['t_eff']:.0f} "
              f"L {r['L_eff']:.0f} Npost {r['N_post']} timeout {r['timeout']}")


if __name__ == "__main__":
    main()
