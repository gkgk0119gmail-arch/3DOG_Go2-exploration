"""Observable upper bound per scene and LiDAR mount tilt (stage 2).

    python scripts/pose_inspection/compute_upper_bound.py [--scenes A_warehouse B_construction] [--tilts 0 15]

Writes <scene dir>/observable_tilt<t>.npz (Cs normaliser) and results/pose_inspection/stage2_upper_bound.json.
"""

import argparse
import json
import os
import sys
import time

SIM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SIM)

from go2_lab.pose_inspection.config import load_config, sim_path  # noqa: E402
from go2_lab.pose_inspection.scene_builder import load_scene  # noqa: E402
from go2_lab.pose_inspection.upper_bound import compute  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["A_warehouse", "B_construction"])
    ap.add_argument("--tilts", nargs="+", type=float, default=[0.0, 15.0])
    ap.add_argument("--config", default="base.yaml")
    ap.add_argument("--out", default="results/pose_inspection/stage2_upper_bound.json")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = {}
    for s in args.scenes:
        meta = load_scene(f"assets/pose_inspection/{s}")
        out[s] = {}
        for tilt in args.tilts:
            t0 = time.time()
            print(f"[{s}] tilt {tilt:g} deg")
            r = compute(cfg, meta, tilt)
            r["seconds"] = round(time.time() - t0, 1)
            out[s][f"tilt{tilt:g}"] = r
            print(f"    Co upper bound: posture {r['co_ub_posture']:.3f}, level {r['co_ub_level']:.3f} | observable volume "
                  f"{r['observable_volume_fraction']:.3f} | {r['seconds']} s")
            for name, v in r["per_object"].items():
                print(f"      {name:16s} posture {v['posture']:.3f} level {v['level']:.3f}")
    path = sim_path(args.out)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
