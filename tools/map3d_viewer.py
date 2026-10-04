"""Interactive 3D page for a map3d.py result: scanned 3D map, GT surface scanned vs missed, coverage over time.

    python tools/map3d.py logs/lidar_scans/<run> --exploration logs/exploration/<run>   # writes map3d_viewer.npz
    python tools/map3d_viewer.py logs/lidar_scans/<run>_map3d [--exploration logs/exploration/<run>] [--grid 0.2]

Writes <dir>/map3d_viewer.html (self-contained, three.js from a CDN). Voxels are re-binned to --grid for the browser;
a coarse GT voxel counts as scanned when at least half of its fine (0.1 m) voxels were covered.
"""

import argparse
import base64
import csv
import json
import os

import numpy as np


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def binned(points, origin, g):
    ijk = np.floor((points - origin) / g).astype(np.int64)
    keys, inv = np.unique(ijk[:, 0] * 10**8 + ijk[:, 1] * 10**4 + ijk[:, 2], return_inverse=True)
    return keys, inv.ravel()


def keys_to_ijk(keys):
    return np.stack([keys // 10**8, (keys // 10**4) % 10**4, keys % 10**4], 1).astype(np.int16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--exploration", default=None)
    ap.add_argument("--grid", type=float, default=0.2)
    args = ap.parse_args()
    d = np.load(os.path.join(args.dir, "map3d_viewer.npz"))
    met = json.load(open(os.path.join(args.dir, "map3d_metrics.json")))
    g = args.grid
    scanned, gt, cov, traj = d["scanned"], d["gt"], d["covered"], d["traj"]
    origin = np.minimum(scanned.min(0), gt.min(0)) - g

    sk, _ = binned(scanned, origin, g)
    gk, ginv = binned(gt, origin, g)
    frac = np.bincount(ginv, weights=cov.astype(float)) / np.bincount(ginv)
    gij = keys_to_ijk(gk)
    hit = frac >= 0.5

    cov2d = []
    if args.exploration and os.path.exists(os.path.join(args.exploration, "exploration_log.csv")):
        with open(os.path.join(args.exploration, "exploration_log.csv")) as f:
            cov2d = [(float(r["time_s"]), float(r["coverage"])) for r in csv.DictReader(f)]
        step = max(1, len(cov2d) // 120)
        cov2d = cov2d[::step] + [cov2d[-1]]
    data = {
        "grid": g, "origin": origin.tolist(),
        "scanned": b64(keys_to_ijk(sk)), "gtHit": b64(gij[hit]), "gtMiss": b64(gij[~hit]),
        "traj": b64(traj[:: max(1, len(traj) // 800), :3].astype(np.float32)),
        "metrics": {k: met[k] for k in ("voxel_m", "scans", "duration_s", "coverage_3d", "c2c_mean_m", "bands",
                                        "coverage_3d_curve")},
        "cov2d": cov2d,
    }
    print(f"scanned {len(sk):,} | GT {len(gk):,} ({hit.mean():.1%} scanned at {g} m) | path {len(traj)}")
    tpl = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "map3d_viewer_template.html")).read()
    out = os.path.join(args.dir, "map3d_viewer.html")
    with open(out, "w") as f:
        f.write(tpl.replace("/*__DATA__*/null", json.dumps(data)))
    print(f"wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
