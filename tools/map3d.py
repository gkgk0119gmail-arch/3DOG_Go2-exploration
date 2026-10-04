"""Build the 3D map of an exploration run from saved VLP-16 scans and score it against the warehouse ground truth.

Usage:
    GO2_SAVE_SCANS=1 GO2_EXPLORE_MAX_S=240 python scripts/rl.py play --task Go2-Warehouse-Explore-Play --num_envs 1
    python tools/map3d.py logs/lidar_scans/<run> [--exploration logs/exploration/<run>] [--voxel 0.1]

3D map  : every return of every scan in the world frame, voxelized (default 0.1 m) -> map3d.ply (+ hit counts)
GT      : surface of the baked warehouse mesh (assets/warehouse/warehouse_raycast.usdc) sampled every ~5 cm, voxelized
Metrics : 3D coverage   = GT surface voxels with a scanned voxel within 1 voxel (26-neighbourhood)
          accuracy      = distance from scanned voxel centres to the nearest GT surface sample (cloud-to-cloud)
          per height band, and 2D vs 3D coverage over time
Figures : map3d_top.png (height of the 3D map), map3d_missed.png (GT surface missed, by height band)
Viewer  : map3d_viewer.npz -> python tools/map3d_viewer.py <out dir> (interactive 3D page)
"""

import argparse
import csv
import glob
import json
import os

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

SIM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BANDS = [(0.0, 0.5, "floor 0-0.5 m"), (0.5, 2.0, "0.5-2 m"), (2.0, 4.0, "2-4 m"), (4.0, 6.0, "4-6 m"),
         (6.0, 8.5, "6-8.5 m"), (8.5, 10.0, "ceiling >8.5 m")]
OFFS = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])


def gt_surface(mesh_usd, bounds, spacing):
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(mesh_usd)  # keep a reference: the prim dies with a temporary stage
    m = UsdGeom.Mesh(stage.GetPrimAtPath("/RaycastMesh/mesh"))
    pts = np.asarray(m.GetPointsAttr().Get(), np.float64)
    tri = np.asarray(m.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
    a, b, c = pts[tri[:, 0]], pts[tri[:, 1]], pts[tri[:, 2]]
    # keep triangles touching the interior box
    lo, hi = np.minimum(np.minimum(a, b), c), np.maximum(np.maximum(a, b), c)
    keep = (hi[:, 0] >= bounds[0]) & (lo[:, 0] <= bounds[2]) & (hi[:, 1] >= bounds[1]) & (lo[:, 1] <= bounds[3])
    a, b, c = a[keep], b[keep], c[keep]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    n = np.maximum(1, np.round(area / spacing**2)).astype(int)
    idx = np.repeat(np.arange(len(a)), n)
    r1, r2 = np.random.default_rng(0).random((2, len(idx), 1))
    s = np.sqrt(r1)
    p = (1 - s) * a[idx] + s * (1 - r2) * b[idx] + s * r2 * c[idx]
    inside = (p[:, 0] >= bounds[0]) & (p[:, 0] <= bounds[2]) & (p[:, 1] >= bounds[1]) & (p[:, 1] <= bounds[3])
    return p[inside]


def keys_of(ijk):
    return (ijk[:, 0].astype(np.int64) << 42) + (ijk[:, 1].astype(np.int64) << 21) + ijk[:, 2].astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scans")
    ap.add_argument("--exploration", default=None, help="exploration run dir (2D coverage log) for the time plot")
    ap.add_argument("--voxel", type=float, default=0.1)
    ap.add_argument("--bounds", type=float, nargs=4, default=(-26.3, -23.6, 5.3, 30.4))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or args.scans.rstrip("/") + "_map3d"
    os.makedirs(out, exist_ok=True)
    v, b = args.voxel, args.bounds
    origin = np.array([b[0] - 1, b[1] - 1, -1.0])

    # --- 3D map from scans --------------------------------------------------------------------------
    files = sorted(glob.glob(os.path.join(args.scans, "*.npz")))
    hits, stamps, per_scan_keys = {}, [], []
    traj = []
    for f in files:
        d = np.load(f)
        pw = R.from_quat(d["quat_w"]).apply(d["points"]) + d["pos_w"]
        ijk = np.floor((pw - origin) / v).astype(np.int64)
        k = keys_of(ijk)
        per_scan_keys.append(np.unique(k))
        stamps.append(float(d["stamp"]))
        traj.append(d["pos_w"])
    t0 = stamps[0]
    t_rel = np.array(stamps) - t0
    all_keys, counts = np.unique(np.concatenate(per_scan_keys), return_counts=True)
    ijk_map = np.stack([all_keys >> 42, (all_keys >> 21) & ((1 << 21) - 1), all_keys & ((1 << 21) - 1)], 1)
    centers = origin + (ijk_map + 0.5) * v
    inside = (centers[:, 0] >= b[0]) & (centers[:, 0] <= b[2]) & (centers[:, 1] >= b[1]) & (centers[:, 1] <= b[3])
    print(f"scans {len(files)} over {t_rel[-1]:.0f} s -> {len(all_keys):,} occupied voxels ({v} m), "
          f"{inside.sum():,} inside the warehouse")

    # --- ground truth surface ------------------------------------------------------------------------
    gt = gt_surface(os.path.join(SIM, "assets", "warehouse", "warehouse_raycast.usdc"), b, v / 2)
    gt_ijk = np.floor((gt - origin) / v).astype(np.int64)
    gt_keys, gt_first = np.unique(keys_of(gt_ijk), return_index=True)
    gt_ijk = gt_ijk[gt_first]
    gt_z = origin[2] + (gt_ijk[:, 2] + 0.5) * v
    print(f"GT surface: {len(gt):,} samples -> {len(gt_keys):,} surface voxels")

    def covered_mask(scanned_keys):
        scanned_keys = np.sort(scanned_keys)
        cov = np.zeros(len(gt_keys), bool)
        for o in OFFS:
            cov |= np.isin(keys_of(gt_ijk + o), scanned_keys, assume_unique=False)
        return cov

    cov = covered_mask(all_keys)
    tree = cKDTree(gt)
    dist, _ = tree.query(centers[inside], k=1, workers=-1)
    res = {"voxel_m": v, "scans": len(files), "duration_s": float(t_rel[-1]),
           "coverage_3d": float(cov.mean()),
           "c2c_mean_m": float(dist.mean()), "c2c_median_m": float(np.median(dist)),
           "c2c_p95_m": float(np.quantile(dist, 0.95)), "spurious_frac_gt_0.2m": float(np.mean(dist > 0.2)),
           "bands": {}}
    print(f"\n3D coverage {cov.mean():.1%} | accuracy (scan->GT) mean {100 * dist.mean():.1f} cm, median "
          f"{100 * np.median(dist):.1f} cm, p95 {100 * np.quantile(dist, 0.95):.1f} cm, >20 cm {np.mean(dist > 0.2):.2%}")
    print(f"{'height band':16s} {'GT voxels':>10s} {'covered':>8s}")
    for lo, hi, name in BANDS:
        m = (gt_z >= lo) & (gt_z < hi)
        res["bands"][name] = {"gt_voxels": int(m.sum()), "coverage": float(cov[m].mean()) if m.any() else None}
        print(f"{name:16s} {m.sum():>10,} {cov[m].mean() if m.any() else float('nan'):>8.1%}")

    # --- coverage over time (3D) vs 2D log -----------------------------------------------------------
    checkpoints = np.unique(np.linspace(0, len(files) - 1, 25).astype(int))
    curve = []
    for i in checkpoints:
        c = covered_mask(np.unique(np.concatenate(per_scan_keys[: i + 1])))
        curve.append((float(t_rel[i]), float(c.mean())))
    res["coverage_3d_curve"] = curve
    cov2d = []
    if args.exploration and os.path.exists(os.path.join(args.exploration, "exploration_log.csv")):
        with open(os.path.join(args.exploration, "exploration_log.csv")) as f:
            cov2d = [(float(r["time_s"]), float(r["coverage"])) for r in csv.DictReader(f)]
    print("\ntime   3D coverage" + ("   2D coverage" if cov2d else ""))
    for t, c in curve[:: max(1, len(curve) // 10)] + [curve[-1]]:
        line = f"{t:5.0f}s  {c:6.1%}"
        if cov2d:
            j = min(range(len(cov2d)), key=lambda k: abs(cov2d[k][0] - t))
            line += f"        {cov2d[j][1]:6.1%}"
        print(line)

    # --- outputs -------------------------------------------------------------------------------------
    with open(os.path.join(out, "map3d_metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    c = centers[inside]
    z = np.clip((c[:, 2] - 0) / 9.0, 0, 1)
    col = (np.stack([z, 1 - np.abs(z - 0.5) * 2, 1 - z], 1) * 255).astype(np.uint8)
    with open(os.path.join(out, "map3d.ply"), "w") as f:
        f.write(f"ply\nformat ascii 1.0\nelement vertex {len(c)}\nproperty float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        np.savetxt(f, np.hstack([c, col]), fmt="%.3f %.3f %.3f %d %d %d")
    render(out, centers[inside], gt_ijk, gt_z, cov, origin, v, b, np.array(traj))
    # everything the interactive viewer (tools/map3d_viewer.py) needs: scanned voxels, GT voxels + covered flag, path
    np.savez_compressed(os.path.join(out, "map3d_viewer.npz"), scanned=c.astype(np.float32),
                        gt=(origin + (gt_ijk + 0.5) * v).astype(np.float32), covered=cov,
                        traj=np.array(traj, np.float32), voxel=v)
    print(f"\nwrote {out}/map3d.ply, map3d_top.png, map3d_missed.png, map3d_metrics.json")


def render(out, centers, gt_ijk, gt_z, cov, origin, v, b, traj):
    from PIL import Image, ImageDraw

    s = 4  # px per 0.1 m column -> scale image
    nx, ny = int((b[2] - b[0]) / v) + 1, int((b[3] - b[1]) / v) + 1
    # top view: max height of the 3D map per column
    hmax = np.full((ny, nx), -1.0)
    ix = ((centers[:, 0] - b[0]) / v).astype(int).clip(0, nx - 1)
    iy = ((centers[:, 1] - b[1]) / v).astype(int).clip(0, ny - 1)
    np.maximum.at(hmax, (iy, ix), centers[:, 2])
    img = np.zeros((ny, nx, 3), np.uint8) + 40
    m = hmax >= 0
    t = np.clip(hmax[m] / 9.0, 0, 1)
    img[m] = (np.stack([t, 1 - np.abs(t - 0.5) * 2, 1 - t], 1) * 255).astype(np.uint8)
    im = Image.fromarray(np.flipud(img)).resize((nx * 2, ny * 2), Image.NEAREST)
    dr = ImageDraw.Draw(im)
    pts = [((p[0] - b[0]) / v * 2, (b[3] - p[1]) / v * 2) for p in traj[::3]]
    if len(pts) > 1:
        dr.line(pts, fill=(255, 255, 255), width=2)
    dr.text((6, 6), "3D map: max height per column (blue 0 m -> red 9 m), white = robot path", fill=(255, 255, 255))
    im.save(os.path.join(out, "map3d_top.png"))

    # missed GT surface per height band (top view, one panel per band)
    gcx = ((origin[0] + (gt_ijk[:, 0] + 0.5) * v - b[0]) / v).astype(int).clip(0, nx - 1)
    gcy = ((origin[1] + (gt_ijk[:, 1] + 0.5) * v - b[1]) / v).astype(int).clip(0, ny - 1)
    panels = []
    for lo, hi, name in BANDS:
        mm = (gt_z >= lo) & (gt_z < hi)
        tot = np.zeros((ny, nx))
        hit = np.zeros((ny, nx))
        np.add.at(tot, (gcy[mm], gcx[mm]), 1)
        np.add.at(hit, (gcy[mm], gcx[mm]), cov[mm])
        p = np.zeros((ny, nx, 3), np.uint8) + 30
        has = tot > 0
        frac = np.where(has, hit / np.maximum(tot, 1), 0)
        p[has] = np.stack([(1 - frac[has]) * 255, frac[has] * 200, np.zeros(has.sum())], 1).astype(np.uint8)
        pim = Image.fromarray(np.flipud(p)).resize((nx, ny), Image.NEAREST)
        d = ImageDraw.Draw(pim)
        cv = cov[mm].mean() if mm.any() else float("nan")
        d.text((4, 4), f"{name}: {cv:.0%}", fill=(255, 255, 255))
        panels.append(pim)
    W = sum(p.width for p in panels) + 4 * (len(panels) - 1)
    sheet = Image.new("RGB", (W, panels[0].height + 18), (0, 0, 0))
    x = 0
    for p in panels:
        sheet.paste(p, (x, 18))
        x += p.width + 4
    ImageDraw.Draw(sheet).text((4, 3), "GT surface per height band: green = scanned, red = missed", fill=(255, 255, 255))
    sheet.save(os.path.join(out, "map3d_missed.png"))


if __name__ == "__main__":
    main()
