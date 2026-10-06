"""How much of a machine's surface (tools/urdf_to_scene.py *_gt.npz) did a recorded run scan?

    python tools/machine_coverage.py assets/excavator/ix35e  level=logs/lidar_scans/<runA>  posture=logs/lidar_scans/<runB>
        [--voxel 0.05] [--out figures/excavator_coverage]

GT surface sampled every voxel/2 and split into elements = (voxel, facing direction: +-x, +-y, +-z of the triangle
normal), so the two faces of a thin plate (canopy roof, bucket wall) are separate. An element is covered when a LiDAR
return lies within one voxel (26-neighbourhood) AND the ray came from the side the face points to. Coverage is split by surface orientation
(up-facing = tops, down-facing = undersides, side) and reported over time. Writes <out>.json and <out>.png
(GT points coloured covered / missed, two views per run, plus coverage-vs-time).
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as R

OFFS = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])


def keys_of(ijk):
    return (ijk[:, 0].astype(np.int64) << 42) + (ijk[:, 1].astype(np.int64) << 21) + ijk[:, 2].astype(np.int64)


def gt_samples(prefix, spacing):
    d = np.load(prefix + "_gt.npz")
    v, t = d["verts"].astype(np.float64), d["tris"]
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    nrm = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(nrm, axis=1)
    unit = nrm / (2 * area[:, None] + 1e-12)
    n = np.maximum(1, np.round(area / spacing**2)).astype(int)
    idx = np.repeat(np.arange(len(a)), n)
    r1, r2 = np.random.default_rng(0).random((2, len(idx), 1))
    s = np.sqrt(r1)
    p = (1 - s) * a[idx] + s * (1 - r2) * b[idx] + s * r2 * c[idx]
    return p, unit[idx]


def load_scans(d):
    out = []
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        z = np.load(f)
        out.append((float(z["stamp"]), R.from_quat(z["quat_w"]).apply(z["points"]) + z["pos_w"], z["pos_w"]))
    t0 = out[0][0]
    return [(t - t0, p, o) for t, p, o in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("machine")
    ap.add_argument("runs", nargs="+", help="label=scan_dir")
    ap.add_argument("--voxel", type=float, default=0.05)
    ap.add_argument("--out", default="figures/excavator_coverage")
    args = ap.parse_args()
    v = args.voxel
    pts, nrm = gt_samples(args.machine, v / 2)
    origin = pts.min(0) - 1.0
    ijk = np.floor((pts - origin) / v).astype(np.int64)
    ax = np.abs(nrm).argmax(1)
    dirn = ax * 2 + (np.take_along_axis(nrm, ax[:, None], 1)[:, 0] < 0)  # 0:+x 1:-x 2:+y 3:-y 4:+z 5:-z
    gkeys, first = np.unique(keys_of(ijk) * 8 + dirn, return_index=True)
    gdir, gpts, gnz = dirn[first], pts[first], nrm[first, 2]
    lo, hi = pts.min(0) - 0.3, pts.max(0) + 0.3
    kind = np.where(gnz > 0.5, "up", np.where(gnz < -0.5, "down", "side"))
    print(f"GT: {len(gkeys):,} surface elements ({v} m voxels x facing direction)")
    res, covs = {}, {}
    for spec in args.runs:
        label, d = spec.split("=", 1)
        scans = load_scans(d)
        curve = []
        cov = np.zeros(len(gkeys), bool)
        for i, (t, p, o) in enumerate(scans):
            ok = np.isfinite(p).all(1)
            p = p[ok]
            p = p[((p >= lo) & (p <= hi)).all(1)]
            if len(p):
                w = np.asarray(o)[None] - p  # towards the sensor
                facing = [w[:, a] > 0 if s_ == 0 else w[:, a] < 0 for a in range(3) for s_ in (0, 1)]  # per dir
                base = np.floor((p - origin) / v).astype(np.int64)
                for off in OFFS:
                    k = keys_of(base + off) * 8
                    for dcode in range(6):
                        m = facing[dcode]
                        if not m.any():
                            continue
                        kk = k[m] + dcode
                        pos = np.searchsorted(gkeys, kk).clip(max=len(gkeys) - 1)
                        hit = gkeys[pos] == kk
                        cov[pos[hit]] = True
            if i % 10 == 0 or i == len(scans) - 1:
                curve.append((t, float(cov.mean())))
        covs[label] = cov
        res[label] = {"coverage": float(cov.mean()), "duration_s": scans[-1][0], "scans": len(scans),
                      **{f"coverage_{k}": float(cov[kind == k].mean()) for k in ("up", "side", "down")},
                      "curve": curve}
        print(f"{label:10s} {cov.mean():6.1%} | up {res[label]['coverage_up']:6.1%} side {res[label]['coverage_side']:6.1%} "
              f"down {res[label]['coverage_down']:6.1%} | {len(scans)} scans, {scans[-1][0]:.0f} s")
    res["_gt"] = {"voxels": int(len(gkeys)), "voxel_m": v, **{f"n_{k}": int((kind == k).sum()) for k in ("up", "side", "down")}}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out + ".json", "w") as f:
        json.dump(res, f, indent=1)
    render(args.out + ".png", gpts, covs, res)


def render(path, gpts, covs, res):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import make_figures as mf

    plt = mf.plt
    labels = list(covs)
    fig = plt.figure(figsize=(16, 4.6 * len(labels)), dpi=110, facecolor=mf.SURF)
    c = gpts.mean(0)
    for r, lab in enumerate(labels):
        cov = covs[lab]
        for j, (elev, azim) in enumerate(((22, -60), (22, 120))):
            ax = fig.add_subplot(len(labels), 3, 3 * r + j + 1, projection="3d")
            sel = np.random.default_rng(1).random(len(gpts)) < min(1.0, 60000 / len(gpts))
            p, k = gpts[sel] - c, cov[sel]
            ax.scatter(p[~k, 0], p[~k, 1], p[~k, 2], s=0.4, c="#e34948", depthshade=False)
            ax.scatter(p[k, 0], p[k, 1], p[k, 2], s=0.4, c="#1baf7a", depthshade=False)
            ax.view_init(elev, azim)
            ax.set_box_aspect(np.ptp(p, 0))
            ax.set_axis_off()
            if j == 0:
                q = res[lab]
                ax.set_title(f"{lab}: {100 * q['coverage']:.1f}%  (윗면 {100 * q['coverage_up']:.0f}% · 옆면 "
                             f"{100 * q['coverage_side']:.0f}% · 아랫면 {100 * q['coverage_down']:.0f}%)",
                             loc="left", fontsize=13, color=mf.INK)
        ax = fig.add_subplot(len(labels), 3, 3 * r + 3)
        for lab2 in labels:
            t, y = np.array(res[lab2]["curve"]).T
            ax.plot(t, 100 * y, lw=2.5 if lab2 == lab else 1.2, color="#1baf7a" if lab2 == lab else "#bdbcb6")
        ax.set_xlabel("시간 [s]", color=mf.INK2)
        ax.set_ylabel("굴착기 표면 커버리지 [%]", color=mf.INK2)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.text(0.01, 0.995, "초록 = 스캔됨, 빨강 = 놓침 (5 cm 복셀, 정답 메시 기준)", va="top", fontsize=12, color=mf.INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, facecolor=mf.SURF)


if __name__ == "__main__":
    main()
