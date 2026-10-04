"""Evaluate an exploration run: 2D map quality against the ground-truth map, and exploration efficiency.

Usage:
    python eval_exploration.py logs/exploration/<run> [--gt assets/maps/full_warehouse.npz]
                               [--bounds -26.3 -23.6 5.3 30.4]

Map metrics (inside ``bounds``, on the run's 0.4 m grid; GT resampled from 0.1 m):
    occupied precision  mapped-occupied cells within 1 cell of a GT obstacle
    occupied recall     GT obstacle cells within 1 cell of a mapped obstacle
    false-free rate     mapped-free cells that are GT obstacles (strict) -- unsafe errors
    coverage            GT free cells that the map knows (free or occupied)
Efficiency (from exploration_log.csv): time and distance to 50/75/90/95/99 % coverage.
"""

import argparse
import csv
import os

import numpy as np
from scipy import ndimage

FREE, OCCUPIED, UNKNOWN = 0, 100, -1


def gt_on_grid(gt_path, origin, cell, shape):
    """Resample the GT map (0.1 m, indexed [x, y], 0 = occupied, 255 = free, 127 = unknown) onto the run grid."""
    d = np.load(gt_path)
    g, go, gr = d["grid"], d["origin"], float(d["resolution"])
    ny, nx = shape
    ys = origin[1] + (np.arange(ny) + 0.5) * cell
    xs = origin[0] + (np.arange(nx) + 0.5) * cell
    k = int(round(cell / gr))
    occ = np.zeros(shape, bool)
    free = np.zeros(shape, bool)
    for dy in range(k):
        for dx in range(k):
            ix = np.floor((xs - cell / 2 + (dx + 0.5) * gr - go[0]) / gr).astype(int)
            iy = np.floor((ys - cell / 2 + (dy + 0.5) * gr - go[1]) / gr).astype(int)
            okx = (ix >= 0) & (ix < g.shape[0])
            oky = (iy >= 0) & (iy < g.shape[1])
            sub = np.full(shape, 127, np.uint8)
            sub[np.ix_(oky, okx)] = g[np.ix_(ix[okx], iy[oky])].T
            occ_count = (sub == 0).astype(int) if dy == 0 and dx == 0 else occ_count + (sub == 0)
            free |= sub == 255
    occ = occ_count >= 2  # >= 2 of 16 sub-cells: ignore single-sample speckle
    return occ, free & ~occ, xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--gt", default=os.path.join(os.path.dirname(__file__), "..", "assets", "maps", "full_warehouse.npz"))
    ap.add_argument("--bounds", type=float, nargs=4, default=(-26.3, -23.6, 5.3, 30.4))
    args = ap.parse_args()

    if not os.path.exists(os.path.join(args.run_dir, "final_map.npz")):
        print("no final_map.npz (older run): efficiency only")
        return efficiency(args.run_dir)
    run = np.load(os.path.join(args.run_dir, "final_map.npz"))
    grid, origin, cell = run["grid"], run["origin"], float(run["cell"])
    gt_occ, gt_free, xs, ys = gt_on_grid(args.gt, origin, cell, grid.shape)
    b = args.bounds
    inside = ((ys >= b[1]) & (ys <= b[3]))[:, None] & ((xs >= b[0]) & (xs <= b[2]))[None, :]

    m_occ, m_free, m_known = grid == OCCUPIED, grid == FREE, grid != UNKNOWN
    near = lambda a: ndimage.binary_dilation(a, iterations=1)  # noqa: E731
    prec = (m_occ & near(gt_occ) & inside).sum() / max((m_occ & inside).sum(), 1)
    rec = (gt_occ & near(m_occ) & inside).sum() / max((gt_occ & inside).sum(), 1)
    false_free = (m_free & ndimage.binary_erosion(gt_occ) & inside).sum() / max((m_free & inside).sum(), 1)
    cov = (m_known & gt_free & inside).sum() / max((gt_free & inside).sum(), 1)
    print(f"map ({cell} m grid, inside {tuple(b)}):")
    print(f"  occupied precision {prec:.1%}   occupied recall {rec:.1%}   false-free {false_free:.2%}   coverage {cov:.1%}")

    efficiency(args.run_dir)
    diff_image(args, grid, m_free, m_occ, gt_occ, inside, near)


def efficiency(run_dir):
    with open(os.path.join(run_dir, "exploration_log.csv")) as f:
        rows = list(csv.DictReader(f))
    if rows:
        t = np.array([float(r["time_s"]) for r in rows])
        c = np.array([float(r["coverage"]) for r in rows])
        d = np.array([float(r["distance_m"]) for r in rows])
        cmax = c.max()
        out = []
        for m in (0.5, 0.75, 0.9, 0.95, 0.99):
            i = np.argmax(c >= m * cmax) if (c >= m * cmax).any() else None
            out.append(f"{int(100 * m)}%: {t[i]:.0f}s/{d[i]:.0f}m" if i is not None else f"{int(100 * m)}%: -")
        print(f"efficiency (relative to final coverage, end t={t[-1]:.0f}s, distance {d[-1]:.0f}m):")
        print("  " + "   ".join(out))



def diff_image(args, grid, m_free, m_occ, gt_occ, inside, near):
    try:
        from PIL import Image

        img = np.full(grid.shape + (3,), 150, np.uint8)
        img[m_free] = 255
        img[m_occ & near(gt_occ)] = (30, 30, 30)  # correct obstacle
        img[m_occ & ~near(gt_occ)] = (230, 120, 0)  # spurious obstacle
        img[gt_occ & ~near(m_occ) & inside] = (0, 140, 255)  # missed obstacle
        img[m_free & ndimage.binary_erosion(gt_occ)] = (255, 0, 0)  # false free
        img[~inside] //= 2
        Image.fromarray(np.flipud(img)).resize((grid.shape[1] * 4, grid.shape[0] * 4), Image.NEAREST).save(
            os.path.join(args.run_dir, "map_eval.png"))
        print(f"  diff image: {os.path.join(args.run_dir, 'map_eval.png')} "
              "(black ok, orange spurious, blue missed, red false-free)")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
