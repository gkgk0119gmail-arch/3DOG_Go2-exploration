"""Top-down occupancy map of a USD scene: cells containing geometry within a height band are occupied.

Usage:
    python usd_occupancy_map.py scene.usd out_prefix [--res 0.1] [--zmin 0.05] [--zmax 1.0]

Writes <out_prefix>.png (white = free, black = occupied, grey = outside geometry bounds) and
<out_prefix>.npz (grid, origin, resolution).
"""

import argparse

import numpy as np
from pxr import Usd, UsdGeom, UsdPhysics


def sample_triangles(pts, tris, spacing):
    """Uniformly sample points on triangles, roughly one per spacing^2 of area (plus the vertices)."""
    a, b, c = pts[tris[:, 0]], pts[tris[:, 1]], pts[tris[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    n = np.clip((area / spacing**2).astype(int), 0, 2000)
    idx = np.repeat(np.arange(len(tris)), n)
    if len(idx) == 0:
        return pts
    r1, r2 = np.random.rand(len(idx), 1), np.random.rand(len(idx), 1)
    s = np.sqrt(r1)
    samples = (1 - s) * a[idx] + s * (1 - r2) * b[idx] + s * r2 * c[idx]
    return np.concatenate([pts, samples])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("usd")
    ap.add_argument("out")
    ap.add_argument("--res", type=float, default=0.1)
    ap.add_argument("--zmin", type=float, default=0.05)
    ap.add_argument("--zmax", type=float, default=1.0)
    args = ap.parse_args()

    stage = Usd.Stage.Open(args.usd, Usd.Stage.LoadAll)
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    cache = UsdGeom.XformCache()
    band, all_xy = [], []
    n_mesh = n_col = 0
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            n_col += 1
        if not prim.IsA(UsdGeom.Mesh) or UsdGeom.Imageable(prim).ComputeVisibility() == "invisible":
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        counts, indices = mesh.GetFaceVertexCountsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get()
        if not pts or not counts:
            continue
        n_mesh += 1
        pts = np.asarray(pts, dtype=np.float64)
        M = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=np.float64)
        pts = (np.c_[pts, np.ones(len(pts))] @ M)[:, :3] * mpu
        # fan-triangulate polygons
        counts, indices = np.asarray(counts), np.asarray(indices)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        tris = [np.stack([indices[s] * np.ones(c - 2, int), indices[s + 1 : s + c - 1], indices[s + 2 : s + c]], 1)
                for s, c in zip(starts, counts) if c >= 3]
        if not tris:
            continue
        samples = sample_triangles(pts, np.concatenate(tris), args.res / 2)
        all_xy.append(samples[:, :2])
        band.append(samples[(samples[:, 2] > args.zmin) & (samples[:, 2] < args.zmax), :2])

    all_xy, band = np.concatenate(all_xy), np.concatenate(band)
    lo, hi = all_xy.min(0), all_xy.max(0)
    shape = np.ceil((hi - lo) / args.res).astype(int) + 1
    grid = np.full(shape, 127, np.uint8)  # unknown
    known = np.floor((all_xy - lo) / args.res).astype(int)
    grid[known[:, 0], known[:, 1]] = 255  # free (some geometry, e.g. floor, below the band)
    occ = np.floor((band - lo) / args.res).astype(int)
    grid[occ[:, 0], occ[:, 1]] = 0  # occupied
    np.savez(args.out + ".npz", grid=grid, origin=lo, resolution=args.res)
    try:
        from PIL import Image

        Image.fromarray(np.flipud(grid.T)).save(args.out + ".png")  # x right, y up
    except ImportError:
        pass
    print(f"meshes={n_mesh} colliders={n_col} bounds x[{lo[0]:.1f},{hi[0]:.1f}] y[{lo[1]:.1f},{hi[1]:.1f}] "
          f"grid={shape.tolist()} occupied={np.mean(grid == 0):.1%} free={np.mean(grid == 255):.1%}")


if __name__ == "__main__":
    main()
