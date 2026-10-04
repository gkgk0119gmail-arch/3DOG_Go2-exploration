"""Bake every mesh of a USD scene into one triangulated world-space mesh (for Isaac Lab's single-mesh RayCaster).

Usage:
    python merge_usd_meshes.py scene.usd out.usdc

The output holds one invisible ``/RaycastMesh/mesh`` prim, so it can be spawned next to the visual scene and used
only as a ray-casting target.
"""

import argparse

import numpy as np
from pxr import Usd, UsdGeom, Vt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    args = ap.parse_args()

    src = Usd.Stage.Open(args.input, Usd.Stage.LoadAll)
    mpu = UsdGeom.GetStageMetersPerUnit(src)
    cache = UsdGeom.XformCache()
    all_pts, all_tris, offset, n_mesh = [], [], 0, 0
    for prim in src.Traverse(Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts, counts, idx = mesh.GetPointsAttr().Get(), mesh.GetFaceVertexCountsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get()
        if not pts or not counts:
            continue
        pts = np.asarray(pts, dtype=np.float64)
        M = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=np.float64)
        pts = (np.c_[pts, np.ones(len(pts))] @ M)[:, :3] * mpu
        counts, idx = np.asarray(counts), np.asarray(idx)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        tris = [np.stack([np.full(c - 2, idx[s]), idx[s + 1:s + c - 1], idx[s + 2:s + c]], 1)
                for s, c in zip(starts, counts) if c >= 3]
        if not tris:
            continue
        all_pts.append(pts.astype(np.float32))
        all_tris.append(np.concatenate(tris) + offset)
        offset += len(pts)
        n_mesh += 1

    pts, tris = np.concatenate(all_pts), np.concatenate(all_tris).astype(np.int32)
    out = Usd.Stage.CreateNew(args.output)
    UsdGeom.SetStageUpAxis(out, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(out, 1.0)
    root = UsdGeom.Xform.Define(out, "/RaycastMesh")
    out.SetDefaultPrim(root.GetPrim())
    m = UsdGeom.Mesh.Define(out, "/RaycastMesh/mesh")
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(pts))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(tris), 3, np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(tris.reshape(-1)))
    m.CreateVisibilityAttr(UsdGeom.Tokens.invisible)
    out.GetRootLayer().Save()
    print(f"merged {n_mesh} meshes -> {len(pts)} vertices, {len(tris)} triangles, bounds "
          f"{pts.min(0).round(2)} {pts.max(0).round(2)} -> {args.output}")


if __name__ == "__main__":
    main()
