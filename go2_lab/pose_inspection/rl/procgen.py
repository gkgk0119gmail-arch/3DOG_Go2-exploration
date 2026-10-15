"""Procedural training scenes for the RL inspection policy (kinematic engine only, no USD).

The machines and clutter are the baked parts of the two hand-built scenes (assets/pose_inspection/{A,B}_*), cut out
once into a part library (local frame: footprint centre at the origin, unrotated). A scene places parts with random
position / heading in one of two layouts and writes the same files a built scene has (gt.npz, surface.npz, grid.npz,
scene.json), so ``Session`` loads it unchanged:

* ``warehouse``: walled hall, rows of open pallet racks (primitive boxes: posts, beams, shelves, some boxed loads)
  with aisles between them, forklifts parked in aisles and in the open area
* ``yard``: fenced yard, 1-4 machines (excavators in two swing poses, crawler dump truck) and clutter (container,
  piles, pallets, pipes)

The two hand-built scenes are never generated: they stay held out for evaluation.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np

from ..scene_builder import sample_triangles, voxel_keys

SIM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SRC = {"A_warehouse": ["forklift_0"],
       "B_construction": ["excavator_a", "excavator_b", "dump_truck", "container", "gravel_pile", "brick_pallets",
                          "pipe_bundle"]}
PART_NAME = {"forklift_0": "forklift", "excavator_a": "excavator_stock", "excavator_b": "excavator_swung"}
SURF_VOXEL = 0.1
GRID_CELL, BAND = 0.1, (0.05, 0.6)
BAND_SPACING = 0.05


# --------------------------------------------------------------------------------------------------------------------
# part library
# --------------------------------------------------------------------------------------------------------------------
def build_part_library(out: str):
    """Cut the machines / clutter out of the built scenes -> one npz per part (local frame)."""
    os.makedirs(out, exist_ok=True)
    for scene, names in SRC.items():
        d = os.path.join(SIM, "assets/pose_inspection", scene)
        meta = json.load(open(os.path.join(d, "scene.json")))
        gt = np.load(os.path.join(d, "gt.npz"))
        su = np.load(os.path.join(d, "surface.npz"))
        V, T, O = gt["verts"], gt["tris"], gt["tri_obj"]
        for ob in meta["objects"]:
            if ob["name"] not in names:
                continue
            i = ob["index"]
            tri = T[O == i]
            used, inv = np.unique(tri.reshape(-1), return_inverse=True)
            v = V[used].astype(np.float64)
            c = (np.asarray(ob["bbox_min"]) + np.asarray(ob["bbox_max"])) / 2
            a = -math.radians(ob.get("yaw_deg", 0.0))
            R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
            to_local = lambda p: np.c_[(p[:, :2] - c[:2]) @ R.T, p[:, 2]]  # noqa: E731
            vl = to_local(v)
            sp = to_local(su["points"][su["obj"] == i].astype(np.float64))
            bp, _, _ = sample_triangles(vl, inv.reshape(-1, 3), BAND_SPACING)
            bp = bp[(bp[:, 2] >= BAND[0]) & (bp[:, 2] <= BAND[1])]
            lo, hi = vl.min(0), vl.max(0)
            name = PART_NAME.get(ob["name"], ob["name"])
            np.savez(os.path.join(out, f"{name}.npz"), verts=vl.astype(np.float32), tris=inv.reshape(-1, 3).astype(
                np.int32), surf=sp.astype(np.float32), band=bp[:, :2].astype(np.float32), lo=lo, hi=hi,
                cls=ob["class"], target=bool(ob["target"]))
            print(f"  part {name}: {len(tri):,} tris, {len(sp):,} surface samples, footprint "
                  f"{hi[0] - lo[0]:.2f} x {hi[1] - lo[1]:.2f} m, h {hi[2]:.2f} m")


def load_parts(lib: str) -> dict:
    out = {}
    for f in sorted(os.listdir(lib)):
        if f.endswith(".npz"):
            d = np.load(os.path.join(lib, f))
            out[f[:-4]] = {k: d[k] for k in d.files}
            out[f[:-4]]["cls"] = str(d["cls"])
            out[f[:-4]]["target"] = bool(d["target"])
    return out


# --------------------------------------------------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------------------------------------------------
_BOX_T = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                   [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])


def box(lo, hi):
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], np.float64)
    return v, _BOX_T.copy()


def rack_row(rng, x0, y0, length, depth, height, levels):
    """Open pallet rack along x: posts every bay, beams + shelf deck per level, loads on some shelves."""
    parts = []
    bay = 2.7
    n = max(1, int(round(length / bay)))
    xs = np.linspace(x0, x0 + length, n + 1)
    for x in xs:
        for y in (y0, y0 + depth):
            parts.append(box((x - 0.05, y - 0.05, 0.0), (x + 0.05, y + 0.05, height)))
    zs = np.linspace(0.15, height - 0.4, levels)
    for z in zs:
        parts.append(box((x0, y0, z), (x0 + length, y0 + depth, z + 0.06)))
        for k in range(n):
            if rng.random() < 0.6:
                h = rng.uniform(0.5, min(1.3, (zs[1] - zs[0]) - 0.2 if levels > 1 else 1.2))
                a, b = xs[k] + 0.15, xs[k + 1] - 0.15
                parts.append(box((a, y0 + 0.08, z + 0.06), (b, y0 + depth - 0.08, z + 0.06 + h)))
    return parts


def walls(b, height, thick=0.2):
    x0, y0, x1, y1 = b
    return [box((x0 - thick, y0 - thick, 0), (x1 + thick, y0, height)), box((x0 - thick, y1, 0), (x1 + thick, y1 + thick,
            height)), box((x0 - thick, y0, 0), (x0, y1, height)), box((x1, y0, 0), (x1 + thick, y1, height))]


def floor(b, pad=1.0):
    x0, y0, x1, y1 = b
    v = np.array([[x0 - pad, y0 - pad, 0], [x1 + pad, y0 - pad, 0], [x1 + pad, y1 + pad, 0], [x0 - pad, y1 + pad, 0]],
                 np.float64)
    return v, np.array([[0, 1, 2], [0, 2, 3]])


# --------------------------------------------------------------------------------------------------------------------
# layouts
# --------------------------------------------------------------------------------------------------------------------
def _footprint(part, x, y, yaw, margin=0.0):
    """Oriented footprint rectangle corners of a placed part (+ margin)."""
    lo, hi = part["lo"][:2] - margin, part["hi"][:2] + margin
    c = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
    return c @ R.T + [x, y]


def _overlap(a, b):
    """Separating-axis test of two convex quads."""
    for poly in (a, b):
        for k in range(4):
            e = poly[(k + 1) % 4] - poly[k]
            n = np.array([-e[1], e[0]])
            pa, pb = a @ n, b @ n
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def _rect(lo, hi):
    return np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]], float)


def layout_warehouse(rng, parts):
    W, L = rng.uniform(22, 34), rng.uniform(30, 48)
    b = (0.0, 0.0, W, L)
    prims, placed, blocked = [floor(b)] + walls(b, 8.0), [], []
    # rack rows along y in the far part of the hall, aisles 3.6-5 m; open area near y = 0
    rack_y0 = rng.uniform(10, 14)
    rack_len = L - rack_y0 - rng.uniform(1.5, 4)
    depth, aisle = 1.1, rng.uniform(3.6, 5.0)
    x = rng.uniform(0.3, 1.5)
    aisles = []
    while x + depth < W - 0.3:
        h = rng.uniform(4.0, 6.5)
        # rack_row builds along x: swap x / y (and the triangle winding) so that rows run along y
        for v, t in rack_row(rng, 0.0, 0.0, rack_len, depth, h, int(rng.integers(3, 5))):
            prims.append((np.c_[v[:, 1] + x, v[:, 0] + rack_y0, v[:, 2]], t[:, [0, 2, 1]]))
        blocked.append(_rect((x - 0.1, rack_y0), (x + depth + 0.1, rack_y0 + rack_len)))
        if x + depth + aisle < W - 0.3:
            aisles.append((x + depth, x + depth + aisle))
        x += depth + aisle
    n_fork = int(rng.integers(3, 9))
    fk = parts["forklift"]
    for _ in range(200):
        if len(placed) >= n_fork:
            break
        if aisles and rng.random() < 0.75:   # parked in an aisle, along it
            a0, a1 = aisles[int(rng.integers(len(aisles)))]
            px = (a0 + a1) / 2 + rng.uniform(-0.3, 0.3)
            py = rng.uniform(rack_y0 + 2, rack_y0 + rack_len - 2)
            yaw = math.pi / 2 * (1 if rng.random() < 0.5 else -1) + rng.normal(0, 0.05)
        else:                                 # open area
            px, py = rng.uniform(3, W - 3), rng.uniform(3, rack_y0 - 2.5)
            yaw = rng.uniform(-math.pi, math.pi)
        fp = _footprint(fk, px, py, yaw, 0.3)
        if any(_overlap(fp, q) for q in blocked):
            continue
        blocked.append(_footprint(fk, px, py, yaw, 1.0))
        placed.append(("forklift", px, py, yaw))
    start = (2.0, 1.5, min(W - 2.0, 8.0), min(rack_y0 - 2.0, 5.0))
    return b, (0.0, 9.0), prims, placed, start


def layout_yard(rng, parts):
    W, L = rng.uniform(24, 36), rng.uniform(24, 36)
    b = (0.0, 0.0, W, L)
    prims = [floor(b)] + walls(b, 2.0, 0.05)
    start = (W / 2 - 4, 1.0, W / 2 + 4, 3.5)
    blocked = [_rect((start[0] - 1, 0), (start[2] + 1, start[3] + 1.5))]
    placed = []
    machines = ["excavator_stock", "excavator_swung", "dump_truck"]
    clutter = ["container", "gravel_pile", "brick_pallets", "pipe_bundle"]
    want = [machines[int(rng.integers(3))] for _ in range(int(rng.integers(1, 5)))]
    want += [clutter[int(rng.integers(4))] for _ in range(int(rng.integers(1, 5)))]
    for name in want:
        p = parts[name]
        for _ in range(100):
            yaw = rng.uniform(-math.pi, math.pi)
            px, py = rng.uniform(3, W - 3), rng.uniform(6, L - 3)
            fp = _footprint(p, px, py, yaw, 0.2)
            if fp.min() < 0.8 or fp[:, 0].max() > W - 0.8 or fp[:, 1].max() > L - 0.8:
                continue
            if any(_overlap(fp, q) for q in blocked):
                continue
            blocked.append(_footprint(p, px, py, yaw, 1.6))
            placed.append((name, px, py, yaw))
            break
    return b, (0.0, 7.0), prims, placed, start


# --------------------------------------------------------------------------------------------------------------------
def make_scene(rng, parts, kind: str, out_dir: str, name: str) -> dict:
    """Generate one scene and write gt.npz / surface.npz / grid.npz / scene.json into ``out_dir``."""
    b, zr, prims, placed, start = (layout_warehouse if kind == "warehouse" else layout_yard)(rng, parts)
    V, T, O, off = [], [], [], 0
    for v, t in prims:
        V.append(v)
        T.append(t + off)
        O.append(np.full(len(t), -1, np.int16))
        off += len(v)
    objects, surf_p, surf_o, band = [], [], [], []
    for i, (pn, x, y, yaw) in enumerate(placed):
        p = parts[pn]
        R = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        v = np.c_[p["verts"][:, :2] @ R.T + [x, y], p["verts"][:, 2]]
        V.append(v)
        T.append(p["tris"] + off)
        O.append(np.full(len(p["tris"]), i, np.int16))
        off += len(v)
        surf_p.append(np.c_[p["surf"][:, :2] @ R.T + [x, y], p["surf"][:, 2]])
        surf_o.append(np.full(len(p["surf"]), i, np.int16))
        band.append(p["band"] @ R.T + [x, y])
        lo, hi = v.min(0), v.max(0)
        objects.append({"name": f"{pn}_{i}", "class": p["cls"], "target": p["target"], "index": i,
                        "center": ((lo + hi) / 2).tolist(), "bbox_min": lo.tolist(), "bbox_max": hi.tolist(),
                        "height_m": float(hi[2]), "yaw_deg": math.degrees(yaw), "part": pn})
    V = np.concatenate(V).astype(np.float32)
    T = np.concatenate(T).astype(np.int32)
    O = np.concatenate(O)
    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "gt.npz"), verts=V, tris=T, tri_obj=O, names=np.array([o["name"] for o in objects]))
    # surface voxels (same encoding as scene_builder.surface_gt)
    origin = np.array([b[0] - 1.0, b[1] - 1.0, zr[0] - 1.0])
    sp = np.concatenate(surf_p) if surf_p else np.zeros((0, 3))
    so = np.concatenate(surf_o) if surf_o else np.zeros(0, np.int16)
    ijk = np.floor((sp - origin) / SURF_VOXEL).astype(np.int64)
    key = voxel_keys(ijk) * 64 + so
    _, first = np.unique(key, return_index=True)
    np.savez(os.path.join(out_dir, "surface.npz"), points=sp.astype(np.float32), obj=so, vox_ijk=ijk[first].astype(
        np.int32), vox_obj=so[first], voxel=np.float32(SURF_VOXEL), origin=origin.astype(np.float32),
        n_obj=np.int32(len(objects)))
    # traversability: primitives rasterised from their triangles, parts from their cached band samples
    lo2, hi2 = np.array(b[:2]) - 1.0, np.array(b[2:]) + 1.0
    nx, ny = int(np.ceil((hi2[0] - lo2[0]) / GRID_CELL)), int(np.ceil((hi2[1] - lo2[1]) / GRID_CELL))
    occ = np.zeros((ny, nx), bool)
    pts = [np.concatenate(band)] if band else []
    for v, t in prims[1:]:   # not the floor
        q, _, _ = sample_triangles(v, t, BAND_SPACING)
        pts.append(q[(q[:, 2] >= BAND[0]) & (q[:, 2] <= BAND[1])][:, :2])
    pts = np.concatenate(pts)
    ix = np.floor((pts[:, 0] - lo2[0]) / GRID_CELL).astype(int)
    iy = np.floor((pts[:, 1] - lo2[1]) / GRID_CELL).astype(int)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    occ[iy[ok], ix[ok]] = True
    xs = lo2[0] + (np.arange(nx) + 0.5) * GRID_CELL
    ys = lo2[1] + (np.arange(ny) + 0.5) * GRID_CELL
    occ |= ~((xs[None, :] >= b[0]) & (xs[None, :] <= b[2]) & (ys[:, None] >= b[1]) & (ys[:, None] <= b[3]))
    np.savez(os.path.join(out_dir, "grid.npz"), occupied=occ, origin=lo2.astype(np.float32), cell=np.float32(GRID_CELL))
    meta = {"name": name, "kind": kind, "bounds": list(b), "z_range": list(zr),
            "robot_start": {"region": list(start), "min_clearance": 0.8}, "objects": objects,
            "triangles": int(len(T)), "surface_voxel": SURF_VOXEL, "procedural": True}
    with open(os.path.join(out_dir, "scene.json"), "w") as f:
        json.dump(meta, f, indent=1)
    meta["dir"] = out_dir
    return meta
