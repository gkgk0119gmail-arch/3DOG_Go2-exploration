"""One storey of a BIM (IFC) model -> a 2.5D training-env world (same format as ariadne3d/procwarehouse.generate).

    python tools/bim_to_25d.py model.ifc out_dir [--storey "Level 1"] [--cell 0.4] [--no-furniture]

Per storey (and per separate free area of at least --min-area, e.g. the two units of a duplex) it writes
<out_dir>/<model>_<storey>[_<k>].npz with
  gt      [y, x] int    255 free / 1 occupied (ARiADNE ground truth)
  height  [y, x] float  obstacle height above the floor [m] (0 on free cells, H outside the building)
  H       float         ceiling height above the floor [m]
  origin  (x0, y0)      world position of cell [0, 0] [m]
and a <model>_<storey>.png preview.

How a cell is classified (cell = 0.4 m, floor = storey elevation):
  free      indoors (a floor slab or room (IfcSpace) floor under it AND a slab, ceiling or roof above it) and no
            element surface in the robot band (0.05-0.7 m above the floor)
  occupied  an element (wall, column, stair, furniture, ...) has a surface in the robot band; its height is the
            tallest surface of that column below the ceiling
  outside   no floor under it or nothing above it (outside, terraces, shafts, stair voids) -> occupied, height = H
Doors are treated as open: cells under a door are cleared even when a finish layer modelled as a separate wall
(without the opening) runs across it. Overhangs above the band (lintels, beams) are dropped: 2.5D has one
height per cell. Free pockets the robot cannot reach from the largest free area become occupied.
"""

import argparse
import os
import re

import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.unit
import numpy as np
from scipy.ndimage import binary_closing
from skimage.measure import label

BAND = (0.05, 0.7)  # [m] above the floor: anything here blocks the Go2
SKIP = {"IfcSpace", "IfcOpeningElement", "IfcDoor", "IfcSite", "IfcBuilding", "IfcBuildingStorey", "IfcAnnotation",
        "IfcGrid", "IfcVirtualElement", "IfcFooting", "IfcPile", "IfcSlab", "IfcCovering", "IfcRoof"}
FURNITURE = {"IfcFurnishingElement", "IfcFurniture", "IfcSystemFurnitureElement"}


def triangles(model, types=None, skip=()):
    """World-coordinate triangles [N, 3, 3] and the IFC class of each, for products of the given classes."""
    settings = ifcopenshell.geom.settings()
    settings.set("use-world-coords", True)
    prods = [p for p in model.by_type("IfcProduct") if p.Representation is not None
             and (types is None or p.is_a() in types) and p.is_a() not in skip]
    if not prods:
        return np.zeros((0, 3, 3)), []
    it = ifcopenshell.geom.iterator(settings, model, include=prods)
    tris, kinds = [], []
    if it.initialize():
        while True:
            shape = it.get()
            v = np.asarray(shape.geometry.verts, float).reshape(-1, 3)
            f = np.asarray(shape.geometry.faces, int).reshape(-1, 3)
            if len(f):
                tris.append(v[f])
                kinds += [model.by_id(shape.id).is_a()] * len(f)
            if not it.next():
                break
    return (np.concatenate(tris) if tris else np.zeros((0, 3, 3))), kinds


def sample(tris, spacing):
    """Points spread over the triangles, about one per spacing^2 of area (plus every vertex)."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    n = np.clip((area / spacing ** 2).astype(int), 0, 5000)
    idx = np.repeat(np.arange(len(tris)), n)
    r1, r2 = np.random.default_rng(0).random((2, len(idx), 1))
    s = np.sqrt(r1)
    pts = (1 - s) * a[idx] + s * (1 - r2) * b[idx] + s * r2 * c[idx]
    return np.concatenate([tris.reshape(-1, 3), pts])


def storey_levels(model):
    """[(name, elevation in m)] sorted by elevation (elevation read in the model's length unit)."""
    scale = ifcopenshell.util.unit.calculate_unit_scale(model)
    lv = [(s.Name or f"storey{s.id()}", float(s.Elevation or 0.0) * scale) for s in model.by_type("IfcBuildingStorey")]
    return sorted(lv, key=lambda t: t[1])


def convert(model, z0, z1, cell=0.4, furniture=True, min_area=15.0):
    """Worlds [(gt, height, H, origin)] for the storey whose floor is at z0 and the next floor at z1 (world metres),
    one per separate free area of at least min_area m^2."""
    skip = set(SKIP) | (set() if furniture else FURNITURE)
    body, _ = triangles(model, skip=skip)
    slabs, _ = triangles(model, types={"IfcSlab", "IfcCovering", "IfcRoof"})
    spaces, _ = triangles(model, types={"IfcSpace"})
    doors, _ = triangles(model, types={"IfcDoor"})
    pb, ps = sample(body, 0.08), sample(slabs, 0.15)
    pr = sample(spaces, 0.15) if len(spaces) else np.zeros((0, 3))
    pd = sample(doors, 0.05) if len(doors) else np.zeros((0, 3))

    # ceiling: underside of the slabs / ceilings above this floor (lowest surface between 2 m and the next floor)
    up = ps[(ps[:, 2] > z0 + 2.0) & (ps[:, 2] < z1 + 1.0)]
    if not len(up):
        raise ValueError("no ceiling above this floor (roof level)")
    H = float(np.percentile(up[:, 2], 10) - z0)
    H = float(np.clip(H, 2.2, 10.0))
    floor = np.concatenate([ps[np.abs(ps[:, 2] - z0) < 0.35], pr[np.abs(pr[:, 2] - z0) < 0.35]])
    if not len(floor):
        raise ValueError("no floor slab at this storey")

    x0, y0 = floor[:, 0].min() - 1.2, floor[:, 1].min() - 1.2
    nx = int(np.ceil((floor[:, 0].max() + 1.2 - x0) / cell)) + 1
    ny = int(np.ceil((floor[:, 1].max() + 1.2 - y0) / cell)) + 1

    def cells(p):
        ix = ((p[:, 0] - x0) / cell).astype(int)
        iy = ((p[:, 1] - y0) / cell).astype(int)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        return iy[ok], ix[ok], p[ok]

    has_floor = np.zeros((ny, nx), bool)
    iy, ix, _ = cells(floor)
    has_floor[iy, ix] = True
    has_floor |= binary_closing(has_floor)  # close 1-cell gaps between sampled slab points
    covered = np.zeros((ny, nx), bool)  # a slab, ceiling or roof overhead (double-height rooms too): indoors
    iy, ix, _ = cells(ps[(ps[:, 2] > z0 + 2.0) & (ps[:, 2] < z0 + 12.0)])
    covered[iy, ix] = True
    has_floor &= covered | binary_closing(covered, iterations=2)

    zrel = pb[:, 2] - z0
    blocking = np.zeros((ny, nx), bool)
    iy, ix, _ = cells(pb[(zrel > BAND[0]) & (zrel < BAND[1])])
    blocking[iy, ix] = True
    zd = pd[:, 2] - z0
    iy, ix, _ = cells(pd[(zd > -0.2) & (zd < 2.5)])
    blocking[iy, ix] = False  # open doors
    hmax = np.zeros((ny, nx), np.float32)
    keep = (zrel > BAND[0]) & (zrel < H - 0.05)
    iy, ix, p = cells(pb[keep])
    np.maximum.at(hmax, (iy, ix), (p[:, 2] - z0).astype(np.float32))

    lab = label(has_floor & ~blocking, connectivity=1)
    sizes = np.bincount(lab.ravel())[1:] * cell ** 2
    worlds = []
    for comp in np.argsort(-sizes) + 1:
        if sizes[comp - 1] < min_area:
            break
        free = lab == comp  # one area the robot can reach; everything else is wall, obstacle or outside
        height = np.where(blocking, np.maximum(hmax, 0.3), H).astype(np.float32)
        height[free] = 0.0
        height = np.minimum(height, H)
        ys, xs = np.nonzero(free)  # crop to this area + 3 cells
        a, b = max(ys.min() - 3, 0), min(ys.max() + 4, ny)
        c, d = max(xs.min() - 3, 0), min(xs.max() + 4, nx)
        free, height = free[a:b, c:d], height[a:b, c:d]
        gt = np.where(free, 255, 1).astype(int)
        gt[[0, -1], :] = 1
        gt[:, [0, -1]] = 1
        worlds.append((gt, height, H, (x0 + c * cell, y0 + a * cell)))
    if not worlds:
        raise ValueError("no free floor area")
    return worlds


def preview(path, gt, height, H, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 8 * gt.shape[0] / max(gt.shape[1], 1) + 0.6))
    img = np.where(gt == 255, np.nan, height)
    ax.imshow(np.where(gt == 255, 1.0, np.nan), cmap="Greys", vmin=0, vmax=1.4, origin="lower")
    im = ax.imshow(img, cmap="viridis", vmin=0, vmax=H, origin="lower")
    fig.colorbar(im, ax=ax, fraction=0.03, label="obstacle height above floor [m]")
    ax.set_title(f"{title}\nfree {(gt == 255).sum() * 0.16:.0f} m², ceiling {H:.2f} m", fontsize=10)
    ax.set_xticks([]), ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ifc")
    ap.add_argument("out_dir")
    ap.add_argument("--storey", help="storey name (default: every storey that has a floor and >= 2.2 m of height)")
    ap.add_argument("--cell", type=float, default=0.4)
    ap.add_argument("--no-furniture", action="store_true")
    ap.add_argument("--min-area", type=float, default=15.0, help="smallest separate free area kept as a world [m^2]")
    args = ap.parse_args()

    model = ifcopenshell.open(args.ifc)
    levels = storey_levels(model)
    os.makedirs(args.out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(args.ifc))[0]
    for i, (name, z0) in enumerate(levels):
        if args.storey and name != args.storey:
            continue
        if not args.storey and re.search(r"roof|dach", name, re.I):
            print(f"{name}: skipped (roof level)")
            continue
        z1 = levels[i + 1][1] if i + 1 < len(levels) else z0 + 6.0
        if z1 - z0 < 2.2:
            continue
        try:
            worlds = convert(model, z0, z1, args.cell, not args.no_furniture, args.min_area)
        except ValueError as e:
            print(f"{name}: skipped ({e})")
            continue
        for k, (gt, height, H, origin) in enumerate(worlds):
            stem = os.path.join(args.out_dir, f"{base}_{re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_')}")
            stem += f"_{k}" if len(worlds) > 1 else ""
            np.savez_compressed(stem + ".npz", gt=gt, height=height, H=H, origin=np.array(origin), cell=args.cell,
                                source=os.path.basename(args.ifc), storey=name, elevation=z0)
            preview(stem + ".png", gt, height, H, f"{base} / {name}" + (f" / area {k}" if len(worlds) > 1 else ""))
            print(f"{name}{f' area {k}' if len(worlds) > 1 else ''}: {gt.shape[1]}x{gt.shape[0]} cells, "
                  f"free {(gt == 255).sum() * args.cell ** 2:.0f} m^2, ceiling {H:.2f} m -> {stem}.npz")


if __name__ == "__main__":
    main()
