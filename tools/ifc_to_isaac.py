"""IFC storey -> Isaac Sim scene for go2_lab/bim.py (step 3): terrain USD + LiDAR raycast USD + spawn / waypoints.

    python tools/ifc_to_isaac.py assets/bim_ifc/dental_clinic.ifc ariadne3d/maps_test/dental_clinic_First_Floor_0.npz \
        assets/bim_ifc/dental_clinic

The 2.5D map (tools/bim_to_25d.py, same IFC) gives the floor elevation and a spawn: its most open free cell. The
building is shifted so the spawn sits at the origin (the Go2 spawns at the env origin), on the storey floor (z = 0).
Writes <out>_terrain.usda (meshes per IFC class + a ground plate, colliders except doors: doors open, as in
bim_to_25d), <out>_raycast.usda (everything in one mesh: the RayCaster takes a single mesh) and <out>_scene.json.
"""

import argparse
import json

import ifcopenshell
import ifcopenshell.geom
import numpy as np
from scipy.ndimage import distance_transform_edt

SKIP = {"IfcSpace", "IfcOpeningElement"}
NO_COLLIDE = {"IfcDoor"}
HIDDEN = {"IfcCovering"}  # ceilings: hidden in the render so the inside is lit and seen from above (LiDAR still hits them)
COLORS = {"IfcWallStandardCase": (0.85, 0.85, 0.8), "IfcWall": (0.85, 0.85, 0.8), "IfcSlab": (0.6, 0.6, 0.62),
          "IfcDoor": (0.55, 0.35, 0.2), "IfcWindow": (0.5, 0.75, 0.95), "IfcPlate": (0.5, 0.75, 0.95),
          "IfcStairFlight": (0.7, 0.55, 0.4), "IfcRailing": (0.3, 0.3, 0.3), "IfcFurnishingElement": (0.9, 0.6, 0.3),
          "IfcBeam": (0.5, 0.5, 0.55), "IfcCovering": (0.75, 0.7, 0.65), "Ground": (0.45, 0.45, 0.45)}


def ifc_meshes(path):
    """{IFC class: (verts [N, 3] m, tris [M, 3])} in world coordinates."""
    f = ifcopenshell.open(path)
    s = ifcopenshell.geom.settings()
    s.set("use-world-coords", True)
    it = ifcopenshell.geom.iterator(s, f, 4)
    acc = {}
    if it.initialize():
        while True:
            sh = it.get()
            cls = f.by_id(sh.id).is_a()
            if cls not in SKIP:
                v = np.array(sh.geometry.verts, float).reshape(-1, 3)
                t = np.array(sh.geometry.faces, int).reshape(-1, 3)
                V, T = acc.setdefault(cls, ([], []))
                T.append(t + sum(len(x) for x in V))
                V.append(v)
            if not it.next():
                break
    return {c: (np.concatenate(V), np.concatenate(T)) for c, (V, T) in acc.items()}


def spawn_and_waypoints(map_path, clear_m=1.0, reach=(3.0, 7.0)):
    """Most open free cell + a loop of up to 4 waypoints around it (line of sight with clear_m clearance), world m."""
    d = np.load(map_path)
    gt, origin, cell = d["gt"], d["origin"], float(d["cell"])
    dist = distance_transform_edt(gt == 255) * cell
    sy, sx = np.unravel_index(dist.argmax(), dist.shape)
    ok = dist >= clear_m

    def visible(a, b):
        n = int(np.abs(np.subtract(b, a)).max()) * 2 + 1
        return bool(ok[np.rint(np.linspace(a[0], b[0], n)).astype(int), np.rint(np.linspace(a[1], b[1], n)).astype(int)].all())

    pts = []
    for ang in np.arange(4) * np.pi / 2 + np.pi / 4:  # NE, NW, SW, SE: farthest visible point in each direction
        best = None
        for r in np.arange(reach[0], reach[1] + 1e-6, cell):
            p = (int(round(sy + np.sin(ang) * r / cell)), int(round(sx + np.cos(ang) * r / cell)))
            if not (0 <= p[0] < gt.shape[0] and 0 <= p[1] < gt.shape[1]) or not visible((sy, sx), p):
                break
            best = p
        if best is not None and (not pts or visible(pts[-1], best)):
            pts.append(best)
    world = lambda p: [float(origin[0] + p[1] * cell), float(origin[1] + p[0] * cell)]  # noqa: E731
    return world((sy, sx)), [world(p) for p in pts], float(d["elevation"]), float(dist.max())


def write_usda(path, meshes, collide=True, lights=False):
    lines = ['#usda 1.0', '(', '    defaultPrim = "World"', '    metersPerUnit = 1', '    upAxis = "Z"', ')', '',
             'def Xform "World"', '{']
    if lights:  # BIM exports carry no lights (the NVIDIA warehouse USD does)
        lines += ['    def DomeLight "Dome"', '    {', '        float inputs:intensity = 1000', '    }',
                  '    def DistantLight "Sun"', '    {', '        float inputs:angle = 1', '        float inputs:intensity = 3000',
                  '        double3 xformOp:rotateXYZ = (30, 20, 0)', '        uniform token[] xformOpOrder = ["xformOp:rotateXYZ"]',
                  '    }']
    for name, (v, t) in meshes.items():
        api = ' (\n        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]\n    )' \
            if collide and name not in NO_COLLIDE else ''
        c = COLORS.get(name, (0.7, 0.7, 0.7))
        lines += [f'    def Mesh "{name}"{api}', '    {',
                  f'        int[] faceVertexCounts = [{", ".join(["3"] * len(t))}]',
                  f'        int[] faceVertexIndices = [{", ".join(map(str, t.reshape(-1)))}]',
                  '        point3f[] points = [' + ", ".join(f"({x:.4f}, {y:.4f}, {z:.4f})" for x, y, z in v) + ']',
                  f'        color3f[] primvars:displayColor = [({c[0]}, {c[1]}, {c[2]})]',
                  '        uniform bool doubleSided = 1']
        if collide and name in HIDDEN:
            lines.append('        token visibility = "invisible"')
        if api:
            lines.append('        uniform token physics:approximation = "none"')
        lines += ['    }']
    lines += ['}', '']
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ifc")
    ap.add_argument("map25d", help="2.5D map of the storey to stand on (bim_to_25d.py output, same IFC)")
    ap.add_argument("out", help="output prefix, e.g. assets/bim_ifc/dental_clinic")
    args = ap.parse_args()
    spawn, waypoints, z0, clear = spawn_and_waypoints(args.map25d)
    shift = np.array([spawn[0], spawn[1], z0])
    meshes = {c: (v - shift, t) for c, (v, t) in ifc_meshes(args.ifc).items()}
    lo = np.min([v.min(0) for v, _ in meshes.values()], 0) - 2.0
    hi = np.max([v.max(0) for v, _ in meshes.values()], 0) + 2.0
    meshes["Ground"] = (np.array([[lo[0], lo[1], -0.01], [hi[0], lo[1], -0.01], [hi[0], hi[1], -0.01],
                                  [lo[0], hi[1], -0.01]]), np.array([[0, 1, 2], [0, 2, 3]]))
    write_usda(args.out + "_terrain.usda", meshes, lights=True)
    allv, allt, n = [], [], 0
    for v, t in meshes.values():
        allv.append(v)
        allt.append(t + n)
        n += len(v)
    write_usda(args.out + "_raycast.usda", {"Raycast": (np.concatenate(allv), np.concatenate(allt))}, collide=False)
    wp = [[x - shift[0], y - shift[1]] for x, y in waypoints] + [[0.0, 0.0]]
    scene = {"ifc": args.ifc, "map25d": args.map25d, "shift": shift.tolist(), "spawn_clearance_m": clear,
             "waypoints": wp}
    with open(args.out + "_scene.json", "w") as f:
        json.dump(scene, f, indent=1)
    print(json.dumps(scene))


if __name__ == "__main__":
    main()
