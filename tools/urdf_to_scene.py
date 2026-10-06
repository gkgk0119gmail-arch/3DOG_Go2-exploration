"""URDF (e.g. an excavator) -> static Isaac scene for go2_lab/excavator.py: textured terrain USD + LiDAR raycast USD +
ground-truth mesh + a scan loop of waypoints around it.

    python tools/urdf_to_scene.py third_party/hr35_modeling/ix35e_description/urdf/ix35e.urdf assets/excavator/ix35e \
        --package ix35e_description=third_party/hr35_modeling/ix35e_description --center 6 0 --yaw -90 --standoff 2.5

The links are posed by forward kinematics at the given joint values (default 0, --joints '{"boom_joint": 0.3}'),
the visual meshes (DAE with textures) are baked into one USD with UsdPreviewSurface materials. The Go2 spawns at the
env origin, so the machine is placed at --center. Outputs:
  <out>_terrain.usda  ground plate (60 x 60 m, collider) + the machine (triangle-mesh collider), lights
  <out>_raycast.usda  ground + machine in one mesh (the RayCaster takes a single mesh)
  <out>_gt.npz        machine triangles (world) for surface-coverage scoring
  <out>_scene.json    footprint, height, waypoints (rectangle loop at --standoff from the footprint)
"""

import argparse
import json
import math
import os
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
from PIL import Image
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Vt


def rpy_matrix(rpy):
    r, p, y = rpy
    return trimesh.transformations.euler_matrix(r, p, y, "sxyz")


def origin_matrix(el):
    if el is None:
        return np.eye(4)
    xyz = [float(v) for v in el.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in el.get("rpy", "0 0 0").split()]
    T = rpy_matrix(rpy)
    T[:3, 3] = xyz
    return T


def link_poses(root, joints):
    """World transform of every link by forward kinematics."""
    kids = {}
    for j in root.findall("joint"):
        kids.setdefault(j.find("parent").get("link"), []).append(j)
    children = {j.find("child").get("link") for j in root.findall("joint")}
    base = [l.get("name") for l in root.findall("link") if l.get("name") not in children][0]
    poses = {base: np.eye(4)}
    stack = [base]
    while stack:
        p = stack.pop()
        for j in kids.get(p, []):
            T = poses[p] @ origin_matrix(j.find("origin"))
            q = joints.get(j.get("name"), 0.0)
            if q and j.get("type") in ("revolute", "continuous"):
                axis = [float(v) for v in (j.find("axis").get("xyz") if j.find("axis") is not None else "1 0 0").split()]
                T = T @ trimesh.transformations.rotation_matrix(q, axis)
            c = j.find("child").get("link")
            poses[c] = T
            stack.append(c)
    return poses


def dae_textures(fn):
    """{geometry id: texture file} from a COLLADA file (trimesh keeps the UVs but may drop the image)."""
    import collada

    d = collada.Collada(fn)
    mat_img = {}
    for m in d.materials:
        samp = getattr(getattr(m.effect, "diffuse", None), "sampler", None)
        if samp is not None:
            mat_img[m.id] = os.path.normpath(os.path.join(os.path.dirname(fn), samp.surface.image.path))
    out = {}
    for g in d.geometries:
        for p in g.primitives:
            if p.material in mat_img:
                out[g.id] = mat_img[p.material]
    return out


def load_visuals(root, packages, poses):
    """[(name, trimesh.Trimesh in world)] for every visual mesh."""
    out = []
    for link in root.findall("link"):
        for k, vis in enumerate(link.findall("visual")):
            m = vis.find("geometry/mesh")
            if m is None:
                continue
            fn = m.get("filename")
            if fn.startswith("package://"):
                pkg, rel = fn[len("package://"):].split("/", 1)
                fn = os.path.join(packages[pkg], rel)
            scale = [float(v) for v in m.get("scale", "1 1 1").split()]
            T = poses[link.get("name")] @ origin_matrix(vis.find("origin"))
            sc = trimesh.load(fn, force="scene")
            tex = dae_textures(fn) if fn.lower().endswith(".dae") else {}
            for g_name, geom in sc.geometry.items():
                for node in sc.graph.geometry_nodes.get(g_name, []):
                    tm = geom.copy()
                    tm.metadata["texture"] = tex.get(g_name)
                    tm.apply_scale(scale)
                    tm.apply_transform(T @ sc.graph.get(node)[0])
                    out.append((f"{link.get('name')}_{k}_{len(out)}", tm))
    return out


def write_scene(path, parts, ground, tex_dir, center, yaw_deg, collide=True):
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
    dome.CreateIntensityAttr(900)
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(2500)
    sun.CreateAngleAttr(1.0)
    UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(35, 15, 0))
    # ground plate: compacted soil
    g = UsdGeom.Mesh.Define(stage, "/World/Ground")
    g.CreatePointsAttr(Vt.Vec3fArray([Gf.Vec3f(*p) for p in ground]))
    g.CreateFaceVertexCountsAttr([4])
    g.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    g.CreateDisplayColorAttr([Gf.Vec3f(0.52, 0.45, 0.36)])
    if collide:
        UsdPhysics.CollisionAPI.Apply(g.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(g.GetPrim()).CreateApproximationAttr("none")
    machine = UsdGeom.Xform.Define(stage, "/World/Machine")
    xf = UsdGeom.Xformable(machine)
    xf.AddTranslateOp().Set(Gf.Vec3d(center[0], center[1], 0.0))
    xf.AddRotateZOp().Set(float(yaw_deg))
    mats = {}
    for name, tm in parts:
        prim_path = f"/World/Machine/{name}"
        mesh = UsdGeom.Mesh.Define(stage, prim_path)
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(tm.vertices, np.float32)))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(tm.faces)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(tm.faces, np.int32).reshape(-1)))
        mesh.CreateDoubleSidedAttr(True)
        if collide:
            UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
            UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("none")
        vis = tm.visual
        img = tm.metadata.get("texture")
        uv = getattr(vis, "uv", None)
        if img is not None and os.path.exists(img) and uv is not None and len(uv) == len(tm.vertices):
            st = UsdGeom.PrimvarsAPI(mesh).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                                                         UsdGeom.Tokens.vertex)
            st.Set(Vt.Vec2fArray.FromNumpy(np.asarray(uv, np.float32)))
            key = img
            if key not in mats:
                tex = os.path.join(tex_dir, os.path.splitext(os.path.basename(img))[0] + ".png")
                im = Image.open(img).convert("RGB")
                im.thumbnail((2048, 2048))
                im.save(tex)
                mats[key] = _material(stage, f"/World/Looks/M{len(mats)}", os.path.relpath(tex, os.path.dirname(path)))
            UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(mats[key])
        else:
            c = np.asarray(getattr(getattr(vis, "material", None), "main_color", [180, 180, 180, 255]))[:3] / 255.0 \
                if vis.kind == "texture" else np.asarray(vis.main_color[:3]) / 255.0
            mesh.CreateDisplayColorAttr([Gf.Vec3f(*map(float, c))])
    stage.GetRootLayer().Save()


def _material(stage, path, tex_rel):
    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    reader = UsdShade.Shader.Define(stage, path + "/st")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex = UsdShade.Shader.Define(stage, path + "/Tex")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(tex_rel)
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def write_raycast(path, verts, tris):
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    m = UsdGeom.Mesh.Define(stage, "/World/Raycast")
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(verts.astype(np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(tris)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(tris.astype(np.int32).reshape(-1)))
    stage.GetRootLayer().Save()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("urdf")
    ap.add_argument("out")
    ap.add_argument("--package", action="append", default=[], help="name=dir for package:// paths")
    ap.add_argument("--joints", default="{}")
    ap.add_argument("--center", type=float, nargs=2, default=(6.0, 0.0))
    ap.add_argument("--yaw", type=float, default=0.0, help="machine yaw [deg] in the world")
    ap.add_argument("--standoff", type=float, default=2.5, help="waypoint loop distance from the footprint [m]")
    ap.add_argument("--ground", type=float, default=60.0)
    args = ap.parse_args()
    packages = dict(p.split("=", 1) for p in args.package)
    root = ET.parse(args.urdf).getroot()
    parts = load_visuals(root, packages, link_poses(root, json.loads(args.joints)))
    allv = np.concatenate([p.vertices for _, p in parts])
    zmin = allv[:, 2].min()
    for _, p in parts:  # stand on the ground
        p.apply_translation([0, 0, -zmin])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tex_dir = os.path.join(os.path.dirname(args.out), "textures")
    os.makedirs(tex_dir, exist_ok=True)
    G = args.ground / 2
    ground = [(-G, -G, 0.0), (G, -G, 0.0), (G, G, 0.0), (-G, G, 0.0)]
    write_scene(args.out + "_terrain.usda", parts, ground, tex_dir, args.center, args.yaw)
    # world-frame machine mesh (for raycast + ground truth)
    c, s = math.cos(math.radians(args.yaw)), math.sin(math.radians(args.yaw))
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    V, T, n = [], [], 0
    for _, p in parts:
        V.append(np.asarray(p.vertices) @ Rz.T + [args.center[0], args.center[1], 0])
        T.append(np.asarray(p.faces) + n)
        n += len(p.vertices)
    V, T = np.concatenate(V), np.concatenate(T)
    gv = np.array(ground)
    write_raycast(args.out + "_raycast.usda", np.concatenate([V, gv]),
                  np.concatenate([T, np.array([[0, 1, 2], [0, 2, 3]]) + len(V)]))
    np.savez_compressed(args.out + "_gt.npz", verts=V.astype(np.float32), tris=T.astype(np.int32))
    lo, hi = V.min(0), V.max(0)
    a = args.standoff
    loop = [(lo[0] - a, lo[1] - a), (hi[0] + a, lo[1] - a), (hi[0] + a, hi[1] + a), (lo[0] - a, hi[1] + a)]
    scene = {"urdf": args.urdf, "joints": json.loads(args.joints), "center": list(args.center), "yaw_deg": args.yaw,
             "bbox_min": lo.tolist(), "bbox_max": hi.tolist(), "height_m": float(hi[2]),
             "waypoints": [[float(x), float(y)] for x, y in loop] + [[float(lo[0] - a), float(lo[1] - a)]],
             "parts": len(parts), "triangles": int(len(T))}
    with open(args.out + "_scene.json", "w") as f:
        json.dump(scene, f, indent=1)
    print(json.dumps({k: v for k, v in scene.items() if k != "waypoints"}, indent=1))
    print("waypoints", scene["waypoints"])


if __name__ == "__main__":
    main()
