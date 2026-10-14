"""Build the POSE-inspection scenes (configs/pose_inspection/scene_*.yaml) into Isaac-ready files.

For a scene config this writes, under ``cfg.out_dir``:

  objects/<name>.usdc   one static (physics-free) USD per object, standing on z = 0 in its own frame
  scene.usda            the visual scene: background + /World/Objects/<name> (UsdSemantics "class" labels)
  raycast.usdc          every visible triangle of the scene baked into one world-space mesh (Isaac Lab RayCaster)
  gt.npz                verts, tris, tri_obj (object index per triangle, -1 = background)
  surface.npz           ground-truth object surface: uniform samples (points, normals, obj) and the surface voxels
                        (Co denominator, POSE S_o^*): vox_ijk, vox_obj, voxel size and grid origin
  grid.npz              2D traversability at ``build.grid_cell`` (rows = y): occupied by any geometry in
                        ``build.grid_band`` or outside ``bounds``
  scene.json            objects (class, target, centre, world AABB, height, counts), bounds, files

Object sources: ``usd`` (flattened, optional child translations, physics stripped), ``urdf`` (links posed by forward
kinematics, textured meshes; tools/urdf_to_scene.py helpers), and simple primitives for clutter (``box``, ``boxes``,
``cone``, ``pipes``).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys

import numpy as np

from .config import SIM_DIR, Cfg, sim_path

SURF_SEED = 0


# --------------------------------------------------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------------------------------------------------
def rot_z(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def sample_triangles(v: np.ndarray, t: np.ndarray, spacing: float, seed: int = SURF_SEED):
    """Uniform surface samples (about one per spacing^2) -> points, unit normals, triangle index."""
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    nrm = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(nrm, axis=1)
    unit = nrm / (2 * area[:, None] + 1e-12)
    rng = np.random.default_rng(seed)
    # stochastic rounding keeps the expected density exact for triangles smaller than spacing^2
    lam = area / spacing**2
    n = np.floor(lam).astype(np.int64) + (rng.random(len(lam)) < (lam - np.floor(lam)))
    idx = np.repeat(np.arange(len(a)), n)
    r1, r2 = rng.random((2, len(idx), 1))
    s = np.sqrt(r1)
    p = (1 - s) * a[idx] + s * (1 - r2) * b[idx] + s * r2 * c[idx]
    return p, unit[idx], idx


def voxel_keys(ijk: np.ndarray) -> np.ndarray:
    ijk = ijk.astype(np.int64)
    return (ijk[:, 0] << 42) + (ijk[:, 1] << 21) + ijk[:, 2]


# --------------------------------------------------------------------------------------------------------------------
# object USDs
# --------------------------------------------------------------------------------------------------------------------
def _usd_object(src: Cfg, out: str):
    """Flatten a USD asset, move named children (e.g. a forklift lift), strip physics -> static prop."""
    from pxr import Gf, Sdf, Usd, UsdGeom

    stage = Usd.Stage.Open(sim_path(src.path))
    root = stage.GetDefaultPrim()
    for child, xyz in (src.get("translate_children") or {}).items():
        prim = stage.GetPrimAtPath(root.GetPath().AppendChild(child))
        ops = [o for o in UsdGeom.Xformable(prim).GetOrderedXformOps() if o.GetOpType() == UsdGeom.XformOp.TypeTranslate]
        if not ops:
            raise ValueError(f"{child}: no translate op to edit")
        ops[0].Set(Gf.Vec3d(*xyz) + ops[0].Get())
    layer = stage.Flatten()
    if src.get("strip_physics", True):
        drop = []
        for prim in stage.Traverse():
            spec = layer.GetPrimAtPath(prim.GetPath())
            if spec is None:
                continue
            if "Joint" in prim.GetTypeName() or prim.GetTypeName() == "PhysicsScene":
                drop.append(prim.GetPath())
                continue
            if spec.HasInfo("apiSchemas"):
                items = spec.GetInfo("apiSchemas").GetAddedOrExplicitItems()
                spec.SetInfo("apiSchemas", Sdf.TokenListOp.CreateExplicit(
                    [a for a in items if not a.startswith(("Physics", "Physx", "Isaac", "Newton"))]))
            for prop in list(spec.properties):
                if prop.name.startswith(("physics:", "physx", "isaac:", "newton")):
                    spec.RemoveProperty(prop)
        for path in sorted(drop, key=lambda p: -len(str(p))):
            spec = layer.GetPrimAtPath(path)
            if spec is not None:
                del spec.nameParent.nameChildren[spec.name]
    layer.Export(out)


def _new_stage(path: str):
    from pxr import Usd, UsdGeom

    if os.path.exists(path):
        os.remove(path)
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/Object")
    stage.SetDefaultPrim(root.GetPrim())
    return stage


def _color_material(stage, path: str, rgb):
    from pxr import Gf, Sdf, UsdShade

    mat = UsdShade.Material.Define(stage, path)
    sh = UsdShade.Shader.Define(stage, path + "/Surface")
    sh.CreateIdAttr("UsdPreviewSurface")
    sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, rgb)))
    sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
    mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
    return mat


def _write_mesh(stage, path: str, verts: np.ndarray, faces: np.ndarray, rgb=None, uv=None, material=None):
    from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt

    m = UsdGeom.Mesh.Define(stage, path)
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(np.asarray(verts, np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray([3] * len(faces)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(np.asarray(faces, np.int32).reshape(-1)))
    m.CreateDoubleSidedAttr(True)
    if rgb is not None:
        m.CreateDisplayColorAttr([Gf.Vec3f(*map(float, rgb))])
    if uv is not None:
        st = UsdGeom.PrimvarsAPI(m).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex)
        st.Set(Vt.Vec2fArray.FromNumpy(np.asarray(uv, np.float32)))
    if material is not None:
        UsdShade.MaterialBindingAPI.Apply(m.GetPrim()).Bind(material)
    return m


def urdf_link_poses(root, joints: dict, axis_override: dict) -> dict:
    """World transform of every URDF link by forward kinematics (``axis_override``: joint -> axis, for URDFs whose
    declared axes do not match their meshes, e.g. ix35e: forward is +y but boom / stick / bucket say +y)."""
    import trimesh

    sys.path.insert(0, str(SIM_DIR / "tools"))
    from urdf_to_scene import origin_matrix  # noqa: PLC0415

    kids = {}
    for j in root.findall("joint"):
        kids.setdefault(j.find("parent").get("link"), []).append(j)
    children = {j.find("child").get("link") for j in root.findall("joint")}
    base = [lk.get("name") for lk in root.findall("link") if lk.get("name") not in children][0]
    poses = {base: np.eye(4)}
    stack = [base]
    while stack:
        p = stack.pop()
        for j in kids.get(p, []):
            T = poses[p] @ origin_matrix(j.find("origin"))
            name, q = j.get("name"), joints.get(j.get("name"), 0.0)
            if q and j.get("type") in ("revolute", "continuous"):
                ax = j.find("axis")
                axis = axis_override.get(name) or [float(v) for v in (ax.get("xyz") if ax is not None else "1 0 0").split()]
                T = T @ trimesh.transformations.rotation_matrix(q, axis)
            c = j.find("child").get("link")
            poses[c] = T
            stack.append(c)
    return poses


def _urdf_object(src: Cfg, out: str):
    """URDF links posed by forward kinematics -> textured meshes on z = 0 (tools/urdf_to_scene.py helpers)."""
    import xml.etree.ElementTree as ET

    from PIL import Image

    sys.path.insert(0, str(SIM_DIR / "tools"))
    from urdf_to_scene import _material, load_visuals  # noqa: PLC0415

    root = ET.parse(sim_path(src.path)).getroot()
    packages = {k: sim_path(v) for k, v in (src.get("packages") or {}).items()}
    poses = urdf_link_poses(root, dict(src.get("joints") or {}), dict(src.get("axis_override") or {}))
    parts = load_visuals(root, packages, poses)
    zmin = min(float(p.vertices[:, 2].min()) for _, p in parts)
    stage = _new_stage(out)
    tex_dir = os.path.join(os.path.dirname(out), "textures")
    os.makedirs(tex_dir, exist_ok=True)
    mats = {}
    for name, tm in parts:
        v = np.asarray(tm.vertices) - [0.0, 0.0, zmin]
        img = tm.metadata.get("texture")
        uv = getattr(tm.visual, "uv", None)
        mat = None
        if img is not None and os.path.exists(img) and uv is not None and len(uv) == len(v):
            if img not in mats:
                tex = os.path.join(tex_dir, os.path.splitext(os.path.basename(img))[0] + ".png")
                if not os.path.exists(tex):
                    im = Image.open(img).convert("RGB")
                    im.thumbnail((2048, 2048))
                    im.save(tex)
                mats[img] = _material(stage, f"/Object/Looks/M{len(mats)}", os.path.relpath(tex, os.path.dirname(out)))
            mat = mats[img]
            rgb = None
        else:
            vis = tm.visual
            rgb = (np.asarray(getattr(getattr(vis, "material", None), "main_color", [180, 180, 180, 255]))[:3]
                   if vis.kind == "texture" else np.asarray(vis.main_color[:3])) / 255.0
            uv = None
        _write_mesh(stage, f"/Object/{name}", v, np.asarray(tm.faces), rgb=rgb, uv=uv, material=mat)
    stage.GetRootLayer().Save()


def _primitive_object(src: Cfg, out: str):
    import trimesh

    kind = src.type
    meshes = []
    if kind == "box":
        sx, sy, sz = src.size
        m = trimesh.creation.box(extents=(sx, sy, sz))
        m.apply_translation([0, 0, sz / 2])
        meshes.append(m)
    elif kind == "boxes":
        for cx, cy, cz, sx, sy, sz in src.boxes:
            m = trimesh.creation.box(extents=(sx, sy, sz))
            m.apply_translation([cx, cy, cz + sz / 2])
            meshes.append(m)
    elif kind == "cone":
        meshes.append(trimesh.creation.cone(radius=src.radius, height=src.height, sections=48))
    elif kind == "pipes":
        r, length = src.radius, src.length
        z = r
        for k, n in enumerate(src.rows):  # rows stacked in a pyramid, axis along x
            for i in range(n):
                y = (i - (n - 1) / 2) * 2 * r
                m = trimesh.creation.cylinder(radius=r, height=length, sections=24)
                m.apply_transform(trimesh.transformations.rotation_matrix(math.pi / 2, [0, 1, 0]))
                m.apply_translation([0, y, z])
                meshes.append(m)
            z += r * math.sqrt(3)
    else:
        raise ValueError(f"unknown object source type {kind}")
    stage = _new_stage(out)
    mat = _color_material(stage, "/Object/Looks/M0", src.color)
    for i, m in enumerate(meshes):
        _write_mesh(stage, f"/Object/part_{i}", m.vertices, m.faces, rgb=src.color, material=mat)
    stage.GetRootLayer().Save()


def build_object(obj: Cfg, out_dir: str, fmt: str) -> str:
    out = os.path.join(out_dir, "objects", f"{obj.name}.{fmt}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    src = obj.source
    if src.type == "usd":
        _usd_object(src, out)
    elif src.type == "urdf":
        _urdf_object(src, out)
    else:
        _primitive_object(src, out)
    return out


def local_bbox(path: str):
    from pxr import Usd, UsdGeom

    st = Usd.Stage.Open(path)
    r = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"]).ComputeWorldBound(
        st.GetPseudoRoot()).ComputeAlignedRange()
    return np.array(r.GetMin()), np.array(r.GetMax())


# --------------------------------------------------------------------------------------------------------------------
# scene composition
# --------------------------------------------------------------------------------------------------------------------
def _background(stage, cfg: Cfg):
    import trimesh
    from pxr import Gf, UsdGeom, UsdLux

    bg = cfg.background
    if bg.type == "usd":
        prim = stage.DefinePrim("/World/Background", "Xform")
        prim.GetReferences().AddReference(sim_path(bg.path))
        for name in bg.get("deactivate") or []:
            stage.OverridePrim(f"/World/Background/{name}").SetActive(False)
    else:
        UsdGeom.Xform.Define(stage, "/World/Background")
    if bg.get("add_ground"):
        g = bg.ground_size / 2
        _write_mesh(stage, "/World/Background/Ground", np.array([(-g, -g, 0), (g, -g, 0), (g, g, 0), (-g, g, 0)]),
                    np.array([[0, 1, 2], [0, 2, 3]]), rgb=bg.ground_color)
    if bg.get("fence"):
        f = bg.fence
        h, t, L = f.height, f.thickness, f.half_size
        for i, (cx, cy, sx, sy) in enumerate([(0, -L, 2 * L, t), (0, L, 2 * L, t), (-L, 0, t, 2 * L), (L, 0, t, 2 * L)]):
            m = trimesh.creation.box(extents=(sx, sy, h))
            m.apply_translation([cx, cy, h / 2])
            _write_mesh(stage, f"/World/Background/Fence_{i}", m.vertices, m.faces, rgb=f.color)
    if bg.get("lights"):
        li = bg.lights
        dome = UsdLux.DomeLight.Define(stage, "/World/Lights/Dome")
        dome.CreateIntensityAttr(float(li.dome_intensity))
        sun = UsdLux.DistantLight.Define(stage, "/World/Lights/Sun")
        sun.CreateIntensityAttr(float(li.sun_intensity))
        sun.CreateAngleAttr(float(li.sun_angle_deg))
        UsdGeom.Xformable(sun).AddRotateXYZOp().Set(Gf.Vec3f(*map(float, li.sun_rotate_xyz_deg)))


def compose_scene(cfg: Cfg, obj_files: dict[str, str], out: str) -> dict[str, dict]:
    from pxr import Gf, Usd, UsdGeom, UsdSemantics

    if os.path.exists(out):
        os.remove(out)
    stage = Usd.Stage.CreateNew(out)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    _background(stage, cfg)
    UsdGeom.Xform.Define(stage, "/World/Objects")
    placed = {}
    for obj in cfg.objects:
        lo, hi = local_bbox(obj_files[obj.name])
        c_local = (lo + hi) / 2
        R = rot_z(obj.yaw_deg)
        t = np.array([*obj.center_xy, 0.0]) - R @ np.array([c_local[0], c_local[1], 0.0])
        x = UsdGeom.Xform.Define(stage, f"/World/Objects/{obj.name}")
        x.AddTranslateOp().Set(Gf.Vec3d(*map(float, t)))
        x.AddRotateZOp().Set(float(obj.yaw_deg))
        prim = x.GetPrim()
        prim.GetReferences().AddReference(os.path.relpath(obj_files[obj.name], os.path.dirname(out)))
        api = UsdSemantics.LabelsAPI.Apply(prim, cfg.build.label_instance)
        api.CreateLabelsAttr().Set([obj["class"]])
        placed[obj.name] = {"translate": t.tolist(), "yaw_deg": float(obj.yaw_deg)}
    stage.GetRootLayer().Save()
    return placed


# --------------------------------------------------------------------------------------------------------------------
# bake + ground truth
# --------------------------------------------------------------------------------------------------------------------
def bake(scene_usd: str, names: list[str]):
    """All visible (default/render purpose) mesh triangles in world space + object index per triangle."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(scene_usd, Usd.Stage.LoadAll)
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    cache = UsdGeom.XformCache()
    index = {f"/World/Objects/{n}": i for i, n in enumerate(names)}
    V, T, O, off = [], [], [], 0
    for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        img = UsdGeom.Imageable(prim)
        if img.ComputePurpose() not in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render):
            continue
        if img.ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        mesh = UsdGeom.Mesh(prim)
        pts, counts, idx = (mesh.GetPointsAttr().Get(), mesh.GetFaceVertexCountsAttr().Get(),
                            mesh.GetFaceVertexIndicesAttr().Get())
        if not pts or not counts:
            continue
        pts = np.asarray(pts, np.float64)
        M = np.asarray(cache.GetLocalToWorldTransform(prim), np.float64)
        pts = (np.c_[pts, np.ones(len(pts))] @ M)[:, :3] * mpu
        counts, idx = np.asarray(counts), np.asarray(idx)
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
        ok = counts >= 3
        if not ok.any():
            continue
        # fan triangulation, vectorised over faces with the same vertex count
        tris = []
        for c in np.unique(counts[ok]):
            s = starts[counts == c]
            for k in range(1, c - 1):
                tris.append(np.stack([idx[s], idx[s + k], idx[s + k + 1]], 1))
        tris = np.concatenate(tris)
        path = str(prim.GetPath())
        oid = next((i for p, i in index.items() if path == p or path.startswith(p + "/")), -1)
        V.append(pts)
        T.append(tris + off)
        O.append(np.full(len(tris), oid, np.int16))
        off += len(pts)
    return np.concatenate(V).astype(np.float32), np.concatenate(T).astype(np.int32), np.concatenate(O)


def write_raycast(path: str, verts: np.ndarray, tris: np.ndarray):
    from pxr import Usd, UsdGeom, Vt

    if os.path.exists(path):
        os.remove(path)
    stage = Usd.Stage.CreateNew(path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    m = UsdGeom.Mesh.Define(stage, "/World/Raycast")
    m.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(verts.astype(np.float32)))
    m.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(np.full(len(tris), 3, np.int32)))
    m.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(tris.astype(np.int32).reshape(-1)))
    stage.GetRootLayer().Save()


def surface_gt(verts, tris, tri_obj, n_obj: int, spacing: float, voxel: float, origin: np.ndarray):
    sel = tri_obj >= 0
    p, n, k = sample_triangles(verts.astype(np.float64), tris[sel], spacing)
    obj = tri_obj[sel][k]
    ijk = np.floor((p - origin) / voxel).astype(np.int64)
    key = voxel_keys(ijk) * 64 + obj  # an object id fits in 6 bits
    ukey, first = np.unique(key, return_index=True)
    return {
        "points": p.astype(np.float32), "normals": n.astype(np.float16), "obj": obj.astype(np.int16),
        "vox_ijk": ijk[first].astype(np.int32), "vox_obj": obj[first].astype(np.int16),
        "vox_normal": n[first].astype(np.float16), "voxel": np.float32(voxel), "origin": origin.astype(np.float32),
        "n_obj": np.int32(n_obj),
    }


def traversability(verts, tris, cfg: Cfg):
    b = cfg.bounds
    cell, (z0, z1) = cfg.build.grid_cell, cfg.build.grid_band
    lo, hi = np.array(b[:2]) - 1.0, np.array(b[2:]) + 1.0
    nx, ny = int(np.ceil((hi[0] - lo[0]) / cell)), int(np.ceil((hi[1] - lo[1]) / cell))
    tz = verts[tris][:, :, 2]
    tx, ty = verts[tris][:, :, 0], verts[tris][:, :, 1]
    keep = (tz.max(1) >= z0) & (tz.min(1) <= z1) & (tx.max(1) >= lo[0]) & (tx.min(1) <= hi[0]) & \
           (ty.max(1) >= lo[1]) & (ty.min(1) <= hi[1])
    occ = np.zeros((ny, nx), bool)
    sub = np.flatnonzero(keep)
    for s in range(0, len(sub), 200_000):
        p, _, _ = sample_triangles(verts.astype(np.float64), tris[sub[s:s + 200_000]], cfg.build.grid_sample_spacing,
                                   seed=s)
        p = p[(p[:, 2] >= z0) & (p[:, 2] <= z1)]
        ix = np.floor((p[:, 0] - lo[0]) / cell).astype(int)
        iy = np.floor((p[:, 1] - lo[1]) / cell).astype(int)
        ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        occ[iy[ok], ix[ok]] = True
    xs = lo[0] + (np.arange(nx) + 0.5) * cell
    ys = lo[1] + (np.arange(ny) + 0.5) * cell
    inside = (xs[None, :] >= b[0]) & (xs[None, :] <= b[2]) & (ys[:, None] >= b[1]) & (ys[:, None] <= b[3])
    occ |= ~inside
    return {"occupied": occ, "origin": lo.astype(np.float32), "cell": np.float32(cell)}


# --------------------------------------------------------------------------------------------------------------------
def build_scene(cfg: Cfg, log=print) -> dict:
    out_dir = sim_path(cfg.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    shutil.rmtree(os.path.join(out_dir, "objects"), ignore_errors=True)
    names = [o.name for o in cfg.objects]
    obj_files = {}
    for obj in cfg.objects:
        log(f"  object {obj.name} ({obj.source.type})")
        obj_files[obj.name] = build_object(obj, out_dir, cfg.build.object_format)
    scene_usd = os.path.join(out_dir, "scene.usda")
    placed = compose_scene(cfg, obj_files, scene_usd)
    log("  baking triangles")
    V, T, O = bake(scene_usd, names)
    write_raycast(os.path.join(out_dir, "raycast.usdc"), V, T)
    np.savez_compressed(os.path.join(out_dir, "gt.npz"), verts=V, tris=T, tri_obj=O, names=np.array(names))
    log(f"  {len(T):,} triangles ({(O >= 0).sum():,} on objects); sampling surface")
    b = cfg.bounds
    origin = np.array([b[0] - 1.0, b[1] - 1.0, cfg.z_range[0] - 1.0])
    surf = surface_gt(V, T, O, len(names), cfg.build.surface_spacing, cfg.build.surface_voxel, origin)
    np.savez_compressed(os.path.join(out_dir, "surface.npz"), **surf)
    log("  rasterising traversability")
    grid = traversability(V, T, cfg)
    np.savez_compressed(os.path.join(out_dir, "grid.npz"), **grid)
    objects = []
    for i, obj in enumerate(cfg.objects):
        tv = V[T[O == i].reshape(-1)]
        lo, hi = tv.min(0), tv.max(0)
        objects.append({
            "name": obj.name, "class": obj["class"], "target": bool(obj.target), "index": i,
            "center": ((lo + hi) / 2).tolist(), "bbox_min": lo.tolist(), "bbox_max": hi.tolist(),
            "height_m": float(hi[2]), "triangles": int((O == i).sum()),
            "surface_points": int((surf["obj"] == i).sum()), "surface_voxels": int((surf["vox_obj"] == i).sum()),
            "file": os.path.relpath(obj_files[obj.name], out_dir), **placed[obj.name],
        })
    meta = {
        "name": cfg.name, "config": cfg._source_, "bounds": list(cfg.bounds), "z_range": list(cfg.z_range),
        "robot_start": dict(cfg.robot_start), "objects": objects, "triangles": int(len(T)),
        "label_instance": cfg.build.label_instance, "surface_voxel": cfg.build.surface_voxel,
        "free_fraction": float(1.0 - grid["occupied"].mean()),
        "files": {"scene": "scene.usda", "raycast": "raycast.usdc", "gt": "gt.npz", "surface": "surface.npz",
                  "grid": "grid.npz"},
    }
    with open(os.path.join(out_dir, "scene.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def load_scene(out_dir: str) -> dict:
    d = sim_path(out_dir)
    with open(os.path.join(d, "scene.json")) as f:
        meta = json.load(f)
    meta["dir"] = d
    return meta
