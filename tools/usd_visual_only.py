"""Flatten a converted asset into a visual-only USD (no rigid bodies, colliders, joints, or mass).

Used to attach sensor models (e.g. the VLP-16 mount) as child geometry of an existing robot link,
so they ride along with that link without adding bodies to the articulation.

Usage:
    python usd_visual_only.py in.usda out.usda
"""

import argparse

from pxr import Sdf, Usd, UsdGeom, UsdPhysics

PHYSICS_API_PREFIXES = ("Physics", "Physx", "Mjc", "Newton", "IsaacRobot", "IsaacLink", "IsaacJoint")


def strip_api_schemas(layer: Sdf.Layer) -> None:
    """Drop physics / robot API schemas from every prim spec's explicit apiSchemas list."""

    def visit(path):
        spec = layer.GetPrimAtPath(path)
        if spec is None or not spec.HasInfo("apiSchemas"):
            return
        op = spec.GetInfo("apiSchemas")
        keep = lambda items: [a for a in items if not a.startswith(PHYSICS_API_PREFIXES)]  # noqa: E731
        new = Sdf.TokenListOp()
        if op.isExplicit:
            new.explicitItems = keep(op.explicitItems)
        else:
            new.prependedItems = keep(op.prependedItems)
            new.appendedItems = keep(op.appendedItems)
        spec.SetInfo("apiSchemas", new)

    layer.Traverse(Sdf.Path.absoluteRootPath, visit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    args = ap.parse_args()

    src = Usd.Stage.Open(args.input, Usd.Stage.LoadAll)
    # de-instance in the session layer (source file untouched) so Flatten() inlines the meshes
    src.SetEditTarget(src.GetSessionLayer())
    while instanced := [p for p in src.Traverse() if p.IsInstanceable()]:
        for prim in instanced:
            prim.SetInstanceable(False)
    stage = Usd.Stage.Open(src.Flatten())
    to_remove = []
    for prim in stage.Traverse():
        if prim.IsA(UsdPhysics.Joint) or prim.GetName() == "Physics":
            to_remove.append(prim.GetPath())
            continue
        for api in list(prim.GetAppliedSchemas()):
            if api.startswith(PHYSICS_API_PREFIXES):
                prim.RemoveAppliedSchema(api)
        for attr in prim.GetAttributes():
            if attr.GetName().startswith(("physics:", "physx", "mjc:", "newton:")):
                prim.RemoveProperty(attr.GetName())
        # colliders are not needed for a visual-only attachment
        if prim.GetName() == "collisions" or UsdGeom.Imageable(prim).GetPurposeAttr().Get() == "guide":
            to_remove.append(prim.GetPath())
    for path in sorted(to_remove, key=lambda p: -len(str(p))):
        stage.RemovePrim(path)
    strip_api_schemas(stage.GetRootLayer())
    stage.GetRootLayer().Export(args.output)
    out = Usd.Stage.Open(args.output)
    n_mesh = sum(1 for p in out.Traverse() if p.IsA(UsdGeom.Mesh))
    n_phys = sum(1 for p in out.Traverse() if any(a.startswith(PHYSICS_API_PREFIXES) for a in p.GetAppliedSchemas()))
    print(f"wrote {args.output}: meshes={n_mesh} prims_with_physics_api={n_phys} defaultPrim={out.GetDefaultPrim().GetPath()}")


if __name__ == "__main__":
    main()
