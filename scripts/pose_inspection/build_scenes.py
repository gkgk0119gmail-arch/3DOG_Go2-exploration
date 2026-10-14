"""Build the POSE-inspection scenes and write the stage-1 summary (no Isaac app needed, pxr + numpy only).

    python scripts/pose_inspection/build_scenes.py [scene_A_warehouse.yaml scene_B_construction.yaml]
        [--results results/pose_inspection]

Outputs per scene: see go2_lab/pose_inspection/scene_builder.py. Summary: <results>/stage1_scenes.json and
<results>/stage1_scenes.png (top view: traversability, object boxes by class, robot start region; GT surface points).
"""

import argparse
import json
import os
import sys
import time

import numpy as np

SIM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SIM)

from go2_lab.pose_inspection.config import load_config, sim_path  # noqa: E402
from go2_lab.pose_inspection.scene_builder import build_scene  # noqa: E402

CLASS_COLORS = {"forklift": "#d95f02", "excavator": "#e6ab02", "dump_truck": "#1b9e77", "material_pile": "#7570b3",
                "container": "#386cb0"}


def figure(metas, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle

    fig, axes = plt.subplots(2, len(metas), figsize=(8 * len(metas), 15), gridspec_kw={"height_ratios": [2.2, 1]})
    axes = np.atleast_2d(axes).reshape(2, -1)
    for j, m in enumerate(metas):
        ax = axes[0, j]
        g = np.load(os.path.join(m["dir"], "grid.npz"))
        occ, o, c = g["occupied"], g["origin"], float(g["cell"])
        ny, nx = occ.shape
        ax.imshow(occ, origin="lower", extent=[o[0], o[0] + nx * c, o[1], o[1] + ny * c], cmap="Greys", vmin=0,
                  vmax=1.6, interpolation="nearest")
        gt = np.load(os.path.join(m["dir"], "gt.npz"))
        V, T, O = gt["verts"], gt["tris"], gt["tri_obj"]
        for ob in m["objects"]:
            # footprint = convex hull of the object's vertices (top view)
            from scipy.spatial import ConvexHull

            pts = V[T[O == ob["index"]].reshape(-1)][:, :2]
            hull = pts[ConvexHull(pts).vertices]
            col = CLASS_COLORS.get(ob["class"], "k")
            ax.add_patch(Polygon(hull, closed=True, fc=col, ec="k", alpha=0.85 if ob["target"] else 0.4, lw=0.8))
            ax.text(ob["center"][0], ob["center"][1], f"{ob['name']}\n{ob['height_m']:.1f} m", ha="center",
                    va="center", fontsize=8, color="k")
        r = m["robot_start"]["region"]
        ax.add_patch(Rectangle((r[0], r[1]), r[2] - r[0], r[3] - r[1], fill=False, ec="#2c7fb8", lw=2, ls="--"))
        ax.text(r[0], r[1] - 0.8, "robot start region", color="#2c7fb8", fontsize=9)
        b = m["bounds"]
        ax.set_xlim(b[0] - 1, b[2] + 1)
        ax.set_ylim(b[1] - 1, b[3] + 1)
        ax.set_aspect("equal")
        ax.set_title(f"{m['name']}: {sum(o['target'] for o in m['objects'])} targets, "
                     f"{m['triangles'] / 1e6:.2f} M triangles", fontsize=12)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        # GT surface samples of the targets, side view (x-z), coloured by object
        ax2 = axes[1, j]
        s = np.load(os.path.join(m["dir"], "surface.npz"))
        rng = np.random.default_rng(0)
        for ob in m["objects"]:
            if not ob["target"]:
                continue
            p = s["points"][s["obj"] == ob["index"]]
            p = p[rng.random(len(p)) < min(1.0, 15000 / max(1, len(p)))]
            ax2.scatter(p[:, 0], p[:, 2], s=0.2, color=CLASS_COLORS.get(ob["class"], "k"))
            ax2.text(ob["center"][0], ob["height_m"] + 0.2, ob["name"], ha="center", fontsize=8)
        ax2.set_aspect("equal")
        ax2.set_xlabel("x [m]")
        ax2.set_ylabel("z [m]")
        ax2.set_title("ground-truth surface samples of the targets (side view)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=80)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", nargs="*", default=["scene_A_warehouse.yaml", "scene_B_construction.yaml"])
    ap.add_argument("--results", default="results/pose_inspection")
    args = ap.parse_args()
    metas = []
    for c in args.configs:
        cfg = load_config(c)
        t0 = time.time()
        print(f"[{cfg.name}] building -> {cfg.out_dir}")
        m = build_scene(cfg)
        m["dir"] = sim_path(cfg.out_dir)
        m["build_s"] = round(time.time() - t0, 1)
        print(f"[{cfg.name}] done in {m['build_s']} s")
        for o in m["objects"]:
            print(f"   {o['name']:18s} {o['class']:14s} target={o['target']!s:5s} h={o['height_m']:.2f} m "
                  f"tris={o['triangles']:>8,} surf_vox={o['surface_voxels']:>7,}")
        metas.append(m)
    res = sim_path(args.results)
    os.makedirs(res, exist_ok=True)
    with open(os.path.join(res, "stage1_scenes.json"), "w") as f:
        json.dump([{k: v for k, v in m.items() if k != "dir"} for m in metas], f, indent=1)
    figure(metas, os.path.join(res, "stage1_scenes.png"))
    print("wrote", os.path.join(res, "stage1_scenes.json"), os.path.join(res, "stage1_scenes.png"))


if __name__ == "__main__":
    main()
