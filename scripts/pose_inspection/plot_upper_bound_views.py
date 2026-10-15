"""Stage 2 figure: where on each machine the theoretical upper bound comes from.

    python scripts/pose_inspection/plot_upper_bound_views.py [--tilt 15] [--out results/pose_inspection/stage2_ub_views]

Every GT surface voxel (0.1 m) of a machine coloured by the sweep of compute_upper_bound.py (every reachable position
on a 0.5 m lattice x 8 headings x the posture grid): seen with the body level / seen only when the body tilts / never
seen. Left column: the reachable lattice (top view) around the machine.
"""

import argparse
import json
import os
import sys

import numpy as np

SIM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SIM)
from go2_lab.pose_inspection.config import load_config  # noqa: E402
from go2_lab.pose_inspection.navgrid import NavGrid  # noqa: E402

COL = {"level": "#2ca25f", "posture": "#f28c28", "never": "#9aa3ad"}
LABEL = {"level": "seen with the body level", "posture": "seen only when the body tilts", "never": "never seen"}
CASES = [("A_warehouse", "forklift_2"), ("B_construction", "excavator_b"), ("B_construction", "dump_truck")]


def decode(keys):
    obj = keys % 64
    vk = keys // 64
    m = (1 << 21) - 1
    return np.stack([vk >> 42, (vk >> 21) & m, vk & m], 1), obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tilt", default="15")
    ap.add_argument("--out", default="results/pose_inspection/stage2_ub_views")
    args = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cfg = load_config(os.path.join(SIM, "configs/pose_inspection/base.yaml"))
    fig = plt.figure(figsize=(22, 6.4 * len(CASES)))
    gs = fig.add_gridspec(len(CASES), 3, width_ratios=[1, 1.25, 1.25], wspace=0.02, hspace=0.12, top=0.94, bottom=0.04)
    out = {}
    for r, (scene, name) in enumerate(CASES):
        d = os.path.join(SIM, "assets/pose_inspection", scene)
        meta = json.load(open(os.path.join(d, "scene.json")))
        ob = next(o for o in meta["objects"] if o["name"] == name)
        s = np.load(os.path.join(d, "surface.npz"))
        u = np.load(os.path.join(d, f"ub_surface_tilt{args.tilt}.npz"))
        ijk, obj = decode(u["keys"])
        sel = obj == ob["index"]
        p = s["origin"] + (ijk[sel] + 0.5) * float(s["voxel"])
        lv, ps = u["level"][sel], u["posture"][sel]
        cls = np.where(lv, "level", np.where(ps, "posture", "never"))
        frac = {k: float((cls == k).mean()) for k in COL}
        out[f"{scene}/{name}"] = frac

        # reachable lattice around the machine (top view)
        ax = fig.add_subplot(gs[r, 0])
        nav = NavGrid(os.path.join(d, "grid.npz"), cfg.robot.radius).restrict_to_region(meta["robot_start"]["region"])
        g = np.load(os.path.join(d, "grid.npz"))
        occ, go, gc = g["occupied"], g["origin"], float(g["cell"])
        ax.imshow(occ, origin="lower", cmap="Greys", vmin=0, vmax=1.6, interpolation="nearest",
                  extent=[go[0], go[0] + occ.shape[1] * gc, go[1], go[1] + occ.shape[0] * gc])
        xy = np.asarray(nav.lattice(cfg.upper_bound.grid_step))
        c = np.asarray(ob["center"][:2])
        win = 7.0
        near = np.all(np.abs(xy - c) < win, 1)
        ax.scatter(xy[near, 0], xy[near, 1], s=6, c="#4a7bd0", label="reachable positions (0.5 m)")
        ax.scatter(p[:, 0], p[:, 1], s=1, c=[COL[k] for k in cls])
        ax.set_xlim(c[0] - win, c[0] + win)
        ax.set_ylim(c[1] - win, c[1] + win)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{scene} / {name}: robot positions tried\n(x 8 headings x 15 postures each)", fontsize=12)
        if r == 0:
            ax.legend(loc="lower left", fontsize=10)

        # two 3D views of the machine surface: each 0.1 m surface voxel drawn as a cube in its class colour
        vi = ijk[sel] - ijk[sel].min(0)
        dims = vi.max(0) + 1
        filled = np.zeros(dims, bool)
        fc = np.zeros(tuple(dims) + (4,))
        rgba = {k: np.array(matplotlib.colors.to_rgba(v)) for k, v in COL.items()}
        filled[vi[:, 0], vi[:, 1], vi[:, 2]] = True
        for k in COL:
            m = cls == k
            fc[vi[m, 0], vi[m, 1], vi[m, 2]] = rgba[k]
        for k, (elev, azim) in enumerate(((28, -55), (28, 125))):
            ax = fig.add_subplot(gs[r, 1 + k], projection="3d")
            ax.voxels(filled, facecolors=fc, edgecolor=(0, 0, 0, 0.08), linewidth=0.2, shade=True)
            ax.view_init(elev=elev, azim=azim)
            ax.set_box_aspect(tuple(dims), zoom=1.15)
            ax.set_axis_off()
            if k == 0:
                ax.set_title(f"level {frac['level']:.0%}  +  tilt only {frac['posture']:.0%}  =  "
                             f"{frac['level'] + frac['posture']:.0%} visible;  never {frac['never']:.0%}",
                             fontsize=14)
    handles = [Line2D([], [], marker="o", ls="", color=COL[k], label=LABEL[k], ms=10) for k in COL]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=16, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(f"Upper bound: which surface can the VLP-16 ({args.tilt} deg mount) reach from anywhere reachable",
                 fontsize=17, y=0.995)
    base = os.path.join(SIM, args.out)
    fig.savefig(base + ".png", dpi=70, bbox_inches="tight")
    json.dump(out, open(base + ".json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
