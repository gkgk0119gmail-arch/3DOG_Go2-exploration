"""Stage 2 figure: how the upper bound is swept -- every reachable robot position, 8 headings, 15 body postures.

    python scripts/pose_inspection/plot_ub_lattice.py [--out results/pose_inspection/stage2_ub_lattice]

Top: each scene from above -- obstacles, machines, the cells the Go2 can reach (clearance > robot radius, connected
to the start region) and the 0.5 m lattice of positions the sweep tries. Bottom: one position close-up (8 headings)
and the side view of the 15 postures (VLP-16 vertical field of view, 15 deg nose-up mount + body pitch) next to a
3.93 m forklift.
"""

import argparse
import json
import math
import os
import sys

import numpy as np

SIM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SIM)
from go2_lab.pose_inspection.config import load_config  # noqa: E402
from go2_lab.pose_inspection.navgrid import NavGrid  # noqa: E402

SCENES = ["A_warehouse", "B_construction"]
C_REACH, C_TOOCLOSE, C_OBST, C_DOT = "#d6e6f5", "#f6d5cf", "#5b6470", "#2b6cb0"
C_TARGET, C_OTHER = "#f28c28", "#a3abb5"


def scene_panel(ax, cfg, scene, hull_cache):
    from matplotlib.patches import Polygon
    from scipy.spatial import ConvexHull

    d = os.path.join(SIM, "assets/pose_inspection", scene)
    meta = json.load(open(os.path.join(d, "scene.json")))
    nav = NavGrid(os.path.join(d, "grid.npz"), cfg.robot.radius)
    clear_all = nav.clear.copy()
    nav.restrict_to_region(meta["robot_start"]["region"])
    img = np.ones(nav.occ.shape + (3,))
    rgb = lambda h: np.array([int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)])  # noqa: E731
    img[~nav.occ & ~nav.reachable] = rgb(C_TOOCLOSE)
    img[nav.reachable] = rgb(C_REACH)
    img[nav.occ] = rgb(C_OBST)
    ext = [nav.origin[0], nav.origin[0] + nav.nx * nav.cell, nav.origin[1], nav.origin[1] + nav.ny * nav.cell]
    ax.imshow(img, origin="lower", extent=ext, interpolation="nearest")
    gt = np.load(os.path.join(d, "gt.npz"))
    V, T, O = gt["verts"], gt["tris"], gt["tri_obj"]
    for ob in meta["objects"]:
        p3 = V[T[O == ob["index"]].reshape(-1)]
        pts = p3[:, :2]
        h = pts[ConvexHull(pts).vertices]
        low = p3[p3[:, 2] < 0.6][:, :2]   # what blocks a 0.4 m tall robot (raised forks do not)
        hull_cache[(scene, ob["name"])] = h
        hull_cache[(scene, ob["name"], "low")] = low[ConvexHull(low).vertices] if len(low) > 3 else h
        ax.add_patch(Polygon(h, closed=True, fc=C_TARGET if ob["target"] else C_OTHER, ec="k", lw=0.6, alpha=0.95))
    xy = np.asarray(nav.lattice(cfg.upper_bound.grid_step))
    ax.scatter(xy[:, 0], xy[:, 1], s=1.6, c=C_DOT, lw=0)
    n_post = len(cfg.upper_bound.pitch_deg) * len(cfg.upper_bound.roll_deg)
    n_yaw = len(cfg.upper_bound.yaws_deg)
    b = meta["bounds"]
    ax.set_xlim(b[0] - 0.5, b[2] + 0.5)
    ax.set_ylim(b[1] - 0.5, b[3] + 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{scene}: {len(xy):,} positions x {n_yaw} headings x {n_post} postures = "
                 f"{len(xy) * n_yaw * n_post:,} LiDAR sweeps", fontsize=12)
    return nav, meta, xy, clear_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/pose_inspection/stage2_ub_lattice")
    args = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Polygon, Wedge

    cfg = load_config(os.path.join(SIM, "configs/pose_inspection/base.yaml"))
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.45, 1], width_ratios=[1, 1, 1.05], hspace=0.12, wspace=0.08)
    hulls = {}
    nav_a, meta_a, xy_a, _ = scene_panel(fig.add_subplot(gs[0, 0]), cfg, "A_warehouse", hulls)
    ax_b = fig.add_subplot(gs[0, 1:])
    scene_panel(ax_b, cfg, "B_construction", hulls)

    # close-up around one warehouse forklift: lattice + 8 headings at one position
    ax = fig.add_subplot(gs[1, 0])
    ob = next(o for o in meta_a["objects"] if o["name"] == "forklift_2")
    c = np.asarray(ob["center"][:2])
    win = 3.2
    img = np.ones(nav_a.occ.shape + (3,))
    img[nav_a.reachable] = matplotlib.colors.to_rgb(C_REACH)
    img[~nav_a.occ & ~nav_a.reachable] = matplotlib.colors.to_rgb(C_TOOCLOSE)
    img[nav_a.occ] = matplotlib.colors.to_rgb(C_OBST)
    ext = [nav_a.origin[0], nav_a.origin[0] + nav_a.nx * nav_a.cell, nav_a.origin[1],
           nav_a.origin[1] + nav_a.ny * nav_a.cell]
    ax.imshow(img, origin="lower", extent=ext, interpolation="nearest")
    ax.add_patch(Polygon(hulls[("A_warehouse", "forklift_2")], closed=True, fc=C_TARGET, ec="k", lw=0.8, ls="--",
                         alpha=0.35))
    ax.add_patch(Polygon(hulls[("A_warehouse", "forklift_2", "low")], closed=True, fc=C_TARGET, ec="k", lw=0.8))
    ax.text(c[0], c[1] - 1.9, "forks raised 1 m:\nthe robot fits under them", ha="center", fontsize=10)
    near = np.all(np.abs(xy_a - c) < win, 1)
    ax.scatter(xy_a[near, 0], xy_a[near, 1], s=28, c=C_DOT, zorder=3)
    p0 = xy_a[near][np.argmin(np.abs(np.linalg.norm(xy_a[near] - c, axis=1) - 2.4))]
    for yd in cfg.upper_bound.yaws_deg:
        a = math.radians(yd)
        ax.annotate("", xy=(p0[0] + 0.9 * math.cos(a), p0[1] + 0.9 * math.sin(a)), xytext=p0,
                    arrowprops=dict(arrowstyle="-|>", color="#c4620a", lw=2.2), zorder=4)
    ax.scatter([p0[0]], [p0[1]], s=90, c="#c4620a", ec="k", zorder=5)
    ax.set_xlim(c[0] - win, c[0] + win)
    ax.set_ylim(c[1] - win, c[1] + win)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("close-up: 0.5 m lattice next to a forklift;\nat every position the robot faces 8 headings",
                 fontsize=12)

    # side view: VLP-16 vertical FOV for the 5 body pitches (roll tilts the same fan sideways, +-15 deg)
    ax = fig.add_subplot(gs[1, 1:])
    li, rb = cfg.lidar, cfg.robot
    h0 = rb.base_height + li.mount_pos[2] + li.scan_height
    dist, H = 3.0, 3.93
    ax.add_patch(plt.Rectangle((dist, 0), 1.2, H, fc=C_TARGET, ec="k", lw=0.8, alpha=0.9))
    ax.text(dist + 0.6, H + 0.12, "forklift 3.93 m", ha="center", fontsize=11)
    ax.plot([-0.8, 7.0], [0, 0], color="k", lw=1)
    ax.add_patch(plt.Rectangle((-0.35, rb.base_height - 0.12), 0.7, 0.2, fc="#d9dde3", ec="k"))
    cols = plt.get_cmap("viridis")(np.linspace(0.1, 0.9, len(cfg.upper_bound.pitch_deg)))
    for pd, col in zip(cfg.upper_bound.pitch_deg, cols):
        # body pitch < 0 = nose up in our convention: the fan centre is mount_tilt - pitch above the horizon
        mid = li.mount_tilt_deg - pd
        lo, hi = mid + li.vfov_deg[0], mid + li.vfov_deg[1]
        ax.add_patch(Wedge((0.2, h0), 6.0, lo, hi, fc=col, alpha=0.13, ec=col, lw=1.4))
        top = h0 + (dist - 0.2) * math.tan(math.radians(min(hi, 80)))
        ax.plot([0.2, dist], [h0, min(top, 5.2)], color=col, lw=1.8,
                label=f"pitch {-pd + 0.0:+.1f} deg: fan {lo:+.0f}..{hi:+.0f} deg"
                      f" -> reaches {min(top, 9.9):.1f} m at 3 m")
    ax.text(6.05, 0.15, "6 m = d_max\n(Co counts\nreturns within)", fontsize=9, color="#555")
    ax.set_xlim(-0.8, 12.5)
    ax.set_ylim(-0.1, 5.4)
    ax.set_aspect("equal")
    ax.set_xlabel("distance [m]")
    ax.set_ylabel("height [m]")
    ax.legend(loc="center right", fontsize=10, title="body pitch (+ = nose up)  x  roll {-15, 0, +15} deg",
              title_fontsize=10)
    ax.set_title("side view: the 15 postures = 5 body pitches x 3 rolls (VLP-16 +-15 deg, mounted 15 deg nose-up)",
                 fontsize=12)

    leg = [Patch(fc=C_REACH, ec="#999", label="reachable by the Go2 (clearance > 0.37 m, connected)"),
           Patch(fc=C_TOOCLOSE, ec="#999", label="free but too close to an obstacle / not connected"),
           Patch(fc=C_OBST, label="obstacle (walls, shelves, fence)"),
           Patch(fc=C_TARGET, ec="k", label="target machine"), Patch(fc=C_OTHER, ec="k", label="other object"),
           plt.Line2D([], [], marker="o", ls="", color=C_DOT, label="lattice position (0.5 m)")]
    fig.legend(handles=leg, loc="upper center", ncol=6, fontsize=11, frameon=False, bbox_to_anchor=(0.5, 0.935))
    fig.suptitle("Upper bound sweep: every reachable position x 8 headings x 15 postures", fontsize=17, y=0.955)
    base = os.path.join(SIM, args.out)
    fig.savefig(base + ".png", dpi=70, bbox_inches="tight")
    print("wrote", base + ".png")


if __name__ == "__main__":
    main()
