"""Figure of the observable upper bound (stage 2): per target, GT surface voxels coloured by what can observe them.

    python scripts/pose_inspection/plot_upper_bound.py [--tilt 15]
"""

import argparse
import json
import os
import sys

import numpy as np

SIM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SIM)


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ap = argparse.ArgumentParser()
    ap.add_argument("--tilt", type=float, default=15.0)
    ap.add_argument("--out", default="results/pose_inspection/stage2_upper_bound.png")
    args = ap.parse_args()
    ub = json.load(open(os.path.join(SIM, "results/pose_inspection/stage2_upper_bound.json")))
    shown = [("A_warehouse", "forklift_0"), ("B_construction", "excavator_a"), ("B_construction", "dump_truck")]
    fig, axes = plt.subplots(1, len(shown), figsize=(6 * len(shown), 5.5), subplot_kw={"projection": "3d"})
    for ax, (scene, name) in zip(axes, shown):
        d = os.path.join(SIM, "assets/pose_inspection", scene)
        meta = json.load(open(os.path.join(d, "scene.json")))
        ob = next(o for o in meta["objects"] if o["name"] == name)
        s = np.load(os.path.join(d, "surface.npz"))
        u = np.load(os.path.join(d, f"ub_surface_tilt{args.tilt:g}.npz"))
        keys = u["keys"]
        ijk = np.stack([(keys // 64) >> 42, ((keys // 64) >> 21) & ((1 << 21) - 1), (keys // 64) & ((1 << 21) - 1)], 1)
        sel = u["obj"] == ob["index"]
        c = s["origin"] + (ijk[sel] + 0.5) * float(s["voxel"])
        lvl, pst = u["level"][sel], u["posture"][sel]
        for m, col, lab in ((lvl, "#1baf7a", "level posture"), (pst & ~lvl, "#f2a900", "only with tilt"),
                            (~pst, "#e34948", "never observable")):
            ax.scatter(c[m, 0], c[m, 1], c[m, 2], s=1.5, c=col, depthshade=False, label=f"{lab} ({m.mean():.0%})")
        ax.set_box_aspect(np.ptp(c, 0))
        ax.view_init(25, -60)
        ax.set_axis_off()
        r = ub[scene][f"tilt{args.tilt:g}"]["per_object"][name]
        ax.set_title(f"{scene} / {name}\nupper bound: level {r['level']:.0%}, with posture {r['posture']:.0%}")
        ax.legend(loc="lower left", fontsize=8, markerscale=6)
    fig.suptitle(f"Observable GT surface voxels (VLP-16 mount tilt {args.tilt:g} deg, all reachable positions)")
    fig.tight_layout()
    fig.savefig(os.path.join(SIM, args.out), dpi=80)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
