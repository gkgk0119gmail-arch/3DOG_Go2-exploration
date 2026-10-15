"""Generate the procedural scene pool for RL training / validation and a preview figure.

    python scripts/pose_inspection/rl_make_scenes.py [--train 200] [--val 20] [--out assets/pose_inspection/rl_scenes]

Scenes alternate warehouse / yard. Seeds: train k -> 1000 + k, val k -> 900000 + k (disjoint). The two hand-built
scenes are not in the pool (held-out test).
"""

import argparse
import json
import os
import sys
import time

import numpy as np

SIM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SIM)
from go2_lab.pose_inspection.config import load_config  # noqa: E402
from go2_lab.pose_inspection.navgrid import NavGrid  # noqa: E402
from go2_lab.pose_inspection.rl.procgen import load_parts, make_scene  # noqa: E402


def preview(scenes, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from scipy.spatial import ConvexHull

    n = len(scenes)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 5.2))
    for ax, m in zip(np.atleast_1d(axes), scenes):
        g = np.load(os.path.join(m["dir"], "grid.npz"))
        gt = np.load(os.path.join(m["dir"], "gt.npz"))
        occ, go, gc = g["occupied"], g["origin"], float(g["cell"])
        ax.imshow(occ, origin="lower", cmap="Greys", vmin=0, vmax=1.4, interpolation="nearest",
                  extent=[go[0], go[0] + occ.shape[1] * gc, go[1], go[1] + occ.shape[0] * gc])
        V, T, O = gt["verts"], gt["tris"], gt["tri_obj"]
        for ob in m["objects"]:
            p = V[T[O == ob["index"]].reshape(-1)][:, :2]
            ax.add_patch(Polygon(p[ConvexHull(p).vertices], fc="#f28c28" if ob["target"] else "#a3abb5", ec="k",
                                 lw=0.5))
        r = m["robot_start"]["region"]
        ax.add_patch(plt.Rectangle((r[0], r[1]), r[2] - r[0], r[3] - r[1], fc="none", ec="#2b6cb0", lw=1.5))
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        nt = sum(o["target"] for o in m["objects"])
        ax.set_title(f"{m['name']} ({m['kind']})\n{nt} machines, {len(m['objects']) - nt} clutter", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=70)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=200)
    ap.add_argument("--val", type=int, default=20)
    ap.add_argument("--out", default="assets/pose_inspection/rl_scenes")
    ap.add_argument("--parts", default="assets/pose_inspection/rl_parts")
    ap.add_argument("--preview", default="results/pose_inspection/rl_scenes_preview.png")
    args = ap.parse_args()
    cfg = load_config(os.path.join(SIM, "configs/pose_inspection/base.yaml"))
    parts = load_parts(os.path.join(SIM, args.parts))
    shown = []
    t0 = time.time()
    for split, n, base in (("train", args.train, 1000), ("val", args.val, 900000)):
        k = 0
        made = 0
        while made < n:
            seed = base + k
            k += 1
            rng = np.random.default_rng(seed)
            kind = "warehouse" if made % 2 == 0 else "yard"
            name = f"{split}_{made:04d}"
            m = make_scene(rng, parts, kind, os.path.join(SIM, args.out, split, name), name)
            # usable: at least one machine, robots can start in the start region
            nav = NavGrid(os.path.join(m["dir"], "grid.npz"), cfg.robot.radius)
            ok = any(o["target"] for o in m["objects"])
            try:
                nav.restrict_to_region(m["robot_start"]["region"])
                nav.sample_starts(m["robot_start"]["region"], 1, 0, m["robot_start"]["min_clearance"])
            except Exception:  # noqa: BLE001
                ok = False
            if not ok:
                continue
            m["seed"] = seed
            with open(os.path.join(m["dir"], "scene.json"), "w") as f:
                json.dump({k_: v for k_, v in m.items() if k_ != "dir"}, f, indent=1)
            made += 1
            if split == "train" and len(shown) < 6:
                shown.append(m)
        print(f"{split}: {n} scenes ({k} tried), {time.time() - t0:.0f} s")
    if shown:
        preview(shown, os.path.join(SIM, args.preview))


if __name__ == "__main__":
    main()
