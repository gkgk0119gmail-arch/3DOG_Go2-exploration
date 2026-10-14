"""Top-down GIF of one inspection run: map, robot path, executed postures, per-object coverage, Co / Cs curves.

    python scripts/pose_inspection/make_run_gif.py <run seed json> [<run2 json> ...] --out media/x.gif
        [--labels "POSE" "planar"] [--step 3] [--fps 12]

Several runs (same scene) are drawn side by side over a common clock.
"""

import argparse
import json
import os
import sys

import numpy as np

SIM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SIM)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=float, default=3.0, help="sim seconds per frame")
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--width", type=float, default=6.0, help="inches per run panel")
    args = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from PIL import Image
    from scipy.spatial import ConvexHull

    runs = [json.load(open(p)) for p in args.runs]
    labels = args.labels or [r["tag"] for r in runs]
    scene = runs[0]["scene"]
    d = os.path.join(SIM, "assets/pose_inspection", scene)
    meta = json.load(open(os.path.join(d, "scene.json")))
    g = np.load(os.path.join(d, "grid.npz"))
    gt = np.load(os.path.join(d, "gt.npz"))
    V, T, O = gt["verts"], gt["tris"], gt["tri_obj"]
    hulls = {}
    for ob in meta["objects"]:
        pts = V[T[O == ob["index"]].reshape(-1)][:, :2]
        hulls[ob["index"]] = pts[ConvexHull(pts).vertices]
    occ, go, gc = g["occupied"], g["origin"], float(g["cell"])
    b = meta["bounds"]
    T_end = max(r["curve"][-1][0] for r in runs)
    times = np.arange(0, T_end + args.step, args.step)
    cmap = plt.get_cmap("RdYlGn")
    frames = []
    n = len(runs)
    aspect = (b[3] - b[1]) / (b[2] - b[0])
    for t in times:
        fig = plt.figure(figsize=(args.width * n + 4.5, args.width * aspect + 0.8), dpi=60)
        gs = fig.add_gridspec(2, n + 1, width_ratios=[args.width] * n + [4.5])
        for i, (r, lab) in enumerate(zip(runs, labels)):
            ax = fig.add_subplot(gs[:, i])
            ax.imshow(occ, origin="lower", extent=[go[0], go[0] + occ.shape[1] * gc, go[1], go[1] + occ.shape[0] * gc],
                      cmap="Greys", vmin=0, vmax=1.8, interpolation="nearest")
            cur = [c for c in r["curve"] if c[0] <= t] or r["curve"][:1]
            per = cur[-1][8]
            for ob in meta["objects"]:
                if not ob["target"]:
                    ax.add_patch(Polygon(hulls[ob["index"]], closed=True, fc="#b8b8c8", ec="none"))
                    continue
                v = per[ob["index"]]
                ax.add_patch(Polygon(hulls[ob["index"]], closed=True, fc=cmap(v), ec="k", lw=0.6))
                ax.text(ob["center"][0], ob["center"][1], f"{v:.0%}", ha="center", va="center", fontsize=8)
            tr = np.array([p for p in r.get("traj", []) if p[0] <= t])
            if len(tr):
                ax.plot(tr[:, 1], tr[:, 2], color="#2c7fb8", lw=1.4)
                x, y, yaw = tr[-1, 1:4]
                ax.plot(x, y, marker=(3, 0, np.degrees(yaw) - 90), ms=11, color="#08306b")
            for ev in r["events"]:
                if ev["type"] == "posture" and ev["t"] <= t:
                    ax.plot(*ev["xy"], marker="*", ms=11, color="#e6550d", mec="k", mew=0.4)
            ax.set_xlim(b[0] - 0.5, b[2] + 0.5)
            ax.set_ylim(b[1] - 0.5, b[3] + 0.5)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{lab}   t = {t:4.0f} s   Co {cur[-1][1]:.0%}   postures {cur[-1][7]}", fontsize=11)
        for k, (col, nm) in enumerate(((1, "object-surface coverage Co"), (4, "scene coverage Cs"))):
            ax = fig.add_subplot(gs[k, n])
            for r, lab, c in zip(runs, labels, ("#1b9e77", "#d95f02", "#7570b3", "#e7298a")):
                cv = np.array([[x[0], x[col]] for x in r["curve"]])
                ax.plot(cv[:, 0], cv[:, 1], color=c, lw=0.8, alpha=0.3)
                m = cv[:, 0] <= t
                ax.plot(cv[m, 0], cv[m, 1], color=c, lw=2, label=lab)
            ax.axvline(t, color="k", lw=0.6)
            ax.set_xlim(0, T_end)
            ax.set_ylim(0, 1)
            ax.set_title(nm, fontsize=10)
            ax.grid(alpha=0.3)
            if k == 0:
                ax.legend(loc="lower right", fontsize=8)
        fig.text(0.01, 0.01, "orange star = executed body posture; object colour = its surface coverage", fontsize=8)
        fig.tight_layout()
        fig.canvas.draw()
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        plt.close(fig)
    out = os.path.join(SIM, args.out) if not os.path.isabs(args.out) else args.out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    pal = [f.convert("P", palette=Image.ADAPTIVE, colors=128) for f in frames]
    pal[0].save(out, save_all=True, append_images=pal[1:] + [pal[-1]] * args.fps * 2, duration=int(1000 / args.fps),
                loop=0, optimize=True)
    print("wrote", out, f"{os.path.getsize(out) / 1e6:.1f} MB, {len(frames)} frames")


if __name__ == "__main__":
    main()
