"""Stage 6 figure: coverage-time trade-off of geometric pruning for each "where is the unseen part" model.

    python scripts/pose_inspection/plot_tradeoff.py [--engine kinematic] [--out results/pose_inspection/stage6_tradeoff]

Each method = one line through its prune thresholds (runs named sweep_<method>_<thr>), plus its no-pruning point
(every sector visited: pose / prior_template / prior_cad). Mean +- std over seeds of t_eff (x) and Co (y); a second
row shows AUCo. Writes <out>.png / .json / .md.
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

SIM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SIM)
from go2_lab.pose_inspection.metrics import auc  # noqa: E402

METHODS = [("box", "box + unknown voxels (POSE)", "pose", "#7f7f7f"),
           ("template", "class template + missing surface (ours)", "prior_template", "#1b9e77"),
           ("cad", "exact CAD + missing surface (upper bound)", "prior_cad", "#7570b3")]


def load(d):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "seed*.json")))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="kinematic")
    ap.add_argument("--scenes", nargs="+", default=["A_warehouse", "B_construction"])
    ap.add_argument("--out", default="results/pose_inspection/stage6_tradeoff")
    args = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = os.path.join(SIM, "results/pose_inspection", args.engine)
    fig, axes = plt.subplots(2, len(args.scenes), figsize=(7 * len(args.scenes), 10), squeeze=False)
    out, md = {}, ["# Stage 6: pruning trade-off per unseen-part model\n"]
    for j, scene in enumerate(args.scenes):
        groups = {}
        for m, _, full, _ in METHODS:
            pts = []
            for d in glob.glob(os.path.join(root, scene, f"sweep_{m}_*")):
                thr = int(re.search(r"_(\d+)$", d).group(1))
                pts.append((thr, load(d)))
            pts.sort()
            if os.path.isdir(os.path.join(root, scene, full)):
                pts.append(("all", load(os.path.join(root, scene, full))))
            groups[m] = [(k, r) for k, r in pts if r]
        allruns = [r for g in groups.values() for _, rr in g for r in rr]
        if not allruns:
            continue
        T = max(r["summary"]["t_eff"] for r in allruns)
        out[scene] = {"T": T}
        md.append(f"\n## {scene} (T = {T:.0f} s)\n")
        md.append("| method | prune threshold | t_eff [s] | Co | AUCo | N_post |")
        md.append("|---|---|---|---|---|---|")
        for m, label, _, col in METHODS:
            rows = []
            for k, rr in groups.get(m, []):
                te = np.array([r["summary"]["t_eff"] for r in rr])
                co = np.array([r["summary"]["Co"] for r in rr])
                au = np.array([auc([(c[0], c[1]) for c in r["curve"]], T, r["summary"]["t_eff"]) for r in rr])
                npst = np.array([r["summary"]["N_post"] for r in rr])
                rows.append((k, te.mean(), te.std(), co.mean(), co.std(), au.mean(), au.std(), npst.mean()))
                md.append(f"| {label} | {k} | {te.mean():.0f} ± {te.std():.0f} | {co.mean():.2f} ± {co.std():.2f} | "
                          f"{au.mean():.2f} ± {au.std():.2f} | {npst.mean():.1f} |")
            out[scene][m] = rows
            if not rows:
                continue
            r = np.array([x[1:] for x in rows], float)
            for row, ax_i, yi in ((0, 0, 2), (1, 1, 4)):
                ax = axes[ax_i, j]
                ax.errorbar(r[:, 0], r[:, yi], xerr=r[:, 1], yerr=r[:, yi + 1], color=col, marker="o", capsize=3,
                            label=label, lw=1.6)
                for x, y, (k, *_) in zip(r[:, 0], r[:, yi], rows):
                    ax.annotate(str(k), (x, y), textcoords="offset points", xytext=(4, 4), fontsize=8, color=col)
        for i, yl in enumerate(("final object-surface coverage Co", "AUCo (common T)")):
            ax = axes[i, j]
            ax.set_xlabel("effective time t_eff [s]")
            ax.set_ylabel(yl)
            ax.grid(alpha=0.3)
            ax.set_title(f"{scene}: {yl.split(' (')[0]} vs time (labels = prune threshold, all = no pruning)",
                         fontsize=10)
            ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    base = os.path.join(SIM, args.out)
    fig.savefig(base + ".png", dpi=80)
    with open(base + ".json", "w") as f:
        json.dump(out, f, indent=1)
    with open(base + ".md", "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
