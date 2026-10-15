"""Learning curves of an RL run (log.csv of rl_train.py): Co, t_eff, AUCo (T = 1200 s), STOP share, per scene kind.

    python scripts/pose_inspection/rl_plot.py results/pose_inspection/rl/run1 [--window 10]
"""

import argparse
import csv
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--window", type=int, default=10)
    args = ap.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = list(csv.DictReader(open(os.path.join(args.run, "log.csv"))))
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.2))
    cols = {"warehouse": "#2b6cb0", "yard": "#c4620a"}
    for kind, c in cols.items():
        rr = [r for r in rows if r["kind"] == kind]
        if not rr:
            continue
        it = np.array([int(r["it"]) for r in rr])
        for ax, key, lab in zip(axes, ("Co", "t_eff", "auc", "stop_frac"),
                                ("final coverage Co", "effective time t_eff [s]", "AUCo (T = 1200 s)",
                                 "episodes ended by STOP")):
            y = np.array([float(r[key]) for r in rr])
            w = max(1, min(args.window, len(y)))
            sm = np.convolve(y, np.ones(w) / w, mode="valid")
            ax.plot(it, y, ".", color=c, alpha=0.3)
            ax.plot(it[w - 1:], sm, color=c, lw=2, label=kind)
            ax.set_title(lab, fontsize=11)
            ax.set_xlabel("iteration (one procedural scene, 12 robots)")
            ax.grid(alpha=0.3)
    axes[0].legend()
    fig.suptitle(f"RL head training: {os.path.basename(args.run)} (dots = one iteration, line = moving mean)")
    fig.tight_layout()
    out = os.path.join(args.run, "curves.png")
    fig.savefig(out, dpi=75)
    print("wrote", out)


if __name__ == "__main__":
    main()
