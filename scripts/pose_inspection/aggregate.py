"""Aggregate inspection runs into POSE-style tables and coverage-over-time figures.

    python scripts/pose_inspection/aggregate.py --engine kinematic --conds pose planar ... \
        [--scenes A_warehouse B_construction] [--name stage3] [--labels "POSE (geometric)" ...]

Per scene, T = the longest effective run time among all compared runs; every run's AUCo is the area under its Co
curve up to its effective terminal, held to T, divided by T (POSE Sec. V-B). Writes
<results>/<name>_<engine>.json, .md (mean +- std over seeds) and .png (Co and Cs over time, mean +- 1 sigma).
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

SIM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SIM)

from go2_lab.pose_inspection.metrics import auc, mean_std, value_at  # noqa: E402

COLS = {"t": 0, "Co": 1, "Co_n26": 2, "Cs_raw": 3, "Cs": 4, "Co_vs_ub": 5, "L": 6, "N_post": 7}


def load(root, scene, cond):
    runs = []
    for f in sorted(glob.glob(os.path.join(root, scene, cond, "seed*.json"))):
        with open(f) as fh:
            runs.append(json.load(fh))
    return runs


def fmt(ms, nd=2):
    m, s = ms
    return "–" if not np.isfinite(m) else f"{m:.{nd}f} ± {s:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="kinematic")
    ap.add_argument("--conds", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None)
    ap.add_argument("--scenes", nargs="+", default=["A_warehouse", "B_construction"])
    ap.add_argument("--name", default="stage3")
    ap.add_argument("--results", default="results/pose_inspection")
    args = ap.parse_args()
    labels = dict(zip(args.conds, args.labels)) if args.labels else {c: c for c in args.conds}
    root = os.path.join(SIM, args.results, args.engine)
    out = {"engine": args.engine, "scenes": {}}
    md = [f"# {args.name} ({args.engine} engine)\n"]
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(args.scenes), 2, figsize=(14, 4.6 * len(args.scenes)), squeeze=False)
    colors = plt.get_cmap("tab10")
    for si, scene in enumerate(args.scenes):
        runs = {c: load(root, scene, c) for c in args.conds}
        runs = {c: r for c, r in runs.items() if r}
        if not runs:
            continue
        T = max(r["summary"]["t_eff"] for rr in runs.values() for r in rr)
        res = {"T": T, "conditions": {}}
        md.append(f"\n## {scene}  (T = {T:.0f} s)\n")
        md.append("| method | AUCo ↑ | t_eff [s] ↓ | L_eff [m] ↓ | N_post | Co ↑ | Co / upper bound ↑ | Cs ↑ | runs (timeouts) |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        tgrid = np.linspace(0, T, 400)
        for ci, (c, rr) in enumerate(runs.items()):
            rows = []
            curves_co, curves_cs = [], []
            for r in rr:
                s = r["summary"]
                cur = [(row[0], row[COLS["Co"]]) for row in r["curve"]]
                curcs = [(row[0], row[COLS["Cs"]]) for row in r["curve"]]
                a = auc(cur, T, t_end=s["t_eff"])
                rows.append({**s, "AUCo": a})
                curves_co.append([value_at([x for x in cur if x[0] <= s["t_eff"]] or cur[:1], t) for t in tgrid])
                curves_cs.append([value_at([x for x in curcs if x[0] <= s["t_eff"]] or curcs[:1], t) for t in tgrid])
            agg = {k: mean_std([x[k] for x in rows]) for k in ("AUCo", "t_eff", "L_eff", "N_post", "Co", "Co_vs_ub", "Cs")}
            per_obj = {}
            for n in rr[0]["targets"]:
                per_obj[n] = mean_std([x["Co_per_object"][n] for x in rows])
            n_to = sum(x["timeout"] for x in rows)
            res["conditions"][c] = {"label": labels[c], "n": len(rows), "timeouts": n_to,
                                    **{k: {"mean": v[0], "std": v[1]} for k, v in agg.items()},
                                    "Co_per_object": {k: {"mean": v[0], "std": v[1]} for k, v in per_obj.items()},
                                    "runs": rows}
            md.append(f"| {labels[c]} | {fmt(agg['AUCo'])} | {fmt(agg['t_eff'], 0)} | {fmt(agg['L_eff'], 0)} | "
                      f"{fmt(agg['N_post'], 1)} | {fmt(agg['Co'])} | {fmt(agg['Co_vs_ub'])} | {fmt(agg['Cs'])} | "
                      f"{len(rows)} ({n_to}) |")
            col = colors(ci)
            for ax, cv, nm in ((axes[si, 0], curves_co, "Co"), (axes[si, 1], curves_cs, "Cs")):
                cv = np.array(cv)
                mu, sd = cv.mean(0), cv.std(0)
                ax.plot(tgrid, mu, color=col, lw=2,
                        label=f"{labels[c]} (AUC {agg['AUCo'][0]:.2f})" if nm == "Co" else labels[c])
                ax.fill_between(tgrid, mu - sd, mu + sd, color=col, alpha=0.15)
        # per-object table
        md.append("\nper-object Co (mean ± std):\n")
        names = next(iter(runs.values()))[0]["targets"]
        md.append("| method | " + " | ".join(names) + " |")
        md.append("|---|" + "---|" * len(names))
        for c, v in res["conditions"].items():
            md.append(f"| {v['label']} | " + " | ".join(f"{v['Co_per_object'][n]['mean']:.2f} ± "
                                                       f"{v['Co_per_object'][n]['std']:.2f}" for n in names) + " |")
        # perception (stage 4) and VLM (stage 5/6) bookkeeping, when present
        extra_rows = []
        for c, rr in runs.items():
            per = rr[0].get("perception")
            if per:
                cls = [k for k in per if isinstance(per[k], dict)]
                extra_rows.append(f"| {labels[c]} | perception | frames {per['frames']}, "
                                  f"{per['wall_s'] / max(1, per['frames']):.2f} s/frame | " + "; ".join(
                                      f"{k}: P {per[k]['precision']:.2f} R {per[k]['recall']:.2f}" for k in cls) + " |")
            v = rr[0].get("vlm")
            if v:
                cl = v["client"]
                acc = sum(p["accepted"] for p in v["per_robot"].values())
                fb = sum(p["fallback_ticks"] for p in v["per_robot"].values())
                vt = sum(p["verdict_ticks"] for p in v["per_robot"].values())
                extra_rows.append(f"| {labels[c]} | VLM | {cl['calls']} calls ({cl['errors']} errors), latency "
                                  f"{cl['latency_mean_s']:.1f} s mean / {cl['latency_max_s']:.1f} s max, tokens "
                                  f"{(cl['tokens_in'] + cl['tokens_out']) / 1000 / max(1, len(rr)):.1f} k per run | "
                                  f"accepted verdicts {acc}, planning ticks on a verdict {vt / max(1, vt + fb):.0%} |")
            res["conditions"][c]["perception"] = per
            res["conditions"][c]["vlm"] = v
        if extra_rows:
            md.append("\n| method | module | load | quality |")
            md.append("|---|---|---|---|")
            md.extend(extra_rows)
        out["scenes"][scene] = res
        axes[si, 0].set_title(f"{scene}: object-surface coverage Co")
        axes[si, 1].set_title(f"{scene}: scene coverage Cs (observable volume)")
        for ax in axes[si]:
            ax.set_xlabel("time [s]")
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.3)
        axes[si, 0].legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    base = os.path.join(SIM, args.results, f"{args.name}_{args.engine}")
    fig.savefig(base + ".png", dpi=80)
    with open(base + ".json", "w") as f:
        json.dump(out, f, indent=1)
    with open(base + ".md", "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print("wrote", base + ".{json,md,png}")


if __name__ == "__main__":
    main()
