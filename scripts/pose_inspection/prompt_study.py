"""Stage 4 prompt study: which words make Grounding DINO find our machines, and how good are the SAM 2 masks?

    python scripts/pose_inspection/prompt_study.py [--scenes A_warehouse B_construction]

Data: logs/pose_inspection/views/<scene>.npz (render_views.py). Per view, GT instances = connected components of the
target class pixels (>= --min_px). A detection matches a component when box IoU >= 0.5 with the component's box.
1) each candidate phrase alone: recall by distance, false positives per image
2) the configured prompt set (base.yaml perception.gsam2.prompts) with SAM 2: per-class pixel IoU / precision /
   recall of the union mask vs the GT class mask
Writes results/pose_inspection/stage4_prompt_study.{json,md,png}.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from scipy import ndimage

SIM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SIM)

from go2_lab.pose_inspection.config import load_config  # noqa: E402

CANDIDATES = {
    "forklift": ["forklift", "reach truck", "pallet stacker", "lift truck", "industrial vehicle"],
    "excavator": ["excavator", "mini excavator", "digger", "backhoe", "construction machine"],
    "dump_truck": ["dump truck", "crawler carrier", "tracked dumper", "crawler dump truck", "truck"],
}


def components(mask, min_px):
    lab, n = ndimage.label(mask)
    out = []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        if len(ys) >= min_px:
            out.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
    return out


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(u, 1e-9)


@torch.inference_mode()
def run_detector(proc, model, images, text, thr, tthr, dev, bs=8):
    from PIL import Image

    out = []
    for s in range(0, len(images), bs):
        ims = [Image.fromarray(im) for im in images[s:s + bs]]
        inp = proc(images=ims, text=[text] * len(ims), return_tensors="pt").to(dev)
        o = model(**inp)
        res = proc.post_process_grounded_object_detection(o, inp.input_ids, threshold=thr, text_threshold=tthr,
                                                          target_sizes=[im.size[::-1] for im in ims])
        for r in res:
            labs = r.get("text_labels", r.get("labels"))
            out.append([(float(sc), [float(v) for v in bx], str(lb)) for sc, bx, lb in
                        zip(r["scores"].tolist(), r["boxes"].tolist(), labs)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", nargs="+", default=["A_warehouse", "B_construction"])
    ap.add_argument("--min_px", type=int, default=300)
    args = ap.parse_args()
    cfg = load_config("base.yaml")
    g = cfg.perception.gsam2
    dev = "cuda:0"
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    proc = AutoProcessor.from_pretrained(g.detector)
    det = AutoModelForZeroShotObjectDetection.from_pretrained(g.detector).to(dev).eval()
    report = {"single_phrase": {}, "prompt_set": {}}
    md = ["# Stage 4 prompt study (Grounding DINO base, SAM 2.1 large)\n"]
    md.append("## 1. Single phrases: recall of GT instances by camera distance, false positives per image\n")
    md.append("| scene | class | phrase | recall 3 m | 6 m | 10 m | 15 m | all | FP / image |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    data = {s: np.load(os.path.join(SIM, "logs/pose_inspection/views", f"{s}.npz")) for s in args.scenes}
    for s, d in data.items():
        classes = list(d["classes"])
        rgb = list(d["rgb"])
        for ci, cls in enumerate(classes):
            gt = [components(m == ci + 1, args.min_px) for m in d["cls"]]
            for ph in CANDIDATES[cls]:
                dets = run_detector(proc, det, rgb, ph + " .", g.box_threshold, g.text_threshold, dev)
                hit_by_dist, fp = {}, 0
                for k, (gg, dd) in enumerate(zip(gt, dets)):
                    used = set()
                    for comp in gg:
                        ok = any(iou(comp, b) >= 0.5 for _, b, _ in dd)
                        hit_by_dist.setdefault(float(d["dist"][k]), []).append(ok)
                    for j, (_, b, _) in enumerate(dd):
                        if not any(iou(comp, b) >= 0.5 for comp in gg):
                            fp += 1
                        used.add(j)
                rec = {k: float(np.mean(v)) for k, v in sorted(hit_by_dist.items())}
                allr = float(np.mean([x for v in hit_by_dist.values() for x in v])) if hit_by_dist else float("nan")
                report["single_phrase"].setdefault(s, {}).setdefault(cls, {})[ph] = {
                    "recall_by_dist": rec, "recall": allr, "fp_per_image": fp / len(rgb)}
                cells = " | ".join(f"{rec.get(x, float('nan')):.2f}" for x in (3.0, 6.0, 10.0, 15.0))
                md.append(f"| {s} | {cls} | {ph} | {cells} | {allr:.2f} | {fp / len(rgb):.2f} |")
                print(md[-1], flush=True)

    # --- configured prompt set with SAM 2 masks ------------------------------------------------------------------
    from go2_lab.pose_inspection.perception import GroundedSam2

    gs = GroundedSam2(g, dev)
    md.append("\n## 2. Configured prompt set + SAM 2: pixel metrics of the class masks\n")
    md.append(f"prompts: `{json.dumps(dict(g.prompts))}`\n")
    md.append("| scene | class | IoU | precision | recall | images with the class |")
    md.append("|---|---|---|---|---|---|")
    examples = []
    for s, d in data.items():
        classes = list(d["classes"])
        stats = {c: [0, 0, 0, 0] for c in classes}  # tp, fp, fn, n_img
        for k in range(len(d["rgb"])):
            res = gs([d["rgb"][k]], classes)[0]
            for ci, c in enumerate(classes):
                gtm = d["cls"][k] == ci + 1
                pm = np.zeros_like(gtm)
                for cc, _, m in res:
                    if cc == c:
                        pm |= m
                stats[c][0] += int((pm & gtm).sum())
                stats[c][1] += int((pm & ~gtm).sum())
                stats[c][2] += int((~pm & gtm).sum())
                stats[c][3] += int(gtm.sum() >= args.min_px)
            if k % 9 == 0 and len(examples) < 8:
                examples.append((s, float(d["dist"][k]), str(d["target"][k]), d["rgb"][k], d["cls"][k], res, classes))
        for c, (tp, fp, fn, n) in stats.items():
            iou_ = tp / max(1, tp + fp + fn)
            pr, rc = tp / max(1, tp + fp), tp / max(1, tp + fn)
            report["prompt_set"].setdefault(s, {})[c] = {"iou": iou_, "precision": pr, "recall": rc, "images": n}
            md.append(f"| {s} | {c} | {iou_:.2f} | {pr:.2f} | {rc:.2f} | {n} |")
            print(md[-1], flush=True)
    res_dir = os.path.join(SIM, "results/pose_inspection")
    with open(os.path.join(res_dir, "stage4_prompt_study.json"), "w") as f:
        json.dump(report, f, indent=1)
    with open(os.path.join(res_dir, "stage4_prompt_study.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(examples), 2, figsize=(10, 3.9 * len(examples)), squeeze=False)
    for r, (s, dist, tgt, im, gtc, res, classes) in enumerate(examples):
        axes[r, 0].imshow(im)
        axes[r, 0].set_title(f"{s} / {tgt} @ {dist:.0f} m")
        ov = im.astype(float) / 255
        for cc, sc, m in res:
            col = np.array(plt.get_cmap("tab10")(classes.index(cc) if cc in classes else 9)[:3])
            ov[m] = 0.5 * ov[m] + 0.5 * col
            ys, xs = np.nonzero(m)
            if len(ys):
                axes[r, 1].text(xs.mean(), ys.mean(), f"{cc} {sc:.2f}", color="w", fontsize=8,
                                bbox={"fc": "k", "alpha": 0.5, "pad": 1})
        axes[r, 1].imshow(ov)
        axes[r, 1].contour(gtc > 0, levels=[0.5], colors="yellow", linewidths=0.8)
        axes[r, 1].set_title("Grounded-SAM 2 masks (yellow = GT outline)")
        for a in axes[r]:
            a.set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, "stage4_prompt_study.png"), dpi=70)
    print("wrote results/pose_inspection/stage4_prompt_study.{json,md,png}")


if __name__ == "__main__":
    main()
