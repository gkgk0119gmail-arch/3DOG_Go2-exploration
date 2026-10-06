"""Isaac clip + live point-cloud panel for a machine scan run (adaptive_scan.py with GO2_SAVE_SCANS=1).

    python tools/scan_video.py <clip.mp4> <scan_dir> <scan_log.json> <machine prefix> <out.mp4>
        [--speed 2] [--title "..."] [--elev 25 --azim -60]

Right panel, synced to the clip's sim time: (top) the machine's accumulated LiDAR points (5 cm voxels, coloured by
height) from a fixed 3D view, (bottom) mini-map: machine footprint, accumulated points, robot path, posture stops
(red stars), and the surface coverage % / curve from the run log.
"""

import argparse
import glob
import json
import os

import imageio.v2 as iio2
import imageio.v3 as iio
import numpy as np
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation as R

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
OUT_FPS = 25


def font(n, b=False):
    return ImageFont.truetype(BOLD if b else FONT, n)


def camera(center, dist, elev, azim):
    e, a = np.deg2rad(elev), np.deg2rad(azim)
    eye = center + dist * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])
    f = center - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return eye, np.stack([r, u, f])


def project(p, eye, Rc, fpx, w, h):
    c = (p - eye) @ Rc.T
    ok = c[:, 2] > 0.1
    x = w / 2 + fpx * c[:, 0] / np.maximum(c[:, 2], 1e-3)
    y = h / 2 - fpx * c[:, 1] / np.maximum(c[:, 2], 1e-3)
    return x, y, c[:, 2], ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("scans")
    ap.add_argument("log")
    ap.add_argument("machine")
    ap.add_argument("out")
    ap.add_argument("--speed", type=float, default=2.0)
    ap.add_argument("--title", default="")
    ap.add_argument("--elev", type=float, default=24)
    ap.add_argument("--azim", type=float, default=-60)
    ap.add_argument("--max_s", type=float, default=1e9)
    args = ap.parse_args()

    gt = np.load(args.machine + "_gt.npz")["verts"]
    lo, hi = gt.min(0) - 0.25, gt.max(0) + 0.25
    center = (gt.min(0) + gt.max(0)) / 2
    size = float(np.linalg.norm(gt.max(0) - gt.min(0)))
    log = json.load(open(args.log))
    cov = np.array(log["coverage"]) if log["coverage"] else np.zeros((1, 2))
    stops = log["stops"]
    taken_t = [d["t"] for d in log["decisions"] if d["taken"]]
    files = sorted(glob.glob(os.path.join(args.scans, "*.npz")))
    scans = []
    for f in files:
        z = np.load(f)
        scans.append((float(z["stamp"]), R.from_quat(z["quat_w"]).apply(z["points"]) + z["pos_w"], z["pos_w"]))
    t0 = scans[0][0] - 0.1  # first sweep is taken one LiDAR period after the clip starts
    src_fps = iio.immeta(args.clip).get("fps", 50)

    PW, PH = 620, 720  # panel
    cmap = colormaps["turbo"]
    eye, Rc = camera(center, 1.25 * size, args.elev, args.azim)
    fpx = 0.85 * PW
    # mini-map frame: machine footprint + loop
    pts_xy = np.array([s[2][:2] for s in scans])
    mlo = np.minimum(gt.min(0)[:2], pts_xy.min(0)) - 0.8
    mhi = np.maximum(gt.max(0)[:2], pts_xy.max(0)) + 0.8
    mh = 240
    ms = min((PW - 40) / (mhi[0] - mlo[0]), mh / (mhi[1] - mlo[1]))

    def mxy(xy):
        return 20 + (xy[..., 0] - mlo[0]) * ms, 436 + mh - (xy[..., 1] - mlo[1]) * ms

    foot = gt[np.random.default_rng(0).choice(len(gt), min(len(gt), 20000), replace=False)]

    vox = {}
    k = 0
    path = []
    w = iio2.get_writer(args.out, fps=OUT_FPS, codec="libx264", quality=8, macro_block_size=8)
    step = src_fps * args.speed / OUT_FPS
    nxt = 0.0
    for i, fr in enumerate(iio.imiter(args.clip)):
        t = i / src_fps
        if t > args.max_s:
            break
        if i < nxt:
            continue
        nxt += step
        while k < len(scans) and scans[k][0] - t0 <= t:
            p = scans[k][1]
            p = p[np.isfinite(p).all(1) & (p >= lo).all(1) & (p <= hi).all(1)]
            for key, q in zip(map(tuple, np.floor(p / 0.05).astype(int)), p):
                vox[key] = q
            path.append(scans[k][2])
            k += 1
        panel = Image.new("RGB", (PW, PH), (20, 22, 26))
        d = ImageDraw.Draw(panel)
        c_now = float(np.interp(t, cov[:, 0], cov[:, 1])) if len(cov) > 1 else 0.0
        d.text((18, 12), f"점군 {len(vox):,} 복셀 · 표면 커버리지 {100 * c_now:4.1f}%", fill=(240, 240, 240),
               font=font(24, True))
        if vox:
            P = np.array(list(vox.values()))
            x, y, depth, ok = project(P, eye, Rc, fpx, PW, 330)
            col = (cmap(np.clip((P[:, 2] - gt.min(0)[2]) / max(np.ptp(gt[:, 2]), 1e-3), 0, 1))[:, :3] * 255).astype(np.uint8)
            order = np.argsort(-depth)  # far first: nearer points overwrite
            order = order[ok[order]]
            xi, yi = x[order].astype(int), y[order].astype(int) + 40
            m = (xi >= 0) & (xi < PW - 1) & (yi >= 50) & (yi < 352)
            xi, yi, cc = xi[m], yi[m], col[order][m]
            arr = np.asarray(panel).copy()
            for dx in (0, 1):
                for dy in (0, 1):
                    arr[yi + dy, xi + dx] = cc
            panel = Image.fromarray(arr)
            d = ImageDraw.Draw(panel)
        # mini-map
        d.rectangle([12, 428, PW - 12, 436 + mh + 8], outline=(70, 72, 78))
        arr = np.asarray(panel).copy()
        fx, fy = mxy(foot[:, :2])
        arr[fy.astype(int).clip(0, PH - 1), fx.astype(int).clip(0, PW - 1)] = (85, 88, 96)
        if vox:
            P = np.array(list(vox.values()))
            px, py = mxy(P[np.argsort(P[:, 2])][:, :2])  # higher points on top
            colm = (cmap(np.clip((np.sort(P[:, 2]) - gt.min(0)[2]) / max(np.ptp(gt[:, 2]), 1e-3), 0, 1))[:, :3] * 255
                    ).astype(np.uint8)
            arr[py.astype(int).clip(0, PH - 1), px.astype(int).clip(0, PW - 1)] = colm
        panel = Image.fromarray(arr)
        d = ImageDraw.Draw(panel)
        if len(path) > 1:
            q = np.array(path)[:, :2]
            qx, qy = mxy(q)
            d.line(list(zip(qx, qy)), fill=(240, 240, 240), width=2)
            d.ellipse([qx[-1] - 6, qy[-1] - 6, qx[-1] + 6, qy[-1] + 6], fill=(27, 175, 122))
        for s in stops:
            if s["t"] <= t:
                sx, sy = mxy(np.array(s["pos"]))
                n_post = sum(1 for tt in taken_t if s["t"] - 0.01 <= tt <= s["t"] + 6 and tt <= t)
                if n_post:
                    d.text((sx - 9, sy - 15), "★", fill=(227, 73, 72), font=font(22))
        d.text((18, 436 + mh + 12), "미니맵: 회색 = 장비 윤곽 · 색 점 = 스캔 (높이) · ★ = 몸 기울여 스캔한 지점",
               fill=(190, 190, 190), font=font(16))
        # coverage curve
        # coverage strip: 0-100 % over the run
        d.rectangle([12, 360, PW - 12, 420], outline=(70, 72, 78))
        d.text((18, 362), "커버리지", fill=(160, 160, 160), font=font(14))
        if len(cov) > 1:
            m = cov[:, 0] <= t
            cx = 20 + cov[m, 0] / max(cov[-1, 0], 1) * (PW - 40)
            cy = 416 - cov[m, 1] * 52
            if m.sum() > 1:
                d.line(list(zip(cx, cy)), fill=(27, 175, 122), width=2)
        left = Image.fromarray(fr).convert("RGB").resize((int(fr.shape[1] * PH / fr.shape[0]), PH))
        canvas = Image.new("RGB", (left.width + PW, PH + 50), (20, 22, 26))
        canvas.paste(left, (0, 50))
        canvas.paste(panel, (left.width, 50))
        ImageDraw.Draw(canvas).text((16, 10), f"{args.title}   t = {t:5.1f} s", fill=(245, 245, 245), font=font(26))
        w.append_data(np.asarray(canvas)[: canvas.height // 8 * 8, : canvas.width // 8 * 8])
    w.close()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
