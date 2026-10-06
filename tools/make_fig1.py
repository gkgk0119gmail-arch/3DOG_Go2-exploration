"""POSE-style overview figure of the machine scan (figures/fig9_posture_scan.png).

    python tools/make_fig1.py <scan out dir (videos_excavator/scan)> [--out figures/fig9_posture_scan.png]

(a) Isaac frame: the Go2 holding a tilted posture next to the excavator (exc_fig1 run)
(b) the same moment: accumulated points coloured by height, the robot pose and the VLP-16 rays of that sweep
(c) the whole adaptive run seen from above: points, path, posture stops (red stars)
(d) surface coverage over time, level vs fixed postures vs gain-driven postures, excavator and dump truck
"""

import argparse
import glob
import json
import os
import sys

import imageio.v3 as iio
import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_figures as mf  # noqa: E402

plt = mf.plt
SIM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
C = {"level": "#8a8984", "fixed": "#2a78d6", "adaptive": "#1baf7a"}
NAME = {"level": "몸 수평", "fixed": "고정 자세 순서", "adaptive": "이득 기반 자세 선택"}


def scans_of(d, t_max=None):
    out = []
    files = sorted(glob.glob(os.path.join(d, "*.npz")))
    t0 = float(np.load(files[0])["stamp"]) - 0.1
    for f in files:
        z = np.load(f)
        t = float(z["stamp"]) - t0
        if t_max is not None and t > t_max:
            break
        out.append((t, R.from_quat(z["quat_w"]).apply(z["points"]) + z["pos_w"], z["pos_w"], z["quat_w"]))
    return out


def machine_points(scans, lo, hi):
    vox = {}
    for _, p, _, _ in scans:
        p = p[np.isfinite(p).all(1) & (p >= lo).all(1) & (p <= hi).all(1)]
        for k, q in zip(map(tuple, np.floor(p / 0.05).astype(int)), p):
            vox[k] = q
    return np.array(list(vox.values()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scan_dir")
    ap.add_argument("--out", default=os.path.join(SIM, "figures", "fig9_posture_scan.png"))
    args = ap.parse_args()
    D = args.scan_dir
    gt = np.load(os.path.join(SIM, "assets", "excavator", "ix35e_gt.npz"))["verts"]
    lo, hi = gt.min(0) - 0.25, gt.max(0) + 0.25
    zr = (gt[:, 2].min(), gt[:, 2].max())
    cmap = plt.get_cmap("turbo")

    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=mf.SURF)
    fig.text(0.03, 0.965, "몸을 기울여 보는 4족 로봇 스캔: Go2 + VLP-16, 굴착기", fontsize=24, weight="bold",
             color=mf.INK, va="top")
    fig.text(0.03, 0.915, "정지 지점마다 자세 후보(앞뒤 5 × 좌우 5)의 새 관측 면을 미리 ray-cast해 가장 이득이 큰 자세로 기울임 "
             "(장비 CAD 모델을 사전 정보로 사용) · Isaac Sim / Isaac Lab", fontsize=12.5, color=mf.INK2, va="top")

    # (a) + (b): the moment of the strongest nose-up posture in the fig1 run
    log = json.load(open(os.path.join(D, "log_exc_fig1", "scan_log.json")))
    taken = [d for d in log["decisions"] if d["taken"]]
    pick = min(taken, key=lambda d: d["best_deg"][0])  # most nose-up
    t_show = pick["t"] + 1.2
    clip = os.path.join(D, "exc_fig1.mp4")
    fps = iio.immeta(clip).get("fps", 50)
    frame = [f for i, f in enumerate(iio.imiter(clip)) if i == int(t_show * fps)][0]
    ax = fig.add_axes([0.02, 0.43, 0.47, 0.44])
    h, w = frame.shape[:2]
    ax.imshow(frame[int(0.12 * h):, int(0.06 * w):int(0.94 * w)])
    ax.axis("off")
    ax.set_title(f"(a) 고개를 {abs(pick['best_deg'][0]):.0f}° 들고 좌우 {pick['best_deg'][1]:+.0f}° 기울여 굴착기 윗부분을 스캔",
                 fontsize=12.5, color=mf.INK, loc="left")

    sc = scans_of(os.path.join(open(os.path.join(D, "scans_exc_fig1.txt")).read().strip()), t_show)
    P = machine_points(sc, lo, hi)
    ax = fig.add_axes([0.5, 0.43, 0.48, 0.44], projection="3d")
    ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=cmap((P[:, 2] - zr[0]) / (zr[1] - zr[0])), s=0.8, depthshade=False)
    _, hits, pos, quat = sc[-1]
    hm = hits[np.isfinite(hits).all(1) & (hits >= lo).all(1) & (hits <= hi).all(1)]
    sel = hm[np.random.default_rng(0).choice(len(hm), min(160, len(hm)), replace=False)] if len(hm) else hm
    for q in sel:
        ax.plot([pos[0], q[0]], [pos[1], q[1]], [pos[2], q[2]], color="#e34948", lw=0.35, alpha=0.55)
    Rb = R.from_quat(quat).as_matrix()
    box = np.array([[x, y, z] for x in (-0.35, 0.35) for y in (-0.15, 0.15) for z in (-0.25, -0.1)])
    bw = (box @ Rb.T) + pos
    for i, j in [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]:
        ax.plot(*zip(bw[i], bw[j]), color=mf.INK, lw=1.4)
    ax.scatter(*pos, color="#e34948", s=25)
    allp = np.vstack([P, pos[None]])
    ax.set_box_aspect(np.ptp(allp, 0) + 0.2)
    ax.view_init(18, -125)
    ax.set_axis_off()
    ax.set_title("(b) 같은 순간의 점군 (높이 색) · 검정 상자 = 기울인 Go2 몸체 · 빨간 선 = 이번 스캔",
                 fontsize=12.5, color=mf.INK, loc="left")

    # (c) whole adaptive run from above
    ax = fig.add_axes([0.02, 0.05, 0.44, 0.34])
    sc = scans_of(open(os.path.join(D, "scans_exc_adaptive_overview.txt")).read().strip())
    P = machine_points(sc, lo, hi)
    o = np.argsort(P[:, 2])
    ax.scatter(P[o, 0], P[o, 1], c=cmap((P[o, 2] - zr[0]) / (zr[1] - zr[0])), s=0.6)
    path = np.array([s[2] for s in sc])
    ax.plot(path[:, 0], path[:, 1], color=mf.INK, lw=1.2)
    lg = json.load(open(os.path.join(D, "log_exc_adaptive_overview", "scan_log.json")))
    for s in lg["stops"]:
        if any(d["taken"] and abs(d["t"] - s["t"]) < 6 and d["t"] >= s["t"] for d in lg["decisions"]):
            ax.plot(*s["pos"], marker="*", color="#e34948", ms=14, mec="white")
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(c) 위에서 본 전체 스캔 · 선 = 경로 · ★ = 몸을 기울여 스캔한 정지 지점", fontsize=12.5, color=mf.INK,
                 loc="left")

    # (d) coverage over time
    for j, (mach, title) in enumerate((("exc", "굴착기 (3.5 t)"), ("dt", "크롤러 덤프트럭 IC120"))):
        ax = fig.add_axes([0.53 + j * 0.24, 0.08, 0.2, 0.25], facecolor=mf.SURF)
        for mode in ("level", "fixed", "adaptive"):
            name = f"{mach}_{mode}" + ("_overview" if mode == "adaptive" else "")
            p = os.path.join(D, f"log_{name}", "scan_log.json")
            if not os.path.exists(p):
                continue
            L = json.load(open(p))
            cv = np.array(L["coverage"])
            ax.plot(cv[:, 0], 100 * cv[:, 1], color=C[mode], lw=2, label=f"{NAME[mode]} {100 * cv[-1, 1]:.1f}%")
        ax.set_title(title, fontsize=12, color=mf.INK, loc="left", pad=6)
        ax.legend(loc="lower right", frameon=False, fontsize=9.5)
        ax.set_xlabel("시간 [s]", color=mf.INK2, fontsize=10.5)
        if j == 0:
            ax.set_ylabel("표면 커버리지 [%]", color=mf.INK2, fontsize=10.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color="#e6e5e0", lw=0.8)
        ax.tick_params(colors=mf.INK2, labelsize=9.5)
    fig.text(0.5, 0.385, "(d) 같은 경로·같은 정지 지점에서 시간별 표면 커버리지 (라이다 15° 장착)", fontsize=12.5, color=mf.INK)
    fig.savefig(args.out, facecolor=mf.SURF)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
