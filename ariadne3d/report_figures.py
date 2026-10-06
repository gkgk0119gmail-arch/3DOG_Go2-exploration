"""Progress figures for the ARiADNE retraining (written to ../figures, style of tools/make_figures.py).

    python report_figures.py [curves] [utility] [worlds] [eval] [gif]

Colors follow the policy everywhere: original = gray, run 1 = blue, run 2 = orange, run 3 = aqua.
"""

import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(SIM, "tools"))
import make_figures as mf  # noqa: E402  (fonts + palette)

plt = mf.plt
INK, INK2, MUTED, SURF = mf.INK, mf.INK2, mf.MUTED, mf.SURF
GRID = "#e6e5e0"
C = {"original": "#8a8984", "run1": "#2a78d6", "run2": "#eb6834", "run3": "#1baf7a"}
NAME = {"original": "원본 (RA-L 2024)", "run1": "1차: 회전 + 3D v1", "run2": "2차: 창고형 2 m (실패)",
        "run3": "3차: 회전 + 3D v2"}
OUT = os.path.join(SIM, "figures")


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#bdbcb6")
    ax.tick_params(colors=INK2, labelsize=11)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def header(fig, title, sub):
    fig.text(0.04, 0.95, title, fontsize=24, weight="bold", color=INK, va="top")
    fig.text(0.04, 0.885, sub, fontsize=13.5, color=INK2, va="top")


def rolling(x, n):
    x = np.asarray(x, float)
    c = np.cumsum(np.insert(np.nan_to_num(x), 0, 0))
    out = (c[n:] - c[:-n]) / n
    return np.concatenate([np.full(n - 1, np.nan), out])


# -- 1. training curves ---------------------------------------------------------------------------------
def curves():
    runs = {"run1": "go2_turn_3d", "run2": "go2_wh_res2", "run3": "go2_turn_3d_view"}
    data = {}
    for k, name in runs.items():
        p = os.path.join(HERE, "train", name, "metrics.csv")
        with open(p) as f:
            rows = list(csv.DictReader(f))
        data[k] = {m: np.array([float(r[m]) for r in rows]) for m in ("episode_reward", "turn_total_rad", "coverage_3d")}
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=SURF)
    header(fig, "재학습 곡선: 1차·3차는 안정, 2차는 붕괴",
           "에피소드 200개 이동평균 · 1·3차 = 원 학습 지도·4 m 노드, 2차 = 창고형 절차 지도·2 m 노드 (환경이 달라 절대값 비교 불가)")
    specs = [("episode_reward", "에피소드 보상", 1.0), ("turn_total_rad", "누적 회전 [°]", 180 / np.pi),
             ("coverage_3d", "3D 표면 커버리지 [%]", 100.0)]
    for i, (m, label, s) in enumerate(specs):
        ax = fig.add_axes([0.06 + i * 0.315, 0.13, 0.26, 0.62], facecolor=SURF)
        for k in ("run1", "run2", "run3"):
            y = rolling(data[k][m] * s, 200)
            x = np.arange(len(y))
            ax.plot(x, y, color=C[k], lw=2)
            j = np.flatnonzero(~np.isnan(y))[-1]
            ax.plot(x[j], y[j], "o", color=C[k], ms=6, mec=SURF, mew=1.5)
        ax.set_title(label, fontsize=14, color=INK, loc="left", pad=10)
        ax.set_xlabel("에피소드", color=INK2, fontsize=11.5)
        style(ax)
        if m == "coverage_3d":
            ax.set_ylim(60, 100)
    fig.legend(handles=[plt.Line2D([], [], color=C[k], lw=2.5, label=NAME[k]) for k in ("run1", "run2", "run3")],
               loc="lower center", ncol=3, frameon=False, fontsize=12.5, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(os.path.join(OUT, "fig6_training_curves.png"), facecolor=SURF)
    plt.close(fig)


# -- 2. 3D utility v1 vs v2 on one state ------------------------------------------------------------------
def utility(ep=5600, at_step=18):
    import torch

    torch.set_num_threads(1)
    from lidar3d import ViewGain
    from nets import load_pretrained
    from worker3d import Worker3D

    pol, *_ = load_pretrained(os.path.join(HERE, "pretrained", "ariadne_ral2024.pth"))
    w = Worker3D(0, pol, ep, greedy=True, record=False)
    w.robot.update_planning_state(w.env.belief_info, w.env.robot_location)
    obs, _, _ = w._observe()
    for _ in range(at_step):
        nxt, _ = w._act(obs)
        w.env.step(nxt)
        w.robot.update_planning_state(w.env.belief_info, w.env.robot_location)
        obs, _, _ = w._observe()
    e, a = w.env, w.robot
    o = np.array([e.belief_origin_x, e.belief_origin_y])
    nodes = a.node_coords
    cells = (nodes - o) / e.cell_size
    v1 = e.belief3d.sample(e.belief3d.belief_utility(e.robot_belief), cells)
    vg = ViewGain(e.belief3d)
    v2 = vg.node_gain(nodes, e.robot_location, o, "belief", e.robot_belief)
    tr = vg.node_gain(nodes, e.robot_location, o, "truth")
    cands = nodes[a.neighbor_indices]
    rc = lambda x, y: np.corrcoef(x, y)[0, 1]  # noqa: E731
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=SURF)
    header(fig, "3D 효용 v1 → v2: \"어디로 가야 3D가 더 보이나\"를 제대로 가리키게",
           f"같은 순간의 그래프 노드 (학습 미사용 지도 · {at_step}스텝째) · 색이 진할수록 값이 큼 · 빨간 원 = 로봇, 테두리 = 다음 후보 노드")
    titles = [("v1: 거리 고리만 (가림 무시)", v1, f"정답과 상관 {rc(v1, tr):+.2f}"),
              ("v2: 노드에서 VLP-16 스캔 모사 (가림 반영)", v2, f"정답과 상관 {rc(v2, tr):+.2f}"),
              ("정답: 그 노드에서 실제로 새로 보일 표면", tr, "critic이 학습에 사용")]
    m = e.robot_belief
    img = np.where(m == 255, 1.0, np.where(m == 1, 0.0, 0.82))
    ys, xs = np.nonzero(m != 127)
    pad = 6
    y0, y1, x0, x1 = max(ys.min() - pad, 0), min(ys.max() + pad, m.shape[0]), max(xs.min() - pad, 0), min(xs.max() + pad, m.shape[1])
    for i, (t, val, note) in enumerate(titles):
        ax = fig.add_axes([0.03 + i * 0.325, 0.08, 0.30, 0.70])
        ax.imshow(img[y0:y1, x0:x1], cmap="gray", vmin=0, vmax=1, origin="lower", interpolation="nearest")
        c = cells - np.array([x0, y0])
        v = (val - val.min()) / (np.ptp(val) + 1e-9)
        ax.scatter(c[:, 0], c[:, 1], c=v, cmap="Blues", vmin=-0.15, vmax=1, s=38, edgecolors="none", zorder=3)
        cc = (cands - o) / e.cell_size - np.array([x0, y0])
        ax.scatter(cc[:, 0], cc[:, 1], facecolors="none", edgecolors=INK, s=70, lw=1.0, zorder=4)
        r = (e.robot_location - o) / e.cell_size - np.array([x0, y0])
        ax.plot(r[0], r[1], "o", color="#e34948", ms=11, mec="white", mew=1.5, zorder=5)
        ax.set_title(t, fontsize=13.5, color=INK, loc="left")
        ax.text(0.01, -0.06, note, transform=ax.transAxes, fontsize=12.5, color=INK, va="top")
        ax.axis("off")
    fig.text(0.04, 0.035, "다음 후보 노드들 사이 순위의 정답 상관 (학습 미사용 지도 2개, 에피소드 전체 평균): v1 0.00 ~ +0.12, v2 +0.46 ~ +0.55",
             fontsize=12, color=INK2)
    fig.savefig(os.path.join(OUT, "fig7_3d_utility_v1_v2.png"), facecolor=SURF)
    plt.close(fig)


# -- 3. training / test worlds --------------------------------------------------------------------------
def worlds():
    from procwarehouse import generate
    from warehouse_env import warehouse

    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=SURF)
    header(fig, "학습 세계: 원 학습 지도 → 창고형 절차 생성 지도, 그리고 실제 창고",
           "왼쪽: 2차 학습용 절차 생성 창고 (색 = 장애물 높이) · 오른쪽: Isaac 창고를 2.5D로 옮긴 평가 세계 위의 그래프 노드 (파랑)")
    import matplotlib.colors as mcolors

    cmap = plt.get_cmap("viridis")
    for i in range(4):
        gt, h, H, st = generate(10_000_019 * 7 + 5600 + i)
        ax = fig.add_axes([0.02 + i * 0.13, 0.12, 0.12, 0.66])
        ax.imshow(np.where(gt == 255, np.nan, h), origin="lower", cmap=cmap, vmin=0, vmax=11, interpolation="nearest")
        ax.plot(st[0], st[1], "o", color="#e34948", ms=7, mec="white")
        ax.set_title(f"{gt.shape[1] * 0.4:.0f}×{gt.shape[0] * 0.4:.0f} m · 천장 {H:.1f} m", fontsize=10.5, color=INK2)
        ax.axis("off")
    fig.text(0.02, 0.81, "창고형 절차 생성 지도 (랙 줄·통로 2.4–4 m·교차 통로·팔레트)", fontsize=13, color=INK)
    gt, h, H, (x0, y0) = warehouse()
    for j, res in enumerate((4.0, 2.0)):
        ax = fig.add_axes([0.56 + j * 0.22, 0.12, 0.2, 0.66])
        shade = np.where(gt == 255, 1.0, np.where(h >= 2.0, 0.35, 0.72))  # tall: dark gray, low clutter: light gray
        ax.imshow(shade, origin="lower", cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        xs = np.arange(np.ceil(x0 / res) * res, x0 + gt.shape[1] * 0.4, res)
        ys = np.arange(np.ceil(y0 / res) * res, y0 + gt.shape[0] * 0.4, res)
        X, Y = np.meshgrid(xs, ys)
        cx, cy = np.rint((X - x0) / 0.4).astype(int), np.rint((Y - y0) / 0.4).astype(int)
        ok = (cx < gt.shape[1]) & (cy < gt.shape[0])
        free = np.zeros_like(ok)
        free[ok] = gt[cy[ok], cx[ok]] == 255
        ax.scatter(cx[free], cy[free], s=9 if res == 2 else 22, color=C["run1"], zorder=3)
        ax.set_title(f"Isaac 창고 · 노드 {res:.0f} m 간격 ({int(free.sum())}개)", fontsize=12, color=INK)
        if j == 0:
            ax.text(0, -0.04, "진회색 = 랙·벽 (≥ 2 m), 연회색 = 팔레트·잡물", transform=ax.transAxes, fontsize=10.5,
                    color=INK2, va="top")
        ax.axis("off")
    cax = fig.add_axes([0.08, 0.06, 0.36, 0.015])
    fig.colorbar(plt.cm.ScalarMappable(mcolors.Normalize(0, 11), cmap), cax=cax, orientation="horizontal").set_label(
        "장애물 높이 [m]", color=INK2)
    fig.savefig(os.path.join(OUT, "fig8_worlds.png"), facecolor=SURF)
    plt.close(fig)


# -- 4. evaluation summary ------------------------------------------------------------------------------
def _load(path):
    with open(path) as f:
        return json.load(f)["per_map"]


def evaluation():
    E = os.path.join(HERE, "eval")
    view = _load(os.path.join(E, "eval_view_maps_tilt15.json"))
    grid = _load(os.path.join(E, "eval_grid_maps_tilt15.json"))
    pol = {"original": view["pretrained/ariadne_ral2024.pth"], "run1": grid["model/go2_turn_3d/checkpoint.pth"],
           "run3": view["model/go2_turn_3d_view/checkpoint.pth"]}
    maps = sorted(set.intersection(*[set(v) for v in pol.values()]))
    ok = [m for m in maps if all(pol[k][m].get("done_2d") == 1 for k in pol)]
    keys = ["original", "run1", "run3"]
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=SURF)
    header(fig, "평가: 학습에 안 쓴 지도 60장 (그리디, 15° 마운트)",
           f"실패 수는 60장 전체 · 나머지는 세 정책 모두 2D를 끝낸 {len(ok)}장에서 같은 지도끼리 비교")
    panels = [
        ("2D 탐사 못 끝낸 지도 [개]", lambda k: sum(pol[k][m].get("done_2d") != 1 for m in maps), "{:.0f}"),
        ("2D 탐사 완료 시간 [s]", lambda k: np.mean([pol[k][m]["time_2d_s"] for m in ok]), "{:.0f}"),
        ("누적 회전 [°]", lambda k: np.rad2deg(np.mean([pol[k][m]["turn_total_rad"] for m in ok])), "{:,.0f}"),
        ("3D 커버리지 @300 s [%]", lambda k: 100 * np.mean([pol[k][m]["cov3d_at_300s"] for m in ok]), "{:.1f}"),
    ]
    for i, (t, fn, fmt) in enumerate(panels):
        ax = fig.add_axes([0.06 + i * 0.235, 0.16, 0.19, 0.58], facecolor=SURF)
        vals = [fn(k) for k in keys]
        bars = ax.bar(range(3), vals, color=[C[k] for k in keys], width=0.62)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(), fmt.format(v), ha="center", va="bottom", fontsize=12,
                    color=INK)
        ax.set_xticks(range(3), ["원본", "1차", "3차"], fontsize=12, color=INK2)
        ax.set_title(t, fontsize=13.5, color=INK, loc="left", pad=12)
        if "3D" in t:
            lo = min(vals)
            ax.set_ylim(max(0, lo - 10), 100)
        style(ax)
    fig.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=C[k], label=NAME[k]) for k in keys], loc="lower center",
               ncol=3, frameon=False, fontsize=12.5)
    fig.savefig(os.path.join(OUT, "fig9_eval.png"), facecolor=SURF)
    plt.close(fig)
    return {k: {"fail": sum(pol[k][m].get("done_2d") != 1 for m in maps),
                **{q: float(np.mean([pol[k][m][q] for m in ok])) for q in
                   ("time_2d_s", "time_s", "turn_total_rad", "travel_dist", "coverage_3d", "cov3d_at_150s",
                    "cov3d_at_300s", "cov3d_at_450s", "cov3d_at_2d")}} for k in keys} | {"n_paired": len(ok)}


# -- 5. side-by-side episode GIF ------------------------------------------------------------------------
def gif(ep=5602, out="media/ariadne3d_original_vs_run3.gif", dt=8.0):
    import torch
    from PIL import Image

    torch.set_num_threads(1)
    from nets import load_pretrained
    from worker3d import Worker3D

    runs = [("original", os.path.join(HERE, "pretrained", "ariadne_ral2024.pth")),
            ("run3", os.path.join(HERE, "model", "go2_turn_3d_view", "checkpoint.pth"))]
    episodes = []
    for key, ck in runs:
        pol, *_ = load_pretrained(ck)
        w = Worker3D(0, pol, ep, greedy=True, record=False)
        e = w.env
        frames = []

        def snap(e=e, frames=frames):
            m = e.robot_belief
            img = np.stack([np.where(m == 255, 252, np.where(m == 1, 60, 210))] * 3, -1).astype(np.uint8)
            sc, b3 = e.scene, e.belief3d
            n_true = sc.wall_true.sum(-1)
            n_seen = ((b3.wall == 1) & sc.wall_true).sum(-1)
            gap = (m == 1) & (n_true > 0) & (n_seen < 0.5 * n_true)  # known wall, less than half its height scanned
            img[gap] = (227, 73, 72)
            frames.append((e.time_s, img, e.robot_cell.copy(), np.rad2deg(e.turn_total), b3.coverage(), e.explored_rate))

        snap()
        step0 = e.step

        def step(wp, step0=step0, snap=snap):
            r = step0(wp)
            snap()
            return r

        e.step = step
        w.run_episode()
        episodes.append((key, frames))
    t_end = max(f[-1][0] for _, f in episodes)
    imgs = []
    for t in np.arange(0, t_end + dt, dt):
        fig = plt.figure(figsize=(12, 6.6), dpi=100, facecolor=SURF)
        fig.text(0.02, 0.955, f"같은 지도·같은 출발점, 같은 시계 (t = {t:3.0f} s) · 빨강 = 아직 높이의 절반도 스캔 안 된 벽 · 선 = 경로",
                 fontsize=11.5, color=INK2)
        for i, (key, frames) in enumerate(episodes):
            k = max(j for j, f in enumerate(frames) if f[0] <= t)
            ts, img, cell, turn, cov, exp = frames[k]
            ax = fig.add_axes([0.02 + i * 0.49, 0.03, 0.47, 0.78])
            ax.imshow(img, origin="lower", interpolation="nearest")
            p = np.array([f[2] for f in frames[: k + 1]])
            ax.plot(p[:, 0], p[:, 1], color=C[key] if key != "original" else "#52514e", lw=2.2)
            ax.plot(cell[0], cell[1], "o", color=INK, ms=9, mec="white", mew=1.5)
            ax.axis("off")
            done = k == len(frames) - 1
            ax.set_title(f"{NAME[key]}{'   탐사 종료' if done else ''}\n시간 {ts:4.0f} s · 회전 {turn:5.0f}° · 2D {100 * exp:3.0f}% · 3D {100 * cov:4.1f}%",
                         fontsize=12.5, color=INK, loc="left")
        fig.canvas.draw()
        imgs.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]).quantize(64))
        plt.close(fig)
    imgs += [imgs[-1]] * 10
    imgs[0].save(os.path.join(SIM, out), save_all=True, append_images=imgs[1:], duration=200, loop=0, optimize=True)
    return [(k, len(f), round(f[-1][0]), round(f[-1][3]), round(f[-1][4], 3)) for k, f in episodes]


if __name__ == "__main__":
    os.chdir(HERE)
    todo = sys.argv[1:] or ["curves", "utility", "worlds"]
    if "curves" in todo:
        curves()
    if "utility" in todo:
        utility()
    if "worlds" in todo:
        worlds()
    if "eval" in todo:
        print(json.dumps(evaluation(), indent=1, ensure_ascii=False))
    if "gif" in todo:
        print(gif())
