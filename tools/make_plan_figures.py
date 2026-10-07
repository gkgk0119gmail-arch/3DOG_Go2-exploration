"""Figures for the research-plan page (plan/README): roadmap, proposed method, BIM-guide training curves.

    python tools/make_plan_figures.py [--metrics ariadne3d/train/bim_guide/metrics.csv]
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_figures as mf  # noqa: E402  (fonts + palette + box / arrow / canvas)

plt = mf.plt
INK, INK2, MUTED, SURF = mf.INK, mf.INK2, mf.MUTED, mf.SURF
GRID = "#e6e5e0"
SIM = mf.SIM
OUT = os.path.join(SIM, "plan")
DONE, NOW, NEXT = mf.BUILT, "#fde7a8", mf.FROZEN  # box fills: done / in progress / planned


def legend_chips(ax, y, items):
    x = 0.04
    for fill, label in items:
        ax.add_patch(mf.FancyBboxPatch((x, y), 0.022, 0.03, boxstyle="round,pad=0.002,rounding_size=0.005",
                                       fc=fill, ec=INK2, lw=1))
        ax.text(x + 0.03, y + 0.015, label, va="center", fontsize=12.5, color=INK2)
        x += 0.03 + 0.012 * len(label) + 0.04


def roadmap(out):
    fig, ax = mf.canvas("연구 로드맵: 걷기 → 탐사 → 3D 스캔 → BIM 활용 → 실제 로봇",
                        "Unitree Go2 + VLP-16 · Isaac Sim / Isaac Lab · RTX 5090 1대 · 2026년 10월 기준")
    stages = [
        ("① 4족 보행", "PPO · 4096 병렬\n1500 iter\n\n속도 오차 0.06 m/s\n99.5 % 안 넘어짐", DONE),
        ("② 2D 자율탐사", "ARiADNE + A*\n+ turn-then-go\n\n창고 탐사 95–125 s\n지도 recall 99.7 %", DONE),
        ("③ 탐사 정책 재학습", "SAC · 회전 비용\n+ 3D 표면 효용\n\n미사용 지도 60장\n2D 완료 83 → 100 %\n시간 −28 % · 회전 −66 %", DONE),
        ("④ 자세 스캔", "몸 기울기 후보별\n관측 이득 예측\n\n굴착기·덤프트럭\n데모 완료", DONE),
        ("⑤ BIM 사전정보", "BIM 계획을 보상으로\n→ 지표 하락\n\n모방학습 +\nRL 미세조정으로 전환", NOW),
        ("⑥ Sim-to-real", "실제 Go2 + VLP-16\n15° 브래킷\nSLAM 위치추정\n\n실내 현장 시험", NEXT),
    ]
    n, x0, gap = len(stages), 0.04, 0.03
    w = (0.92 - gap * (n - 1)) / n
    y, h = 0.4, 0.34
    for i, (t, b, fill) in enumerate(stages):
        x = x0 + i * (w + gap)
        mf.box(ax, x, y, w, h, t, b, fill, title_size=14.5, body_size=12)
        if i:
            mf.arrow(ax, x - gap + 0.001, y + h / 2, x - 0.001, y + h / 2)
    legend_chips(ax, 0.3, [(DONE, "완료"), (NOW, "진행 중"), (NEXT, "예정")])
    fig.savefig(out, facecolor=SURF, bbox_inches=mf.matplotlib.transforms.Bbox([[0, 2.3], [16, 9]]))
    plt.close(fig)


def method(out):
    fig, ax = mf.canvas("제안: BIM 계획을 '보상'이 아니라 '선생'으로 쓰기",
                        "도면을 보는 전문가 계획으로 먼저 모방학습하고, 같은 시간·3D 보상으로 강화학습 미세조정한다")
    h = 0.24
    top, bot = 0.52, 0.16
    w = 0.2
    xs = [0.04, 0.28, 0.52, 0.76]
    mf.box(ax, xs[0], top, w, h, "BIM (IFC) 도면", "건물 층별 벽·천장 높이\n→ 2.5D 학습 세계\n(tools/bim_to_25d.py)", DONE)
    mf.box(ax, xs[1], top, w, h, "BIM 전문가 계획", "정답 지도를 보고\n초당 새 관측 면이 최대인\n정지 지점 순서 (탐욕)", DONE)
    mf.box(ax, xs[2], top, w, h, "1. 모방학습", "전문가가 고른 다음 노드를\n라벨로 BC → DAgger로\n정책 자신의 경로에서 재라벨", NOW)
    mf.box(ax, xs[3], top, w, h, "2. RL 미세조정", "SAC (회전 시간 + 3D 보상)\n+ BC 정규화로\n모방 정책에서 멀어지지 않게", NOW)
    mf.box(ax, xs[0], bot, w, h, "현재 시도", "같은 계획을 보상 항으로만\n추가 (진행률 −1…0)\n→ 9,000 에피소드 동안 하락", NEXT,
           edge=mf.ORANGE)
    mf.box(ax, xs[2], bot, w, h, "3. 도면을 입력으로", "도면은 현장에서도 주어짐\n→ 정책 입력 채널로 추가\n가구·장비는 무작위로 추가", NEXT)
    mf.box(ax, xs[3], bot, w, h, "평가", "학습에 안 쓴 치과 도면\nIsaac BIM 장면 (Go2 보행)\n→ 실제 Go2", NEXT)
    for i in range(3):
        mf.arrow(ax, xs[i] + w + 0.004, top + h / 2, xs[i + 1] - 0.004, top + h / 2)
    mf.arrow(ax, xs[1] + w / 2, top - 0.004, xs[0] + w * 0.75, bot + h + 0.004, color=mf.ORANGE)
    mf.arrow(ax, xs[3] + w * 0.3, top - 0.004, xs[2] + w * 0.7, bot + h + 0.004)
    mf.arrow(ax, xs[2] + w + 0.004, bot + h / 2, xs[3] - 0.004, bot + h / 2)
    legend_chips(ax, 0.06, [(DONE, "구현됨"), (NOW, "다음 단계"), (NEXT, "현재 결과 / 이후")])
    fig.savefig(out, facecolor=SURF)
    plt.close(fig)


def rolling(x, n):
    c = np.cumsum(np.insert(x, 0, 0))
    return np.concatenate([np.full(n - 1, np.nan), (c[n:] - c[:-n]) / n])


def style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#bdbcb6")
    ax.tick_params(colors=INK2, labelsize=11)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def bim_curves(metrics, out, n=300):
    rows = {}
    with open(metrics) as f:
        for r in csv.DictReader(f):
            rows[int(r["episode"])] = r  # resumed runs repeat a few episodes: keep the last
    eps = np.array(sorted(rows))
    get = lambda k: np.array([float(rows[e][k]) for e in eps])  # noqa: E731
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=SURF)
    fig.text(0.04, 0.95, "⑤ 현재 결과: BIM 계획을 보상에 넣었더니 오히려 나빠졌다", fontsize=24, weight="bold", color=INK, va="top")
    fig.text(0.04, 0.885, f"BIM 학습 지도 8장 (Duplex, DigitalHub) · 에피소드 {n}개 이동평균 · 0–{eps[-1]:,} 에피소드 · "
             "원본 공개 체크포인트에서 시작 · 노드 2 m 간격", fontsize=13.5, color=INK2, va="top")
    specs = [(get("explored_rate") * 100, "2D 탐사율 [%]"),
             (get("guide_frac") * 100, "BIM 계획 지점 통과 비율 [%]"),
             (get("turn_time_s"), "에피소드당 회전 시간 [s]")]
    for i, (y, label) in enumerate(specs):
        ax = fig.add_axes([0.06 + i * 0.315, 0.14, 0.26, 0.62], facecolor=SURF)
        r = rolling(y, n)
        ax.plot(eps, r, color=mf.BLUE, lw=2)
        j0, j1 = n - 1, len(r) - 1
        rising = r[j0 + 200] > r[j0]
        for j, dy in ((j0, -20 if rising else 10), (j1, 10)):
            ax.plot(eps[j], r[j], "o", color=mf.BLUE, ms=7, mec=SURF, mew=1.5)
            ax.annotate(f"{r[j]:.0f}", (eps[j], r[j]), xytext=(6, dy), textcoords="offset points", ha="left",
                        fontsize=12.5, color=INK)
        ax.set_title(label, fontsize=14, color=INK, loc="left", pad=10)
        ax.set_xlabel("에피소드", color=INK2, fontsize=11.5)
        ax.set_ylim(0, np.nanmax(r) * 1.2)
        style(ax)
    fig.text(0.04, 0.035, "보상: 매 스텝 계획의 다음 지점 쪽으로 간 정도에 따라 −1…0 추가. 탐사율과 계획 통과율은 내려가고 "
             "회전 시간은 늘었다 (원인 분석 중).", fontsize=12.5, color=INK2)
    fig.savefig(out, facecolor=SURF)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default=os.path.join(SIM, "ariadne3d", "train", "bim_guide", "metrics.csv"))
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    roadmap(os.path.join(OUT, "fig_roadmap.png"))
    method(os.path.join(OUT, "fig_method.png"))
    bim_curves(args.metrics, os.path.join(OUT, "fig_bim_guide_training.png"))
