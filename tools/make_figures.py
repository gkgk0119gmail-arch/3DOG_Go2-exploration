"""Presentation figures (16:9 PNG, Korean labels) -> sim/figures/.

    python tools/make_figures.py [--exploration logs/exploration/<run>] [--map3d logs/lidar_scans/<run>_map3d]

fig1_ariadne_pipeline.png   how the public ARiADNE checkpoint is used in our stack (frozen / built / trained by us)
fig2_ariadne_snapshot.png   one real replanning tick on the live map, explained step by step
fig3_paper_vs_ours.png      paper (point robot, 2D) vs ours (Go2 turning cost + 3D coverage reward) retraining plan
fig4_2d_vs_3d.png           2D coverage vs 3D surface coverage over time + 3D coverage per height band
"""

import argparse
import csv
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from PIL import Image  # noqa: E402

SIM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Noto Sans CJK ships as one .ttc; matplotlib registers its first face ("JP"), which also covers Hangul
from matplotlib import font_manager  # noqa: E402

for _f in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"):
    if os.path.exists(_f):
        font_manager.fontManager.addfont(_f)
plt.rcParams["font.family"] = "Noto Sans CJK JP"
plt.rcParams["axes.unicode_minus"] = False

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SURF = "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # categorical slots 1, 2 (reference palette)
FROZEN, BUILT, TRAINED = "#d9d8d3", "#cde2fb", "#fbd9c9"  # box fills: public & frozen / we built / we trained


def box(ax, x, y, w, h, title, body, fill, edge=INK2, title_size=15, body_size=11.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.008,rounding_size=0.015", fc=fill, ec=edge, lw=1.2))
    ax.text(x + w / 2, y + h - 0.035, title, ha="center", va="top", fontsize=title_size, weight="bold", color=INK)
    ax.text(x + w / 2, y + h - 0.095, body, ha="center", va="top", fontsize=body_size, color=INK2, linespacing=1.45)


def arrow(ax, x0, y0, x1, y1, label=None, color=INK2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=18, lw=1.6, color=color))
    if label:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.018, label, ha="center", va="bottom", fontsize=10.5, color=MUTED)


def canvas(title, subtitle=None):
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=SURF)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.04, 0.95, title, fontsize=26, weight="bold", color=INK, va="top")
    if subtitle:
        ax.text(0.04, 0.885, subtitle, fontsize=14, color=INK2, va="top")
    return fig, ax


def fig1(out):
    fig, ax = canvas("ARiADNE 공개 체크포인트를 어떻게 쓰는가",
                     "상위: 어디로 갈지 (공개 체크포인트, 고정)  ·  하위: 어떻게 갈지 (우리가 구현 / 학습)")
    y1, h = 0.55, 0.23
    w, gap, x0 = 0.16, 0.025, 0.04
    top = [
        ("VLP-16 스캔", "3D 점군 10 Hz\n28,800점/스캔\n(Go2 등 위 장착)", BUILT),
        ("2D 점유 지도", "높이 0.15–1.2 m 투영\n0.4 m 격자 · 로그오즈\n(octomap 대체, GPU)", BUILT),
        ("탐사 그래프", "빈 공간에 2 m 격자 노드\n노드 가치 = 10 m 안에서\n보이는 탐사 경계 수", FROZEN),
        ("정책 신경망", "어텐션 인코더–디코더\n+ 이웃 노드 포인터\nRA-L 2024 · 145만 파라미터", FROZEN),
        ("다음 웨이포인트", "현재 노드의 이웃 중 하나\n0.4 s마다 재계획 (2.5 Hz)\n가치 합 0 → 탐사 완료", FROZEN),
    ]
    for i, (t, b, c) in enumerate(top):
        x = x0 + i * (w + gap)
        box(ax, x, y1, w, h, t, b, c)
        if i:
            arrow(ax, x - gap + 0.002, y1 + h / 2, x - 0.002, y1 + h / 2)
    y2 = 0.15
    bottom = [
        ("목표 확정 규칙", "도착 1.8 m · 진행방향 33° 이내\n· 막힘/정체 5.4 s일 때만 교체\n(병렬 32마리 CEM으로 탐색)", TRAINED),
        ("A* 경로", "장애물 0.37 m 차단\n벽 0.69 m 이내 추가 비용\n8방향 + 시야선 단축", BUILT),
        ("회전 후 전진", "방향 오차 25° 초과 → 제자리 회전\n1.3 m 앞 지점 추종\n0.3–1.0 m/s · 최대 1.5 rad/s", BUILT),
        ("보행 정책", "Isaac Lab PPO\n4096마리 병렬 학습\n(실로봇: Go2 내장 보행)", TRAINED),
        ("Go2 + VLP-16", "Isaac Sim 창고 디지털 트윈\n실로봇: Jetson Orin NX\n+ go2_ros2_sdk", BUILT),
    ]
    for i, (t, b, c) in enumerate(bottom):
        x = x0 + i * (w + gap)
        box(ax, x, y2, w, h, t, b, c)
        if i:
            arrow(ax, x - gap + 0.002, y2 + h / 2, x - 0.002, y2 + h / 2)
    # waypoint -> commit rule (wrap around)
    # waypoint (top right) -> commit rule (bottom left), routed through the gap between the rows
    xr, xl, ymid = x0 + 4 * (w + gap) + w / 2, x0 + w / 2, (y1 + y2 + h) / 2
    ax.plot([xr, xr, xl], [y1 - 0.004, ymid, ymid], color=INK2, lw=1.6)
    arrow(ax, xl, ymid, xl, y2 + h + 0.004)
    ax.text((xr + xl) / 2, ymid + 0.012, "웨이포인트 (x, y)", ha="center", va="bottom", fontsize=11.5, color=MUTED)
    # legend
    for i, (c, t) in enumerate([(FROZEN, "공개 체크포인트 (재학습 없이 그대로 사용)"), (BUILT, "우리가 구현"),
                                (TRAINED, "우리가 학습 / 탐색 (Isaac Lab 병렬)")]):
        ax.add_patch(FancyBboxPatch((0.04 + i * 0.3, 0.045), 0.022, 0.03, boxstyle="round,pad=0.003", fc=c, ec=INK2))
        ax.text(0.068 + i * 0.3, 0.06, t, va="center", fontsize=12, color=INK2)
    fig.savefig(os.path.join(out, "fig1_ariadne_pipeline.png"), facecolor=SURF)
    plt.close(fig)


def fig2(out, exploration):
    snaps = sorted(f for f in os.listdir(exploration) if f.startswith("map_"))
    pick = min(snaps, key=lambda f: abs(float(f[4:-5]) - 40.0))
    t = float(pick[4:-5])
    fig, ax = canvas("재계획 한 번(0.4초)에 일어나는 일", f"실제 실행 화면 · 리스폰 후 {t:.0f}초 · 로봇 1마리의 자기 지도")
    img = Image.open(os.path.join(exploration, pick))
    iax = fig.add_axes([0.04, 0.06, 0.40, 0.78])
    iax.imshow(img, interpolation="nearest")
    iax.axis("off")
    legend = [("#ffffff", "빈 공간 (스캔됨)"), ("#1e1e1e", "장애물 (선반·벽)"), ("#969696", "미탐사"),
              ("#e63c3c", "탐사 경계 (빈칸–미탐사 경계)"), ("#009600", "그래프 노드 (가치 있음)"),
              ("#5a5a5a", "그래프 노드 (가치 0)"), ("#1e5aff", "지나온 궤적"), ("#00c8dc", "A* 경로"),
              ("#ff8c00", "선택된 웨이포인트")]
    for i, (c, txt) in enumerate(legend):
        y = 0.80 - i * 0.042
        ax.add_patch(FancyBboxPatch((0.47, y - 0.012), 0.018, 0.024, boxstyle="round,pad=0.002", fc=c, ec=INK2, lw=0.8))
        ax.text(0.495, y, txt, va="center", fontsize=12, color=INK2)
    steps = [
        ("①  지도 갱신", "VLP-16 스캔 4번(0.1 s 간격)을 2D 점유 지도에 누적"),
        ("②  그래프 갱신", "빈 공간 2 m 격자 노드 · 충돌 없는 간선 · 노드마다\n    10 m 안에서 보이는 탐사 경계 수 = 가치"),
        ("③  신경망 추론", "공개 체크포인트가 현재 노드의 이웃 중\n    장기적으로 가장 유리한 하나를 고름 (0.1–0.2 s)"),
        ("④  확정 · 경로", "진행 방향과 맞거나 도착했을 때만 목표 교체 →\n    A*로 벽에서 떨어진 경로 생성"),
        ("⑤  주행", "필요하면 제자리 회전 → 1.3 m 앞 지점을 따라 전진\n    → 보행 정책이 관절 명령으로 변환"),
    ]
    for i, (h, b) in enumerate(steps):
        y = 0.80 - i * 0.14
        ax.text(0.73, y + 0.012, h, fontsize=15, weight="bold", color=INK, va="center")
        ax.text(0.73, y - 0.022, b, fontsize=11.5, color=INK2, va="top", linespacing=1.4)
    ax.text(0.73, 0.08, "가치 있는 노드가 하나도 없으면 → \"탐사 완료\" (2D 기준)", fontsize=12.5, color=ORANGE,
            weight="bold")
    fig.savefig(os.path.join(out, "fig2_ariadne_snapshot.png"), facecolor=SURF)
    plt.close(fig)


def fig3(out, m3d, cov2d_end):
    fig, ax = canvas("논문 그대로 vs 우리: 무엇을 바꿔 재학습하는가",
                     "ARiADNE (RA-L 2024)는 회전 비용 없는 점 로봇 · 2D 지도 기준 → 4족 Go2와 3D 스캔 목표에 맞게 재학습")
    colw, x1, x2, top = 0.42, 0.04, 0.54, 0.80
    for x, title, fill in [(x1, "논문: 점 로봇", FROZEN), (x2, "우리: Go2 4족 + VLP-16", TRAINED)]:
        ax.add_patch(FancyBboxPatch((x, 0.09), colw, 0.74, boxstyle="round,pad=0.008,rounding_size=0.02", fc=fill,
                                    ec=INK2, lw=1.2, alpha=0.55))
        ax.text(x + colw / 2, top, title, ha="center", va="top", fontsize=19, weight="bold", color=INK)
    rows = [
        ("이동 모델", "어느 방향이든 즉시 이동\n회전 비용 = 0", "방향을 바꾸려면 멈춰서 제자리 회전\n(최대 1.5 rad/s) → 시간 비용"),
        ("탐사 목표", "2D 점유 지도의 빈 공간", "3D 표면 스캔 (건설 현장 디지털 트윈)"),
        ("보상", "새로 본 2D 면적 (탐사 효율)",
         r"$\Delta A_{2D} + \lambda_{3D}\,\Delta C_{3D} - \lambda_{turn}\,t_{turn} - \lambda_{d}\,d$"),
        ("학습 환경", "2D 격자 시뮬레이터\n지도 이미지 5,663장", "Isaac Lab 병렬 복제 (각자 독립 지도)\nIsaac Sim 창고 + VLP-16 · 크리틱에 정답 지도"),
        ("완료 판정", "탐사 경계(2D)가 남지 않으면 끝", "3D 표면 커버리지 목표 달성 시 끝"),
    ]
    for i, (name, a, b) in enumerate(rows):
        y = top - 0.10 - i * 0.135
        ax.text(0.5, y - 0.005, name, ha="center", va="top", fontsize=12.5, weight="bold", color=MUTED)
        ax.text(x1 + colw / 2, y - 0.035, a, ha="center", va="top", fontsize=13.5, color=INK2, linespacing=1.4)
        ax.text(x2 + colw / 2, y - 0.035, b, ha="center", va="top", fontsize=13.5, color=INK, linespacing=1.4,
                weight="bold" if name == "보상" else "normal")
    if m3d:
        ax.text(0.5, 0.045, f"근거 (실측): 2D 탐사 \"완료\" 시점에 2D {cov2d_end:.0%} · 3D 표면 {m3d['coverage_3d']:.0%}"
                f"  —  선반 높이(0.5–4 m)에서 가장 많이 놓침", ha="center", fontsize=14, color=ORANGE, weight="bold")
    fig.savefig(os.path.join(out, "fig3_paper_vs_ours.png"), facecolor=SURF)
    plt.close(fig)


def fig4(out, exploration, m3d):
    with open(os.path.join(exploration, "exploration_log.csv")) as f:
        rows = list(csv.DictReader(f))
    t2 = [float(r["time_s"]) for r in rows]
    c2 = [100 * min(float(r["coverage"]), 1.0) for r in rows]
    t3 = [t for t, _ in m3d["coverage_3d_curve"]]
    c3 = [100 * c for _, c in m3d["coverage_3d_curve"]]
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=SURF)
    fig.text(0.04, 0.95, "2D로는 \"완료\"인데 3D로는 아직 비어 있다", fontsize=26, weight="bold", color=INK, va="top")
    fig.text(0.04, 0.885, "같은 탐사 1회 (ARiADNE + A*, 창고) · VLP-16 스캔 1,246개를 3D 복셀(0.1 m)로 누적해 정답 메시 표면과 비교",
             fontsize=14, color=INK2, va="top")
    a1 = fig.add_axes([0.06, 0.12, 0.50, 0.66], facecolor=SURF)
    a1.plot(t2, c2, color=BLUE, lw=2)
    a1.plot(t3, c3, color=ORANGE, lw=2)
    a1.text(t2[-1] + 2, c2[-1], f"2D 탐사율 {c2[-1]:.0f}%", color=INK, fontsize=13, va="center")
    a1.text(t3[-1] + 2, c3[-1], f"3D 표면 {c3[-1]:.0f}%", color=INK, fontsize=13, va="center")
    done = t2[-1]
    a1.axvline(done, color=MUTED, lw=1, ls="--")
    a1.text(done - 2, 32, "ARiADNE\n\"탐사 완료\"", ha="right", color=INK2, fontsize=11.5)
    a1.set_xlim(0, done + 28)
    a1.set_ylim(0, 105)
    a1.set_xlabel("시간 [s]", color=INK2, fontsize=12)
    a1.set_ylabel("커버리지 [%]", color=INK2, fontsize=12)
    a1.grid(axis="y", color="#e6e5e0", lw=0.8)
    for s in ("top", "right"):
        a1.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        a1.spines[s].set_color("#bdbcb6")
    a1.tick_params(colors=INK2)
    a1.legend(handles=[plt.Line2D([], [], color=BLUE, lw=2, label="2D 탐사율 (빈 공간)"),
                       plt.Line2D([], [], color=ORANGE, lw=2, label="3D 표면 커버리지")],
              loc="lower center", bbox_to_anchor=(0.45, 0.02), frameon=False, fontsize=12)
    a2 = fig.add_axes([0.66, 0.12, 0.30, 0.66], facecolor=SURF)
    names = list(m3d["bands"])
    vals = [100 * m3d["bands"][n]["coverage"] for n in names]
    labels = ["바닥 0–0.5", "0.5–2", "2–4", "4–6", "6–8.5", "천장 >8.5"]
    ys = list(range(len(vals)))[::-1]
    a2.barh(ys, vals, color=ORANGE, height=0.62)
    for y, v in zip(ys, vals):
        a2.text(v + 1.5, y, f"{v:.0f}%", va="center", fontsize=12, color=INK)
    a2.set_yticks(ys, labels, fontsize=12, color=INK2)
    a2.set_xlim(0, 112)
    a2.set_xlabel("3D 표면 커버리지 [%]  (높이 [m])", color=INK2, fontsize=12)
    a2.grid(axis="x", color="#e6e5e0", lw=0.8)
    for s in ("top", "right"):
        a2.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        a2.spines[s].set_color("#bdbcb6")
    a2.tick_params(colors=INK2)
    fig.text(0.66, 0.06, "천장: VLP-16 시야 ±15°라 로봇 근처 천장은 원리상 안 보임 (센서 장착 문제)\n"
             "선반 0.5–4 m: 경로 선택으로 줄일 수 있는 공백 → 3D 커버리지 보상의 대상",
             fontsize=11, color=INK2, va="top", linespacing=1.5)
    fig.savefig(os.path.join(out, "fig4_2d_vs_3d.png"), facecolor=SURF)
    plt.close(fig)


def _style(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#bdbcb6")
    ax.tick_params(colors=INK2, labelsize=11.5)


def fig5(out, study):
    tilts = sorted(int(k[5:]) for k in study if k.startswith("pitch"))
    rep = [100 * study[f"pitch{t}"]["replay"]["all"] for t in tilts]
    bnd = [100 * study[f"pitch{t}"]["bound"]["all"] for t in tilts]
    ceil = [100 * study[f"pitch{t}"]["replay"]["ceiling >8.5 m"] for t in tilts]
    low0 = study["pitch0"]["replay"]["front_low_cells_per_scan"]
    low = [100 * study[f"pitch{t}"]["replay"]["front_low_cells_per_scan"] / low0 for t in tilts]
    pick = 15 if 15 in tilts else tilts[len(tilts) // 2]
    i = tilts.index(pick)
    fig = plt.figure(figsize=(16, 9), dpi=120, facecolor=SURF)
    fig.text(0.04, 0.95, "천장 공백은 정책이 아니라 마운트 문제: VLP-16을 15° 들면", fontsize=26, weight="bold", color=INK, va="top")
    fig.text(0.04, 0.885, "같은 탐사 경로(1,246 스캔)를 마운트 각도만 바꿔 GPU ray-cast · 0.1 m 복셀로 정답 메시와 비교 "
             "(0°에서 실제 시뮬 결과 77.0%와 일치)", fontsize=14, color=INK2, va="top")
    a1 = fig.add_axes([0.06, 0.14, 0.48, 0.62], facecolor=SURF)
    a1.axvspan(pick - 2.5, pick + 2.5, color="#f1f0ec", zorder=0)
    a1.plot(tilts, bnd, color=BLUE, lw=2, marker="o", ms=7)
    a1.plot(tilts, rep, color=ORANGE, lw=2, marker="o", ms=7)
    a1.text(tilts[-1] + 1.2, bnd[-1], "상한: 창고 어디서든 볼 수 있는 표면", color=INK, fontsize=12, va="center")
    a1.text(tilts[-1] + 1.2, rep[-1] - 1.5, "지금 경로 그대로 (ARiADNE)", color=INK, fontsize=12, va="center")
    a1.annotate(f"{rep[0]:.1f}%", (0, rep[0]), xytext=(6, -16), textcoords="offset points", fontsize=12, color=INK)
    a1.annotate(f"{rep[i]:.1f}%", (pick, rep[i]), xytext=(0, -24), textcoords="offset points", ha="center",
                fontsize=13, weight="bold", color=INK)
    a1.annotate(f"{bnd[i]:.1f}%", (pick, bnd[i]), xytext=(0, 10), textcoords="offset points", ha="center",
                fontsize=12, color=INK)
    a1.annotate("", (pick + 0.8, bnd[i] - 0.6), (pick + 0.8, rep[i] + 0.6),
                arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.2))
    a1.text(pick + 1.6, (rep[i] + bnd[i]) / 2, "보상 재학습으로\n줄일 몫", fontsize=11, color=INK2, va="center")
    a1.set_xlim(-2, tilts[-1] + 22)
    a1.set_ylim(70, 100)
    a1.set_xticks(tilts, [f"{t}°" for t in tilts])
    a1.set_xlabel("마운트 기울기 (앞쪽 위로)", color=INK2, fontsize=12)
    a1.set_ylabel("3D 표면 커버리지 [%]", color=INK2, fontsize=12)
    a1.grid(axis="y", color="#e6e5e0", lw=0.8)
    _style(a1)
    a2 = fig.add_axes([0.70, 0.14, 0.26, 0.62], facecolor=SURF)
    a2.axvspan(pick - 2.5, pick + 2.5, color="#f1f0ec", zorder=0)
    a2.plot(tilts, low, color=INK2, lw=2, marker="o", ms=7)
    for t, v in zip(tilts, low):
        a2.annotate(f"{v:.0f}%", (t, v), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=11, color=INK)
    a2.set_xticks(tilts, [f"{t}°" for t in tilts])
    a2.set_ylim(-5, 120)
    a2.set_xlim(-4, tilts[-1] + 4)
    a2.set_xlabel("마운트 기울기", color=INK2, fontsize=12)
    a2.set_title("앞쪽 낮은 장애물(<0.45 m) 검출, 0° 대비", fontsize=13, color=INK, loc="left", pad=12)
    a2.grid(axis="y", color="#e6e5e0", lw=0.8)
    _style(a2)
    fig.text(0.70, 0.06, f"{pick}°: 천장 {ceil[0]:.0f}% → {ceil[i]:.0f}%, 앞쪽 저상 장애물 검출 유지\n"
             "20° 이상: 앞쪽 바닥이 안 보여 팔레트·턱에 부딪힐 위험",
             fontsize=11, color=INK2, va="top", linespacing=1.5)
    fig.savefig(os.path.join(out, "fig5_lidar_mount_tilt.png"), facecolor=SURF)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exploration", default=os.path.join(SIM, "logs", "exploration", "2026-10-03_00-31-54"))
    ap.add_argument("--map3d", default=os.path.join(SIM, "logs", "lidar_scans", "2026-10-03_00-31-56_map3d"))
    ap.add_argument("--mount", default=os.path.join(SIM, "logs", "mount_study", "mount_study.json"))
    args = ap.parse_args()
    out = os.path.join(SIM, "figures")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(args.map3d, "map3d_metrics.json")) as f:
        m3d = json.load(f)
    with open(os.path.join(args.exploration, "exploration_log.csv")) as f:
        cov2d_end = min(float(list(csv.DictReader(f))[-1]["coverage"]), 1.0)
    fig1(out)
    fig2(out, args.exploration)
    fig3(out, m3d, cov2d_end)
    fig4(out, args.exploration, m3d)
    if os.path.exists(args.mount):
        with open(args.mount) as f:
            fig5(out, json.load(f))
    print("wrote", sorted(os.listdir(out)))


if __name__ == "__main__":
    main()
