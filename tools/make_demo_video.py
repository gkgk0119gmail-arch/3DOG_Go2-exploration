"""Cut the excavator-scan demo into one captioned video (title card, clips with a caption bar, result card).

    python tools/make_demo_video.py <videos dir> <coverage json> <out.mp4>

Clips (from the recording scripts) are looked up by name in <videos dir>; missing ones are skipped.
"""

import json
import os
import sys

import imageio.v2 as iio2
import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
W, H, FPS = 1280, 720, 25
BG, INK, INK2, ACC, MISS = (252, 252, 251), (11, 11, 11), (82, 81, 78), (27, 175, 122), (227, 73, 72)

SEGMENTS = [  # (file, speed, caption, max seconds of source)
    ("exc1_level_follow.mp4", 2.0, "① 굴착기 주변 주행 스캔 — Go2 + VLP-16 (라이다 15° 장착, 몸 수평) · 2배속", 20),
    ("posture_policy_grid.mp4", 1.0, "② 몸 앞뒤·좌우 기울이기 정책 — Isaac Lab PPO 4,096 병렬 학습 (2,000 iter)", 15),
    ("exc2_posture_m15_follow.mp4", 1.5, "③ 멈춰서 몸을 기울여 스캔 — 좌우 기울여 굴착기 쪽 위로, 고개 들기 · 1.5배속", 24),
    ("exc2_posture_m15_overview.mp4", 3.0, "④ 전체 경로 — 측면 중앙 4곳에서 자세 스캔 · 3배속", 80),
]


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def card(lines, sub=None):
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    y = 250
    for i, line in enumerate(lines):
        f = font(46 if i == 0 else 30, bold=i == 0)
        d.text((80, y), line, fill=INK if i == 0 else INK2, font=f)
        y += 80 if i == 0 else 50
    if sub:
        d.text((80, H - 90), sub, fill=INK2, font=font(22))
    return np.asarray(im)


def captioned(frame, caption):
    im = Image.fromarray(frame).convert("RGB").resize((W, H - 60))
    canvas = Image.new("RGB", (W, H), (24, 24, 24))
    canvas.paste(im, (0, 60))
    ImageDraw.Draw(canvas).text((24, 14), caption, fill=(245, 245, 245), font=font(26))
    return np.asarray(canvas)


def result_card(cov):
    labels = [("level_m0", "라이다 수평 장착 · 몸 수평"), ("posture_m0", "라이다 수평 장착 · 몸 기울이기"),
              ("level_m15", "라이다 15° 장착 · 몸 수평"), ("posture_m15", "라이다 15° 장착 · 몸 기울이기")]
    labels = [(k, n) for k, n in labels if k in cov]
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((60, 40), "결과: 굴착기 표면 커버리지 (같은 경로·같은 정지 지점)", fill=INK, font=font(38, True))
    d.text((60, 100), "정답 메시 5 cm 복셀 × 면 방향 · 그 면 앞쪽에서 온 레이저만 인정", fill=INK2, font=font(22))
    cols = [("coverage", "전체"), ("coverage_up", "윗면"), ("coverage_side", "옆면"), ("coverage_down", "아랫면")]
    x0, y0, bw = 470, 175, 170
    for j, (_, cn) in enumerate(cols):
        d.text((x0 + j * bw + 20, y0 - 45), cn, fill=INK2, font=font(24))
    for i, (k, name) in enumerate(labels):
        y = y0 + i * 115
        d.text((60, y + 18), name, fill=INK, font=font(25, "기울이기" in name))
        for j, (key, _) in enumerate(cols):
            v = cov[k][key]
            x = x0 + j * bw
            d.rectangle([x, y + 10, x + bw - 30, y + 70], fill=(236, 235, 230))
            fill_w = int((bw - 30) * v)
            d.rectangle([x, y + 10, x + fill_w, y + 70], fill=ACC if "posture" in k else (138, 137, 132))
            txt = f"{100 * v:.1f}%"
            if fill_w >= 90:
                d.text((x + 8, y + 22), txt, fill=(255, 255, 255), font=font(24, True))
            else:  # short bar: dark label on the track, right of the fill
                d.text((x + fill_w + 6, y + 22), txt, fill=INK, font=font(24, True))
    if {"level_m0", "posture_m0", "level_m15", "posture_m15"} <= set(cov):
        g0 = 100 * (cov["posture_m0"]["coverage"] - cov["level_m0"]["coverage"])
        g15 = 100 * (cov["posture_m15"]["coverage"] - cov["level_m15"]["coverage"])
        d.text((60, H - 118), f"몸 기울이기: 수평 장착에서 +{g0:.1f}%p, 15° 장착에서 +{g15:.1f}%p · 윗면은 센서(높이 0.44 m)보다 높아 "
               "지상 로봇으로는 원리상 보기 어려움", fill=INK, font=font(21))
    d.text((60, H - 70), "Isaac Sim 6.0 / Isaac Lab 3.0 · RTX 5090 · 굴착기 모델: ix35e (3.5 t, orientpine/hr35_modeling)",
           fill=INK2, font=font(20))
    return np.asarray(im)


def main():
    vdir, cov_json, out = sys.argv[1:4]
    cov = json.load(open(cov_json)) if os.path.exists(cov_json) else {}
    w = iio2.get_writer(out, fps=FPS, codec="libx264", quality=8, macro_block_size=8)
    for _ in range(FPS * 4):
        w.append_data(card(["건설장비 3D 스캔: 4족 로봇 Go2 + VLP-16",
                            "몸을 앞뒤·좌우로 기울여 센서 시야를 넓히는 자세 스캔",
                            "Isaac Sim / Isaac Lab 시뮬레이션 데모"], "연세대 PRAXIS 실전문제연구팀 3DOG"))
    for fn, speed, cap, max_s in SEGMENTS:
        p = os.path.join(vdir, fn)
        if not os.path.exists(p):
            print("skip", fn)
            continue
        src_fps = iio.immeta(p).get("fps", 50)
        step = src_fps * speed / FPS
        t = 0.0
        for i, fr in enumerate(iio.imiter(p)):
            if i > max_s * src_fps:
                break
            if i >= t:
                w.append_data(captioned(fr, cap))
                t += step
    if cov:
        rc = result_card(cov)
        for _ in range(FPS * 8):
            w.append_data(rc)
    w.close()
    print("wrote", out)


if __name__ == "__main__":
    main()
