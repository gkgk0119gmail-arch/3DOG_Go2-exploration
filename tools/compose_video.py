"""Put a recorded viewport clip side by side with time-stamped panel images (2D map snapshots / map tiles).

    python tools/compose_video.py VIEWPORT.mp4 IMAGE_DIR OUT.mp4 [--prefix map_] [--fps 50] [--title "..."]

Panel images must be named <prefix><seconds>s.png (as written by the exploration commands); for each video frame at
t = frame / fps the newest image with time <= t is shown.
"""

import argparse
import glob
import os
import re

import imageio.v2 as iio2
import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("images")
    ap.add_argument("out")
    ap.add_argument("--prefix", default="map_")
    ap.add_argument("--fps", type=float, default=50.0, help="viewport clip frame rate (= 1 / env step dt)")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed-up (frames are skipped)")
    ap.add_argument("--title", default="")
    ap.add_argument("--panel_title", default="")
    args = ap.parse_args()

    pat = re.compile(re.escape(args.prefix) + r"([0-9.]+)s\.png$")
    imgs = sorted((float(pat.search(f).group(1)), f) for f in glob.glob(os.path.join(args.images, args.prefix + "*s.png"))
                  if pat.search(f))
    if not imgs:
        raise SystemExit(f"no {args.prefix}*s.png images in {args.images}")
    times = np.array([t for t, _ in imgs])
    # panels are cropped to the observed area, so their size grows: letterbox all into one fixed box
    aspect = max(w / h for w, h in (Image.open(f).size for _, f in imgs))
    font = ImageFont.truetype(FONT, 26) if os.path.exists(FONT) else None
    small = ImageFont.truetype(FONT, 20) if os.path.exists(FONT) else None
    cache = {}
    step = max(1, int(round(args.speed)))
    writer = iio2.get_writer(args.out, fps=args.fps / step * min(args.speed, 1.0) if args.speed < 1 else args.fps,
                             codec="libx264", quality=8, macro_block_size=1)
    n = 0
    for i, frame in enumerate(iio.imiter(args.video)):
        if i % step:
            continue
        t = i / args.fps
        k = max(0, int(np.searchsorted(times, t, side="right")) - 1)
        if k not in cache:
            p = Image.open(imgs[k][1]).convert("RGB")
            h = frame.shape[0] - 40
            box = Image.new("RGB", (int(h * aspect), h), (18, 18, 18))
            f = min(box.width / p.width, h / p.height)
            p = p.resize((max(1, int(p.width * f)), max(1, int(p.height * f))), Image.NEAREST)
            box.paste(p, ((box.width - p.width) // 2, (h - p.height) // 2))
            cache = {k: box}
        panel = cache[k]
        W = frame.shape[1] + panel.width + 20
        canvas = Image.new("RGB", (W, frame.shape[0] + 50), (18, 18, 18))
        canvas.paste(Image.fromarray(frame), (0, 50))
        canvas.paste(panel, (frame.shape[1] + 10, 90))
        d = ImageDraw.Draw(canvas)
        d.text((14, 10), args.title, fill=(255, 255, 255), font=font)
        d.text((frame.shape[1] + 14, 58), f"{args.panel_title}  t = {t:5.1f} s", fill=(220, 220, 220), font=small)
        writer.append_data(np.asarray(canvas)[: canvas.height // 16 * 16, : canvas.width // 16 * 16])
        n += 1
    writer.close()
    print(f"wrote {args.out}: {n} frames, {n / args.fps:.1f} s clip ({n * step / args.fps:.1f} s sim)")


if __name__ == "__main__":
    main()
