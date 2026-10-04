"""Stitch exploration map snapshots (logs/exploration/<run>/map_*.png) into an animated GIF.

Usage:
    python snapshots_to_gif.py logs/exploration/<run> [--fps 8] [--width 480]
"""

import argparse
import glob
import os

from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--width", type=int, default=480)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.run_dir, "map_*.png")))
    if os.path.exists(os.path.join(args.run_dir, "final.png")):
        files += [os.path.join(args.run_dir, "final.png")] * int(2 * args.fps)  # hold the final map
    frames = []
    for f in files:
        im = Image.open(f).convert("RGB")
        im = im.resize((args.width, int(im.height * args.width / im.width)), Image.NEAREST)
        frames.append(im.convert("P", palette=Image.ADAPTIVE, colors=64))
    out = os.path.join(args.run_dir, "exploration.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / args.fps), loop=0, optimize=True)
    print(f"wrote {out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
