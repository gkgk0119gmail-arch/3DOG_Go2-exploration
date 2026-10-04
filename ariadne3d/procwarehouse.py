"""Procedural warehouses for training: rack rows with aisles, cross aisles, pallets, a few partitions.

Each episode index gives one building (seeded): outer walls up to the ceiling (6-11 m), 1-3 rack blocks of parallel
rows (depth 0.8-1.4 m, height 3-8 m, aisles 2.4-4 m), cross aisles that cut the rows, scattered pallets / boxes
(0.3-1.5 m) and optional partition walls (2-4 m). Free space is kept connected so every area can be reached.
"""

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt
from skimage.measure import label

from env3d import Env3D
from lidar3d import CEIL_AREA, DZ, FOUR, NB, WALL_AREA, Scene25D

CELL = 0.4


def generate(seed, cell=CELL):
    rng = np.random.default_rng(seed)
    W, L = rng.uniform(22, 40), rng.uniform(32, 60)  # building [m] (the Isaac warehouse is ~32 x 54 m)
    nx, ny = int(W / cell) + 2, int(L / cell) + 2
    H = float(rng.uniform(6.0, 11.0))
    height = np.zeros((ny, nx), np.float32)
    height[[0, -1], :] = H
    height[:, [0, -1]] = H

    def box(x0, y0, x1, y1, h):
        i0, i1 = int(np.clip(y0 / cell, 1, ny - 1)), int(np.clip(y1 / cell, 1, ny - 1))
        j0, j1 = int(np.clip(x0 / cell, 1, nx - 1)), int(np.clip(x1 / cell, 1, nx - 1))
        if i1 > i0 and j1 > j0:
            height[i0:i1, j0:j1] = np.maximum(height[i0:i1, j0:j1], h)

    # rack blocks: split the hall along its length, each block with its own row orientation
    n_blocks = int(rng.integers(1, 4))
    edges = np.sort(rng.uniform(0.25, 0.75, n_blocks - 1)) * L
    ys = np.concatenate([[0.0], edges, [L]])
    for b in range(n_blocks):
        y0, y1 = ys[b] + rng.uniform(2.5, 4.0), ys[b + 1] - rng.uniform(2.5, 4.0)
        x0, x1 = rng.uniform(2.5, 4.0), W - rng.uniform(2.5, 4.0)
        if y1 - y0 < 6 or x1 - x0 < 6:
            continue
        depth, aisle, rack_h = rng.uniform(0.8, 1.4), rng.uniform(2.4, 4.0), rng.uniform(3.0, min(8.0, H - 1.0))
        along_x = rng.random() < 0.5  # rows parallel to x
        n_cross = int(rng.integers(0, 3))
        if along_x:
            cross = np.sort(rng.uniform(x0 + 4, x1 - 4, n_cross)) if x1 - x0 > 10 else []
            y = y0
            while y + depth <= y1:
                segs = np.concatenate([[x0], np.repeat(cross, 2) + np.tile([-1.5, 1.5], len(cross)), [x1]]).reshape(-1, 2)
                for a, c in segs:
                    box(a, y, c, y + depth, rack_h * rng.uniform(0.9, 1.0))
                y += depth + aisle
        else:
            cross = np.sort(rng.uniform(y0 + 4, y1 - 4, n_cross)) if y1 - y0 > 10 else []
            x = x0
            while x + depth <= x1:
                segs = np.concatenate([[y0], np.repeat(cross, 2) + np.tile([-1.5, 1.5], len(cross)), [y1]]).reshape(-1, 2)
                for a, c in segs:
                    box(x, a, x + depth, c, rack_h * rng.uniform(0.9, 1.0))
                x += depth + aisle
    # partitions (low walls with a gap)
    for _ in range(int(rng.integers(0, 3))):
        if rng.random() < 0.5:
            y = rng.uniform(4, L - 4)
            g = rng.uniform(3, W - 3)
            box(0, y, g - 1.5, y + 0.3, rng.uniform(2, 4))
            box(g + 1.5, y, W, y + 0.3, rng.uniform(2, 4))
        else:
            x = rng.uniform(4, W - 4)
            g = rng.uniform(3, L - 3)
            box(x, 0, x + 0.3, g - 1.5, rng.uniform(2, 4))
            box(x, g + 1.5, x + 0.3, L, rng.uniform(2, 4))
    # pallets / boxes in the free space
    free0 = height == 0
    for _ in range(int(rng.integers(5, 30))):
        cy, cx = np.argwhere(free0)[rng.integers(free0.sum())]
        s = rng.uniform(0.8, 1.6)
        box(cx * cell, cy * cell, cx * cell + s, cy * cell + s, rng.uniform(0.3, 1.5))
    # keep the largest free component (unreachable pockets become solid)
    free = height == 0
    lab = label(free, connectivity=1)
    if lab.max() > 1:
        big = np.argmax(np.bincount(lab.ravel())[1:]) + 1
        height[free & (lab != big)] = 1.0
    free = height == 0
    gt = np.where(free, 255, 1).astype(int)
    dist = distance_transform_edt(free)
    cand = np.argwhere(dist >= 3)
    y, x = cand[rng.integers(len(cand))]
    return gt, height, H, np.array([x, y])


class ProcScene(Scene25D):
    def __init__(self, ground_truth, height, H):  # noqa: D107
        self.free = ground_truth == 255
        self.H, self.height = H, height
        boundary = (ground_truth == 1) & binary_dilation(self.free, FOUR)
        z = (np.arange(NB) + 0.5) * DZ
        self.wall_true = boundary[..., None] & (z[None, None, :] < height[..., None])
        self.ceil_true = height < H - 1e-3
        self.floor_true = self.free
        self.total_area = self.wall_true.sum() * WALL_AREA + (self.ceil_true.sum() + self.floor_true.sum()) * CEIL_AREA


class ProcWarehouseEnv3D(Env3D):
    def import_ground_truth(self, episode_index):
        self._gen = generate(10_000_019 * 7 + episode_index)
        gt, _, _, start = self._gen
        return gt.copy(), start

    def make_scene(self, rng):
        gt, height, H, _ = self._gen
        return ProcScene(self.ground_truth, height, H)
