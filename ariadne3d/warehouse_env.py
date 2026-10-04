"""The Isaac warehouse as a 2.5D training-env world (layout from the GT occupancy map, heights from the mesh).

Lets the 2D-graph policies run on the same building the Go2 explores in Isaac, in seconds instead of minutes.
Episode index = seed of the random free start.
"""

import functools
import os
import sys

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

from env3d import Env3D
from lidar3d import DZ, NB, CEIL_AREA, FOUR, WALL_AREA, Scene25D

SIM = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GT_MAP = os.path.join(SIM, "assets", "maps", "full_warehouse.npz")
MESH = os.path.join(SIM, "assets", "warehouse", "warehouse_raycast.usdc")
INTERIOR = (-26.3, -23.6, 5.3, 30.4)


@functools.lru_cache(maxsize=1)
def warehouse(cell=0.4):
    d = np.load(GT_MAP)
    g, go, res = d["grid"], d["origin"], float(d["resolution"])  # [x, y], 0 occ / 255 free / 127 unknown
    k = int(round(cell / res))
    b = INTERIOR
    x0, y0 = b[0] - 1.2, b[1] - 1.2
    nx, ny = int(np.ceil((b[2] - b[0] + 2.4) / cell)), int(np.ceil((b[3] - b[1] + 2.4) / cell))
    free = np.zeros((ny, nx), bool)
    for iy in range(ny):
        for ix in range(nx):
            gx = int(round((x0 + ix * cell - go[0]) / res))
            gy = int(round((y0 + iy * cell - go[1]) / res))
            blk = g[max(gx, 0):gx + k, max(gy, 0):gy + k]
            free[iy, ix] = blk.size > 0 and (blk == 255).all()
    gt = np.where(free, 255, 1).astype(int)
    gt[[0, -1], :] = 1
    gt[:, [0, -1]] = 1
    # heights from the mesh surface: tallest non-ceiling surface per column
    sys.path.insert(0, os.path.join(SIM, "tools"))
    from map3d import gt_surface

    pts = gt_surface(MESH, INTERIOR, 0.1)
    H = float(np.percentile(pts[pts[:, 2] > 5.0, 2], 50))
    ix = np.clip(((pts[:, 0] - x0) / cell).round().astype(int), 0, nx - 1)
    iy = np.clip(((pts[:, 1] - y0) / cell).round().astype(int), 0, ny - 1)
    body = (pts[:, 2] > 0.15) & (pts[:, 2] < H - 0.3)
    hmax = np.zeros((ny, nx), np.float32)
    np.maximum.at(hmax, (iy[body], ix[body]), pts[body, 2].astype(np.float32))
    occ = gt == 1
    # racks are hollow at 0.4 m: every obstacle component is a solid column as tall as its tallest surface
    from skimage.measure import label

    lab, n = label(occ, connectivity=2, return_num=True)
    comp_h = np.zeros(n + 1, np.float32)
    np.maximum.at(comp_h, lab[occ], hmax[occ])
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    for c in border:
        comp_h[c] = H  # outer walls
    comp_h[(comp_h == 0)] = 1.0
    comp_h[0] = 0.0
    height = comp_h[lab].astype(np.float32)
    return gt, height, H, (x0, y0)


class WarehouseScene(Scene25D):
    def __init__(self, ground_truth, height, H):  # noqa: D107 (same fields as Scene25D, fixed heights)
        self.free = ground_truth == 255
        self.H = H
        self.height = height
        occ = ground_truth == 1
        boundary = occ & binary_dilation(self.free, FOUR)
        z = (np.arange(NB) + 0.5) * DZ
        self.wall_true = boundary[..., None] & (z[None, None, :] < self.height[..., None])
        self.ceil_true = self.height < self.H - 1e-3
        self.floor_true = self.free
        self.total_area = self.wall_true.sum() * WALL_AREA + (self.ceil_true.sum() + self.floor_true.sum()) * CEIL_AREA


class WarehouseEnv3D(Env3D):
    def import_ground_truth(self, episode_index):
        gt, _, _, _ = warehouse()
        dist = distance_transform_edt(gt == 255)
        cand = np.argwhere(dist >= 3)  # >= 1.2 m from obstacles
        y, x = cand[np.random.default_rng(1000 + episode_index).integers(len(cand))]
        return gt.copy(), np.array([x, y])

    def make_scene(self, rng):
        gt, height, H, _ = warehouse()
        return WarehouseScene(self.ground_truth, height, H)
