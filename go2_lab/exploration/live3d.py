"""3D surface belief from real VLP-16 hits, per robot, on the GPU (deployment side of ariadne3d/lidar3d.py).

Same grid as the 2D occupancy mapper (rows = y), same height bins / utility classes / annulus kernels as training:
  wall bin  : a hit between the floor and the ceiling lands in (cell, 0.5 m bin)       -> seen
              a ray passing through (cell, bin) before its hit                         -> empty
  ceiling   : a hit within 0.4 m of the ceiling estimate (highest return seen, > 5 m)
  utility   : unknown wall bins on known obstacle edges + unseen ceiling over known free cells, counted on a 2x pooled
              grid and spread with the annulus of view distances for each height class (torch conv2d)

``GTCoverage3D`` scores every robot online the way tools/map3d.py does offline: 0.1 m voxels of the warehouse mesh
surface, covered when a hit voxel is within one voxel (26-neighbourhood).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

SIM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(SIM, "ariadne3d"))
from lidar3d import CLASSES, DZ, H_MAX, NB, annulus  # noqa: E402

sys.path.pop(0)

FREE, OCCUPIED = 0, 100  # occupancy_mapper / ROS layout


class Live3D:
    def __init__(self, num_envs, origin, nx, ny, cell=0.4, tilt_deg=0.0, z_sensor=0.44, utility_range=20.0,
                 device="cuda", ray_samples=12):
        self.E, self.nx, self.ny, self.cell = num_envs, nx, ny, cell
        self.origin = torch.tensor(origin, dtype=torch.float32, device=device)
        self.device = device
        self.zs = z_sensor
        self.ray_t = torch.linspace(0.05, 0.95, ray_samples, device=device)
        self.wall = torch.zeros(num_envs, ny, nx, NB, dtype=torch.uint8, device=device)
        self.ceil = torch.zeros(num_envs, ny, nx, dtype=torch.bool, device=device)
        self.zmax = torch.zeros(num_envs, device=device)
        self.seen_area = torch.zeros(num_envs, device=device)
        tilt = np.deg2rad(tilt_deg)
        up, down = np.tan(np.deg2rad(15.0) + tilt), np.tan(np.deg2rad(15.0))
        self._r_min = lambda z: (z - z_sensor) / up if z > z_sensor else (z_sensor - z) / down
        self.utility_range = utility_range
        self.ucell = 2 * cell
        self.kernels = [torch.as_tensor(annulus(min(self._r_min(lo), utility_range), utility_range, self.ucell),
                                        device=device)[None, None] for lo, _ in CLASSES]
        self._ceil_k = {}

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else torch.as_tensor(env_ids, device=self.device)
        self.wall[ids] = 0
        self.ceil[ids] = False
        self.zmax[ids] = 0.0
        self.seen_area[ids] = 0.0

    def ceiling(self, e):
        z = float(self.zmax[e])
        return (z, True) if z > 5.0 else (H_MAX, False)

    def insert(self, env_ids, origins, hits):
        """origins [n, 3], hits [n, R, 3] world frame (non-finite = no return)."""
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        ok = torch.isfinite(hits).all(-1)
        z = torch.where(ok, hits[..., 2], torch.zeros_like(hits[..., 2]))
        self.zmax[ids] = torch.maximum(self.zmax[ids], torch.where(ok & (z < 15.0), z, torch.zeros_like(z)).amax(1))
        H = torch.where(self.zmax[ids] > 5.0, self.zmax[ids], torch.full_like(self.zmax[ids], H_MAX))[:, None]
        e_idx = ids[:, None].expand_as(z)
        cx = ((hits[..., 0] - self.origin[0]) / self.cell).floor().long()
        cy = ((hits[..., 1] - self.origin[1]) / self.cell).floor().long()
        inside = ok & (cx >= 0) & (cx < self.nx) & (cy >= 0) & (cy < self.ny)
        before_w = (self.wall == 1).sum(dim=(1, 2, 3))[ids].float()
        before_c = self.ceil.sum(dim=(1, 2))[ids].float()
        # empty bins along the rays (before the hit)
        o = origins[:, None, None, :]
        p = o + self.ray_t[None, None, :, None] * (torch.where(ok[..., None], hits, origins[:, None, :]) - origins[:, None, :])[:, :, None, :]
        px = ((p[..., 0] - self.origin[0]) / self.cell).floor().long()
        py = ((p[..., 1] - self.origin[1]) / self.cell).floor().long()
        pb = (p[..., 2] / DZ).floor().long()
        m = ok[..., None] & (px >= 0) & (px < self.nx) & (py >= 0) & (py < self.ny) & (pb >= 0) & (pb < NB) \
            & (p[..., 2] < H[..., None])
        ee = ids[:, None, None].expand_as(px)
        cur = self.wall[ee[m], py[m], px[m], pb[m]]
        self.wall[ee[m], py[m], px[m], pb[m]] = torch.where(cur == 0, torch.full_like(cur, 2), cur)
        # wall hits / ceiling hits
        is_ceil = inside & (z >= H - 0.4) & (H < H_MAX)
        is_wall = inside & (z > 0.1) & (z < H - 0.4)
        b = (z / DZ).floor().long().clamp(0, NB - 1)
        self.wall[e_idx[is_wall], cy[is_wall], cx[is_wall], b[is_wall]] = 1
        self.ceil[e_idx[is_ceil], cy[is_ceil], cx[is_ceil]] = True
        new_w = (self.wall == 1).sum(dim=(1, 2, 3))[ids].float() - before_w
        new_c = self.ceil.sum(dim=(1, 2))[ids].float() - before_c
        self.seen_area[ids] += new_w * self.cell * DZ + new_c * self.cell**2

    def utility(self, e, grid2d: np.ndarray) -> np.ndarray:
        """3D utility grid [ny // 2, nx // 2] (unseen elements in view) for robot ``e`` with its 2D map."""
        g = torch.as_tensor(grid2d, device=self.device)
        free = g == FREE
        edge = (g == OCCUPIED) & (F.max_pool2d(free[None, None].float(), 3, 1, 1)[0, 0] > 0)
        H, known = self.ceiling(e)
        nb_h = int(np.ceil(H / DZ))
        unk = (self.wall[e] == 0) & edge[..., None]
        unk[..., nb_h:] = False
        ny2, nx2 = self.ny // 2, self.nx // 2
        u = torch.zeros(ny2, nx2, device=self.device)

        def pooled(a):
            return a[: ny2 * 2, : nx2 * 2].reshape(ny2, 2, nx2, 2).sum((1, 3))

        for (lo, hi), k in zip(CLASSES, self.kernels):
            b0, b1 = int(lo / DZ), min(int(hi / DZ), nb_h)
            if b1 <= b0:
                continue
            cnt = pooled(unk[..., b0:b1].sum(-1).float())
            if cnt.any():
                u += F.conv2d(cnt[None, None], k, padding=k.shape[-1] // 2)[0, 0]
        key = round(H, 1)
        if key not in self._ceil_k:
            r0 = self._r_min(H)
            self._ceil_k[key] = None if r0 >= self.utility_range else torch.as_tensor(
                annulus(r0, self.utility_range, self.ucell), device=self.device)[None, None]
        k = self._ceil_k[key]
        if k is not None:
            cnt = pooled((free & ~self.ceil[e]).float())
            if cnt.any():
                u += F.conv2d(cnt[None, None], k, padding=k.shape[-1] // 2)[0, 0]
        return u.clamp_min(0).cpu().numpy()


class GTCoverage3D:
    """Online 3D surface coverage against the warehouse mesh (same definition as tools/map3d.py)."""

    def __init__(self, num_envs, bounds, voxel=0.1, device="cuda"):
        sys.path.insert(0, os.path.join(SIM, "tools"))
        from map3d import OFFS, gt_surface, keys_of

        sys.path.pop(0)
        mesh = os.path.join(SIM, "assets", "warehouse", "warehouse_raycast.usdc")
        self.v, self.bounds, self.device = voxel, bounds, device
        self.origin = np.array([bounds[0] - 1, bounds[1] - 1, -1.0])
        gt = gt_surface(mesh, bounds, voxel / 2)
        keys = np.unique(keys_of(np.floor((gt - self.origin) / voxel).astype(np.int64)))
        self.keys = torch.as_tensor(keys, device=device)
        self.offs = torch.as_tensor(OFFS, device=device)
        self.covered = torch.zeros(num_envs, len(keys), dtype=torch.bool, device=device)
        self._o = torch.as_tensor(self.origin, dtype=torch.float32, device=device)

    def reset(self, env_ids=None):
        self.covered[slice(None) if env_ids is None else torch.as_tensor(env_ids, device=self.device)] = False

    def insert(self, env_ids, hits):
        b = self.bounds
        for i, e in enumerate(env_ids):
            h = hits[i]
            h = h[torch.isfinite(h).all(-1)]
            h = h[(h[:, 0] >= b[0] - 0.2) & (h[:, 0] <= b[2] + 0.2) & (h[:, 1] >= b[1] - 0.2) & (h[:, 1] <= b[3] + 0.2)]
            if len(h) == 0:
                continue
            ijk = torch.unique(torch.floor((h - self._o) / self.v).long(), dim=0)
            nb = (ijk[:, None, :] + self.offs[None]).reshape(-1, 3)
            k = (nb[:, 0] << 42) + (nb[:, 1] << 21) + nb[:, 2]
            pos = torch.searchsorted(self.keys, k).clamp(max=len(self.keys) - 1)
            hit = self.keys[pos] == k
            self.covered[e, pos[hit]] = True

    def coverage(self) -> np.ndarray:
        return self.covered.float().mean(1).cpu().numpy()
