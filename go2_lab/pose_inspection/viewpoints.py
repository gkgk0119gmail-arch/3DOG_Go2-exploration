"""POSE pose-aware semantic viewpoint sampling (Sec. IV-B) on the online voxel map.

a) candidates: ``n_sectors`` angular sectors around the object centroid, ``standoffs`` [m] on each sector's centre
   ray, base yaw facing the object; colliding / unreachable candidates are dropped by the caller (navigation)
b) visibility score S(v): mapped object voxels ray-cast from the candidate's (level) sensor origin; S = panoramic
   solid angle [deg^2] of the unoccluded ones; S < s_min is discarded
c) posture gain G(v, pitch, roll): unique unknown voxels inside the inflated object box B_o (+delta_b) reached by the
   tilted VLP-16 band within rho_max (rays stop at occupied voxels); the best posture is adopted only if G > G0
d) aim-aligned execution (Sec. IV-D): keep the sampled scan-band axis n* = R(psi_vp, theta*, phi*) e_z, choose the
   smallest yaw change from the arrival heading that reproduces it within the posture limits

The box region is pluggable (``region_fn``) for the stage-6 priors (box inflation / shape template / CAD).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import warp as wp

from .raycast import rot_ypr


@wp.kernel
def _visible_kernel(origins: wp.array(dtype=wp.vec3), targets: wp.array(dtype=wp.vec3), n_t: int,
                    grid: wp.array(dtype=wp.int32), base: int, lo: wp.vec3, voxel: float, nx: int, ny: int, nz: int,
                    out: wp.array(dtype=wp.int32)):
    tid = wp.tid()
    c = tid // n_t
    o = origins[c]
    p = targets[tid % n_t]
    d = p - o
    dist = wp.length(d)
    d = d / dist
    step = voxel * 0.5
    n = int((dist - 1.5 * voxel) / step)
    vis = int(1)
    for k in range(1, n):
        x = o + d * (float(k) * step)
        i = int(wp.floor((x[0] - lo[0]) / voxel))
        j = int(wp.floor((x[1] - lo[1]) / voxel))
        m = int(wp.floor((x[2] - lo[2]) / voxel))
        if i >= 0 and i < nx and j >= 0 and j < ny and m >= 0 and m < nz:
            if grid[base + (i * ny + j) * nz + m] == 2:
                vis = 0
                break
    out[tid] = vis


@wp.kernel
def _gain_kernel(origins: wp.array(dtype=wp.vec3), rots: wp.array(dtype=wp.mat33), local: wp.array(dtype=wp.vec3),
                 n_dirs: int, rho: float, grid: wp.array(dtype=wp.int32), base: int, lo: wp.vec3, voxel: float,
                 nx: int, ny: int, nz: int, region: wp.array(dtype=wp.uint8), b0: wp.vec3i, bs: wp.vec3i,
                 scratch: wp.array(dtype=wp.uint8)):
    """Mark unknown voxels of the region (box index range b0 + [0, bs)) reached by each pose's rays."""
    tid = wp.tid()
    p = tid // n_dirs
    o = origins[p]
    d = rots[p] * local[tid % n_dirs]
    step = voxel * 0.5
    n = int(rho / step)
    nb = bs[0] * bs[1] * bs[2]
    for k in range(1, n):
        x = o + d * (float(k) * step)
        i = int(wp.floor((x[0] - lo[0]) / voxel))
        j = int(wp.floor((x[1] - lo[1]) / voxel))
        m = int(wp.floor((x[2] - lo[2]) / voxel))
        if i < 0 or i >= nx or j < 0 or j >= ny or m < 0 or m >= nz:
            break
        g = grid[base + (i * ny + j) * nz + m]
        if g == 2:
            break
        li = i - b0[0]
        lj = j - b0[1]
        lm = m - b0[2]
        if g == 0 and li >= 0 and li < bs[0] and lj >= 0 and lj < bs[1] and lm >= 0 and lm < bs[2]:
            q = (li * bs[1] + lj) * bs[2] + lm
            if region[q] != wp.uint8(0):
                scratch[p * nb + q] = wp.uint8(1)


@wp.kernel
def _fine_gain_kernel(origins: wp.array(dtype=wp.vec3), rots: wp.array(dtype=wp.mat33), local: wp.array(dtype=wp.vec3),
                      n_dirs: int, rho: float, grid: wp.array(dtype=wp.int32), base: int, lo: wp.vec3, voxel: float,
                      nx: int, ny: int, nz: int, missing: wp.array(dtype=wp.uint8), f0: wp.vec3i, fs: wp.vec3i,
                      fvox: float, scratch: wp.array(dtype=wp.uint8)):
    """Mark missing predicted-surface fine voxels reached by each pose's rays; a ray ends in the first occupied
    coarse voxel, after testing the fine voxel it entered there (the surface it would hit)."""
    tid = wp.tid()
    p = tid // n_dirs
    o = origins[p]
    d = rots[p] * local[tid % n_dirs]
    step = fvox * 0.5
    n = int(rho / step)
    nb = fs[0] * fs[1] * fs[2]
    for k in range(1, n):
        x = o + d * (float(k) * step)
        i = int(wp.floor((x[0] - lo[0]) / voxel))
        j = int(wp.floor((x[1] - lo[1]) / voxel))
        m = int(wp.floor((x[2] - lo[2]) / voxel))
        if i < 0 or i >= nx or j < 0 or j >= ny or m < 0 or m >= nz:
            break
        g = grid[base + (i * ny + j) * nz + m]
        li = int(wp.floor((x[0] - lo[0]) / fvox)) - f0[0]
        lj = int(wp.floor((x[1] - lo[1]) / fvox)) - f0[1]
        lm = int(wp.floor((x[2] - lo[2]) / fvox)) - f0[2]
        if li >= 0 and li < fs[0] and lj >= 0 and lj < fs[1] and lm >= 0 and lm < fs[2]:
            q = (li * fs[1] + lj) * fs[2] + lm
            if missing[q] != wp.uint8(0):
                scratch[p * nb + q] = wp.uint8(1)
        if g == 2:
            break


def sample_candidates(centroid, n_sectors: int, standoffs) -> np.ndarray:
    """[n_sectors * len(standoffs), 4] rows (x, y, yaw facing the object, sector)."""
    out = []
    for s in range(n_sectors):
        a = 2 * math.pi * (s + 0.5) / n_sectors
        for d in standoffs:
            out.append([centroid[0] + d * math.cos(a), centroid[1] + d * math.sin(a), a + math.pi, s])
    return np.array(out)


class ViewpointScorer:
    """Visibility scores and posture gains against one environment's voxel map."""

    def __init__(self, world, cfg, lidar_dirs: torch.Tensor, mount_pos: torch.Tensor, mount_R: torch.Tensor,
                 postures: list[tuple[float, float]]):
        self.w, self.cfg = world, cfg
        self.dirs = lidar_dirs
        self.mount_pos, self.mount_R = mount_pos.to(world.device), mount_R.to(world.device)
        self.postures = postures
        self.dev = world.device

    def _base(self, e: int):
        nx, ny, nz = self.w.vg.shape
        return e * nx * ny * nz

    def sensor_origin(self, xy, yaw, pitch=0.0, roll=0.0):
        R = rot_ypr(torch.as_tensor(yaw), torch.as_tensor(pitch), torch.as_tensor(roll)).to(self.dev)
        b = torch.tensor([xy[0], xy[1], self.cfg.robot.base_height], dtype=torch.float32, device=self.dev)
        return b + R @ self.mount_pos, R @ self.mount_R

    # ---------------------------------------------------------------------------------------------------------------
    def visibility(self, e: int, cand_xy_yaw: np.ndarray, obj_voxels: torch.Tensor) -> np.ndarray:
        """S(v) [deg^2] for each candidate (level posture) over the object's mapped voxels."""
        if len(cand_xy_yaw) == 0 or len(obj_voxels) == 0:
            return np.zeros(len(cand_xy_yaw))
        o = torch.stack([self.sensor_origin(c[:2], c[2])[0] for c in cand_xy_yaw])
        t = obj_voxels.contiguous().float()
        nx, ny, nz = self.w.vg.shape
        out = torch.empty(len(o) * len(t), dtype=torch.int32, device=self.dev)
        wp.launch(_visible_kernel, dim=len(o) * len(t), device=self.dev,
                  inputs=[wp.from_torch(o.contiguous(), dtype=wp.vec3), wp.from_torch(t, dtype=wp.vec3), len(t),
                          wp.from_torch(self.w.vg.grid.view(-1)), self._base(e), wp.vec3(*map(float, self.w.vg.lo)),
                          self.w.vg.voxel, nx, ny, nz, wp.from_torch(out)])
        vis = out.view(len(o), len(t)).bool()
        d2 = ((t[None] - o[:, None]) ** 2).sum(-1).clamp(min=1e-3)
        sa = (self.w.vg.voxel ** 2) / d2 * (180.0 / math.pi) ** 2  # deg^2 per voxel
        return (sa * vis).sum(1).cpu().numpy()

    # ---------------------------------------------------------------------------------------------------------------
    def region_box(self, lo: np.ndarray, hi: np.ndarray, inflate: float):
        """Voxel index box (b0, size) of [lo - inflate, hi + inflate] clipped to the grid, with an all-ones mask."""
        v, glo = self.w.vg.voxel, self.w.vg.lo
        shape = np.array(self.w.vg.shape)
        b0 = np.clip(np.floor((lo - inflate - glo) / v).astype(int), 0, shape - 1)
        b1 = np.clip(np.ceil((hi + inflate - glo) / v).astype(int), 1, shape)
        lo_z = np.floor((max(lo[2] - inflate, glo[2]) - glo[2]) / v).astype(int)
        b0[2] = max(b0[2], lo_z)
        size = np.maximum(b1 - b0, 1)
        return b0, size, torch.ones(int(np.prod(size)), dtype=torch.uint8, device=self.dev)

    def fine_missing(self, e: int, region: dict) -> torch.Tensor:
        """Predicted-surface fine voxels not hit yet and not inside known-free coarse voxels (uint8, flat)."""
        w = self.w
        f0, fs = region["f0"], region["fs"]
        sl = tuple(slice(int(a), int(a + b)) for a, b in zip(f0, fs))
        seen = w.fine[e][sl]
        idx = torch.stack(torch.meshgrid(*[torch.arange(int(a), int(a + b), device=self.dev) for a, b in zip(f0, fs)],
                                         indexing="ij"), -1)
        c = ((idx.float() + 0.5) * w.fine_voxel / w.vg.voxel).long()
        shape = torch.tensor(w.vg.shape, device=self.dev)
        c = torch.minimum(c, shape - 1)
        free = w.vg.grid[e][c[..., 0], c[..., 1], c[..., 2]] == 1
        return (region["mask"].view(*map(int, fs)) & ~seen & ~free).reshape(-1).to(torch.uint8)

    def gains(self, e: int, poses: list[tuple], region) -> np.ndarray:
        """G for each pose (x, y, yaw, pitch, roll) over ``region``: (b0, size, mask[uint8]) on the coarse map
        (POSE: unknown voxels), or a dict {"f0", "fs", "mask"} of predicted surface on the fine grid (missing
        surface, stage-6 priors)."""
        if isinstance(region, dict):
            return self._fine_gains(e, poses, region)
        b0, size, mask = region
        rho = self.cfg.planner.posture.rho_max
        o, R = zip(*(self.sensor_origin(p[:2], p[2], p[3], p[4]) for p in poses))
        o, R = torch.stack(o), torch.stack(R)
        nb = int(np.prod(size))
        scratch = torch.zeros(len(poses) * nb, dtype=torch.uint8, device=self.dev)
        nx, ny, nz = self.w.vg.shape
        D = len(self.dirs)
        wp.launch(_gain_kernel, dim=len(poses) * D, device=self.dev,
                  inputs=[wp.from_torch(o.contiguous(), dtype=wp.vec3), wp.from_torch(R.contiguous(), dtype=wp.mat33),
                          wp.from_torch(self.dirs, dtype=wp.vec3), D, float(rho),
                          wp.from_torch(self.w.vg.grid.view(-1)), self._base(e),
                          wp.vec3(*map(float, self.w.vg.lo)), self.w.vg.voxel, nx, ny, nz,
                          wp.from_torch(mask), wp.vec3i(*map(int, b0)), wp.vec3i(*map(int, size)),
                          wp.from_torch(scratch)])
        return scratch.view(len(poses), nb).sum(1).cpu().numpy()

    def _fine_gains(self, e: int, poses, region) -> np.ndarray:
        missing = self.fine_missing(e, region)
        rho = self.cfg.planner.posture.rho_max
        o, R = zip(*(self.sensor_origin(p[:2], p[2], p[3], p[4]) for p in poses))
        o, R = torch.stack(o), torch.stack(R)
        nb = int(np.prod(region["fs"]))
        scratch = torch.zeros(len(poses) * nb, dtype=torch.uint8, device=self.dev)
        nx, ny, nz = self.w.vg.shape
        D = len(self.dirs)
        wp.launch(_fine_gain_kernel, dim=len(poses) * D, device=self.dev,
                  inputs=[wp.from_torch(o.contiguous(), dtype=wp.vec3), wp.from_torch(R.contiguous(), dtype=wp.mat33),
                          wp.from_torch(self.dirs, dtype=wp.vec3), D, float(rho),
                          wp.from_torch(self.w.vg.grid.view(-1)), self._base(e),
                          wp.vec3(*map(float, self.w.vg.lo)), self.w.vg.voxel, nx, ny, nz,
                          wp.from_torch(missing), wp.vec3i(*map(int, region["f0"])), wp.vec3i(*map(int, region["fs"])),
                          float(self.w.fine_voxel), wp.from_torch(scratch)])
        return scratch.view(len(poses), nb).sum(1).cpu().numpy()

    def best_posture(self, e: int, xy, yaw: float, region):
        """(pitch*, roll*, G*, G0): argmax over the grid; level if no posture beats G0."""
        poses = [(xy[0], xy[1], yaw, p, r) for (p, r) in self.postures]
        g = self.gains(e, poses, region)
        g0 = float(g[0])  # postures[0] is the level posture
        k = int(np.argmax(g))
        if g[k] > g0 and k != 0:
            return self.postures[k][0], self.postures[k][1], float(g[k]), g0
        return 0.0, 0.0, g0, g0


def aim_aligned(psi_vp: float, theta: float, phi: float, psi0: float, step: float, theta_max: float,
                phi_max: float):
    """Smallest yaw change from the arrival heading psi0 that keeps the scan-band axis of (psi_vp, theta, phi)
    within the posture limits -> (psi*, theta*, phi*) (POSE Eq. 3-6)."""
    n = (rot_ypr(psi_vp, theta, phi) @ torch.tensor([0.0, 0.0, 1.0])).numpy()

    def tilt_at(psi):
        m = rot_ypr(-psi, 0.0, 0.0).numpy() @ n
        return math.atan2(m[0], m[2]), -math.asin(max(-1.0, min(1.0, m[1])))

    beta0 = (psi_vp - psi0 + math.pi) % (2 * math.pi) - math.pi
    cands = sorted({0.0, beta0} | {k * step * s for k in range(1, int(abs(beta0) / step) + 1) for s in (1, -1)
                                   if k * step <= abs(beta0) + 1e-9}, key=abs)
    for dpsi in cands:
        th, ph = tilt_at(psi0 + dpsi)
        if abs(th) <= theta_max + 1e-6 and abs(ph) <= phi_max + 1e-6:
            return psi0 + dpsi, th, ph
    return psi_vp, theta, phi
