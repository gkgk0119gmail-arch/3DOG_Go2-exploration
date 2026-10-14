"""GPU ray casting against a baked scene (warp): the VLP-16 used by the pipeline, the upper-bound sweep and the
free-space bookkeeping of the scene-coverage map.

The LiDAR is cast here instead of through Isaac Lab's RayCaster so every return carries the triangle it hit, hence
the ground-truth object id (exact GT segmentation, POSE's evaluation setting) -- geometry and ray pattern are the
same as the RayCaster (single static mesh, instantaneous 10 Hz sweep, the robot itself is not hit).

Conventions: body rotation R(yaw, pitch, roll) = Rz(yaw) Ry(pitch) Rx(roll) (POSE Eq. 3; pitch > 0 = nose down,
go2_lab/posture.py); the sensor is mounted at ``lidar.mount_pos`` with a nose-up tilt and its scan origin
``lidar.scan_height`` above the plate (go2_lab/factory.py).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import warp as wp

wp.config.quiet = True


# --------------------------------------------------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------------------------------------------------
@wp.kernel
def _cast_kernel(mesh: wp.uint64, origins: wp.array(dtype=wp.vec3), rots: wp.array(dtype=wp.mat33),
                 local: wp.array(dtype=wp.vec3), n_dirs: int, max_d: float,
                 t_out: wp.array(dtype=float), face_out: wp.array(dtype=int)):
    tid = wp.tid()
    p = tid // n_dirs
    d = rots[p] * local[tid % n_dirs]
    q = wp.mesh_query_ray(mesh, origins[p], d, max_d)
    if q.result:
        t_out[tid] = q.t
        face_out[tid] = q.face
    else:
        t_out[tid] = 1.0e9
        face_out[tid] = -1


@wp.kernel
def _mark_kernel(origins: wp.array(dtype=wp.vec3), rots: wp.array(dtype=wp.mat33), local: wp.array(dtype=wp.vec3),
                 n_dirs: int, ray_t: wp.array(dtype=float), env_of_pose: wp.array(dtype=int),
                 lo: wp.vec3, voxel: float, nx: int, ny: int, nz: int, free_range: float, step: float,
                 grid: wp.array(dtype=wp.int32)):
    """Free voxels along each ray (value 1), the voxel of a return occupied (value 2); max keeps occupied."""
    tid = wp.tid()
    p = tid // n_dirs
    o = origins[p]
    d = rots[p] * local[tid % n_dirs]
    t = ray_t[tid]
    hit = t < 1.0e8
    end = wp.min(t, free_range)
    base = env_of_pose[p] * nx * ny * nz
    n = int(end / step)
    for k in range(n):
        x = o + d * (float(k) * step)
        i = int(wp.floor((x[0] - lo[0]) / voxel))
        j = int(wp.floor((x[1] - lo[1]) / voxel))
        m = int(wp.floor((x[2] - lo[2]) / voxel))
        if i >= 0 and i < nx and j >= 0 and j < ny and m >= 0 and m < nz:
            wp.atomic_max(grid, base + (i * ny + j) * nz + m, 1)
    if hit:
        x = o + d * t
        i = int(wp.floor((x[0] - lo[0]) / voxel))
        j = int(wp.floor((x[1] - lo[1]) / voxel))
        m = int(wp.floor((x[2] - lo[2]) / voxel))
        if i >= 0 and i < nx and j >= 0 and j < ny and m >= 0 and m < nz:
            wp.atomic_max(grid, base + (i * ny + j) * nz + m, 2)


# --------------------------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------------------------
def vlp16_dirs(channels: int, vfov_deg, az_res_deg: float) -> np.ndarray:
    """Unit ray directions in the sensor frame (x forward, z up), ring-major."""
    el = np.deg2rad(np.linspace(vfov_deg[0], vfov_deg[1], channels))
    az = np.deg2rad(np.arange(-180.0, 180.0, az_res_deg))
    E, A = np.meshgrid(el, az, indexing="ij")
    return np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1).reshape(-1, 3)


def rot_ypr(yaw, pitch, roll):
    """Rz(yaw) Ry(pitch) Rx(roll) for broadcastable tensors -> [..., 3, 3]."""
    yaw, pitch, roll = torch.broadcast_tensors(*(torch.as_tensor(v, dtype=torch.float32) for v in (yaw, pitch, roll)))
    cy, sy, cp, sp, cr, sr = yaw.cos(), yaw.sin(), pitch.cos(), pitch.sin(), roll.cos(), roll.sin()
    R = torch.stack([
        torch.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr], -1),
        torch.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr], -1),
        torch.stack([-sp, cp * sr, cp * cr], -1),
    ], -2)
    return R


def mount(lidar_cfg, tilt_deg: float | None = None):
    """(scan origin in the base frame [3], sensor rotation in the base frame [3, 3])."""
    t = math.radians(lidar_cfg.mount_tilt_deg if tilt_deg is None else tilt_deg)
    mp, h = lidar_cfg.mount_pos, lidar_cfg.scan_height
    pos = torch.tensor([mp[0] - h * math.sin(t), mp[1], mp[2] + h * math.cos(t)], dtype=torch.float32)
    R = rot_ypr(0.0, -t, 0.0)  # nose-up tilt = negative pitch about y
    return pos, R


def sensor_poses(base_xyz: torch.Tensor, R_body: torch.Tensor, mount_pos: torch.Tensor, R_mount: torch.Tensor):
    """World scan origins [P, 3] and sensor rotations [P, 3, 3] from body poses."""
    dev = base_xyz.device
    o = base_xyz + (R_body @ mount_pos.to(dev))
    return o, R_body @ R_mount.to(dev)


class SceneMesh:
    """The baked scene (gt.npz) as a warp mesh + per-triangle object ids."""

    def __init__(self, gt_npz: str, device: str = "cuda:0"):
        d = np.load(gt_npz)
        self.device = device
        self.verts = d["verts"]
        self.tris = d["tris"]
        self.tri_obj = torch.as_tensor(d["tri_obj"].astype(np.int64), device=device)
        self.mesh = wp.Mesh(points=wp.array(self.verts, dtype=wp.vec3, device=device),
                            indices=wp.array(self.tris.reshape(-1), dtype=wp.int32, device=device))

    def cast(self, origins: torch.Tensor, rots: torch.Tensor, local: torch.Tensor, max_d: float):
        """Rays local[D] from P poses -> t [P, D] (1e9 = no return), face [P, D] (-1), obj [P, D] (-1 background)."""
        P, D = len(origins), len(local)
        dev = self.device
        t = torch.empty(P * D, dtype=torch.float32, device=dev)
        f = torch.empty(P * D, dtype=torch.int32, device=dev)
        wp.launch(_cast_kernel, dim=P * D, device=dev,
                  inputs=[self.mesh.id, wp.from_torch(origins.contiguous().float(), dtype=wp.vec3),
                          wp.from_torch(rots.contiguous().float(), dtype=wp.mat33),
                          wp.from_torch(local.contiguous().float(), dtype=wp.vec3), D, float(max_d),
                          wp.from_torch(t), wp.from_torch(f)])
        t, f = t.view(P, D), f.view(P, D).long()
        obj = torch.where(f >= 0, self.tri_obj[f.clamp(min=0)], torch.full_like(f, -1))
        return t, f, obj


class VoxelGrid:
    """Per-environment unknown (0) / free (1) / occupied (2) voxel map over the scene volume, on the GPU."""

    def __init__(self, bounds, z_range, voxel: float, n_envs: int, device: str = "cuda:0"):
        self.lo = np.array([bounds[0], bounds[1], z_range[0]], np.float32)
        hi = np.array([bounds[2], bounds[3], z_range[1]], np.float32)
        self.voxel = float(voxel)
        self.shape = tuple(int(math.ceil(v)) for v in (hi - self.lo) / voxel)
        self.n_envs, self.device = n_envs, device
        self.grid = torch.zeros((n_envs, *self.shape), dtype=torch.int32, device=device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.grid.zero_()
        else:
            self.grid[env_ids] = 0

    def insert(self, origins, rots, local, t, env_of_pose, free_range: float):
        P, D = len(origins), len(local)
        nx, ny, nz = self.shape
        wp.launch(_mark_kernel, dim=P * D, device=self.device,
                  inputs=[wp.from_torch(origins.contiguous().float(), dtype=wp.vec3),
                          wp.from_torch(rots.contiguous().float(), dtype=wp.mat33),
                          wp.from_torch(local.contiguous().float(), dtype=wp.vec3), D,
                          wp.from_torch(t.contiguous().view(-1).float()),
                          wp.from_torch(env_of_pose.to(torch.int32).contiguous()),
                          wp.vec3(*map(float, self.lo)), self.voxel, nx, ny, nz, float(free_range),
                          self.voxel * 0.5, wp.from_torch(self.grid.view(-1))])

    def known_fraction(self, mask: torch.Tensor | None = None) -> torch.Tensor:
        """(free + occupied) / total per env; with ``mask`` [nx, ny, nz] only over the masked voxels."""
        known = self.grid > 0
        if mask is None:
            return known.flatten(1).float().mean(1)
        m = mask.to(known.device)
        return (known & m).flatten(1).sum(1).float() / m.sum().clamp(min=1).float()
