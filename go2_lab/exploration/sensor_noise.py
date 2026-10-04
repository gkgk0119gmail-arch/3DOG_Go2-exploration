"""VLP-16 measurement model applied to ideal ray-cast hits (cheap stand-in for a physically based RTX lidar).

Applied per scan before mapping:
  * range noise           Gaussian along the beam, sigma = 3 cm (VLP-16 datasheet accuracy +-3 cm)
  * min range / dropout   returns closer than 0.9 m are discarded (velodyne driver default); random 2 % dropout
  * rolling-shutter skew  a VLP-16 sweeps 360 deg in 0.1 s; a beam fired dt before the end of the sweep was taken
                          from where the robot was dt ago, but is interpreted in the end-of-sweep frame (no deskew)
  * pose error (optional) per-scan localization error of the mapping pose (stand-in for SLAM jitter)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from isaaclab.utils.math import quat_apply_inverse


@dataclass
class VLP16NoiseCfg:
    range_sigma: float = 0.03
    """Range noise standard deviation [m]."""
    min_range: float = 0.9
    """Returns closer than this [m] are dropped."""
    dropout: float = 0.02
    """Probability that a return is lost."""
    sweep_period: float = 0.1
    """Time of one 360 deg revolution [s] (10 Hz)."""
    pose_sigma_xy: float = 0.0
    """Per-scan position error of the mapping pose [m] (0 disables)."""
    pose_sigma_yaw_deg: float = 0.0
    """Per-scan heading error of the mapping pose [deg] (0 disables)."""


def _rotz(v: torch.Tensor, ang: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(ang), torch.sin(ang)
    return torch.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1], v[..., 2]], dim=-1)


@torch.no_grad()
def apply_vlp16_noise(cfg: VLP16NoiseCfg, origins: torch.Tensor, quats: torch.Tensor, hits: torch.Tensor,
                      lin_vel_w: torch.Tensor, yaw_rate: torch.Tensor, generator: torch.Generator | None = None):
    """Corrupt ideal hits like a real VLP-16 scan.

    Args:
        origins: Scan origins [E, 3] (end of sweep).
        quats: Sensor orientations [E, 4] (x, y, z, w).
        hits: Ideal ray hits [E, R, 3]; non-finite rows have no return.
        lin_vel_w: Sensor linear velocity in the world frame [E, 3] [m/s].
        yaw_rate: Yaw rate [E] [rad/s].

    Returns:
        (noisy hits [E, R, 3] with dropped returns set to inf, mapping origins [E, 3], mapping yaw offsets [E])
    """
    E, R, _ = hits.shape
    dev = hits.device
    rnd = lambda *shape: torch.rand(*shape, device=dev, generator=generator)  # noqa: E731
    nrm = lambda *shape: torch.randn(*shape, device=dev, generator=generator)  # noqa: E731

    valid = torch.isfinite(hits).all(-1)
    vec = torch.where(valid[..., None], hits - origins[:, None, :], torch.zeros_like(hits))
    rng = torch.linalg.norm(vec, dim=-1).clamp(min=1e-6)
    dirs = vec / rng[..., None]

    # range noise, min range, dropout
    rng_noisy = rng + cfg.range_sigma * nrm(E, R)
    keep = valid & (rng_noisy >= cfg.min_range) & (rnd(E, R) >= cfg.dropout)
    pts = origins[:, None, :] + dirs * rng_noisy[..., None]

    # rolling shutter: beam azimuth (sensor frame) -> time before end of sweep
    local = quat_apply_inverse(quats[:, None, :].expand(E, R, 4).reshape(-1, 4), dirs.reshape(-1, 3)).reshape(E, R, 3)
    az = torch.atan2(local[..., 1], local[..., 0])  # [-pi, pi], sweep assumed to run -pi -> pi
    dt = cfg.sweep_period * (1.0 - (az + math.pi) / (2 * math.pi))  # beam fired dt before the sweep ended
    # the point was seen from (origin - v dt, yaw - w dt) but is reported in the end-of-sweep frame
    rel = pts - origins[:, None, :] + lin_vel_w[:, None, :] * dt[..., None]
    pts = origins[:, None, :] + _rotz(rel, yaw_rate[:, None] * dt)

    # localization error of the mapping pose: rotate/shift the whole scan about the origin
    map_origins = origins.clone()
    dyaw = torch.zeros(E, device=dev)
    if cfg.pose_sigma_xy > 0 or cfg.pose_sigma_yaw_deg > 0:
        dxy = torch.zeros(E, 3, device=dev)
        dxy[:, :2] = cfg.pose_sigma_xy * nrm(E, 2)
        dyaw = math.radians(cfg.pose_sigma_yaw_deg) * nrm(E)
        pts = origins[:, None, :] + dxy[:, None, :] + _rotz(pts - origins[:, None, :], dyaw[:, None])
        map_origins = origins + dxy

    pts = torch.where(keep[..., None], pts, torch.full_like(pts, float("inf")))
    return pts, map_origins, dyaw


@torch.no_grad()
def deskew(origins: torch.Tensor, quats: torch.Tensor, hits: torch.Tensor, lin_vel_est: torch.Tensor,
           yaw_rate_est: torch.Tensor, sweep_period: float = 0.1) -> torch.Tensor:
    """Undo rolling-shutter skew with an odometry/IMU velocity estimate (what LIO pipelines do per scan).

    Inverse of the skew in :func:`apply_vlp16_noise`: p = o + Rz(-w dt) (p' - o) - v dt.
    """
    E, R, _ = hits.shape
    ok = torch.isfinite(hits).all(-1)
    rel = torch.where(ok[..., None], hits - origins[:, None, :], torch.zeros_like(hits))
    local = quat_apply_inverse(quats[:, None, :].expand(E, R, 4).reshape(-1, 4), rel.reshape(-1, 3)).reshape(E, R, 3)
    dt = sweep_period * (1.0 - (torch.atan2(local[..., 1], local[..., 0]) + math.pi) / (2 * math.pi))
    rel = _rotz(rel, -yaw_rate_est[:, None] * dt) - lin_vel_est[:, None, :] * dt[..., None]
    return torch.where(ok[..., None], origins[:, None, :] + rel, hits)
