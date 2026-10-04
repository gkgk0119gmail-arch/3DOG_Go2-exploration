"""Save ray-cast LiDAR scans like a Velodyne driver would publish them (points in the sensor frame).

Registered as an interval event term, so every scan period it writes one ``.npz`` per environment:
    points   [N, 3] float32  x, y, z in the sensor ("velodyne") frame [m]
    ring     [N]    uint8    laser channel (0 = lowest beam)
    pos_w    [3]    float32  sensor position in the world frame [m]
    quat_w   [4]    float32  sensor orientation in the world frame, (x, y, z, w)
    stamp    float           simulation time [s]
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_mul

_RUN_DIRS: dict[str, str] = {}


def save_lidar_scan(env, env_ids, sensor_cfg: SceneEntityCfg, out_dir: str, channels: int, vertical_fov: tuple):
    sensor = env.scene[sensor_cfg.name]
    run_dir = _RUN_DIRS.setdefault(out_dir, os.path.join(out_dir, time.strftime("%Y-%m-%d_%H-%M-%S")))
    os.makedirs(run_dir, exist_ok=True)

    hits_w = sensor.data.ray_hits_w.torch  # [E, R, 3], inf where nothing was hit
    # data.pos_w / quat_w are the pose of the link the sensor is attached to; the rays start at the
    # cfg offset from it, so the scan origin ("velodyne" frame) is link pose * offset.
    link_pos_w = sensor.data.pos_w.torch  # [E, 3]
    link_quat_w = sensor.data.quat_w.torch  # [E, 4] (x, y, z, w)
    off_pos = torch.tensor(sensor.cfg.offset.pos, device=link_pos_w.device).expand_as(link_pos_w)
    off_quat = torch.tensor(sensor.cfg.offset.rot, device=link_pos_w.device).expand_as(link_quat_w)
    pos_w = link_pos_w + quat_apply(link_quat_w, off_pos)
    quat_w = quat_mul(link_quat_w, off_quat)
    stamp = float(env.sim.current_time) if hasattr(env.sim, "current_time") else float(env.common_step_counter * env.step_dt)
    step = env.common_step_counter

    ids = range(hits_w.shape[0]) if env_ids is None else env_ids.tolist()
    for e in ids:
        valid = torch.isfinite(hits_w[e]).all(dim=-1)
        rel = hits_w[e][valid] - pos_w[e]
        pts = quat_apply_inverse(quat_w[e].expand(rel.shape[0], 4), rel)
        elev = torch.rad2deg(torch.atan2(pts[:, 2], torch.linalg.norm(pts[:, :2], dim=1)))
        step_deg = (vertical_fov[1] - vertical_fov[0]) / (channels - 1)
        ring = torch.clamp(torch.round((elev - vertical_fov[0]) / step_deg), 0, channels - 1)
        np.savez_compressed(
            os.path.join(run_dir, f"env{e}_scan_{step:07d}.npz"),
            points=pts.cpu().numpy().astype(np.float32),
            ring=ring.cpu().numpy().astype(np.uint8),
            pos_w=pos_w[e].cpu().numpy().astype(np.float32),
            quat_w=quat_w[e].cpu().numpy().astype(np.float32),
            stamp=stamp,
        )
