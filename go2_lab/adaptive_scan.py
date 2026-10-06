"""Gain-driven posture scanning around a known machine (POSE-style posture selection with a CAD prior).

The robot walks a loop of stops around the machine (corners + side midpoints). Live, every LiDAR sweep marks which
surface elements of the machine have been seen (tools/urdf_to_scene.py *_gt.npz; element = 5 cm voxel x facing
direction, seen only from the side it faces -- tools/machine_coverage.py). At a stop it predicts, for every candidate
body posture (pitch x roll grid), how many still-unseen elements one VLP-16 sweep would hit (GPU ray cast against the
machine mesh) and
    tilts to the best posture if its gain beats the level posture by ``margin`` and exceeds ``min_gain``,
    holds it for ``hold_s`` (the sweeps update the seen set), re-plans (up to ``max_postures`` per stop),
    otherwise walks on at once -- no time spent where tilting adds nothing.
GO2_SCAN_MODE=adaptive (default) | level (same stops, never tilts) | fixed (the scripted list of posture.py).
Logs <out_dir>/scan_log.json: coverage over time and every posture decision.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Sequence
from dataclasses import MISSING

import numpy as np
import torch
import warp as wp

from isaaclab.utils import configclass

from .posture import WaypointScanCommand, WaypointScanCommandCfg

OFFS = torch.tensor([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])


@wp.kernel
def _cast(mesh: wp.uint64, origins: wp.array(dtype=wp.vec3), quats: wp.array(dtype=wp.quat),
          local: wp.array(dtype=wp.vec3), n_dirs: int, max_d: float, out: wp.array(dtype=wp.vec3)):
    tid = wp.tid()
    p = tid // n_dirs
    o = origins[p]
    d = wp.quat_rotate(quats[p], local[tid % n_dirs])
    q = wp.mesh_query_ray(mesh, o, d, max_d)
    if q.result:
        out[tid] = o + q.t * d
    else:
        out[tid] = wp.vec3(1.0e9, 1.0e9, 1.0e9)


def _quat_xyzw(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(R).as_quat()


class MachineSurface:
    """Face-aware surface elements of the machine + the seen set, on the GPU."""

    def __init__(self, gt_npz: str, voxel: float, device: str):
        d = np.load(gt_npz)
        v, t = d["verts"].astype(np.float64), d["tris"]
        a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
        nrm = np.cross(b - a, c - a)
        area = 0.5 * np.linalg.norm(nrm, axis=1)
        unit = nrm / (2 * area[:, None] + 1e-12)
        n = np.maximum(1, np.round(area / (voxel / 2) ** 2)).astype(int)
        idx = np.repeat(np.arange(len(a)), n)
        r1, r2 = np.random.default_rng(0).random((2, len(idx), 1))
        s = np.sqrt(r1)
        pts = (1 - s) * a[idx] + s * (1 - r2) * b[idx] + s * r2 * c[idx]
        nr = unit[idx]
        self.origin = pts.min(0) - 1.0
        ax = np.abs(nr).argmax(1)
        dirn = ax * 2 + (np.take_along_axis(nr, ax[:, None], 1)[:, 0] < 0)
        ijk = np.floor((pts - self.origin) / voxel).astype(np.int64)
        keys = ((ijk[:, 0] << 42) + (ijk[:, 1] << 21) + ijk[:, 2]) * 8 + dirn
        keys, first = np.unique(keys, return_index=True)
        self.device, self.voxel = device, voxel
        self.keys = torch.as_tensor(keys, device=device)
        self.nz = torch.as_tensor(nr[first, 2], device=device)
        self.seen = torch.zeros(len(keys), dtype=torch.bool, device=device)
        self._o = torch.as_tensor(self.origin, dtype=torch.float32, device=device)
        self._offs = OFFS.to(device)
        self.lo = torch.as_tensor(pts.min(0) - 0.3, dtype=torch.float32, device=device)
        self.hi = torch.as_tensor(pts.max(0) + 0.3, dtype=torch.float32, device=device)
        self.mesh = wp.Mesh(points=wp.array(v.astype(np.float32), dtype=wp.vec3, device=device),
                            indices=wp.array(t.astype(np.int32).reshape(-1), dtype=wp.int32, device=device))

    def match(self, hits: torch.Tensor, origin: torch.Tensor) -> torch.Tensor:
        """Element indices hit by returns ``hits`` [N, 3] from sensor ``origin`` [3] (face-aware, 26-neighbourhood)."""
        ok = torch.isfinite(hits).all(-1) & (hits >= self.lo).all(-1) & (hits <= self.hi).all(-1)
        p = hits[ok]
        if len(p) == 0:
            return torch.zeros(0, dtype=torch.long, device=self.device)
        w = origin[None] - p
        facing = torch.stack([w[:, 0] > 0, w[:, 0] < 0, w[:, 1] > 0, w[:, 1] < 0, w[:, 2] > 0, w[:, 2] < 0], 1)  # [N, 6]
        base = torch.floor((p - self._o) / self.voxel).long()
        nb = base[:, None, :] + self._offs[None]  # [N, 27, 3]
        k = ((nb[..., 0] << 42) + (nb[..., 1] << 21) + nb[..., 2]) * 8  # [N, 27]
        kk = (k[..., None] + torch.arange(6, device=self.device)).reshape(len(p), -1)  # [N, 27*6]
        m = facing[:, None, :].expand(-1, 27, -1).reshape(len(p), -1)
        kk = kk[m]
        pos = torch.searchsorted(self.keys, kk).clamp(max=len(self.keys) - 1)
        return torch.unique(pos[self.keys[pos] == kk])

    def coverage(self) -> float:
        return float(self.seen.float().mean())


class AdaptiveScanCommand(WaypointScanCommand):
    cfg: AdaptiveScanCommandCfg

    def __init__(self, cfg: AdaptiveScanCommandCfg, env):
        super().__init__(cfg, env)
        self._env = env
        dev = str(self.device)
        self.surface = MachineSurface(cfg.gt_npz, cfg.voxel, dev)
        self.sensor = env.scene[cfg.lidar_name]
        el = np.deg2rad(np.linspace(-15, 15, 16))
        az = np.deg2rad(np.arange(0, 360, cfg.plan_az_res_deg))
        E, A = np.meshgrid(el, az, indexing="ij")
        local = np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1).reshape(-1, 3)
        off = self.sensor.cfg.offset
        from scipy.spatial.transform import Rotation

        self._off_pos = np.asarray(off.pos)
        self._off_R = Rotation.from_quat(off.rot).as_matrix()
        self._local = wp.array(local.astype(np.float32), dtype=wp.vec3, device=dev)
        self._n_dirs = len(local)
        pitch = np.deg2rad(cfg.pitch_grid_deg)
        roll = np.deg2rad(cfg.roll_grid_deg)
        self.candidates = np.array([(p, r) for p in pitch for r in roll])
        self.mode = os.environ.get("GO2_SCAN_MODE", "adaptive")
        self.phase = "walk"  # walk | hold | settle
        self.phase_t = 0.0
        self.n_post = 0
        self.used: list[int] = []
        self.log = {"mode": self.mode, "machine": cfg.gt_npz, "coverage": [], "decisions": [], "stops": []}
        self._last_cov_t = -1.0
        self._t = 0.0
        self.out_dir = os.environ.get("GO2_SCAN_LOG", os.path.join(os.path.dirname(cfg.gt_npz), "runs",
                                                                   time.strftime("%Y-%m-%d_%H-%M-%S") + f"_{self.mode}"))
        os.makedirs(self.out_dir, exist_ok=True)

    # -- live coverage -----------------------------------------------------------------------------------
    def _sensor_pose(self):
        from isaaclab.utils.math import quat_apply

        pos = self.sensor.data.pos_w.torch[0]
        q = self.sensor.data.quat_w.torch[0]
        off = torch.as_tensor(self._off_pos, dtype=torch.float32, device=self.device)
        return pos + quat_apply(q[None], off[None])[0]

    def _update_seen(self):
        hits = self.sensor.data.ray_hits_w.torch[0]
        idx = self.surface.match(hits, self._sensor_pose())
        self.surface.seen[idx] = True

    # -- posture gain prediction -------------------------------------------------------------------------
    def _predict_gains(self) -> np.ndarray:
        from scipy.spatial.transform import Rotation

        base = self.robot.data.root_pos_w.torch[0].cpu().numpy()
        yaw = float(self.robot.data.heading_w.torch[0])
        origins, quats = [], []
        for p, r in self.candidates:
            Rb = Rotation.from_euler("zyx", [yaw, p, r]).as_matrix()  # yaw, then pitch about y, roll about x
            Rs = Rb @ self._off_R
            origins.append(base + Rb @ self._off_pos)
            quats.append(_quat_xyzw(Rs))
        dev = str(self.device)
        o = wp.array(np.asarray(origins, np.float32), dtype=wp.vec3, device=dev)
        q = wp.array(np.asarray(quats, np.float32), dtype=wp.quat, device=dev)
        out = wp.empty(len(origins) * self._n_dirs, dtype=wp.vec3, device=dev)
        wp.launch(_cast, dim=len(origins) * self._n_dirs,
                  inputs=[self.surface.mesh.id, o, q, self._local, self._n_dirs, 30.0, out], device=dev)
        hits = wp.to_torch(out).view(len(origins), self._n_dirs, 3)
        gains = []
        for i in range(len(origins)):
            idx = self.surface.match(hits[i], torch.as_tensor(origins[i], dtype=torch.float32, device=self.device))
            gains.append(int((~self.surface.seen[idx]).sum()))
        return np.array(gains)

    # -- driver ------------------------------------------------------------------------------------------
    def _update_command(self):
        dt = self._dt
        self._t += dt
        if self._t - self._last_cov_t >= 0.1 - 1e-6:  # LiDAR rate
            self._last_cov_t = self._t
            self._update_seen()
            self.log["coverage"].append((round(self._t, 2), self.surface.coverage()))
            if len(self.log["coverage"]) % 100 == 0:  # every 10 s
                self.save_log()
        pos = self.robot.data.root_pos_w.torch[:, :2] - self._env_origins[:, :2]
        if self.phase == "walk":
            dist = float(torch.linalg.norm(self.waypoints[self.waypoint_idx[0]] - pos[0]))
            if dist < self.cfg.reach_radius:
                self.log["stops"].append({"t": round(self._t, 2), "waypoint": int(self.waypoint_idx[0]),
                                          "pos": pos[0].tolist(), "coverage": self.surface.coverage()})
                self.phase, self.phase_t, self.n_post, self.used = "settle", 0.0, 0, []
                self._plan_next()
                self.save_log()
        elif self.phase in ("hold", "settle"):
            self.phase_t += dt
            if self.phase == "hold" and self.phase_t >= self.cfg.hold_s:
                self._plan_next()
            elif self.phase == "settle" and self.phase_t >= self.cfg.settle_s:
                self.phase = "walk"
                self.posture_target[:] = 0.0
                self.waypoint_idx[:] = (self.waypoint_idx + 1) % len(self.waypoints)
                self.turning[:] = True
        self._drive_to(self.waypoints[self.waypoint_idx])
        if self.phase != "walk":
            self.vel_command_b[:] = 0.0

    def _plan_next(self):
        """Pick the next posture at this stop (or decide to leave)."""
        if self.mode == "level" or self.n_post >= self.cfg.max_postures:
            return self._leave()
        if self.mode == "fixed":
            if self.n_post >= len(self.postures):
                return self._leave()
            self.posture_target[:] = self.postures[self.n_post]
            self.n_post += 1
            self.phase, self.phase_t = "hold", 0.0
            return
        g = self._predict_gains()
        level = int(np.flatnonzero((np.abs(self.candidates) < 1e-6).all(1))[0])
        order = [i for i in np.argsort(-g) if i not in self.used]
        best = order[0]
        take = g[best] >= self.cfg.min_gain and g[best] >= (1.0 + self.cfg.margin) * g[level] and best != level
        self.log["decisions"].append({"t": round(self._t, 2), "stop": len(self.log["stops"]) - 1,
                                      "best_deg": np.rad2deg(self.candidates[best]).round(1).tolist(),
                                      "gain_best": int(g[best]), "gain_level": int(g[level]), "taken": bool(take)})
        if not take:
            return self._leave()
        self.used.append(best)
        self.posture_target[:] = torch.as_tensor(self.candidates[best], dtype=torch.float32, device=self.device)
        self.n_post += 1
        self.phase, self.phase_t = "hold", 0.0

    def _leave(self):
        self.posture_target[:] = 0.0
        self.phase, self.phase_t = "settle", 0.0 if self.n_post else self.cfg.settle_s - 0.3

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        out = super().reset(env_ids)
        self.phase = "walk"
        return out

    def __del__(self):
        try:
            self.save_log()
        except Exception:  # noqa: BLE001
            pass

    def save_log(self):
        with open(os.path.join(self.out_dir, "scan_log.json"), "w") as f:
            json.dump(self.log, f)


@configclass
class AdaptiveScanCommandCfg(WaypointScanCommandCfg):
    class_type: type = AdaptiveScanCommand
    gt_npz: str = MISSING
    lidar_name: str = "lidar"
    voxel: float = 0.05
    plan_az_res_deg: float = 1.0
    pitch_grid_deg: tuple[float, ...] = (-30.0, -20.0, -10.0, 0.0, 10.0)
    roll_grid_deg: tuple[float, ...] = (-20.0, -10.0, 0.0, 10.0, 20.0)
    min_gain: int = 40
    """Unseen elements (5 cm voxel faces) a posture must reach to be worth holding."""
    margin: float = 0.3
    max_postures: int = 3
