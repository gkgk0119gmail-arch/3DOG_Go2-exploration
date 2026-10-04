"""Turn-then-go waypoint follower, exposed as an Isaac Lab velocity command term.

The walking policy only sees ``(vx, vy, wz)``. This term turns in place until the robot faces the next
waypoint, then walks forward while correcting heading -- the same way the real Go2 is driven by a
high-level planner (ARiADNE waypoints) through its sport-mode velocity interface.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils import configclass
from isaaclab.utils.math import wrap_to_pi


class WaypointTurnThenGoCommand(UniformVelocityCommand):
    cfg: WaypointTurnThenGoCommandCfg

    def __init__(self, cfg: WaypointTurnThenGoCommandCfg, env):
        super().__init__(cfg, env)
        self._env_origins = env.scene.env_origins
        self.waypoints = torch.tensor(cfg.waypoints, dtype=torch.float32, device=self.device)  # [N, 2], env frame
        self.waypoint_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.turning = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        ids = slice(None) if env_ids is None else env_ids
        self.waypoint_idx[ids] = 0
        self.turning[ids] = True
        return super().reset(env_ids)

    def _resample_command(self, env_ids: Sequence[int]):
        # commands come from the waypoint controller, never from random sampling
        self.is_standing_env[env_ids] = False

    def _update_command(self):
        pos = self.robot.data.root_pos_w.torch[:, :2] - self._env_origins[:, :2]

        # advance to the next waypoint once inside the reach radius
        dist = torch.linalg.norm(self.waypoints[self.waypoint_idx] - pos, dim=1)
        reached = dist < self.cfg.reach_radius
        self.waypoint_idx = torch.where(reached, (self.waypoint_idx + 1) % len(self.waypoints), self.waypoint_idx)
        self.turning |= reached
        self._drive_to(self.waypoints[self.waypoint_idx])

    def _drive_to(self, target: torch.Tensor, speed_dist: torch.Tensor | None = None):
        """Turn-then-go velocity command towards ``target`` [E, 2] (env frame).

        ``speed_dist`` [E] sets the forward speed (e.g. remaining path length); defaults to the distance to target.
        """
        cfg = self.cfg
        pos = self.robot.data.root_pos_w.torch[:, :2] - self._env_origins[:, :2]
        yaw = self.robot.data.heading_w.torch
        delta = target - pos
        dist = torch.linalg.norm(delta, dim=1) if speed_dist is None else speed_dist
        heading_err = wrap_to_pi(torch.atan2(delta[:, 1], delta[:, 0]) - yaw)

        # hysteresis: start turning above turn_threshold, resume walking below go_threshold
        # per-robot thresholds when set (parallel parameter search), else the cfg values
        turn_thr = getattr(self, "turn_threshold_env", cfg.turn_threshold)
        go_thr = getattr(self, "go_threshold_env", cfg.go_threshold)
        self.turning = torch.where(self.turning, heading_err.abs() > go_thr, heading_err.abs() > turn_thr)

        wz = torch.clamp(cfg.yaw_gain * heading_err, -cfg.max_yaw_rate, cfg.max_yaw_rate)
        vx = torch.clamp(cfg.lin_gain * dist, cfg.min_lin_vel, cfg.max_lin_vel)
        self.vel_command_b[:, 0] = torch.where(self.turning, torch.zeros_like(vx), vx)
        self.vel_command_b[:, 1] = 0.0
        self.vel_command_b[:, 2] = wz


@configclass
class WaypointTurnThenGoCommandCfg(UniformVelocityCommandCfg):
    class_type: type = WaypointTurnThenGoCommand

    waypoints: list[tuple[float, float]] = MISSING
    """Waypoints (x, y) [m] in the environment frame, visited in a loop."""

    reach_radius: float = 0.4
    """Distance [m] at which a waypoint counts as reached."""

    turn_threshold: float = 0.35
    """Heading error [rad] above which the robot stops and turns in place."""

    go_threshold: float = 0.12
    """Heading error [rad] below which the robot resumes walking forward."""

    yaw_gain: float = 1.5
    max_yaw_rate: float = 1.0
    """Maximum yaw-rate command [rad/s]."""

    lin_gain: float = 0.8
    min_lin_vel: float = 0.3
    max_lin_vel: float = 0.8
    """Forward velocity command limits [m/s]."""
