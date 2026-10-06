"""Body-posture commands for the Go2: pitch / roll targets the walking policy tracks (POSE-style posture scanning).

Convention: pitch > 0 = nose down, pitch < 0 = nose up (looks up), roll > 0 = right side down (rotation about the
body x axis). A level robot has projected gravity (0, 0, -1); a body tilted by (pitch p, roll r) has
    g_b = (sin p, -cos p sin r, -cos p cos r)
so tracking is a distance between unit vectors, independent of yaw.

* ``BodyPostureCommand``   random targets for training (full range when standing, scaled down while walking)
* ``Go2PostureWalkEnvCfg`` the factory-walk task + posture command / observation / tracking reward
* ``WaypointScanCommand``  demo driver: walk the waypoint loop, stop at each waypoint and step through a list of
                           postures (the posture term reads its target), then walk on
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch

import isaaclab.envs.mdp as mdp
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.utils import configclass

from .locomotion import Go2FactoryWalkEnvCfg, Go2FactoryWalkPPORunnerCfg, _play
from .waypoint_command import WaypointTurnThenGoCommand, WaypointTurnThenGoCommandCfg


def posture_gravity(pitch: torch.Tensor, roll: torch.Tensor) -> torch.Tensor:
    cp = torch.cos(pitch)
    return torch.stack([torch.sin(pitch), -cp * torch.sin(roll), -cp * torch.cos(roll)], dim=-1)


class BodyPostureCommand(CommandTerm):
    cfg: BodyPostureCommandCfg

    def __init__(self, cfg: BodyPostureCommandCfg, env):
        super().__init__(cfg, env)
        self._env = env
        self.robot = env.scene[cfg.asset_name]
        self.target = torch.zeros(self.num_envs, 2, device=self.device)  # sampled (pitch, roll)
        self.effective = torch.zeros(self.num_envs, 2, device=self.device)  # what the policy must track now
        self.metrics["error_deg"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return f"BodyPostureCommand: pitch {self.cfg.pitch_range}, roll {self.cfg.roll_range} rad"

    @property
    def command(self) -> torch.Tensor:
        return self.effective

    def _resample_command(self, env_ids: Sequence[int]):
        n = len(env_ids)
        p = torch.empty(n, device=self.device).uniform_(*self.cfg.pitch_range)
        r = torch.empty(n, device=self.device).uniform_(*self.cfg.roll_range)
        level = torch.rand(n, device=self.device) < self.cfg.rel_level_envs
        self.target[env_ids, 0] = torch.where(level, torch.zeros_like(p), p)
        self.target[env_ids, 1] = torch.where(level, torch.zeros_like(r), r)

    def _update_command(self):
        scale = torch.ones(self.num_envs, device=self.device)
        if self.cfg.velocity_command:
            v = self._env.command_manager.get_command(self.cfg.velocity_command)
            speed = torch.linalg.norm(v[:, :2], dim=1) + 0.3 * v[:, 2].abs()
            scale = torch.clamp(1.0 - speed / self.cfg.full_tilt_speed, self.cfg.min_scale_moving, 1.0)
        self.effective = self.target * scale[:, None]

    def _update_metrics(self):
        g = self.robot.data.projected_gravity_b.torch
        gt = posture_gravity(self.effective[:, 0], self.effective[:, 1])
        cos = torch.clamp((g * gt).sum(-1), -1.0, 1.0)
        self.metrics["error_deg"] = torch.rad2deg(torch.acos(cos))


@configclass
class BodyPostureCommandCfg(CommandTermCfg):
    class_type: type = BodyPostureCommand
    asset_name: str = "robot"
    pitch_range: tuple[float, float] = (-0.45, 0.45)
    roll_range: tuple[float, float] = (-0.3, 0.3)
    rel_level_envs: float = 0.25
    """Fraction of commands that ask for the level posture."""
    velocity_command: str | None = "base_velocity"
    full_tilt_speed: float = 0.6
    """Commanded speed [m/s] (+0.3 x |yaw rate|) at which the posture target is scaled down to ``min_scale_moving``."""
    min_scale_moving: float = 0.3


def track_posture_exp(env, command_name: str, std: float, asset_name: str = "robot") -> torch.Tensor:
    g = env.scene[asset_name].data.projected_gravity_b.torch
    cmd = env.command_manager.get_command(command_name)
    err = ((g - posture_gravity(cmd[:, 0], cmd[:, 1])) ** 2).sum(-1)
    return torch.exp(-err / std**2)


@configclass
class Go2PostureWalkEnvCfg(Go2FactoryWalkEnvCfg):
    """Factory walk + body pitch / roll tracking (more standing so the posture is learned at rest)."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.rel_standing_envs = 0.35
        self.commands.base_posture = BodyPostureCommandCfg(resampling_time_range=(2.5, 5.0))
        self.observations.policy.base_posture = ObsTerm(func=mdp.generated_commands,
                                                        params={"command_name": "base_posture"})
        rew = self.rewards
        rew.flat_orientation_l2 = None  # replaced by posture tracking
        rew.stand_still = None  # standing still now means holding a commanded tilt, not the default joints
        rew.hip_deviation.weight = -0.1  # roll needs hip abduction
        rew.base_height.weight = -5.0
        rew.track_posture = RewTerm(func=track_posture_exp, weight=2.0, params={"command_name": "base_posture",
                                                                               "std": 0.2})


@configclass
class Go2PostureWalkEnvCfg_PLAY(Go2PostureWalkEnvCfg):
    """GO2_SHOW_LAYOUT=grid: robots standing on a flat floor 2.5 m apart, new random posture every 2.5 s, fixed
    oblique camera (GO2_CAM_EYE) -- for recording what the posture policy learned."""

    def __post_init__(self):
        super().__post_init__()
        _play(self)
        import os

        if os.environ.get("GO2_SHOW_LAYOUT") == "grid":
            self.scene.terrain.terrain_type = "plane"
            self.scene.terrain.terrain_generator = None
            self.curriculum.terrain_levels = None
            self.scene.env_spacing = 2.5
            v = self.commands.base_velocity
            v.rel_standing_envs = 1.0  # stand and tilt
            self.commands.base_posture.rel_level_envs = 0.0
            self.commands.base_posture.resampling_time_range = (2.5, 2.5)
            eye = os.environ.get("GO2_CAM_EYE")
            self.viewer.origin_type = "world"
            self.viewer.eye = tuple(float(x) for x in eye.split(",")) if eye else (-4.5, -6.5, 3.0)
            self.viewer.lookat = (0.0, 0.0, 0.25)


@configclass
class Go2PostureWalkPPORunnerCfg(Go2FactoryWalkPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "go2_posture_walk"
        self.max_iterations = 2000


# -- demo: stop at each waypoint and scan through a list of postures ----------------------------------------
class WaypointScanCommand(WaypointTurnThenGoCommand):
    cfg: WaypointScanCommandCfg

    def __init__(self, cfg: WaypointScanCommandCfg, env):
        super().__init__(cfg, env)
        self.postures = torch.tensor(cfg.scan_postures, dtype=torch.float32, device=self.device)  # [P, 2]
        self.scan_t = torch.full((self.num_envs,), -1.0, device=self.device)  # <0: walking
        self.posture_target = torch.zeros(self.num_envs, 2, device=self.device)
        mask = cfg.scan_mask if cfg.scan_mask is not None else [True] * len(cfg.waypoints)
        self.scan_mask = torch.tensor(mask, dtype=torch.bool, device=self.device)
        self._dt = env.step_dt

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        ids = slice(None) if env_ids is None else env_ids
        self.scan_t[ids] = -1.0
        self.posture_target[ids] = 0.0
        return super().reset(env_ids)

    def _update_command(self):
        pos = self.robot.data.root_pos_w.torch[:, :2] - self._env_origins[:, :2]
        dist = torch.linalg.norm(self.waypoints[self.waypoint_idx] - pos, dim=1)
        walking = self.scan_t < 0
        reached = walking & (dist < self.cfg.reach_radius)
        scan_here = self.scan_mask[self.waypoint_idx]
        # waypoints without a scan: just move on
        self.waypoint_idx = torch.where(reached & ~scan_here, (self.waypoint_idx + 1) % len(self.waypoints),
                                        self.waypoint_idx)
        self.turning |= reached & ~scan_here
        arrived = reached & scan_here
        self.scan_t = torch.where(arrived, torch.zeros_like(self.scan_t), self.scan_t)
        scanning = self.scan_t >= 0
        hold = self.cfg.hold_s
        total = hold * len(self.postures) + self.cfg.settle_s
        k = torch.clamp((self.scan_t / hold).long(), 0, len(self.postures) - 1)
        in_tail = self.scan_t >= hold * len(self.postures)
        tgt = torch.where(in_tail[:, None], torch.zeros_like(self.posture_target), self.postures[k])
        self.posture_target = torch.where(scanning[:, None], tgt, torch.zeros_like(self.posture_target))
        done = scanning & (self.scan_t >= total)
        self.scan_t = torch.where(scanning, self.scan_t + self._dt, self.scan_t)
        self.scan_t = torch.where(done, torch.full_like(self.scan_t, -1.0), self.scan_t)
        self.waypoint_idx = torch.where(done, (self.waypoint_idx + 1) % len(self.waypoints), self.waypoint_idx)
        self.turning |= done
        self._drive_to(self.waypoints[self.waypoint_idx])
        still = self.scan_t >= 0
        self.vel_command_b[still] = 0.0


@configclass
class WaypointScanCommandCfg(WaypointTurnThenGoCommandCfg):
    class_type: type = WaypointScanCommand
    scan_postures: list[tuple[float, float]] = MISSING
    """(pitch, roll) [rad] held one after another at every waypoint."""
    hold_s: float = 1.5
    settle_s: float = 1.0
    """Seconds back at the level posture before walking on."""
    scan_mask: list[bool] | None = None
    """Per waypoint: stop and scan there (default: everywhere)."""


class ScriptedPostureCommand(BodyPostureCommand):
    """Posture target taken from the WaypointScanCommand (demo); no random sampling."""

    def _resample_command(self, env_ids: Sequence[int]):
        pass

    def _update_command(self):
        src = self._env.command_manager.get_term(self.cfg.source_command)
        self.effective = src.posture_target.clone()


@configclass
class ScriptedPostureCommandCfg(BodyPostureCommandCfg):
    class_type: type = ScriptedPostureCommand
    source_command: str = "base_velocity"
