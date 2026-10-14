"""Isaac Lab task for the POSE-inspection experiments: N Go2 robots (posture-walk policy + VLP-16 visual) in one
scene, one per seed, each driven by its own InspectionAgent through the shared Session.

Physics: robots walk on a plane (``env_spacing = 0``: all clones share the scene frame, collisions between clones
are filtered); the scene geometry is used by our warp VLP-16 and the GT traversability grid (time spent within
``collision_clearance`` of GT obstacles is reported). ``GO2_PI_VISUAL=1`` also loads the textured scene USD
(no colliders) for the GUI, videos and the stage-4 cameras.

Run configuration through the environment (scripts/pose_inspection/run_isaac.py sets these):
  GO2_PI_EXP    experiment YAML (configs/pose_inspection/...)     GO2_PI_SCENE  built scene name
  GO2_PI_SET    JSON dict of dotted overrides                       GO2_PI_SEEDS  comma-separated seeds
  GO2_PI_OUT    output directory                                     GO2_PI_TAG    run label
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence

import numpy as np
import torch

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import matrix_from_quat, wrap_to_pi

from ..factory import Go2WarehouseDemoEnvCfg, _go2_vlp16_usd
from ..posture import ScriptedPostureCommandCfg
from ..waypoint_command import WaypointTurnThenGoCommand, WaypointTurnThenGoCommandCfg
from .config import load_config
from .navgrid import NavGrid
from .scene_builder import load_scene


def run_settings():
    exp = load_config(os.environ.get("GO2_PI_EXP", "exp/pose.yaml"), json.loads(os.environ.get("GO2_PI_SET", "{}")))
    scene = os.environ.get("GO2_PI_SCENE", "B_construction")
    seeds = [int(s) for s in os.environ.get("GO2_PI_SEEDS", "").split(",") if s.strip()] or list(
        range(exp.experiment.seed_base, exp.experiment.seed_base + exp.experiment.n_seeds))
    return exp, scene, seeds


def needs_cameras(exp) -> bool:
    """Ring cameras: Grounded-SAM 2 perception (stage 4) or VLM photos (stage 5 pruning, stage 6 template hints)."""
    return exp.perception.mode == "gsam2" or uses_vlm(exp)


def uses_vlm(exp) -> bool:
    return exp.planner.semantic.prune == "vlm" or exp.priors.method == "template_vlm"


class PoseInspectionCommand(WaypointTurnThenGoCommand):
    cfg: PoseInspectionCommandCfg

    def __init__(self, cfg: PoseInspectionCommandCfg, env):
        super().__init__(cfg, env)
        from .session import Session  # noqa: PLC0415

        exp, scene, seeds = run_settings()
        assert len(seeds) == self.num_envs, f"{len(seeds)} seeds for {self.num_envs} envs"
        self.exp = exp
        self.meta = load_scene(f"assets/pose_inspection/{scene}")
        out = os.environ.get("GO2_PI_OUT", os.path.join(exp.experiment.results_dir, "isaac", scene, "run"))
        self.perc, self.vlm, self.pruners, self.cams = None, None, {}, None
        if exp.perception.mode == "gsam2":
            from .perception import Gsam2Perception  # noqa: PLC0415

            self.perc = Gsam2Perception(exp, self.meta, len(seeds), str(self.device))
        if needs_cameras(exp):
            self.cams = [env.scene[f"pi_cam{k}"] for k in range(exp.perception.cameras.n)]
            self._last_frame = torch.full((len(seeds),), -1, dtype=torch.long, device=self.device)
        factory = None
        if uses_vlm(exp):
            from .vlm import VlmClient  # noqa: PLC0415
            from .vlm_pruner import VlmPruner  # noqa: PLC0415

            self.vlm = VlmClient(exp.planner.vlm, str(self.device))
            objs = self.meta["objects"]
            for e in range(len(seeds)):
                if self.perc is None:
                    cls_fn = (lambda o: objs[o]["class"])
                else:
                    cls_fn = (lambda o, e=e: self.perc.classes[self.perc.assoc[e].boxes[o]["cls"]]
                              if o < len(self.perc.assoc[e].boxes) and self.perc.assoc[e].boxes[o] is not None
                              else "machine")
                self.pruners[e] = VlmPruner(exp, self.vlm, e, cls_fn,
                                            log_images=exp.planner.vlm.log_consults if e == 0 else 0,
                                            image_dir=os.path.join(out, "vlm_inputs"))
            if exp.planner.semantic.prune == "vlm":
                factory = self.pruners.__getitem__   # else the pruners only keep photos (template hints)
        self.hints: dict[tuple[int, int], dict] = {}
        hint_fn = self._vlm_hint if exp.priors.method == "template_vlm" else None
        self.session = Session(exp, self.meta, seeds, out, device=str(self.device), perception=self.perc,
                               pruner_factory=factory, tag=os.environ.get("GO2_PI_TAG", ""), vlm_hint_fn=hint_fn)
        rs = self.meta["robot_start"]
        self.nav_gt = NavGrid(os.path.join(self.meta["dir"], "grid.npz"), exp.robot.radius).restrict_to_region(rs["region"])
        self.starts = np.array([self.nav_gt.sample_starts(rs["region"], 1, s, rs["min_clearance"])[0] for s in seeds])
        self.posture_target = torch.zeros(self.num_envs, 2, device=self.device)
        self._dt = env.step_dt
        self.t = 0.0
        self._k = 0
        self._scan_every = max(1, int(round(1.0 / (exp.lidar.rate_hz * self._dt))))
        self._placed = False
        self.finished = False
        self.collision_s = np.zeros(self.num_envs)
        self.fell = np.zeros(self.num_envs, bool)

    # ---------------------------------------------------------------------------------------------------------------
    def _place(self):
        """Teleport every robot to its seed's start pose (once, after the first reset)."""
        E = self.num_envs
        st = torch.as_tensor(self.starts, dtype=torch.float32, device=self.device)
        z = self.robot.data.root_pos_w.torch[:, 2].clone()
        half = st[:, 2] / 2
        quat = torch.stack([torch.zeros(E, device=self.device), torch.zeros(E, device=self.device), half.sin(),
                            half.cos()], -1)  # (x, y, z, w)
        pose = torch.cat([torch.stack([st[:, 0], st[:, 1], z], -1), quat], -1)
        ids = torch.arange(E, device=self.device)
        self.robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(E, 6, device=self.device), env_ids=ids)
        self._placed = True

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        out = super().reset(env_ids)
        self.posture_target[:] = 0.0
        return out

    def _measured(self):
        pos = self.robot.data.root_pos_w.torch
        yaw = self.robot.data.heading_w.torch
        g = self.robot.data.projected_gravity_b.torch
        pitch = torch.asin(g[:, 0].clamp(-1, 1))
        roll = torch.atan2(-g[:, 1], -g[:, 2])
        poses = torch.stack([pos[:, 0], pos[:, 1], yaw], -1).cpu().numpy()
        return pos, poses, torch.stack([pitch, roll], -1).cpu().numpy()

    def _camera_frames(self) -> bool:
        """Hand new camera frames (all ring cameras of a robot) to the perception hook; True if any."""
        for cam in self.cams:  # sensors refresh lazily: reading .data runs the pending update (and bumps .frame)
            _ = cam.data
        frame = self.cams[0].frame.torch.long()
        new = torch.nonzero(frame != self._last_frame).flatten().tolist()
        if not new:
            return False
        self._last_frame = frame.clone()
        data = []
        for cam in self.cams:
            d = cam.data
            Rw = matrix_from_quat(d.quat_w_ros.torch)          # camera (ROS: z fwd, x right, y down) -> world
            data.append((d.output["rgb"].torch, d.output["distance_to_image_plane"].torch, d.intrinsic_matrices.torch,
                         Rw, d.pos_w.torch))
        yaw = self.robot.data.heading_w.torch.cpu().numpy()
        xy = self.robot.data.root_pos_w.torch[:, :2].cpu().numpy()
        for e in new:
            if self.session.term.done[e]:
                continue
            cams = []
            for rgb, depth, K, Rw, p in data:
                R_wc = Rw[e].T
                cams.append({"rgb": rgb[e, ..., :3].cpu().numpy(), "depth": depth[e, ..., 0], "K": K[e],
                             "R_wc": R_wc, "t_wc": -R_wc @ p[e]})
            if self.perc is not None:
                self.perc.set_frames(e, cams)
            if e in self.pruners:
                self._store_vlm_photos(e, cams, xy[e], float(yaw[e]))
        return bool(new) and self.perc is not None

    def _store_vlm_photos(self, e, cams, xy, yaw):
        """Keep, per (object, sector), the latest photo of the camera that faces the object best."""
        from .vlm_pruner import best_camera, sector_of  # noqa: PLC0415

        agent = self.session.agents[e]
        n = len(cams)
        cam_yaws = [yaw + 2 * math.pi * k / n for k in range(n)]
        for o, m in agent.models.items():
            c = m["centroid"]
            d = float(np.hypot(c[0] - xy[0], c[1] - xy[1]))
            v = self.exp.planner.vlm
            if not v.photo_min_range <= d <= v.photo_range:
                continue
            key = (o, sector_of(c, xy, self.exp.planner.semantic.n_sectors))
            old = self.pruners[e].frames.get(key)
            if old is not None and old.get("dist", 0.0) > d + v.photo_keep_margin:
                continue  # keep the farther photo of this sector: it shows the whole machine
            k = best_camera(cam_yaws, math.atan2(c[1] - xy[1], c[0] - xy[0]))
            f = dict(cams[k])
            f["depth"] = f["depth"].cpu().numpy()
            f["t"] = self.t
            f["cam_yaw"] = cam_yaws[k]
            f["dist"] = d
            self.pruners[e].store_frame(*key, f)

    HINT_PROMPT = ("This robot camera photo shows a {cls}. Where does its boom / arm (or its tallest part) point in "
                   "the image? Reply with JSON only: {{\"direction\": \"left|right|toward camera|away from camera|"
                   "none\", \"height_m\": <approximate height of the machine in metres>}}")

    def _vlm_hint(self, agent, o, m):
        """Template hint from the first photo of an object (stage 6 template_vlm): boom heading + height."""
        key = (agent.e, o)
        h = self.hints.get(key)
        if h is not None and h.get("result") is not None:
            return h["result"]
        if h is not None:
            r = h["req"]
            if not r.ready(self.t):
                return None
            res = None
            if r.error is None and isinstance(r.parsed, dict):
                d = str(r.parsed.get("direction", "none")).lower()
                off = {"left": math.pi / 2, "right": -math.pi / 2, "toward camera": math.pi, "away from camera": 0.0}
                hm = r.parsed.get("height_m")
                res = {"upper_yaw": h["cam_yaw"] + off[d] if d in off else None,
                       "height": float(hm) if isinstance(hm, (int, float)) else None}
            h["result"] = res or {"upper_yaw": None, "height": None}
            return h["result"]
        frames = [f for (ob, _), f in self.pruners[agent.e].frames.items() if ob == o]
        if not frames:
            return None
        f = min(frames, key=lambda x: x["t"])
        from PIL import Image  # noqa: PLC0415

        cls = self.meta["objects"][o]["class"] if self.perc is None else "machine"
        req = self.vlm.submit([Image.fromarray(f["rgb"])], self.HINT_PROMPT.format(cls=cls.replace("_", " ")), self.t,
                              meta={"env": agent.e, "obj": int(o), "kind": "template_hint"})
        self.hints[key] = {"req": req, "cam_yaw": f["cam_yaw"], "result": None}
        return None

    def _update_command(self):
        if not self._placed:
            self._place()
            self.vel_command_b[:] = 0.0
            return
        s = self.session
        pos, poses, postures = self._measured()
        # warm-up: let the gait settle after the teleport before the clock starts
        if self._k * self._dt < self.cfg.warmup_s:
            self._k += 1
            self.vel_command_b[:] = 0.0
            return
        fresh = self._camera_frames() if self.cams is not None else False
        if self._k % self._scan_every == 0 or fresh:
            R = matrix_from_quat(self.robot.data.root_quat_w.torch)
            s.scan(self.t, pos, R)
        cmds = s.control(self.t, poses, postures)
        targets = torch.as_tensor(poses[:, :2], dtype=torch.float32, device=self.device).clone()
        mode = []
        for e, c in enumerate(cmds):
            if c is None:
                mode.append("hold")
                self.posture_target[e] = 0.0
                continue
            mode.append(c["mode"])
            self.posture_target[e] = torch.as_tensor(c.get("posture", (0.0, 0.0)), device=self.device)
            if c["mode"] == "move":
                targets[e] = torch.as_tensor(c["waypoint"], dtype=torch.float32, device=self.device)
        self._drive_to(targets)
        yaw = self.robot.data.heading_w.torch
        for e, c in enumerate(cmds):
            if mode[e] == "turn":
                err = wrap_to_pi(torch.tensor(c["yaw"], device=self.device) - yaw[e])
                self.vel_command_b[e, 0] = 0.0
                self.vel_command_b[e, 1] = 0.0
                self.vel_command_b[e, 2] = torch.clamp(self.cfg.yaw_gain * err, -self.cfg.max_yaw_rate,
                                                       self.cfg.max_yaw_rate)
                self.turning[e] = True
            elif mode[e] == "hold":
                self.vel_command_b[e] = 0.0
                self.turning[e] = True
        # GT safety bookkeeping (no physical colliders for the scene objects)
        clear = self.nav_gt.clearance(poses[:, :2])
        active = ~s.term.done
        self.collision_s += np.where(active & (clear < self.cfg.collision_clearance), self._dt, 0.0)
        self.fell |= pos[:, 2].cpu().numpy() < self.cfg.fall_height
        s.log(self.t, poses)
        self.t += self._dt
        self._k += 1
        if s.term.done.all() and not self.finished:
            self.finished = True
            extra = {"engine": "isaac", "collision_s": self.collision_s.round(2).tolist(), "fell": self.fell.tolist()}
            if self.perc is not None:
                extra["perception"] = self.perc.summary()
            if self.vlm is not None:
                extra["vlm"] = {"client": self.vlm.summary(),
                                "per_robot": {str(e): p.summary() for e, p in self.pruners.items()}}
                os.makedirs(s.out_dir, exist_ok=True)
                with open(os.path.join(s.out_dir, "vlm_calls.json"), "w") as f:
                    json.dump(self.vlm.log, f, indent=1, default=str)
            s.save(extra)


@configclass
class PoseInspectionCommandCfg(WaypointTurnThenGoCommandCfg):
    class_type: type = PoseInspectionCommand
    waypoints: list = [(0.0, 0.0)]
    warmup_s: float = 2.0
    collision_clearance: float = 0.15
    """Base centre closer than this to a GT obstacle cell counts as a collision [m]."""
    fall_height: float = 0.15


@configclass
class Go2PoseInspectionEnvCfg(Go2WarehouseDemoEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        exp, scene, seeds = run_settings()
        meta = load_scene(f"assets/pose_inspection/{scene}")
        n = len(seeds)
        self.scene.num_envs = n
        self.scene.env_spacing = 0.0
        self.episode_length_s = exp.termination.time_limit_s + 600.0
        self.terminations.base_contact = None
        # walk on a plane; the scene is only seen through our ray caster (and optionally rendered)
        self.scene.terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", env_spacing=0.0)
        self.scene.raycast_mesh = None
        self.scene.lidar = None
        self.events.save_lidar = None
        gsam2 = needs_cameras(exp)
        if os.environ.get("GO2_PI_VISUAL", "0") == "1" or gsam2:
            self.scene.visual = AssetBaseCfg(
                prim_path="/World/Visual",
                spawn=sim_utils.UsdFileCfg(usd_path=os.path.join(meta["dir"], meta["files"]["scene"]),
                                           collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False)))
        if gsam2:
            self._add_cameras(exp)
        if os.environ.get("GO2_PI_VIDEO", "0") == "1":  # chase camera for recordings (moved by run_isaac.py)
            from isaaclab.sensors import CameraCfg  # noqa: PLC0415

            w, h = (int(v) for v in os.environ.get("GO2_PI_VIDEO_RES", "960,540").split(","))
            self.scene.pi_video = CameraCfg(prim_path="{ENV_REGEX_NS}/VideoCam", update_period=0.0, height=h, width=w,
                                            data_types=["rgb"], update_latest_camera_pose=True,
                                            spawn=sim_utils.PinholeCameraCfg(focal_length=14.0,
                                                                             horizontal_aperture=20.955,
                                                                             clipping_range=(0.1, 200.0)))
        # VLP-16 visual on the robot at the experiment's mount tilt
        self.scene.robot.spawn = self.scene.robot.spawn.replace(usd_path=_go2_vlp16_usd(exp.lidar.mount_tilt_deg))
        old = self.commands.base_velocity
        self.commands.base_velocity = PoseInspectionCommandCfg(
            asset_name="robot", resampling_time_range=(1.0e6, 1.0e6), heading_command=False, rel_standing_envs=0.0,
            rel_heading_envs=0.0, ranges=old.ranges, debug_vis=False)
        self.commands.base_posture = ScriptedPostureCommandCfg(resampling_time_range=(1.0e6, 1.0e6))
        self.observations.policy.base_posture = ObsTerm(func=mdp.generated_commands,
                                                        params={"command_name": "base_posture"})
        self.viewer.origin_type = "world"
        b = meta["bounds"]
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        self.viewer.eye = (cx, b[1] - 8.0, 22.0)
        self.viewer.lookat = (cx, cy, 0.0)
        if os.environ.get("GO2_PI_VIDEO", "0") == "1":
            self.sim.render_interval = self.decimation
        elif not gsam2:
            self.sim.render_interval = 50
        self.seed = seeds[0]

    def _add_cameras(self, exp):
        """Ring of pinhole cameras on the base (stage 4), rendered once per perception period."""
        from scipy.spatial.transform import Rotation  # noqa: PLC0415

        from isaaclab.sensors import CameraCfg  # noqa: PLC0415

        c = exp.perception.cameras
        f = 20.955 / 2 / math.tan(math.radians(c.hfov_deg) / 2)
        for k in range(c.n):
            yaw = 360.0 * k / c.n
            q = Rotation.from_euler("ZY", [yaw, -c.pitch_up_deg], degrees=True).as_quat()  # (x, y, z, w)
            setattr(self.scene, f"pi_cam{k}", CameraCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/base/pi_cam{k}", update_period=exp.perception.gsam2.period_s,
                height=c.height, width=c.width, data_types=["rgb", "distance_to_image_plane"],
                update_latest_camera_pose=True,
                offset=CameraCfg.OffsetCfg(pos=tuple(c.pos), rot=tuple(float(v) for v in q), convention="world"),
                spawn=sim_utils.PinholeCameraCfg(focal_length=f, horizontal_aperture=20.955,
                                                 clipping_range=(0.1, 200.0))))
        self.sim.render_interval = max(1, int(round(exp.perception.gsam2.period_s / self.sim.dt)))
