"""Warehouse scenes: factory-walk Go2 policy + VLP-16 LiDAR.

* ``Go2WarehouseDemoEnvCfg``    -- fixed turn-then-go waypoint loop
* ``Go2WarehouseExploreEnvCfg`` -- autonomous exploration: VLP-16 -> 2D occupancy map -> ARiADNE -> turn-then-go

Set ``GO2_SAVE_SCANS=1`` to also write every VLP-16 scan to ``logs/lidar_scans`` (about 2.7 GB per hour).
"""

import math
import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.ray_caster import RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from .exploration.exploration_command import AriadneExplorationCommandCfg
from .lidar_io import save_lidar_scan
from .locomotion import Go2FactoryWalkEnvCfg_PLAY
from .waypoint_command import WaypointTurnThenGoCommandCfg

SIM_DIR = Path(__file__).resolve().parents[1]
ASSET_DIR = SIM_DIR / "assets"
WAREHOUSE_USD = str(next((ASSET_DIR / "warehouse").rglob("full_warehouse.usd"), ""))

# Velodyne VLP-16: 16 channels, +-15 deg vertical (2 deg steps), 360 deg horizontal, 0.2 deg @ 10 Hz, 100 m range
VLP16_CHANNELS = 16
VLP16_VFOV = (-15.0, 15.0)
VLP16_PATTERN = patterns.LidarPatternCfg(
    channels=VLP16_CHANNELS, vertical_fov_range=VLP16_VFOV, horizontal_fov_range=(-180.0, 180.0), horizontal_res=0.2
)

# Mount from anujjain-dev/unitree-go2-ros2 go2_description/xacro/velodyne.xacro:
#   base_link -> velodyne_base_link: xyz (0.2, 0, 0.08); velodyne_base_link -> velodyne (scan origin): z +0.0377
VLP16_MOUNT_POS = (0.2, 0.0, 0.08)
VLP16_SCAN_HEIGHT = 0.0377
# Nose-up mount tilt [deg] (GO2_LIDAR_TILT). tools/lidar_mount_study.py on a recorded run: 15 deg lifts 3D surface
# coverage 76.9 % -> 89.5 % (ceiling 50 % -> 91 %) with the low obstacles ahead still seen; >= 20 deg loses them.
VLP16_TILT_DEG = float(os.environ.get("GO2_LIDAR_TILT", "0"))
_t = math.radians(VLP16_TILT_DEG)
# the sensor pivots on its base plate: the scan origin (0.0377 m above it) leans back with the tilt
VLP16_SCAN_POS = (VLP16_MOUNT_POS[0] - VLP16_SCAN_HEIGHT * math.sin(_t), 0.0,
                  VLP16_MOUNT_POS[2] + VLP16_SCAN_HEIGHT * math.cos(_t))
VLP16_SCAN_ROT = (0.0, -math.sin(_t / 2), 0.0, math.cos(_t / 2))  # (x, y, z, w): -tilt about y = nose up


def _go2_vlp16_usd(tilt_deg: float) -> str:
    """go2_vlp16.usda (vlp16/vlp16_mount_visual.usda at VLP16_MOUNT_POS), with the mount visual tilted if asked."""
    base = ASSET_DIR / "go2_vlp16.usda"
    if abs(tilt_deg) < 1e-6:
        return str(base)
    out = ASSET_DIR / f"go2_vlp16_tilt{tilt_deg:g}.usda"
    text = base.read_text().replace(
        'uniform token[] xformOpOrder = ["xformOp:translate"]',
        f'float3 xformOp:rotateXYZ = (0, {-tilt_deg:g}, 0)\n'
        '            uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]')
    if not out.exists() or out.read_text() != text:
        out.write_text(text)
    return str(out)


GO2_VLP16_USD = _go2_vlp16_usd(VLP16_TILT_DEG)
VLP16_PAYLOAD_MASS = 0.83 + 0.15  # sensor + bracket [kg]

GT_MAP = str(ASSET_DIR / "maps" / "full_warehouse.npz")
WAREHOUSE_RAYCAST_USD = str(ASSET_DIR / "warehouse" / "warehouse_raycast.usdc")  # tools/merge_usd_meshes.py
WAREHOUSE_INTERIOR = (-26.3, -23.6, 5.3, 30.4)  # inside the outer walls [m], read off the ground-truth map

# loop checked against assets/maps/full_warehouse.npz with 0.45 m obstacle inflation
WAREHOUSE_WAYPOINTS = [(-13.0, 5.0), (-13.0, 27.0), (-18.0, 27.0), (-18.0, 5.0), (-8.0, -10.0), (0.0, 0.0)]


@configclass
class Go2WarehouseDemoEnvCfg(Go2FactoryWalkEnvCfg_PLAY):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 3600.0

        # warehouse as static terrain (its meshes carry their own colliders)
        self.scene.terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="usd", usd_path=WAREHOUSE_USD)
        self.curriculum.terrain_levels = None
        self.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}
        self.events.reset_base.params["velocity_range"] = {}

        # Go2 with the VLP-16 + bracket model composed under its base link (visual only; mass added below)
        self.scene.robot.spawn = self.scene.robot.spawn.replace(usd_path=GO2_VLP16_USD)
        # payload: fixed added mass and the resulting CoM shift (both inside the training randomization)
        base_mass = 6.921
        self.events.add_base_mass.params.update(
            mass_distribution_params=(VLP16_PAYLOAD_MASS, VLP16_PAYLOAD_MASS), operation="add", distribution="uniform"
        )
        total = base_mass + VLP16_PAYLOAD_MASS
        com_x = VLP16_PAYLOAD_MASS * VLP16_MOUNT_POS[0] / total
        com_z = VLP16_PAYLOAD_MASS * (VLP16_MOUNT_POS[2] + 0.03) / total
        self.events.base_com.params["com_range"] = {"x": (com_x, com_x), "y": (0.0, 0.0), "z": (com_z, com_z)}

        # VLP-16 beams from the scan origin of the mounted sensor, cast against the warehouse baked into one
        # invisible triangle mesh (tools/merge_usd_meshes.py); ~500x faster than MultiMeshRayCaster here
        self.scene.raycast_mesh = AssetBaseCfg(
            prim_path="/World/RaycastMesh", spawn=sim_utils.UsdFileCfg(usd_path=WAREHOUSE_RAYCAST_USD)
        )
        self.scene.lidar = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            offset=RayCasterCfg.OffsetCfg(pos=VLP16_SCAN_POS, rot=VLP16_SCAN_ROT),
            ray_alignment="base",
            max_distance=100.0,
            update_period=0.1,
            mesh_prim_paths=["/World/RaycastMesh"],
            pattern_cfg=VLP16_PATTERN,
            debug_vis=True,
        )

        # turn-then-go waypoint follower drives the walking policy
        old = self.commands.base_velocity
        self.commands.base_velocity = WaypointTurnThenGoCommandCfg(
            asset_name="robot",
            resampling_time_range=(1.0e6, 1.0e6),
            heading_command=False,
            rel_standing_envs=0.0,
            rel_heading_envs=0.0,
            ranges=old.ranges,
            debug_vis=True,
            waypoints=WAREHOUSE_WAYPOINTS,
        )

        # record every VLP-16 scan (10 Hz) in the sensor frame, like /velodyne_points (opt-in)
        self.events.save_lidar = None if os.environ.get("GO2_SAVE_SCANS", "0") != "1" else EventTerm(
            func=save_lidar_scan,
            mode="interval",
            interval_range_s=(0.1, 0.1),
            params={
                "sensor_cfg": SceneEntityCfg("lidar"),
                "out_dir": str(SIM_DIR / "logs" / "lidar_scans"),
                "channels": VLP16_CHANNELS,
                "vertical_fov": VLP16_VFOV,
            },
        )

        # camera behind-above the robot (GO2_CAM_EYE="dx,dy,dz" to move it)
        eye = os.environ.get("GO2_CAM_EYE")
        self.viewer.eye = tuple(float(v) for v in eye.split(",")) if eye else (-2.5, -2.5, 2.0)


@configclass
class Go2WarehouseExploreEnvCfg(Go2WarehouseDemoEnvCfg):
    """Autonomous exploration of the warehouse with the RA-L 2024 ARiADNE checkpoint."""

    def __post_init__(self):
        super().__post_init__()
        ranges = self.commands.base_velocity.ranges
        self.commands.base_velocity = AriadneExplorationCommandCfg(
            asset_name="robot",
            resampling_time_range=(1.0e6, 1.0e6),
            heading_command=False,
            rel_standing_envs=0.0,
            rel_heading_envs=0.0,
            ranges=ranges,
            debug_vis=True,
            reach_radius=0.5,
            max_lin_vel=1.0,  # walking policy was trained on vx <= 1.0 m/s, |wz| <= 1.5 rad/s
            max_yaw_rate=1.5,
            # parallel CEM search (logs/parallel/cem_best_2026-10-01.json); vs the previous hand-set values on 48
            # paired random starts: finish 95 s vs 102 s, completion 100 % vs 96 %, map recall 0.988 vs 0.975
            switch_heading=0.57,
            commit_reach=1.8,
            lookahead=1.3,
            clearance=0.7,
            robot_radius=0.37,
            stall_s=5.4,
            turn_threshold=0.44,
            go_threshold=0.15,
            planner_overrides={"THR_NEXT_WAYPOINT": 4.4},
            map_bounds=(-40.0, -50.0, 20.0, 45.0),
            out_dir=str(SIM_DIR / "logs" / "exploration"),
            gt_map=GT_MAP,
            gt_bounds=WAREHOUSE_INTERIOR,
        )
        # 28.8k hit markers per frame dominate render time; the occupancy snapshots show the map instead
        self.scene.lidar.debug_vis = os.environ.get("GO2_LIDAR_VIS", "0") == "1"


@configclass
class Go2WarehouseParallelEnvCfg(Go2WarehouseDemoEnvCfg):
    """N independent Go2 clones exploring the same warehouse (own map / planner each) + reward-driven search.

    Env vars: GO2_PAR_MODE=search|eval|compare, GO2_PAR_GENS (generations), GO2_PAR_EPISODE_S, GO2_PAR_SEED,
    GO2_PAR_STARTS (shared starts per candidate), GO2_PAR_PARAMS (best.json for eval/compare/warm start),
    GO2_FAST_PHYSICS (default 1: plane physics + GT collision proxy).
    """

    def __post_init__(self):
        super().__post_init__()
        from .exploration.parallel_command import ParallelExplorationCommandCfg

        self.scene.num_envs = 16
        self.scene.env_spacing = 0.0  # every clone lives in the same warehouse frame
        self.scene.terrain.env_spacing = 0.0
        self.scene.lidar.debug_vis = False
        if os.environ.get("GO2_FAST_PHYSICS", "1") == "1":
            # Robot feet against the 677k-triangle warehouse collider dominate GPU time. Walk on a plane instead, keep
            # the warehouse for visuals + LiDAR only, and judge collisions against the ground-truth map.
            self.scene.terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="plane", env_spacing=0.0)
            self.scene.warehouse = AssetBaseCfg(
                prim_path="/World/Warehouse",
                spawn=sim_utils.UsdFileCfg(
                    usd_path=WAREHOUSE_USD, collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False)
                ),
            )
            # the LiDAR already casts against the baked /World/RaycastMesh (set in Go2WarehouseDemoEnvCfg)
        ranges = self.commands.base_velocity.ranges
        self.commands.base_velocity = ParallelExplorationCommandCfg(
            asset_name="robot",
            resampling_time_range=(1.0e6, 1.0e6),
            heading_command=False,
            rel_standing_envs=0.0,
            rel_heading_envs=0.0,
            ranges=ranges,
            debug_vis=False,
            reach_radius=0.5,
            max_lin_vel=1.0,
            max_yaw_rate=1.5,
            mode=os.environ.get("GO2_PAR_MODE", "search"),
            generations=int(os.environ.get("GO2_PAR_GENS", "8")),
            episode_s=float(os.environ.get("GO2_PAR_EPISODE_S", "180")),
            seed=int(os.environ.get("GO2_PAR_SEED", "0")),
            starts_per_candidate=int(os.environ.get("GO2_PAR_STARTS", "1")),
            map_bounds=(-40.0, -50.0, 20.0, 45.0),
            gt_map=GT_MAP,
            gt_bounds=WAREHOUSE_INTERIOR,
            out_dir=str(SIM_DIR / "logs" / "parallel"),
        )
        # overview camera, low and oblique from the south wall (the panel's camera buttons switch to follow mode)
        self.viewer.origin_type = "world"
        self.viewer.eye = (-10.5, -29.6, 18.0)
        self.viewer.lookat = (-10.5, -0.6, 0.0)
