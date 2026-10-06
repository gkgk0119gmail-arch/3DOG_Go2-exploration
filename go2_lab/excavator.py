"""Construction-site scan demo: the warehouse demo (factory-walk Go2 + VLP-16 + turn-then-go waypoint loop) around
an excavator converted by tools/urdf_to_scene.py (default assets/excavator/ix35e: 3.5 t mini excavator).

GO2_MACHINE=<prefix> picks the scene, GO2_CAM_WORLD="ex,ey,ez,lx,ly,lz" fixes the camera in the world (default:
behind the robot, GO2_CAM_EYE), GO2_LIDAR_TILT tilts the VLP-16 mount (factory.py).
"""

import json
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from .factory import SIM_DIR, Go2WarehouseDemoEnvCfg

MACHINE_PREFIX = os.environ.get("GO2_MACHINE", str(SIM_DIR / "assets" / "excavator" / "ix35e"))


@configclass
class Go2ExcavatorDemoEnvCfg(Go2WarehouseDemoEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        with open(MACHINE_PREFIX + "_scene.json") as f:
            scene = json.load(f)
        self.scene.terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="usd",
                                                usd_path=MACHINE_PREFIX + "_terrain.usda")
        self.scene.raycast_mesh = AssetBaseCfg(prim_path="/World/RaycastMesh",
                                               spawn=sim_utils.UsdFileCfg(usd_path=MACHINE_PREFIX + "_raycast.usda",
                                                                          visible=False))
        self.commands.base_velocity.waypoints = [tuple(p) for p in scene["waypoints"]]
        cam = os.environ.get("GO2_CAM_WORLD")
        if cam:
            v = [float(x) for x in cam.split(",")]
            self.viewer.origin_type = "world"
            self.viewer.eye, self.viewer.lookat = tuple(v[:3]), tuple(v[3:6])


@configclass
class Go2ExcavatorPostureDemoEnvCfg(Go2ExcavatorDemoEnvCfg):
    """Posture-tracking gait (posture.py). Loop of corners + side midpoints around the machine (counter-clockwise,
    machine on the robot's left); at each midpoint the robot stops and holds GO2_SCAN_POSTURES
    ("p,r;p,r;..." in degrees, default roll toward the machine, nose up, both) before walking on."""

    def __post_init__(self):
        super().__post_init__()
        import math

        import isaaclab.envs.mdp as mdp
        from isaaclab.managers import ObservationTermCfg as ObsTerm

        from .posture import ScriptedPostureCommandCfg, WaypointScanCommandCfg

        with open(MACHINE_PREFIX + "_scene.json") as f:
            scene = json.load(f)
        lo, hi, a = scene["bbox_min"], scene["bbox_max"], float(os.environ.get("GO2_STANDOFF", "2.0"))
        x0, x1, y0, y1 = lo[0] - a, hi[0] + a, lo[1] - a, hi[1] + a
        loop = [(x0, y0), ((x0 + x1) / 2, y0), (x1, y0), (x1, (y0 + y1) / 2), (x1, y1), ((x0 + x1) / 2, y1),
                (x0, y1), (x0, (y0 + y1) / 2)]
        spec = os.environ.get("GO2_SCAN_POSTURES", "0,17;-22,0;-18,14")
        postures = [tuple(math.radians(float(v)) for v in pr.split(",")) for pr in spec.split(";")]
        old = self.commands.base_velocity
        self.commands.base_velocity = WaypointScanCommandCfg(
            asset_name=old.asset_name, resampling_time_range=old.resampling_time_range, heading_command=False,
            rel_standing_envs=0.0, rel_heading_envs=0.0, ranges=old.ranges, debug_vis=old.debug_vis,
            waypoints=loop, scan_mask=[i % 2 == 1 for i in range(len(loop))], scan_postures=postures,
            hold_s=float(os.environ.get("GO2_SCAN_HOLD", "2.0")))
        self.commands.base_posture = ScriptedPostureCommandCfg(resampling_time_range=(1.0e6, 1.0e6))
        self.observations.policy.base_posture = ObsTerm(func=mdp.generated_commands,
                                                        params={"command_name": "base_posture"})
