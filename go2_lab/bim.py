"""BIM scenes (step 3): the warehouse demo (factory-walk Go2 + VLP-16 + turn-then-go waypoints) in a building
converted by tools/ifc_to_isaac.py. GO2_BIM=<prefix> picks the scene (default assets/bim_ifc/dental_clinic)."""

import json
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from .factory import SIM_DIR, Go2WarehouseDemoEnvCfg

BIM_PREFIX = os.environ.get("GO2_BIM", str(SIM_DIR / "assets" / "bim_ifc" / "dental_clinic"))


@configclass
class Go2BimDemoEnvCfg(Go2WarehouseDemoEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        with open(BIM_PREFIX + "_scene.json") as f:
            scene = json.load(f)
        self.scene.terrain = TerrainImporterCfg(prim_path="/World/ground", terrain_type="usd",
                                                usd_path=BIM_PREFIX + "_terrain.usda")
        self.scene.raycast_mesh = AssetBaseCfg(prim_path="/World/RaycastMesh",
                                               spawn=sim_utils.UsdFileCfg(usd_path=BIM_PREFIX + "_raycast.usda",
                                                                          visible=False))
        self.commands.base_velocity.waypoints = [tuple(p) for p in scene["waypoints"]]
