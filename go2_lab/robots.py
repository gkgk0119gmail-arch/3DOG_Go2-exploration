"""Go2 articulation built from the official Unitree URDF (unitree_ros/robots/go2_description)."""

from pathlib import Path

import isaaclab.sim as sim_utils

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

ASSET_DIR = Path(__file__).resolve().parents[1] / "assets"
GO2_USD_PATH = str(ASSET_DIR / "go2" / "usd" / "go2.usd" / "go2_merged" / "go2_merged.usda")

# Same DC-motor actuator model and default pose as Isaac Lab's Go2, but spawned from our own URDF conversion.
GO2_URDF_CFG = UNITREE_GO2_CFG.replace(
    spawn=UNITREE_GO2_CFG.spawn.replace(usd_path=GO2_USD_PATH),
)
