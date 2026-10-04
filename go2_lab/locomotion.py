"""Blind rough-terrain velocity-tracking task for the URDF-converted Go2.

The policy sees no height scan, so it can be dropped into arbitrary USD scenes (factory, warehouse)
where the terrain ray caster has no single mesh to hit.
"""

import isaaclab.terrains as terrain_gen
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.core.velocity.mdp as mdp
from isaaclab_tasks.core.velocity.config.go2.agents.rsl_rl_ppo_cfg import UnitreeGo2RoughPPORunnerCfg
from isaaclab_tasks.core.velocity.config.go2.rough_env_cfg import UnitreeGo2RoughEnvCfg

from .robots import GO2_URDF_CFG


@configclass
class Go2StockBlindRoughEnvCfg(UnitreeGo2RoughEnvCfg):
    """Isaac Lab's Go2 asset (converted from the same unitree_ros URDF), blind, PhysX."""

    def __post_init__(self):
        super().__post_init__()
        # blind policy: drop the terrain height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # same backend as the factory scene (Isaac Sim USD assets)
        self.sim.physics.default = self.sim.physics.isaacsim_physx
        # GUI camera follows robot 0 (terrain tiles put robots far from the world origin)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        self.viewer.env_index = 0
        self.viewer.eye = (2.0, 2.0, 1.2)
        self.viewer.lookat = (0.0, 0.0, 0.0)


@configclass
class Go2BlindRoughEnvCfg(Go2StockBlindRoughEnvCfg):
    """Same task with our own URDF->USD conversion (currently unstable, see README notes)."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = GO2_URDF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def _play(cfg):
    cfg.scene.num_envs = 50
    cfg.scene.env_spacing = 2.5
    cfg.scene.terrain.max_init_terrain_level = None
    if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.curriculum = False
    cfg.observations.policy.enable_corruption = False
    cfg.events.base_external_force_torque = None
    cfg.events.push_robot = None


@configclass
class Go2StockBlindRoughEnvCfg_PLAY(Go2StockBlindRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play(self)


@configclass
class Go2BlindRoughEnvCfg_PLAY(Go2BlindRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play(self)


@configclass
class Go2StockBlindRoughPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "go2_stock_blind_rough"


@configclass
class Go2BlindRoughPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "go2_urdf_blind_rough"


##
# Factory walking: forward/turn commands, near-flat floors, posture-shaping rewards.
##


@configclass
class Go2FactoryWalkEnvCfg(Go2StockBlindRoughEnvCfg):
    """Blind Go2 gait for factory floors, driven by a turn-then-go waypoint controller.

    Compared with the stock rough task: little lateral velocity, direct yaw-rate commands (no heading
    tracking), near-flat terrain, and extra rewards that keep a natural standing height and posture.
    """

    def __post_init__(self):
        super().__post_init__()

        # terrain: flat floor, small roughness, low sills / cable covers
        gen = self.scene.terrain.terrain_generator
        gen.sub_terrains = {
            "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.5),
            "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=0.35, noise_range=(0.005, 0.03), noise_step=0.005, border_width=0.25
            ),
            "sills": terrain_gen.MeshRandomGridTerrainCfg(
                proportion=0.15, grid_width=0.45, grid_height_range=(0.01, 0.04), platform_width=2.0
            ),
        }

        # commands: mostly forward + turning, small sideways, 10% standing
        cmd = self.commands.base_velocity
        cmd.heading_command = False
        cmd.rel_heading_envs = 0.0
        cmd.rel_standing_envs = 0.1
        cmd.ranges.lin_vel_x = (-0.5, 1.0)
        cmd.ranges.lin_vel_y = (-0.3, 0.3)
        cmd.ranges.ang_vel_z = (-1.5, 1.5)

        # posture / gait shaping
        rew = self.rewards
        rew.flat_orientation_l2.weight = -2.5
        rew.feet_air_time.weight = 0.25
        rew.dof_pos_limits.weight = -1.0
        rew.base_height = RewTerm(func=mdp.base_height_l2, weight=-10.0, params={"target_height": 0.30})
        rew.hip_deviation = RewTerm(
            func=mdp.joint_deviation_l1, weight=-0.3, params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint")}
        )
        rew.thigh_contact = RewTerm(
            func=mdp.undesired_contacts,
            weight=-1.0,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_thigh"), "threshold": 1.0},
        )
        rew.feet_slide = RewTerm(
            func=mdp.feet_slide,
            weight=-0.1,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
                "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            },
        )
        rew.stand_still = RewTerm(
            func=mdp.stand_still_joint_deviation_l1, weight=-0.5, params={"command_name": "base_velocity"}
        )


@configclass
class Go2FactoryWalkEnvCfg_PLAY(Go2FactoryWalkEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _play(self)


@configclass
class Go2FactoryWalkPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "go2_factory_walk"
        self.max_iterations = 1500


@configclass
class Go2FactoryWalkShowcaseEnvCfg(Go2FactoryWalkEnvCfg):
    """The training scene as-is (4096 robots, curriculum terrain, random commands) for recording checkpoints.

    GO2_SHOW_CAM=overview (default, whole terrain) | close (follow robot 0).
    """

    def __post_init__(self):
        super().__post_init__()
        import os

        eye = os.environ.get("GO2_CAM_EYE")  # "x,y,z": camera offset (close) or position (overview) [m]
        eye = tuple(float(v) for v in eye.split(",")) if eye else None
        if os.environ.get("GO2_SHOW_LAYOUT", "terrain") == "grid":
            # ~36 robots on a flat floor, 2.5 m apart, seen by one fixed oblique camera (recording clips)
            self.scene.terrain.terrain_type = "plane"
            self.scene.terrain.terrain_generator = None
            self.curriculum.terrain_levels = None
            self.scene.env_spacing = 2.5
            self.viewer.origin_type = "world"
            self.viewer.eye = eye or (-7.5, -10.0, 7.0)
            self.viewer.lookat = (0.0, 0.0, 0.0)
            return
        if os.environ.get("GO2_SHOW_CAM", "overview") == "close":
            self.viewer.origin_type = "asset_root"
            self.viewer.asset_name = "robot"
            self.viewer.env_index = 0
            self.viewer.eye = eye or (4.0, 4.0, 3.0)
            self.viewer.lookat = (0.0, 0.0, 0.0)
        else:
            self.viewer.origin_type = "world"
            self.viewer.eye = eye or (-55.0, -10.0, 38.0)
            self.viewer.lookat = (0.0, 25.0, 0.0)
