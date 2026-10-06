"""3DOG Isaac Lab tasks: Go2 locomotion and factory exploration scenes."""

import gymnasium as gym

_TASKS = {
    # Isaac Lab's Go2 asset (converted from the same unitree_ros URDF) -- known-good
    "Go2-Stock-Blind-Rough": ("Go2StockBlindRoughEnvCfg", "Go2StockBlindRoughPPORunnerCfg"),
    "Go2-Stock-Blind-Rough-Play": ("Go2StockBlindRoughEnvCfg_PLAY", "Go2StockBlindRoughPPORunnerCfg"),
    # factory gait: forward/turn commands, posture-shaping rewards
    "Go2-Factory-Walk": ("Go2FactoryWalkEnvCfg", "Go2FactoryWalkPPORunnerCfg"),
    "Go2-Factory-Walk-Play": ("Go2FactoryWalkEnvCfg_PLAY", "Go2FactoryWalkPPORunnerCfg"),
    # training scene for recording checkpoints (4096 robots)
    "Go2-Factory-Walk-Showcase": ("Go2FactoryWalkShowcaseEnvCfg", "Go2FactoryWalkPPORunnerCfg"),
    # warehouse demo: factory-walk policy + VLP-16 + waypoint loop (play only)
    "Go2-Warehouse-Demo-Play": ("factory:Go2WarehouseDemoEnvCfg", "Go2FactoryWalkPPORunnerCfg"),
    # autonomous exploration: VLP-16 -> occupancy map -> ARiADNE waypoints (play only)
    "Go2-Warehouse-Explore-Play": ("factory:Go2WarehouseExploreEnvCfg", "Go2FactoryWalkPPORunnerCfg"),
    # N independent clones in the same warehouse + reward-driven parameter search (play only)
    "Go2-Warehouse-Parallel-Play": ("factory:Go2WarehouseParallelEnvCfg", "Go2FactoryWalkPPORunnerCfg"),
    # warehouse demo in a BIM building (tools/ifc_to_isaac.py; GO2_BIM picks the scene, default dental clinic)
    "Go2-Bim-Demo-Play": ("bim:Go2BimDemoEnvCfg", "Go2FactoryWalkPPORunnerCfg"),
    # construction-site scan around an excavator (tools/urdf_to_scene.py; GO2_MACHINE picks the scene)
    "Go2-Excavator-Demo-Play": ("excavator:Go2ExcavatorDemoEnvCfg", "Go2FactoryWalkPPORunnerCfg"),
    # body pitch / roll command tracking on top of the factory gait (posture.py)
    "Go2-Posture-Walk": ("posture:Go2PostureWalkEnvCfg", "posture:Go2PostureWalkPPORunnerCfg"),
    "Go2-Posture-Walk-Play": ("posture:Go2PostureWalkEnvCfg_PLAY", "posture:Go2PostureWalkPPORunnerCfg"),
    # excavator scan with posture stops: walk the loop, tilt the body at the side midpoints (play only)
    "Go2-Excavator-Posture-Demo-Play": ("excavator:Go2ExcavatorPostureDemoEnvCfg", "posture:Go2PostureWalkPPORunnerCfg"),
    # our own URDF->USD conversion -- joints currently unstable in training
    "Go2-URDF-Blind-Rough": ("Go2BlindRoughEnvCfg", "Go2BlindRoughPPORunnerCfg"),
    "Go2-URDF-Blind-Rough-Play": ("Go2BlindRoughEnvCfg_PLAY", "Go2BlindRoughPPORunnerCfg"),
}

for _task_id, (_env_cfg, _agent_cfg) in _TASKS.items():
    gym.register(
        id=_task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.{_env_cfg}" if ":" in _env_cfg else f"{__name__}.locomotion:{_env_cfg}",
            "rsl_rl_cfg_entry_point": f"{__name__}.{_agent_cfg}" if ":" in _agent_cfg else f"{__name__}.locomotion:{_agent_cfg}",
            "default_agent": "rsl_rl",
        },
    )
