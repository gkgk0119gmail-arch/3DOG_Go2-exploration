"""Isaac Lab RL entrypoint with the 3DOG tasks registered.

Usage:
    python sim/scripts/rl.py train --task Go2-URDF-Blind-Rough --num_envs 4096
    python sim/scripts/rl.py play  --task Go2-URDF-Blind-Rough-Play --num_envs 16
"""

import sys
from pathlib import Path

import warp as wp

wp.config.enable_backward = False

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import go2_lab  # noqa: E402,F401  (registers gym tasks)
from isaaclab_rl.entrypoints.dispatch import run_cli  # noqa: E402

if __name__ == "__main__":
    action, argv = sys.argv[1], sys.argv[2:]
    if "--rl_library" not in argv:
        argv = ["--rl_library", "rsl_rl", *argv]
    raise SystemExit(run_cli(action, argv))
