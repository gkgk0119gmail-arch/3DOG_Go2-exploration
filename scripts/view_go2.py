"""Side-by-side viewer: Isaac Lab stock Go2 (left) vs. our URDF-converted Go2 (right), standing on a flat floor.

Usage:
    python sim/scripts/view_go2.py                    # GUI window
    python sim/scripts/view_go2.py --screenshot out.png   # also save a viewport capture after it settles
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--physics", default="isaacsim_physx", choices=["isaacsim_physx"])
parser.add_argument("--screenshot", type=str, default=None)
parser.add_argument("--only", choices=["stock", "ours"], default=None, help="Spawn a single robot.")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(visualizer=["kit"])
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # noqa: E402

from go2_lab.robots import GO2_URDF_CFG  # noqa: E402


@configclass
class ViewSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0))
    stock = UNITREE_GO2_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Stock", init_state=UNITREE_GO2_CFG.init_state.replace(pos=(0.0, -0.5, 0.4))
    )
    ours = GO2_URDF_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Ours", init_state=GO2_URDF_CFG.init_state.replace(pos=(0.0, 0.5, 0.4))
    )


def main():
    names = [args_cli.only] if args_cli.only else ["stock", "ours"]
    if args_cli.only:
        setattr(ViewSceneCfg, "ours" if args_cli.only == "stock" else "stock", None)
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    # match the RL env: run the DC-motor actuators through the backend-native path
    sim_cfg.use_newton_actuators = True
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.2, 0.0, 1.0], target=[0.0, 0.0, 0.25])
    scene = InteractiveScene(ViewSceneCfg(num_envs=1, env_spacing=3.0))
    sim.reset()
    # write the default root / joint state explicitly, as the RL env's reset events do
    for name in names:
        robot = scene[name]
        root_pose = robot.data.default_root_pose.torch.clone()
        root_pose[:, :3] += scene.env_origins
        robot.write_root_pose_to_sim_index(root_pose=root_pose)
        robot.write_root_velocity_to_sim_index(root_velocity=robot.data.default_root_vel.torch.clone())
        robot.write_joint_position_to_sim_index(position=robot.data.default_joint_pos.torch.clone())
        robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.torch.clone())
    scene.reset()
    dt = sim.get_physics_dt()
    step = 0
    while simulation_app.is_running():
        for name in names:
            robot = scene[name]
            robot.set_joint_position_target_index(target=robot.data.default_joint_pos.torch)
        scene.write_data_to_sim()
        sim.step()
        scene.update(dt)
        step += 1
        if step in (1, 50) or step % 400 == 0:
            for name in names:
                d = scene[name].data
                z = d.root_pos_w.torch[0, 2].item()
                q = d.joint_pos.torch[0, :3].tolist()
                qt = d.joint_pos_target.torch[0, :3].tolist()
                tau = d.applied_torque.torch[0, :3].tolist()
                print(
                    f"[view] step {step} {name} z={z:.3f} q={[round(v, 2) for v in q]}"
                    f" q_target={[round(v, 2) for v in qt]} tau={[round(v, 2) for v in tau]}",
                    flush=True,
                )
        if args_cli.screenshot and step == 600:
            from omni.kit.viewport.utility import capture_viewport_to_file, get_active_viewport

            capture_viewport_to_file(get_active_viewport(), args_cli.screenshot)
            print(f"[view] screenshot -> {args_cli.screenshot}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
