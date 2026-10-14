"""Run an inspection experiment in Isaac Lab: one Go2 (posture-walk policy) per seed, all in the same scene.

    python scripts/pose_inspection/run_isaac.py --exp exp/pose.yaml --scene A_warehouse [--seeds 0 1 2 3 4 5]
        [--set planner.posture.enabled=false ...] [--name pose] [--visual] [--viz kit]

Results: <experiment.results_dir>/isaac/<scene>/<name>/seed<k>.json (same format as the kinematic engine).
"""

import argparse
import json
import os
import sys
import time

from isaaclab.app import AppLauncher

SIM = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, SIM)

ap = argparse.ArgumentParser()
ap.add_argument("--exp", default="exp/pose.yaml")
ap.add_argument("--scene", default="B_construction")
ap.add_argument("--seeds", nargs="*", type=int, default=None)
ap.add_argument("--set", nargs="*", default=[], help="dotted overrides key=value (YAML values)")
ap.add_argument("--name", default=None)
ap.add_argument("--visual", action="store_true", help="also load the textured scene (GUI / video / cameras)")
ap.add_argument("--policy", default="weights/go2_posture_walk_model_1999.pt")
ap.add_argument("--max_wall_s", type=float, default=7200.0)
ap.add_argument("--video_out", default=None, help="mp4 of robot 0 from a following camera (implies --visual)")
ap.add_argument("--video_every_s", type=float, default=0.2, help="sim seconds between video frames")
ap.add_argument("--video_max_s", type=float, default=600.0)
ap.add_argument("--video_eye", type=float, nargs=3, default=(-3.2, -2.4, 2.4), help="camera offset from the robot")
AppLauncher.add_app_launcher_args(ap)
args = ap.parse_args()

import yaml  # noqa: E402

over = {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in args.set)}
sys.path.insert(0, SIM)
from go2_lab.pose_inspection.config import load_config, sim_path  # noqa: E402

cfg = load_config(args.exp, over)
seeds = args.seeds if args.seeds is not None else list(range(cfg.experiment.seed_base,
                                                             cfg.experiment.seed_base + cfg.experiment.n_seeds))
name = args.name or os.path.splitext(os.path.basename(args.exp))[0]
out = sim_path(os.path.join(cfg.experiment.results_dir, "isaac", args.scene, name))
if args.video_out:
    os.environ["GO2_PI_VIDEO"] = "1"
    args.video_out = os.path.abspath(args.video_out)
    args.visual = True
os.environ.update({"GO2_PI_EXP": args.exp, "GO2_PI_SCENE": args.scene, "GO2_PI_SET": json.dumps(over),
                   "GO2_PI_SEEDS": ",".join(map(str, seeds)), "GO2_PI_OUT": out, "GO2_PI_TAG": name,
                   "GO2_PI_VISUAL": "1" if args.visual else "0"})
if args.visual or cfg.perception.mode == "gsam2" or cfg.planner.semantic.prune == "vlm" or \
        cfg.priors.method == "template_vlm":
    args.enable_cameras = True
app = AppLauncher(args).app

import importlib.metadata as metadata  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import warp as wp  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

wp.config.enable_backward = False

import go2_lab  # noqa: E402,F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils.hydra import resolve_task_config  # noqa: E402


def main():
    # same config path as scripts/rl.py play: physics / renderer presets resolved to their defaults
    env_cfg, agent_cfg = resolve_task_config("Go2-Pose-Inspection-Play", "rsl_rl_cfg_entry_point", overrides=[])
    env_cfg.sim.device = args.device if args.device is not None else env_cfg.sim.device
    writer = None
    env = gym.make("Go2-Pose-Inspection-Play", cfg=env_cfg)
    if args.video_out:
        import imageio.v2 as iio

        os.makedirs(os.path.dirname(os.path.abspath(args.video_out)), exist_ok=True)
        writer = iio.get_writer(args.video_out, fps=25, codec="libx264", quality=7)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    venv = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(venv, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(sim_path(args.policy))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    term = env.unwrapped.command_manager.get_term("base_velocity")
    print(f"[{name} / {args.scene}] seeds {seeds} -> {out}")
    obs = venv.get_observations()
    t0, last = time.time(), 0.0
    next_frame = 0.0
    while app.is_running() and not term.finished:
        with torch.inference_mode():
            obs, _, dones, _ = venv.step(policy(obs))
            policy.reset(dones)
        if writer is not None and term.t >= next_frame and term.t <= args.video_max_s:
            next_frame = term.t + args.video_every_s
            cam = env.unwrapped.scene["pi_video"]
            if term.t > 0:
                writer.append_data(cam.data.output["rgb"].torch[0, ..., :3].cpu().numpy())
            # chase view for the next frame: behind / beside / above robot 0 in its heading frame
            rob = env.unwrapped.scene["robot"]
            p = rob.data.root_pos_w.torch[0]
            yaw = float(rob.data.heading_w.torch[0])
            ex, ey, ez = args.video_eye
            c, s_ = np.cos(yaw), np.sin(yaw)
            eye = torch.tensor([[p[0] + c * ex - s_ * ey, p[1] + s_ * ex + c * ey, ez]], device=p.device)
            look = torch.tensor([[p[0], p[1], 0.7]], device=p.device)
            cam.set_world_poses_from_view(eye.float(), look.float())
        if term.t - last >= 60.0:
            last = term.t
            co = term.session.surf.co().cpu().numpy().round(3)
            _, cs = term.session.cs()
            print(f"  t={term.t:6.0f}s  wall {time.time() - t0:5.0f}s  Co {co}  Cs {cs.round(3)}  "
                  f"done {term.session.term.done.sum()}/{len(seeds)}", flush=True)
        if time.time() - t0 > args.max_wall_s:
            print("wall-clock limit reached; saving partial runs")
            term.session.save({"engine": "isaac", "partial": True})
            break
    if writer is not None:
        writer.close()
        print("wrote", args.video_out)
    s = term.session
    print(f"done: sim {term.t:.0f} s in {time.time() - t0:.0f} s wall")
    for e in range(len(seeds)):
        r = s.summary(e)
        print(f"  seed {r['seed']}: Co {r['Co']:.3f} (vs UB {r['Co_vs_ub']:.3f}) Cs {r['Cs']:.3f} t_eff {r['t_eff']:.0f} "
              f"L {r['L_eff']:.0f} Npost {r['N_post']} timeout {r['timeout']} collision {term.collision_s[e]:.1f}s "
              f"fell {term.fell[e]}")
    env.close()


if __name__ == "__main__":
    main()
    app.close()
