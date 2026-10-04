"""Parallel, independent exploration with N Go2 clones in the same site + reward-driven parameter search.

Every robot explores on its own: its own occupancy map, its own ARiADNE planner process (planner_worker.py),
its own A* path. Robots share the warehouse geometry only (inter-robot collisions are filtered and the LiDAR
only sees the warehouse), so they never see or help each other.

Generations: all robots start from random free poses; a robot finishes when ARiADNE reports exploration done,
when ``episode_s`` elapses, or when it falls. Each robot is scored by how fast and how well it mapped
(area under its coverage-vs-time curve over ``episode_s``); its map is also checked against the ground truth.
  * mode "search": cross-entropy method -- the best quarter of the population ("the robots that mapped best")
    defines the parameter distribution of the next generation
  * mode "eval":   every robot uses the same parameters; generations only add random starts (statistics)
  * mode "loop":   continuous -- every robot that finishes shows its map for ``respawn_delay`` s, then respawns
                   alone at a random free pose with a fresh map/planner (no generations; episodes.csv)
"""

from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import MISSING

import numpy as np
import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_mul

from ..waypoint_command import WaypointTurnThenGoCommand, WaypointTurnThenGoCommandCfg
from .occupancy_mapper import FREE, OCCUPIED, OccupancyMapper2D

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# name: (default, low, high, kind) -- "follow" params act in the simulator, "planner" params go to ARiADNE
PARAM_SPACE = {
    "switch_heading": (0.785, 0.3, 1.4, "follow"),   # [rad] accept a new waypoint without turning below this
    "commit_reach": (2.0, 1.0, 4.0, "follow"),       # [m] always accept a new waypoint this close to the target
    "lookahead": (1.2, 0.6, 2.0, "follow"),          # [m] path-following lookahead
    "clearance": (1.0, 0.5, 1.5, "follow"),          # [m] A* wall-distance cost range
    "robot_radius": (0.35, 0.30, 0.45, "follow"),    # [m] A* hard clearance
    "turn_threshold": (0.35, 0.2, 0.7, "follow"),    # [rad] stop and turn in place above this heading error
    "stall_s": (4.0, 2.0, 8.0, "follow"),            # [s] drop a target that is not getting closer
    "THR_NEXT_WAYPOINT": (4.0, 2.0, 8.0, "planner"),  # [m] ARiADNE plans waypoints at least this far
}


class ParallelExplorationCommand(WaypointTurnThenGoCommand):
    cfg: ParallelExplorationCommandCfg

    def __init__(self, cfg: ParallelExplorationCommandCfg, env):
        super().__init__(cfg, env)
        from .metrics import MapScorer
        from .planner_worker import PlannerWorker

        self._env = env
        E = self.num_envs
        self.mapper = OccupancyMapper2D(bounds=cfg.map_bounds, cell_size=cfg.cell_size, max_range=cfg.sensor_range,
                                        obstacle_z=cfg.obstacle_z, num_envs=E, device=str(self.device))
        # LiDAR measurement model: GO2_LIDAR_NOISE=0 ideal | 1 VLP-16 noise | 2 + localization error
        from .sensor_noise import VLP16NoiseCfg

        level = int(os.environ.get("GO2_LIDAR_NOISE", "0"))
        self._compare_kind = os.environ.get("GO2_COMPARE", "params")  # compare mode: "params" or "noise"
        if cfg.mode == "compare" and self._compare_kind == "noise":
            level = max(level, 1)
        self.noise_cfg = VLP16NoiseCfg(pose_sigma_xy=0.05 if level >= 2 else 0.0,
                                       pose_sigma_yaw_deg=0.5 if level >= 2 else 0.0) if level > 0 else None
        self.noisy = np.full(E, level > 0)
        if cfg.mode == "compare" and self._compare_kind == "noise":
            self.noisy[: E // 2] = False  # A = ideal LiDAR, B = VLP-16 model (same parameters, paired starts)
        self._noise_gen = torch.Generator(device=str(self.device)).manual_seed(cfg.seed)
        self._deskew = os.environ.get("GO2_DESKEW", "0") == "1"  # motion-compensate noisy scans with an IMU estimate
        self._skip_turn = float(os.environ.get("GO2_SKIP_TURN", "0"))  # [rad/s] skip scans above this yaw rate
        print(f"[parallel] lidar noise level {level} on {int(self.noisy.sum())}/{E} robots, deskew "
              f"{os.environ.get('GO2_DESKEW', '0')}, skip-turn {os.environ.get('GO2_SKIP_TURN', '0')}", flush=True)
        self.scorer = MapScorer(cfg.gt_map, self.mapper.origin, self.mapper.cell, (self.mapper.ny, self.mapper.nx),
                                cfg.gt_bounds)
        self._gt_free_area = float((self.scorer.gt_free & self.scorer.inside).sum()) * cfg.cell_size**2
        self._spawn_xy = self._spawn_candidates()
        self._gt_dist = self._gt_distance_map()
        self._rng = np.random.default_rng(cfg.seed)

        # parameter search state (normalized space [0, 1])
        self.names = list(PARAM_SPACE)
        lo = np.array([PARAM_SPACE[n][1] for n in self.names])
        hi = np.array([PARAM_SPACE[n][2] for n in self.names])
        self._lo, self._hi = lo, hi
        defaults = np.array([PARAM_SPACE[n][0] for n in self.names])
        self.mu = (defaults - lo) / (hi - lo)
        self._default_z = self.mu.copy()
        # GO2_PAR_PARAMS=<best.json>: start from (eval: use) the "mean" parameters found by a previous search
        params_file = os.environ.get("GO2_PAR_PARAMS")
        if params_file:
            with open(params_file) as f:
                found = json.load(f)["mean"]
            vals = np.array([found.get(n, d) for n, d in zip(self.names, defaults)])
            self.mu = np.clip((vals - lo) / (hi - lo), 0, 1)
            print(f"[parallel] parameters from {params_file}: {dict(zip(self.names, np.round(vals, 3).tolist()))}",
                  flush=True)
        self.sigma = np.full(len(self.names), cfg.cem_init_std)
        self.generation = 0
        self.best = {"score": -1.0}

        self.out_dir = os.path.join(cfg.out_dir, time.strftime("%Y-%m-%d_%H-%M-%S") + f"_{cfg.mode}")
        os.makedirs(self.out_dir, exist_ok=True)
        self._gen_log = open(os.path.join(self.out_dir, "generations.csv"), "w", newline="")
        self._gen_csv = csv.writer(self._gen_log)
        self._gen_csv.writerow(["generation", "env", *self.names, "score", "coverage", "time_s", "distance_m",
                                "fell", "collided", "precision", "recall", "false_free", "plan_ms", "noisy", "planner",
                                "cov3d", "turn_deg"])

        # planners: GO2_PLANNER for every robot, or GO2_COMPARE=planner: GO2_PLANNER_A (first half) vs GO2_PLANNER_B
        self.planner_spec = self._planner_specs(E)
        self.live3d = self.gtcov3d = None
        if any(sp["kind"] == "dense" for sp in self.planner_spec) or os.environ.get("GO2_EVAL3D") == "1":
            from .live3d import GTCoverage3D, Live3D

            self.live3d = Live3D(E, self.mapper.origin, self.mapper.nx, self.mapper.ny, cfg.cell_size,
                                 tilt_deg=float(os.environ.get("GO2_LIDAR_TILT", "0")), device=str(self.device))
            self.gtcov3d = GTCoverage3D(E, cfg.gt_bounds, device=str(self.device))
            print(f"[parallel] 3D belief + GT 3D coverage on ({len(self.gtcov3d.keys):,} surface voxels)", flush=True)
        print(f"[parallel] starting {E} planner workers: "
              + ", ".join(sorted({self._planner_name(sp) for sp in self.planner_spec})), flush=True)
        self.workers = [PlannerWorker(self._planner_params(defaults, e)) for e in range(E)]
        # the env's own startup reset would overwrite our spawn poses: start generation 0 on the first step
        self._pending_start = True
        self.finished = np.ones(E, bool)
        self.t0 = 0.0
        self._init_ui()
        print(f"[parallel] {E} robots, mode={cfg.mode}, logging to {self.out_dir}", flush=True)

    # -- setup ----------------------------------------------------------------------------------------

    def _spawn_candidates(self) -> np.ndarray:
        """Free cells of the GT map at least ``spawn_clearance`` from any obstacle, inside the GT bounds."""
        from scipy import ndimage

        d = np.load(self.cfg.gt_map)
        g, o, r = d["grid"], d["origin"], float(d["resolution"])
        occ = ndimage.binary_opening(g == 0, np.ones((2, 2)))  # drop speckle
        dist = ndimage.distance_transform_edt(~occ) * r
        ok = (g == 255) & (dist >= self.cfg.spawn_clearance)
        ij = np.argwhere(ok)
        xy = o + (ij + 0.5) * r
        b = self.cfg.gt_bounds
        if b is not None:
            xy = xy[(xy[:, 0] > b[0] + 1) & (xy[:, 0] < b[2] - 1) & (xy[:, 1] > b[1] + 1) & (xy[:, 1] < b[3] - 1)]
        return xy

    def _gt_distance_map(self):
        """(distance [m] to the nearest GT obstacle on the 0.1 m GT grid, origin, resolution)."""
        from scipy import ndimage

        d = np.load(self.cfg.gt_map)
        g, o, r = d["grid"], d["origin"], float(d["resolution"])
        occ = ndimage.binary_opening(g == 0, np.ones((2, 2)))
        return ndimage.distance_transform_edt(~occ) * r, o, r

    def _collided(self, pos: np.ndarray) -> np.ndarray:
        dist, o, r = self._gt_dist
        ij = np.clip(np.floor((pos - o) / r).astype(int), 0, np.array(dist.shape) - 1)
        return dist[ij[:, 0], ij[:, 1]] < self.cfg.collision_radius

    def _planner_params(self, values: np.ndarray, e: int = 0) -> dict:
        p = {"SENSOR_RANGE": self.cfg.sensor_range, "CELL_SIZE": self.cfg.cell_size}
        for n, v in zip(self.names, values):
            if PARAM_SPACE[n][3] == "planner":
                p[n] = float(v)
        return {**p, **self.planner_spec[e]} if hasattr(self, "planner_spec") else p

    @staticmethod
    def _parse_planner(spec: str) -> dict:
        """"ros" | "dense" (fine-tuned ariadne3d checkpoint) | "dense:pretrained" | "dense:<checkpoint path>"."""
        if spec in ("", "ros"):
            return {"kind": "ros"}
        kind, _, ck = spec.partition(":")
        if kind != "dense":
            raise ValueError(f"unknown planner {spec!r}")
        sim = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        ck = {"": None, "pretrained": os.path.join(sim, "ariadne3d", "pretrained", "ariadne_ral2024.pth")}.get(ck, ck)
        out = {"kind": "dense", "checkpoint": ck}
        if os.environ.get("GO2_DENSE_RES"):
            out["node_res"] = float(os.environ["GO2_DENSE_RES"])
        return out

    def _planner_specs(self, E: int) -> list[dict]:
        if self.cfg.mode == "compare" and self._compare_kind == "planner":
            a = self._parse_planner(os.environ.get("GO2_PLANNER_A", "ros"))
            b = self._parse_planner(os.environ.get("GO2_PLANNER_B", "dense"))
            return [a] * (E // 2) + [b] * (E - E // 2)
        return [self._parse_planner(os.environ.get("GO2_PLANNER", "ros"))] * E

    @staticmethod
    def _planner_name(sp: dict) -> str:
        if sp["kind"] == "ros":
            return "ros"
        ck = sp.get("checkpoint")
        return "dense:" + ("finetuned" if ck is None else "pretrained" if ck.endswith("ariadne_ral2024.pth") else os.path.basename(ck))

    @property
    def n_starts(self) -> int:
        """Shared start poses per candidate (every candidate of a generation runs from the same K starts)."""
        return 2 if self.cfg.mode == "compare" else max(1, min(self.cfg.starts_per_candidate, self.num_envs))

    def _candidate_of(self, e: int) -> int:
        return e // (self.num_envs // self.n_candidates)

    @property
    def n_candidates(self) -> int:
        if self.cfg.mode == "compare":
            return 2
        return self.num_envs // self.n_starts

    def _sample_params(self) -> np.ndarray:
        """[E, P] parameter values; robots of one candidate share its parameters.

        search : candidate 0 is the current mean, the rest are sampled around it
        eval   : every robot uses the current mean (defaults or GO2_PAR_PARAMS)
        compare: first half defaults, second half GO2_PAR_PARAMS (paired starts)
        """
        C, P = self.n_candidates, len(self.names)
        if self.cfg.mode in ("eval", "loop"):
            z = np.repeat(self.mu[None], C, 0)
        elif self.cfg.mode == "compare":
            z = np.stack([self.mu, self.mu]) if self._compare_kind in ("noise", "planner") else np.stack([self._default_z, self.mu])
        else:
            z = np.clip(self.mu + self.sigma * self._rng.standard_normal((C, P)), 0, 1)
            z[0] = self.mu
        per_env = np.repeat(z, self.num_envs // C, axis=0)
        return self._lo + per_env * (self._hi - self._lo)

    # -- generations ----------------------------------------------------------------------------------

    def _start_generation(self):
        E = self.num_envs
        self.params = self._sample_params()
        P = {n: self.params[:, i] for i, n in enumerate(self.names)}
        self.turn_threshold_env = torch.tensor(P["turn_threshold"], dtype=torch.float32, device=self.device)
        self.go_threshold_env = self.turn_threshold_env * (self.cfg.go_threshold / self.cfg.turn_threshold)
        self.P = P
        for e, w in enumerate(self.workers):
            w.reset(self._planner_params(self.params[e], e), wait=False)
        self._worker_ready = np.zeros(E, bool)
        self.mapper.reset()
        if self.live3d is not None:
            self.live3d.reset()
            self.gtcov3d.reset()
        self.turn_total = np.zeros(E)
        self.prev_yaw = None
        self._respawn(np.arange(E))
        now = self._sim_time()
        self.t0 = now
        self.t0_env = np.full(E, now)
        self.finish_abs = np.full(E, np.nan)
        self.episode_idx = np.zeros(E, int)
        self._skip_dist = np.zeros(E, bool)
        self.finished = np.zeros(E, bool)
        self.fell = np.zeros(E, bool)
        self.collided = np.zeros(E, bool)
        self.done = np.zeros(E, bool)
        self.finish_time = np.full(E, np.nan)
        self.target = np.full((E, 2), np.nan)
        self.paths: list[np.ndarray | None] = [None] * E
        self.req = np.zeros(E, int)  # latest request id per robot (stale replies are ignored)
        self.plan_pending = np.zeros(E, bool)
        self.path_pending = np.zeros(E, bool)
        self.last_plan = np.full(E, now + self.cfg.warmup_s - self.cfg.replan_period)
        self.last_path = np.full(E, -1e9)
        self.target_set = np.zeros(E)
        self.target_best = np.full(E, np.inf)
        self.distance = np.zeros(E)
        self.prev_xy = None
        self.traj: list[list] = [[] for _ in range(E)]
        self.cov_hist: list[list] = [[] for _ in range(E)]
        self.plan_ms: list[list] = [[] for _ in range(E)]
        self.graphs = [None] * E
        self._last_scan = self._last_ui = self._last_cov = -1e9
        print(f"[parallel] generation {self.generation} started", flush=True)

    def _respawn(self, env_ids: np.ndarray):
        """Random free pose + default joints for the given robots (written straight to the simulation)."""
        n = len(env_ids)
        if self.cfg.mode in ("eval", "loop"):
            # independent random start per robot
            xy = self._spawn_xy[self._rng.choice(len(self._spawn_xy), n)]
            yaw = self._rng.uniform(-math.pi, math.pi, n)
        else:
            # search/compare: the k-th robot of every candidate shares the same start (fair comparison)
            per_cand = self.num_envs // self.n_candidates
            starts_xy = self._spawn_xy[self._rng.choice(len(self._spawn_xy), per_cand)]
            starts_yaw = self._rng.uniform(-math.pi, math.pi, per_cand)
            k = np.asarray(env_ids) % per_cand
            xy, yaw = starts_xy[k], starts_yaw[k]
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        pose = self.robot.data.default_root_pose.torch[ids].clone()
        pose[:, 0:2] = torch.as_tensor(xy, dtype=torch.float32, device=self.device) + self._env_origins[ids, :2]
        quat = np.stack([np.zeros(n), np.zeros(n), np.sin(yaw / 2), np.cos(yaw / 2)], axis=1)  # (x, y, z, w)
        pose[:, 3:7] = torch.as_tensor(quat, dtype=torch.float32, device=self.device)
        self.robot.write_root_pose_to_sim_index(root_pose=pose, env_ids=ids)
        self.robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=self.device), env_ids=ids)
        self.robot.write_joint_position_to_sim_index(position=self.robot.data.default_joint_pos.torch[ids].clone(),
                                                     env_ids=ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(self.robot.data.default_joint_vel.torch[ids]),
                                                     env_ids=ids)

    def _restart_robot(self, e: int, now: float):
        """Loop mode: respawn one robot at a random free pose with a fresh map and planner."""
        self.workers[e].reset(self._planner_params(self.params[e], e), wait=False)
        self._worker_ready[e] = False
        self.mapper.reset([e])
        if self.live3d is not None:
            self.live3d.reset([e])
            self.gtcov3d.reset([e])
        self.turn_total[e] = 0.0
        self._respawn(np.array([e]))
        self.req[e] += 1  # ignore replies still in flight for the old episode
        self.t0_env[e] = now
        self.finish_abs[e] = np.nan
        self.episode_idx[e] += 1
        self.finished[e] = self.fell[e] = self.collided[e] = self.done[e] = False
        self.finish_time[e] = np.nan
        self.target[e] = np.nan
        self.paths[e] = None
        self.plan_pending[e] = self.path_pending[e] = False
        self.last_plan[e] = now + self.cfg.warmup_s - self.cfg.replan_period
        self.last_path[e] = -1e9
        self.target_set[e], self.target_best[e] = 0.0, np.inf
        self.distance[e] = 0.0
        self._skip_dist[e] = True  # the teleport is not walking distance
        self.traj[e], self.cov_hist[e], self.plan_ms[e], self.graphs[e] = [], [], [], None

    def _log_episode(self, e: int):
        """Loop mode: one CSV row + running statistics per finished robot episode."""
        if not hasattr(self, "_ep_log"):
            self._ep_log = open(os.path.join(self.out_dir, "episodes.csv"), "w", newline="")
            self._ep_csv = csv.writer(self._ep_log)
            self._ep_csv.writerow(["env", "episode", "result", "time_s", "coverage", "score", "distance_m", "precision",
                                   "recall", "false_free", "noisy"])
            self.stats = []
        q = self.scorer(self.mapper.grid(e))
        result = "done" if self.done[e] else "fell" if self.fell[e] else "collided" if self.collided[e] else "timeout"
        self._ep_csv.writerow([e, int(self.episode_idx[e]), result, f"{self.finish_time[e]:.1f}", f"{q['coverage']:.4f}",
                               f"{self._score(e):.4f}", f"{self.distance[e]:.1f}", f"{q['precision']:.4f}",
                               f"{q['recall']:.4f}", f"{q['false_free']:.4f}", int(self.noisy[e])])
        self._ep_log.flush()
        self.stats.append((result, self.finish_time[e], q["coverage"], q["recall"]))
        n = len(self.stats)
        if n % 16 == 0:
            st = np.array([(r == "done", t, c, rc) for r, t, c, rc in self.stats], dtype=float)
            done_t = st[st[:, 0] == 1, 1]
            print(f"[parallel] loop: {n} episodes | done {st[:, 0].mean():.0%} | finish median "
                  f"{np.median(done_t) if len(done_t) else float('nan'):.0f}s | coverage {st[:, 2].mean():.3f} | "
                  f"recall {st[:, 3].mean():.3f}", flush=True)

    def _finish(self, e: int, reason: str):
        if self.finished[e]:
            return
        self.finished[e] = True
        self.finish_time[e] = self._sim_time() - self.t0_env[e]
        self.finish_abs[e] = self._sim_time()
        self.fell[e] = reason == "fell"
        self.collided[e] = reason == "collided"
        self.done[e] = reason == "done"
        self.target[e] = np.nan
        self.paths[e] = None

    def _score(self, e: int) -> float:
        """Mean coverage over [0, episode_s] (coverage frozen after finishing): rewards mapping fast and fully."""
        h = np.array(self.cov_hist[e]) if self.cov_hist[e] else np.zeros((1, 2))
        T = self.cfg.episode_s
        ts = np.linspace(0, T, 200)
        cov = np.interp(ts, h[:, 0], h[:, 1], left=0.0, right=h[-1, 1])
        return float(cov.mean())

    def _end_generation(self):
        E = self.num_envs
        scores = np.array([self._score(e) for e in range(E)])
        quality = [self.scorer(self.mapper.grid(e)) for e in range(E)]
        cov3d = self.gtcov3d.coverage() if self.gtcov3d is not None else None
        for e in range(E):
            q = quality[e]
            self._gen_csv.writerow([self.generation, e, *np.round(self.params[e], 4), f"{scores[e]:.4f}",
                                    f"{q['coverage']:.4f}", f"{self.finish_time[e]:.1f}", f"{self.distance[e]:.1f}",
                                    int(self.fell[e]), int(self.collided[e]), f"{q['precision']:.4f}", f"{q['recall']:.4f}",
                                    f"{q['false_free']:.4f}",
                                    f"{np.mean(self.plan_ms[e]):.0f}" if self.plan_ms[e] else "", int(self.noisy[e]),
                                    self._planner_name(self.planner_spec[e]),
                                    f"{cov3d[e]:.4f}" if cov3d is not None else "", f"{np.rad2deg(self.turn_total[e]):.0f}"])
        self._gen_log.flush()
        C = self.n_candidates
        cand_scores = scores.reshape(C, -1).mean(1)
        order = np.argsort(-cand_scores) * (E // C)  # first robot of each candidate, best first
        b = order[0]
        line = (f"[parallel] generation {self.generation}: score mean {scores.mean():.3f} best candidate "
                f"{cand_scores.max():.3f} (cand {b // (E // C)}, {E // C} starts) "
                f"| done {self.done.sum()}/{E} fell {self.fell.sum()} collided {self.collided.sum()} | finish time median "
                f"{np.nanmedian(np.where(self.done, self.finish_time, np.nan)) if self.done.any() else float('nan'):.0f}s "
                f"| recall {np.mean([q['recall'] for q in quality]):.3f} false-free {np.mean([q['false_free'] for q in quality]):.4f}")
        if self.cfg.mode == "compare":
            d = scores.reshape(2, -1)
            diff = d[1] - d[0]
            name_a, name_b = {"noise": ("ideal", "vlp16-noise"),
                              "planner": (self._planner_name(self.planner_spec[0]), self._planner_name(self.planner_spec[-1]))
                              }.get(self._compare_kind, ("default", "params"))
            line += (f"\n[parallel] compare: {name_a} {d[0].mean():.3f} vs {name_b} {d[1].mean():.3f} | paired diff "
                     f"{diff.mean():+.3f} +- {diff.std(ddof=1) / np.sqrt(len(diff)):.3f} "
                     f"(better in {int((diff > 0).sum())}/{len(diff)})")
            self._compare = getattr(self, "_compare", []) + diff.tolist()
            allc = np.array(self._compare)
            se = allc.std(ddof=1) / np.sqrt(len(allc)) if len(allc) > 1 else float("nan")
            line += f"\n[parallel] compare cumulative ({len(allc)} pairs): diff {allc.mean():+.3f} +- {se:.3f}"
            # time / 3D / turning per half (finish time of robots that finished, NaN otherwise)
            ft = np.where(self.done, self.finish_time, np.nan).reshape(2, -1)
            tu = np.rad2deg(self.turn_total).reshape(2, -1)
            parts = [f"done {int(self.done.reshape(2, -1)[i].sum())}/{E // 2} finish {np.nanmean(ft[i]):.0f}s "
                     f"turn {tu[i].mean():.0f}deg" + (f" 3D {cov3d.reshape(2, -1)[i].mean():.3f}" if cov3d is not None else "")
                     for i in range(2)]
            line += f"\n[parallel] compare detail: {name_a}: {parts[0]} | {name_b}: {parts[1]}"
        print(line, flush=True)
        if cand_scores.max() > self.best["score"]:
            self.best = {"score": float(cand_scores.max()), "generation": self.generation,
                         "params": dict(zip(self.names, map(float, self.params[b])))}
        if self.cfg.mode == "search":
            elite = order[: max(2, int(round(C * self.cfg.cem_elite_frac)))]
            z = (self.params[elite] - self._lo) / (self._hi - self._lo)
            a = self.cfg.cem_smoothing
            self.mu = a * z.mean(0) + (1 - a) * self.mu
            self.sigma = np.maximum(a * z.std(0) + (1 - a) * self.sigma, self.cfg.cem_min_std)
            mean_params = dict(zip(self.names, np.round(self._lo + self.mu * (self._hi - self._lo), 3).tolist()))
            print(f"[parallel] next mean: {mean_params}", flush=True)
        with open(os.path.join(self.out_dir, "best.json"), "w") as f:
            json.dump({"best_single": self.best, "generation": self.generation,
                       "mean": dict(zip(self.names, (self._lo + self.mu * (self._hi - self._lo)).tolist())),
                       "std_normalized": dict(zip(self.names, self.sigma.tolist()))}, f, indent=2)
        self._last_gen_summary = line
        self.generation += 1
        if self.generation >= self.cfg.generations and self.cfg.exit_when_done:
            import signal

            print("[parallel] all generations finished", flush=True)
            os.kill(os.getpid(), signal.SIGINT)
            return
        self._start_generation()

    # -- per-step -------------------------------------------------------------------------------------

    def _sim_time(self) -> float:
        return self._env.common_step_counter * self._env.step_dt

    def reset(self, env_ids=None):
        # Isaac Lab resets an env after a fall (base contact); that robot is out for this generation
        out = super().reset(env_ids)
        if not getattr(self, "_pending_start", True) and env_ids is not None:
            for e in (env_ids.tolist() if hasattr(env_ids, "tolist") else list(env_ids)):
                if self._sim_time() - self.t0_env[int(e)] > 0.5:
                    self._finish(int(e), "fell")
        return out

    def _resample_command(self, env_ids):
        self.is_standing_env[env_ids] = False

    def _insert_scans(self, active: np.ndarray):
        sensor = self._env.scene[self.cfg.lidar_name]
        ids = torch.as_tensor(np.flatnonzero(active), device=self.device)
        if len(ids) == 0:
            return
        prof = os.environ.get("GO2_PROFILE") == "1"
        if prof:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        pos, quat = sensor.data.pos_w.torch[ids], sensor.data.quat_w.torch[ids]
        hits = sensor.data.ray_hits_w.torch[ids]
        if prof:
            torch.cuda.synchronize()
            t1 = time.perf_counter()
        off = torch.tensor(sensor.cfg.offset.pos, device=self.device).expand(len(ids), 3)
        origins = pos + quat_apply(quat, off)
        quat = quat_mul(quat, torch.tensor(sensor.cfg.offset.rot, device=self.device).expand(len(ids), 4))  # sensor frame
        if self.noise_cfg is not None and self.noisy[ids.cpu().numpy()].any():
            from .sensor_noise import apply_vlp16_noise

            m = torch.as_tensor(self.noisy[ids.cpu().numpy()], device=self.device)
            n_ids = ids[m]
            v_true = self.robot.data.root_lin_vel_w.torch[n_ids]
            w_true = self.robot.data.root_ang_vel_w.torch[n_ids, 2]
            noisy_hits, n_origins, _ = apply_vlp16_noise(self.noise_cfg, origins[m], quat[m], hits[m], v_true, w_true,
                                                          generator=self._noise_gen)
            if self._deskew:
                from .sensor_noise import deskew

                # odometry/IMU estimate: true velocity + noise (Go2 IMU-grade)
                g = self._noise_gen
                v_est = v_true + 0.05 * torch.randn(v_true.shape, device=self.device, generator=g)
                w_est = w_true + 0.05 * torch.randn(w_true.shape, device=self.device, generator=g)
                noisy_hits = deskew(n_origins, quat[m], noisy_hits, v_est, w_est)
            if self._skip_turn > 0:  # drop scans taken while turning fast
                fast = (w_true.abs() > self._skip_turn)[:, None, None]
                noisy_hits = torch.where(fast, torch.full_like(noisy_hits, float("inf")), noisy_hits)
            hits = hits.clone()
            hits[m] = noisy_hits
            origins = origins.clone()
            origins[m] = n_origins
        self.mapper.insert_scans(origins, hits, ids.tolist())
        if self.live3d is not None:
            self.live3d.insert(ids.tolist(), origins, hits)
            self.gtcov3d.insert(ids.tolist(), hits)
        if prof:
            torch.cuda.synchronize()
            sec = self._prof["sections"]
            sec["  raycast"] = sec.get("  raycast", 0.0) + t1 - t0
            sec["  map_insert"] = sec.get("  map_insert", 0.0) + time.perf_counter() - t1

    def _handle_replies(self, pos: np.ndarray, yaw: np.ndarray, t: float):
        for e, w in enumerate(self.workers):
            for msg in w.poll():
                if msg[0] == "reset":
                    self._worker_ready[e] = True
                elif msg[0] == "plan" and msg[1] == self.req[e] and not self.finished[e]:
                    self.plan_pending[e] = False
                    _, _, wp, done, graph, dt = msg
                    self.graphs[e] = graph
                    self.plan_ms[e].append(1000 * dt)
                    if wp is None and done:
                        self._finish(e, "done")
                    elif wp is not None and self._accept(e, wp, pos[e], yaw[e], t):
                        self.target[e] = wp
                        self.target_set[e] = t
                        self.target_best[e] = np.linalg.norm(wp - pos[e])
                        self._request_path(e, pos[e], t)
                elif msg[0] == "path" and msg[1] == self.req[e] and not self.finished[e]:
                    self.path_pending[e] = False
                    path = msg[2]
                    if path is None:
                        self.target[e], self.paths[e] = np.nan, None
                    else:
                        path[-1] = self.target[e]
                        self.paths[e] = path

    def _accept(self, e, wp, pos, yaw, t) -> bool:
        P = self.P
        if np.isnan(self.target[e, 0]) or self.paths[e] is None:
            return True
        if np.allclose(self.target[e], wp):
            return False
        dist = np.linalg.norm(self.target[e] - pos)
        if dist < P["commit_reach"][e]:
            return True
        d = wp - pos
        if abs((math.atan2(d[1], d[0]) - yaw + math.pi) % (2 * math.pi) - math.pi) < P["switch_heading"][e]:
            return True
        self.target_best[e] = min(self.target_best[e], dist)
        stalled = t - self.target_set[e] > P["stall_s"][e] and dist > self.target_best[e] - 0.1
        return bool(dist > self.target_best[e] + 0.3 or stalled)

    def _request_path(self, e, pos, t):
        self.req[e] += 1
        self.path_pending[e] = True
        self.last_path[e] = t
        self.workers[e].path(self.req[e], self.mapper.grid(e), self.mapper.origin, self.mapper.cell, pos,
                             self.target[e], robot_radius=float(self.P["robot_radius"][e]),
                             clearance=float(self.P["clearance"][e]))

    def _lookahead(self, e, pos):
        p = self.paths[e]
        if len(p) < 2:
            return p[-1], float(np.linalg.norm(p[-1] - pos))
        seg = p[1:] - p[:-1]
        L = np.linalg.norm(seg, axis=1) + 1e-9
        tp = np.clip(np.einsum("ij,ij->i", pos - p[:-1], seg) / L**2, 0, 1)
        k = int(np.argmin(np.linalg.norm(p[:-1] + seg * tp[:, None] - pos, axis=1)))
        remaining = float(L[k] * (1 - tp[k]) + L[k + 1:].sum())
        s = self.P["lookahead"][e] + tp[k] * L[k]
        for j in range(k, len(seg)):
            if s <= L[j]:
                return p[j] + seg[j] / L[j] * s, remaining
            s -= L[j]
        return p[-1], remaining

    def _update_command(self):
        # profiling: time inside this term vs the whole env step (GO2_PROFILE=1)
        if os.environ.get("GO2_PROFILE") == "1":
            now = time.perf_counter()
            prof = self.__dict__.setdefault("_prof", {"ours": 0.0, "total": 0.0, "last": now, "sections": {}})
            prof["total"] += now - prof["last"]
            t0 = now
            self._update_command_impl()
            prof["ours"] += time.perf_counter() - t0
            prof["last"] = time.perf_counter()
            return
        self._update_command_impl()

    def _tic(self, name):
        if os.environ.get("GO2_PROFILE") == "1":
            now = time.perf_counter()
            prof = self.__dict__.setdefault("_prof", {"ours": 0.0, "total": 0.0, "last": now, "sections": {}})
            if "_sec" in prof:
                n, t = prof["_sec"]
                prof["sections"][n] = prof["sections"].get(n, 0.0) + now - t
            prof["_sec"] = (name, now)

    def _update_command_impl(self):
        if self._pending_start:
            self._pending_start = False
            self._start_generation()
        t = self._sim_time()
        E = self.num_envs
        pos = (self.robot.data.root_pos_w.torch[:, :2] - self._env_origins[:, :2]).cpu().numpy().astype(np.float64)
        yaw = self.robot.data.heading_w.torch.cpu().numpy()
        active = ~self.finished
        if self.prev_xy is not None:
            step = np.where(active & ~self._skip_dist, np.linalg.norm(pos - self.prev_xy, axis=1), 0.0)
            self.distance += step
        if self.prev_yaw is not None:
            dyaw = np.abs((yaw - self.prev_yaw + math.pi) % (2 * math.pi) - math.pi)
            self.turn_total += np.where(active & ~self._skip_dist, dyaw, 0.0)
        self.prev_yaw = yaw.copy()
        self._skip_dist[:] = False
        self.prev_xy = pos.copy()
        el = t - self.t0
        el_env = t - self.t0_env  # per-robot episode time (differs from el in loop mode)
        for e in np.flatnonzero(active):
            if el_env[e] > self.cfg.episode_s:
                self._finish(e, "timeout")

        self._tic("scan")
        if t - self._last_scan >= self.cfg.scan_period - 1e-6:
            self._last_scan = t
            self._insert_scans(~self.finished)
        if el - getattr(self, "_last_progress", -1e9) >= 30.0:
            self._last_progress = el
            area = self.mapper.explored_area(None) / self._gt_free_area
            print(f"[parallel] {'loop' if self.cfg.mode == 'loop' else 'gen ' + str(self.generation)} t={el:.0f}s active {int(active.sum())}/{E} "
                  f"coverage mean {area.mean():.2f} max {area.max():.2f} (wall {time.strftime('%H:%M:%S')})", flush=True)
            if "_prof" in self.__dict__ and self._prof["total"] > 0:
                pr = self._prof
                secs = " ".join(f"{k} {v:.1f}s" for k, v in sorted(pr["sections"].items(), key=lambda kv: -kv[1]))
                print(f"[parallel] profile: env step total {pr['total']:.1f}s, command term {pr['ours']:.1f}s | {secs}",
                      flush=True)
        if t - self._last_cov >= 0.5:
            self._last_cov = t
            for e in np.flatnonzero(self._collided(pos) & ~self.finished & (el_env > 1.0)):
                self._finish(e, "collided")
            area = self.mapper.explored_area(None)
            for e in np.flatnonzero(~self.finished):
                self.cov_hist[e].append((el_env[e], area[e] / self._gt_free_area))
                if len(self.traj[e]) == 0 or np.linalg.norm(self.traj[e][-1] - pos[e]) > 0.3:
                    self.traj[e].append(pos[e].copy())

        self._tic("replies")
        self._handle_replies(pos, yaw, t)
        for e in np.flatnonzero(~self.finished & self._worker_ready):
            if not self.plan_pending[e] and not self.path_pending[e] and t - self.last_plan[e] >= self.cfg.replan_period:
                self.last_plan[e] = t
                self.req[e] += 1
                self.plan_pending[e] = True
                extra = None
                if self.planner_spec[e]["kind"] == "dense":
                    extra = {"heading": float(yaw[e]), "util_grid": self.live3d.utility(e, self.mapper.grid(e)),
                             "seen_area": float(self.live3d.seen_area[e])}
                self.workers[e].plan(self.req[e], self.mapper.grid(e, self.cfg.inflate_cells), self.mapper.origin, pos[e],
                                     extra)
            elif (not np.isnan(self.target[e, 0]) and not self.plan_pending[e] and not self.path_pending[e]
                  and t - self.last_path[e] >= self.cfg.path_period):
                self._request_path(e, pos[e], t)

        self._tic("ui")
        if self._ui is not None and t - self._last_ui >= self.cfg.ui_period:
            self._last_ui = t
            self._update_ui()
            capture = os.environ.get("GO2_UI_CAPTURE")  # one screenshot of the whole app window (verification)
            if capture and not getattr(self, "_captured", False) and el >= float(os.environ.get("GO2_UI_CAPTURE_T", "30")):
                self._captured = True
                import omni.kit.renderer_capture as rc

                rc.acquire_renderer_capture_interface().capture_next_frame_swapchain(capture)
                print(f"[parallel] app window capture -> {capture}", flush=True)

        if self.cfg.mode == "loop":
            for e in np.flatnonzero(self.finished & (t - self.finish_abs >= self.cfg.respawn_delay)):
                self._log_episode(e)
                self._restart_robot(e, t)
        elif self.finished.all():
            self._end_generation()
            self.vel_command_b[:] = 0.0
            return

        self._hide_visual_clutter()
        tiles_dir = os.environ.get("GO2_SAVE_TILES")
        if tiles_dir and t - getattr(self, "_last_tiles", -1e9) >= 1.0:
            self._last_tiles = t
            os.makedirs(tiles_dir, exist_ok=True)
            self._render_tiles().save(os.path.join(tiles_dir, f"tiles_{t - self.t0:07.1f}s.png"))
        self._tic("drive")
        if self._ui is not None:
            cov_now = np.array([h[-1][1] if h else 0.0 for h in self.cov_hist])
            self._update_camera(pos, yaw, cov_now)
        # turn-then-go towards each robot's lookahead point
        carrots = pos.copy()
        remaining = np.zeros(E)
        drive = np.zeros(E, bool)
        for e in range(E):
            if self.finished[e] or self.paths[e] is None or np.isnan(self.target[e, 0]):
                continue
            carrots[e], remaining[e] = self._lookahead(e, pos[e])
            drive[e] = remaining[e] >= self.cfg.reach_radius
        self._drive_to(torch.as_tensor(carrots, dtype=torch.float32, device=self.device),
                       speed_dist=torch.as_tensor(remaining, dtype=torch.float32, device=self.device))
        idle = torch.as_tensor(~drive, device=self.device)
        self.vel_command_b[idle] = 0.0
        self.turning[idle] = True
        self._tic("end")

    # -- GUI: tiled maps + leaderboard ----------------------------------------------------------------

    def _init_ui(self):
        """GUI only: the panel itself is created lazily on the first refresh, after Kit has settled its layout."""
        self._ui = None
        try:
            from isaaclab.app import AppLauncher

            if AppLauncher.has_gui():
                self._ui = {}
        except Exception:  # noqa: BLE001
            pass

    # -- GUI camera: overview / follow a robot -----------------------------------------------------------

    def _set_camera(self, mode: str, robot: int | None = None):
        self._cam = {"mode": mode, "robot": robot if robot is not None else getattr(self, "_cam", {}).get("robot", 0),
                     "eye": None}
        if mode == "overview":
            b = self.cfg.gt_bounds
            cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
            self._env.sim.set_camera_view(eye=(cx, b[1] - 6.0, 18.0), target=(cx, cy - 4.0, 0.0))

    def _step_camera(self, d: int):
        cur = getattr(self, "_cam", {}).get("robot", 0)
        self._set_camera("robot", (cur + d) % self.num_envs)

    def _update_camera(self, pos: np.ndarray, yaw: np.ndarray, cov: np.ndarray | None = None):
        cam = getattr(self, "_cam", None)
        if cam is None or cam["mode"] == "overview":
            return
        if cam["mode"] == "leader" and cov is not None:
            cam["robot"] = int(np.argmax(cov))
        e = cam["robot"]
        if getattr(self, "_cam_models", None) is not None:
            back, up = (m.get_value_as_float() for m in self._cam_models)
        else:
            back, up = self.cfg.follow_offset
        target = np.array([pos[e, 0], pos[e, 1], 0.3])
        eye = target + np.array([-back * math.cos(yaw[e]), -back * math.sin(yaw[e]), up])
        cam["eye"] = eye if cam["eye"] is None else cam["eye"] + 0.1 * (eye - cam["eye"])  # smooth
        self._env.sim.set_camera_view(eye=tuple(cam["eye"]), target=tuple(target))

    def _camera_desc(self) -> str:
        cam = getattr(self, "_cam", {"mode": "overview"})
        return "overview" if cam["mode"] == "overview" else f"following #{cam['robot']}" + (
            " (leader)" if cam["mode"] == "leader" else "")

    def _hide_visual_clutter(self):
        """Hide the warehouse ceiling and the physics-only plane (visual only; LiDAR uses its own mesh)."""
        if getattr(self, "_clutter_hidden", False):
            return
        self._clutter_hidden = True
        import omni.usd
        from pxr import Usd, UsdGeom

        stage = omni.usd.get_context().get_stage()
        for prim in stage.Traverse():
            if prim.GetName().startswith("SM_Ceiling"):
                UsdGeom.Imageable(prim).MakeInvisible()
        ground = stage.GetPrimAtPath("/World/ground")
        if stage.GetPrimAtPath("/World/Warehouse").IsValid() and ground.IsValid():
            for prim in Usd.PrimRange(ground):  # z-fights with the warehouse floor
                if prim.IsA(UsdGeom.Imageable):
                    UsdGeom.Imageable(prim).MakeInvisible()

    def _ensure_window(self):
        if self._ui.get("provider") is not None:
            return
        import omni.ui as ui
        import omni.usd
        from pxr import Usd, UsdGeom

        self._hide_visual_clutter()

        provider = ui.ByteImageProvider()
        window = ui.Window("Parallel Exploration", width=600, height=900)
        with window.frame:
            with ui.VStack(spacing=4):
                with ui.HStack(height=28, spacing=4):
                    ui.Label("camera:", width=60)
                    ui.Button("Overview", clicked_fn=lambda: self._set_camera("overview"))
                    ui.Button("Follow leader", clicked_fn=lambda: self._set_camera("leader"))
                    ui.Button("<", width=30, clicked_fn=lambda: self._step_camera(-1))
                    ui.Button(">", width=30, clicked_fn=lambda: self._step_camera(+1))
                back0, up0 = self.cfg.follow_offset
                with ui.HStack(height=22, spacing=6):
                    ui.Label("follow distance [m]", width=130)
                    back_model = ui.FloatSlider(min=1.0, max=25.0, step=0.5).model
                    back_model.set_value(back0)
                with ui.HStack(height=22, spacing=6):
                    ui.Label("follow height [m]", width=130)
                    up_model = ui.FloatSlider(min=0.5, max=30.0, step=0.5).model
                    up_model.set_value(up0)
                self._cam_models = (back_model, up_model)
                label = ui.Label("", height=60, word_wrap=True)
                ui.ImageWithProvider(provider, fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)
        window.deferred_dock_in("Stage", ui.DockPolicy.CURRENT_WINDOW_IS_ACTIVE)
        self._ui.update({"window": window, "provider": provider, "label": label})
        mode = os.environ.get("GO2_CAM", "overview")
        if mode.isdigit():
            self._set_camera("robot", int(mode) % self.num_envs)
        else:
            self._set_camera(mode)

    def _update_ui(self):
        self._ensure_window()
        canvas = self._render_tiles()
        rgba = np.asarray(canvas.convert("RGBA"), dtype=np.uint8)
        self._ui["provider"].set_data_array(rgba, [rgba.shape[1], rgba.shape[0]])
        self._update_label()

    def _render_tiles(self):
        """Mosaic of every robot's own map (+ path, trajectory, coverage, state); leader framed green."""
        from PIL import Image, ImageDraw

        E = self.num_envs
        el = self._sim_time() - self.t0
        scores = np.array([self._score(e) for e in range(E)])
        cov = np.array([h[-1][1] if h else 0.0 for h in self.cov_hist])
        lead = int(np.argmax(cov))
        b = self.cfg.gt_bounds
        c = self.mapper.cell
        r0, r1 = int((b[1] - self.mapper.origin[1]) / c), int((b[3] - self.mapper.origin[1]) / c)
        c0, c1 = int((b[0] - self.mapper.origin[0]) / c), int((b[2] - self.mapper.origin[0]) / c)
        cols = int(math.ceil(math.sqrt(E)))
        rows = int(math.ceil(E / cols))
        tw, th = (c1 - c0) * 2, (r1 - r0) * 2
        canvas = Image.new("RGB", (cols * (tw + 6), rows * (th + 18)), (40, 40, 40))
        for e in range(E):
            g = self.mapper.grid(e)[r0:r1, c0:c1]
            img = np.full(g.shape + (3,), 150, np.uint8)
            img[g == FREE] = 255
            img[g == OCCUPIED] = 30
            tile = Image.fromarray(img).resize((tw, th), Image.NEAREST)
            dr = ImageDraw.Draw(tile)
            px = lambda p: ((p[0] - b[0]) / c * 2, (p[1] - b[1]) / c * 2)  # noqa: E731
            if len(self.traj[e]) > 1:
                dr.line([px(p) for p in self.traj[e]], fill=(30, 90, 255), width=1)
            if self.paths[e] is not None:
                dr.line([px(p) for p in self.paths[e]], fill=(0, 200, 220), width=2)
            tile = tile.transpose(Image.FLIP_TOP_BOTTOM)
            x, y = (e % cols) * (tw + 6) + 3, (e // cols) * (th + 18) + 15
            canvas.paste(tile, (x, y))
            ep = f" ep{int(self.episode_idx[e]) + 1}" if self.cfg.mode == "loop" else ""
            state = ep + " " + ("fell" if self.fell[e] else "hit" if self.collided[e] else ("done" if self.done[e] else ("end" if self.finished[e] else "")))
            followed = getattr(self, "_cam", {}).get("mode") in ("robot", "leader") and self._cam.get("robot") == e
            col = (90, 230, 90) if e == lead else (255, 200, 0) if followed else (220, 220, 220)
            ImageDraw.Draw(canvas).text((x, y - 13), f"#{e} {cov[e]:.0%} {state}", fill=col)
            if e == lead or followed:
                ImageDraw.Draw(canvas).rectangle([x - 2, y - 2, x + tw + 1, y + th + 1],
                                                 outline=(255, 200, 0) if followed else (90, 230, 90), width=2)
        return canvas

    def _update_label(self):
        E = self.num_envs
        el = self._sim_time() - self.t0
        scores = np.array([self._score(e) for e in range(E)])
        top = np.argsort(-scores)[:3]
        if self.cfg.mode == "loop":
            stats = getattr(self, "stats", [])
            done_t = [t for r, t, _, _ in stats if r == "done"]
            covs = np.array([h[-1][1] if h else 0.0 for h in self.cov_hist])
            self._ui["label"].text = (
                f"loop  t = {el:.0f} s  episodes finished {len(stats)}  "
                f"(done {len(done_t) / max(len(stats), 1):.0%}, median {np.median(done_t) if done_t else float('nan'):.0f} s)  "
                f"camera: {self._camera_desc()}\n"
                f"exploring {int((~self.finished).sum())}/{E}  live coverage mean {covs.mean():.0%}  "
                f"respawn after finishing ({self.cfg.respawn_delay:.0f} s pause)")
            return
        self._ui["label"].text = (
            f"gen {self.generation} ({self.cfg.mode})  t = {el:.0f}/{self.cfg.episode_s:.0f} s  "
            f"active {int((~self.finished).sum())}/{E}  best so far {self.best['score']:.3f}  "
            f"camera: {self._camera_desc()}\n"
            f"leaders (score): " + "  ".join(f"#{e} {scores[e]:.3f}" for e in top) + "\n"
            + getattr(self, "_last_gen_summary", ""))


@configclass
class ParallelExplorationCommandCfg(WaypointTurnThenGoCommandCfg):
    class_type: type = ParallelExplorationCommand

    waypoints: list[tuple[float, float]] = [(0.0, 0.0)]  # unused; targets come from ARiADNE

    respawn_delay: float = 3.0
    """loop mode: seconds a finished robot keeps showing its final map before respawning."""
    mode: str = "search"
    """"search" (CEM over PARAM_SPACE), "eval" (one parameter set, random starts), "compare"
    (defaults vs GO2_PAR_PARAMS on paired starts) or "loop" (continuous per-robot respawn)."""
    starts_per_candidate: int = 1
    """search: shared start poses per candidate; candidates = num_envs / starts_per_candidate."""
    generations: int = 8
    exit_when_done: bool = True
    episode_s: float = 180.0
    seed: int = 0

    cem_init_std: float = 0.25
    cem_elite_frac: float = 0.25
    cem_smoothing: float = 0.7
    cem_min_std: float = 0.05

    lidar_name: str = "lidar"
    map_bounds: tuple[float, float, float, float] = MISSING
    cell_size: float = 0.4
    sensor_range: float = 20.0
    obstacle_z: tuple[float, float] = (0.15, 1.2)
    inflate_cells: int = 1
    scan_period: float = 0.1
    replan_period: float = 0.4
    path_period: float = 1.0
    warmup_s: float = 1.0

    gt_map: str = MISSING
    gt_bounds: tuple[float, float, float, float] = MISSING
    spawn_clearance: float = 1.0
    collision_radius: float = 0.25
    """A robot whose centre gets closer than this [m] to a GT obstacle counts as collided (fast-physics mode)."""
    out_dir: str = MISSING
    ui_period: float = 1.0
    follow_offset: tuple[float, float] = (6.0, 5.0)
    """Follow camera: distance behind and height above the robot [m]."""
