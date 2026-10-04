"""Autonomous exploration command: VLP-16 -> 2D occupancy map -> ARiADNE waypoint -> A* path -> turn-then-go.

Runs inside the Isaac Lab env as the ``base_velocity`` command term (single environment):
  * every ``scan_period`` s the LiDAR scan is inserted into :class:`OccupancyMapper2D`
  * every ``replan_period`` s ARiADNE replans on the (inflated) map in a background thread
  * a committed waypoint is reached along a grid A* path (clearance-aware), followed with a lookahead point
  * the walking policy receives turn-then-go velocity commands
  * a live "ARiADNE Map" panel (GUI), snapshots, and an exploration log are written to ``out_dir``

Environment variables (all optional):
  GO2_EXPLORE_MAX_S=<s>   headless: stop the app after completion or this sim time
  GO2_RESTART_AT=<s>      test hook: press "Restart exploration" at this sim time
  GO2_UI_CAPTURE=<png>    save one screenshot of the whole app window (at GO2_UI_CAPTURE_T, default 30 s)
"""

from __future__ import annotations

import csv
import os
import signal
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import MISSING

import numpy as np
import torch

from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

from ..waypoint_command import WaypointTurnThenGoCommand, WaypointTurnThenGoCommandCfg
from .occupancy_mapper import FREE, OCCUPIED, OccupancyMapper2D

# The ARiADNE core and the A* cost map pull in scipy/scikit-image (OpenBLAS). Importing them before Kit starts
# lets OpenBLAS's fork handler crash Kit's telemetry fork, so they are imported lazily inside the command term.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


class AriadneExplorationCommand(WaypointTurnThenGoCommand):
    cfg: AriadneExplorationCommandCfg

    def __init__(self, cfg: AriadneExplorationCommandCfg, env):
        super().__init__(cfg, env)
        assert self.num_envs == 1, "exploration runs with a single environment"
        self._env = env
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ariadne")
        self.out_dir = os.path.join(cfg.out_dir, time.strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(self.out_dir, exist_ok=True)
        self._log = open(os.path.join(self.out_dir, "exploration_log.csv"), "w", newline="")
        self._csv = csv.writer(self._log)
        self._csv.writerow(["time_s", "explored_m2", "coverage", "distance_m", "x", "y", "waypoint_x", "waypoint_y",
                            "plan_ms", "path_m"])
        self._gt_free_area = self._load_gt_free_area(cfg.gt_map, cfg.gt_bounds)
        self._max_s = float(os.environ.get("GO2_EXPLORE_MAX_S", "0")) or None
        self._reset_state()
        self._init_ui()
        print(f"[exploration] logging to {self.out_dir}", flush=True)

    def _reset_state(self):
        """(Re)initialize map, planner and bookkeeping; used at start and by "Restart exploration"."""
        from .ariadne_planner import AriadnePlanner

        cfg = self.cfg
        self.mapper = OccupancyMapper2D(bounds=cfg.map_bounds, cell_size=cfg.cell_size, max_range=cfg.sensor_range,
                                        obstacle_z=cfg.obstacle_z, device=str(self.device))
        self.planner = AriadnePlanner(SENSOR_RANGE=cfg.sensor_range, CELL_SIZE=cfg.cell_size, **cfg.planner_overrides)
        self._generation = getattr(self, "_generation", 0) + 1
        self._plan_future: Future | None = None
        self._graph = (np.zeros((0, 2)), np.zeros(0), [], np.zeros((0, 2)))
        self.target: torch.Tensor | None = None
        self.path: np.ndarray | None = None
        self.done = False
        self._done_time: float | None = None
        self._t0 = self._sim_time()
        self._last_scan = self._last_snap = self._last_ui = self._last_path = -1e9
        self._last_plan = self._t0 + cfg.warmup_s - cfg.replan_period
        self._target_set_time, self._target_best_dist = 0.0, float("inf")
        self.trajectory: list[np.ndarray] = []
        self.distance = 0.0
        self.plan_times: list[float] = []
        self.milestones: dict[float, tuple[float, float]] = {}
        self._restart_requested = False

    # -- helpers --------------------------------------------------------------------------------------

    @staticmethod
    def _load_gt_free_area(path: str | None, bounds) -> float | None:
        """Free area [m^2] of the ground-truth map (tools/usd_occupancy_map.py), optionally inside bounds."""
        if not path or not os.path.exists(path):
            return None
        d = np.load(path)
        grid, origin, res = d["grid"], d["origin"], float(d["resolution"])  # grid indexed [x, y]
        free = grid == 255
        if bounds is not None:
            xs = origin[0] + (np.arange(grid.shape[0]) + 0.5) * res
            ys = origin[1] + (np.arange(grid.shape[1]) + 0.5) * res
            free &= ((xs >= bounds[0]) & (xs <= bounds[2]))[:, None] & ((ys >= bounds[1]) & (ys <= bounds[3]))[None, :]
        return float(free.sum()) * res**2

    def _sim_time(self) -> float:
        return self._env.common_step_counter * self._env.step_dt

    def _elapsed(self) -> float:
        return self._sim_time() - self._t0

    def _coverage(self) -> float:
        return self.mapper.explored_area() / self._gt_free_area if self._gt_free_area else float("nan")

    def _insert_scan(self):
        sensor = self._env.scene[self.cfg.lidar_name]
        link_pos, link_quat = sensor.data.pos_w.torch[0], sensor.data.quat_w.torch[0]
        offset = torch.tensor([sensor.cfg.offset.pos], device=link_pos.device)
        self.mapper.insert_scan(link_pos + quat_apply(link_quat[None], offset)[0], sensor.data.ray_hits_w.torch[0])

    # -- ARiADNE (background thread) ------------------------------------------------------------------

    def _plan_job(self, generation: int, grid: np.ndarray, origin: np.ndarray, pos_xy: np.ndarray):
        t0 = time.time()
        self.planner.update_map(grid, origin)
        self.planner.update_location(pos_xy)
        wp = self.planner.plan()
        return generation, wp, self.planner.done, self.planner.graph(), time.time() - t0

    def _poll_planner(self, pos_xy: np.ndarray):
        t = self._sim_time()
        if self._plan_future is not None and self._plan_future.done():
            generation, wp, done, graph, dt = self._plan_future.result()
            self._plan_future = None
            if generation == self._generation:
                self._graph = graph
                self.plan_times.append(dt)
                if wp is None and done:
                    self.done, self._done_time, self.target, self.path = True, t, None, None
                    print(f"[exploration] completed at t={self._elapsed():.1f}s, explored {self.mapper.explored_area():.0f}"
                          f" m2, distance {self.distance:.1f} m", flush=True)
                    self._snapshot(final=True)
                elif wp is not None and self._accept_waypoint(np.asarray(wp, dtype=np.float64), pos_xy):
                    self.target = torch.as_tensor(np.asarray(wp, dtype=np.float32)[None], device=self.device)
                    self._target_set_time = t
                    self._target_best_dist = float(np.linalg.norm(np.asarray(wp) - pos_xy))
                    self._plan_path(pos_xy)
                self._log_row(pos_xy)
        if not self.done and self._plan_future is None and t - self._last_plan >= self.cfg.replan_period - 1e-6:
            self._last_plan = t
            grid = self.mapper.grid(inflate_cells=self.cfg.inflate_cells)
            self._plan_future = self._executor.submit(self._plan_job, self._generation, grid, self.mapper.origin.copy(),
                                                      pos_xy.copy())

    def _accept_waypoint(self, wp: np.ndarray, pos_xy: np.ndarray) -> bool:
        """Commit to the current target unless switching is cheap or necessary.

        ARiADNE replans at 2.5 Hz; the original stack smooths this with a local planner. A turn-then-go
        follower would stop and re-turn on every change, so a new waypoint is taken only when:
          1. there is no target yet, or the current one is reached (within THR_TO_WAYPOINT),
          2. the new waypoint lies within ``switch_heading`` of the current walking direction,
          3. the current target became an obstacle, has no A* path, or progress towards it stalled.
        """
        if self.target is None:
            return True
        cur = self.target[0].cpu().numpy().astype(np.float64)
        if np.allclose(cur, wp):
            return False
        dist = float(np.linalg.norm(cur - pos_xy))
        if dist < self.cfg.commit_reach or self.path is None:
            return True
        yaw = float(self.robot.data.heading_w.torch[0])
        d = wp - pos_xy
        if abs((np.arctan2(d[1], d[0]) - yaw + np.pi) % (2 * np.pi) - np.pi) < self.cfg.switch_heading:
            return True
        cell = np.floor((cur - self.mapper.origin) / self.mapper.cell).astype(int)
        if self.mapper.grid(inflate_cells=self.cfg.inflate_cells)[cell[1], cell[0]] == OCCUPIED:
            return True
        self._target_best_dist = min(self._target_best_dist, dist)
        no_progress = self._sim_time() - self._target_set_time > self.cfg.stall_s and dist > self._target_best_dist - 0.1
        return bool(dist > self._target_best_dist + 0.3 or no_progress)

    # -- A* path following ----------------------------------------------------------------------------

    def _plan_path(self, pos_xy: np.ndarray):
        from .astar import plan_path

        self._last_path = self._sim_time()
        if self.target is None:
            self.path = None
            return
        goal = self.target[0].cpu().numpy()
        path, _ = plan_path(self.mapper.grid(), self.mapper.origin, self.mapper.cell, pos_xy, goal,
                            robot_radius=self.cfg.robot_radius, clearance=self.cfg.clearance)
        if path is None:
            self.path = None
            self.target = None  # unreachable on the current map: let ARiADNE pick again
            return
        path[-1] = goal
        self.path = path

    def _lookahead(self, pos_xy: np.ndarray) -> tuple[np.ndarray, float]:
        """Point ``lookahead`` metres ahead of the robot's projection on the path, and the remaining length."""
        p = self.path
        if len(p) < 2:
            return p[-1], float(np.linalg.norm(p[-1] - pos_xy))
        seg = p[1:] - p[:-1]
        seg_len = np.linalg.norm(seg, axis=1) + 1e-9
        tproj = np.clip(np.einsum("ij,ij->i", pos_xy - p[:-1], seg) / seg_len**2, 0, 1)
        proj = p[:-1] + seg * tproj[:, None]
        k = int(np.argmin(np.linalg.norm(proj - pos_xy, axis=1)))
        remaining = float(seg_len[k] * (1 - tproj[k]) + seg_len[k + 1:].sum())
        s = self.cfg.lookahead + tproj[k] * seg_len[k]
        for j in range(k, len(seg)):
            if s <= seg_len[j]:
                return p[j] + seg[j] / seg_len[j] * s, remaining
            s -= seg_len[j]
        return p[-1], remaining

    # -- logging / visualization ----------------------------------------------------------------------

    def _log_row(self, pos_xy):
        cov = self._coverage()
        for m in self.cfg.coverage_milestones:
            if m not in self.milestones and cov >= m:
                self.milestones[m] = (self._elapsed(), self.distance)
        wp = self.target[0].tolist() if self.target is not None else [float("nan")] * 2
        path_len = float(np.linalg.norm(np.diff(self.path, axis=0), axis=1).sum()) if self.path is not None else float("nan")
        self._csv.writerow([f"{self._elapsed():.2f}", f"{self.mapper.explored_area():.1f}", f"{cov:.4f}",
                            f"{self.distance:.2f}", *np.round(pos_xy, 2), *np.round(wp, 2),
                            f"{1000 * self.plan_times[-1]:.1f}" if self.plan_times else "", f"{path_len:.2f}"])
        self._log.flush()

    def summary(self) -> str:
        ms = "  ".join(f"{int(100 * m)}%@{t:.0f}s/{d:.0f}m" for m, (t, d) in sorted(self.milestones.items()))
        plan = f"{1000 * np.mean(self.plan_times):.0f}ms" if self.plan_times else "-"
        return (f"t={self._elapsed():.1f}s done={self.done} explored={self.mapper.explored_area():.0f}m2 "
                f"coverage={self._coverage():.1%} distance={self.distance:.1f}m plan={plan}  milestones: {ms}")

    def _snapshot(self, final: bool = False):
        im = self._render_map()
        name = "final.png" if final else f"map_{self._elapsed():07.1f}s.png"
        im.save(os.path.join(self.out_dir, name))
        im.save(os.path.join(self.out_dir, "latest.png"))
        if final:
            np.savez_compressed(os.path.join(self.out_dir, "final_map.npz"), grid=self.mapper.grid(),
                                origin=self.mapper.origin, cell=self.mapper.cell, trajectory=np.array(self.trajectory))
        if self._ui is not None:
            self._update_ui(im)

    def _render_map(self):
        from PIL import Image, ImageDraw

        g = self.mapper.grid()
        scale, c = self.cfg.snapshot_scale, self.mapper.cell
        img = np.full(g.shape + (3,), 150, np.uint8)
        img[g == FREE] = 255
        img[g == OCCUPIED] = 30
        im = Image.fromarray(img).resize((g.shape[1] * scale, g.shape[0] * scale), Image.NEAREST)
        draw = ImageDraw.Draw(im)

        def px(p):
            return ((p[0] - self.mapper.origin[0]) / c * scale, (p[1] - self.mapper.origin[1]) / c * scale)

        nodes, util, edges, frontier = self._graph
        for a, b in edges:
            draw.line([px(a), px(b)], fill=(150, 210, 150), width=1)
        for f in frontier:
            x, y = px(f)
            draw.rectangle([x - 1, y - 1, x + 1, y + 1], fill=(230, 60, 60))
        for n, u in zip(nodes, util):
            x, y = px(n)
            draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill=(0, 150, 0) if u > 0 else (90, 90, 90))
        if len(self.trajectory) > 1:
            draw.line([px(p) for p in self.trajectory[::5]] + [px(self.trajectory[-1])], fill=(30, 90, 255), width=2)
        if self.path is not None:
            draw.line([px(p) for p in self.path], fill=(0, 200, 220), width=3)
        if self.target is not None:
            x, y = px(self.target[0].cpu().numpy())
            draw.ellipse([x - 5, y - 5, x + 5, y + 5], outline=(255, 140, 0), width=3)
        if self.trajectory:
            x, y = px(self.trajectory[-1])
            draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(30, 90, 255))
        obs = np.argwhere(g != -1)  # crop to the observed area (+ margin) so the map fills the panel
        if len(obs):
            m = int(self.cfg.view_margin / c)
            r0, c0 = np.maximum(obs.min(0) - m, 0)
            r1, c1 = np.minimum(obs.max(0) + m + 1, g.shape)
            im = im.crop((c0 * scale, r0 * scale, c1 * scale, r1 * scale))
        return im.transpose(Image.FLIP_TOP_BOTTOM)  # y up

    # -- live 2D map panel inside the Isaac Sim GUI ---------------------------------------------------

    def _init_ui(self):
        self._ui = None
        try:
            from isaaclab.app import AppLauncher

            if not AppLauncher.has_gui():
                return
            import omni.ui as ui
        except Exception:  # noqa: BLE001
            return
        provider = ui.ByteImageProvider()
        window = ui.Window("ARiADNE Map", width=560, height=900)
        with window.frame:
            with ui.VStack(spacing=4):
                with ui.HStack(height=40):
                    label = ui.Label("waiting for first scan...", word_wrap=True)
                    ui.Button("Restart exploration", width=150, clicked_fn=self._request_restart)
                ui.Label("white free | black occupied | grey unknown | red frontier | green node (utility) | "
                         "blue trajectory | cyan A* path | orange target",
                         height=30, word_wrap=True, style={"font_size": 13, "color": 0xFF999999})
                ui.ImageWithProvider(provider, fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT)
        try:  # tab next to the Stage panel so the 3D viewport stays unobstructed
            window.deferred_dock_in("Stage", ui.DockPolicy.CURRENT_WINDOW_IS_ACTIVE)
        except Exception:  # noqa: BLE001
            pass
        self._ui = {"window": window, "provider": provider, "label": label}

    def _request_restart(self):
        self._restart_requested = True

    def _status(self) -> str:
        cov = f"{self._coverage():.0%}" if self._gt_free_area else "n/a"
        plan = f"{1000 * self.plan_times[-1]:.0f} ms" if self.plan_times else "-"
        state = "DONE" if self.done else ("turning" if bool(self.turning[0]) else "walking")
        return (f"t = {self._elapsed():.1f} s   [{state}]\n"
                f"explored {self.mapper.explored_area():.0f} m2 ({cov} of GT)   distance {self.distance:.1f} m   plan {plan}")

    def _update_ui(self, im=None):
        im = im if im is not None else self._render_map()
        rgba = np.asarray(im.convert("RGBA"), dtype=np.uint8)
        self._ui["provider"].set_data_array(rgba, [rgba.shape[1], rgba.shape[0]])
        self._ui["label"].text = self._status()

    # -- hooks ----------------------------------------------------------------------------------------

    def _hooks(self):
        if os.environ.get("GO2_HIDE_CEILING") == "1" and not getattr(self, "_ceiling_hidden", False):
            self._ceiling_hidden = True  # visual only (LiDAR rays hit the baked raycast mesh): lets a high camera see in
            import omni.usd
            from pxr import UsdGeom

            for prim in omni.usd.get_context().get_stage().Traverse():
                if prim.GetName().startswith("SM_Ceiling"):
                    UsdGeom.Imageable(prim).MakeInvisible()
        restart_at = os.environ.get("GO2_RESTART_AT")
        if restart_at and not getattr(self, "_restart_hook_fired", False) and self._sim_time() >= float(restart_at):
            self._restart_hook_fired = True
            self._restart_requested = True
        if self._restart_requested:
            self._reset_state()
            print(f"[exploration] restarted at t={self._sim_time():.1f}s", flush=True)
        capture = os.environ.get("GO2_UI_CAPTURE")
        if capture and self._ui is not None and not getattr(self, "_captured", False) \
                and self._sim_time() >= float(os.environ.get("GO2_UI_CAPTURE_T", "30")):
            self._captured = True
            try:
                import omni.kit.renderer_capture as rc

                rc.acquire_renderer_capture_interface().capture_next_frame_swapchain(capture)
                print(f"[exploration] app window capture -> {capture}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[exploration] app window capture failed: {e}", flush=True)
        if self._max_s is not None and ((self._done_time is not None and self._sim_time() - self._done_time > 2.0)
                                        or self._elapsed() > self._max_s):
            print(f"[exploration] summary: {self.summary()}", flush=True)
            self._snapshot(final=True)
            self._max_s = None
            os.kill(os.getpid(), signal.SIGINT)

    # -- command term ---------------------------------------------------------------------------------

    def _update_command(self):
        self._hooks()
        t = self._sim_time()
        pos = (self.robot.data.root_pos_w.torch[0, :2] - self._env_origins[0, :2]).cpu().numpy()
        if self.trajectory:
            self.distance += float(np.linalg.norm(pos - self.trajectory[-1]))
        self.trajectory.append(pos.copy())

        if t - self._last_scan >= self.cfg.scan_period - 1e-6:
            self._last_scan = t
            self._insert_scan()
        self._poll_planner(pos)
        if self.target is not None and t - self._last_path >= self.cfg.path_period:
            self._plan_path(pos)  # refresh A* as the map grows

        if t - self._last_snap >= self.cfg.snapshot_period:
            self._last_snap = self._last_ui = t
            self._snapshot()
        elif self._ui is not None and t - self._last_ui >= self.cfg.ui_period:
            self._last_ui = t
            self._update_ui()

        if self.target is None or self.path is None or self.done:
            self.vel_command_b[:] = 0.0
            self.turning[:] = True
            return
        carrot, remaining = self._lookahead(pos)
        if remaining < self.cfg.reach_radius:
            self.vel_command_b[:] = 0.0  # arrived: wait for the next waypoint
            self.turning[:] = True
            return
        self._drive_to(torch.as_tensor(carrot[None], dtype=torch.float32, device=self.device),
                       speed_dist=torch.tensor([remaining], dtype=torch.float32, device=self.device))


@configclass
class AriadneExplorationCommandCfg(WaypointTurnThenGoCommandCfg):
    class_type: type = AriadneExplorationCommand

    waypoints: list[tuple[float, float]] = [(0.0, 0.0)]  # unused; targets come from ARiADNE

    lidar_name: str = "lidar"
    map_bounds: tuple[float, float, float, float] = MISSING
    """(x_min, y_min, x_max, y_max) [m] of the mapped area (env frame)."""
    planner_overrides: dict = {}
    """Extra ARiADNE ``parameter`` overrides, e.g. {"THR_NEXT_WAYPOINT": 4.4}."""
    cell_size: float = 0.4
    sensor_range: float = 20.0
    obstacle_z: tuple[float, float] = (0.15, 1.2)
    inflate_cells: int = 1
    """Obstacle inflation [cells] of the map handed to ARiADNE."""
    scan_period: float = 0.1
    replan_period: float = 0.4
    warmup_s: float = 1.0
    """Let the robot settle and collect a few scans before the first plan."""

    commit_reach: float = 2.0
    """Distance [m] to the current target below which a new waypoint is always accepted (THR_TO_WAYPOINT)."""
    switch_heading: float = 0.785
    """New waypoints within this heading error [rad] of the robot's yaw replace the target without turning."""
    stall_s: float = 4.0
    """Replace the target if no progress was made towards it for this long [s]."""

    robot_radius: float = 0.35
    """A* keeps at least this distance [m] from obstacles (Go2 half-width + margin)."""
    clearance: float = 1.0
    """Within this distance [m] of obstacles A* adds a cost, keeping the robot mid-aisle."""
    lookahead: float = 1.2
    """Path-following lookahead [m]."""
    path_period: float = 1.0
    """A* refresh period [s] while following a waypoint."""

    out_dir: str = MISSING
    gt_map: str | None = None
    gt_bounds: tuple[float, float, float, float] | None = None
    """Region (x_min, y_min, x_max, y_max) [m] of the ground-truth map that counts as explorable."""
    coverage_milestones: tuple[float, ...] = (0.5, 0.75, 0.9, 0.95, 0.99)
    snapshot_period: float = 2.0
    snapshot_scale: int = 4
    view_margin: float = 3.0
    """Margin [m] kept around the observed area when rendering the map."""
    ui_period: float = 0.5
    """Refresh period [s] of the live map panel (GUI only)."""
