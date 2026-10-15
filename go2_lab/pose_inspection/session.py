"""Engine-agnostic experiment session: N robots (one per seed) in one scene, each with its own map and planner.

The engine (kinematic simulator or Isaac Lab) owns the robot motion and calls
  ``scan(t, base_xyz, R_body)``   every LiDAR period: cast the VLP-16, label returns, update maps and coverage
  ``control(t, poses, postures)`` every control step: one command per robot (agent.InspectionAgent)
  ``log(t, poses)``               every step: path length, coverage curves (log period), termination
  ``save()``                      per-run JSON (curves, events, summary) under ``out_dir``
"""

from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import torch

from .agent import InspectionAgent
from .mapping import WorldModel
from .metrics import SurfaceCoverage, Termination
from .raycast import SceneMesh, mount, sensor_poses, vlp16_dirs
from .upper_bound import posture_grid
from .viewpoints import ViewpointScorer


def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.tolist()
    raise TypeError(type(o).__name__)


class Session:
    def __init__(self, cfg, scene_meta: dict, seeds: list[int], out_dir: str, device: str = "cuda:0",
                 region_fn=None, pruner_factory=None, perception=None, tag: str = "", class_fn=None,
                 vlm_hint_fn=None, selector_factory=None):
        self.cfg, self.meta, self.seeds, self.dev = cfg, scene_meta, list(seeds), device
        self.E = len(seeds)
        self.out_dir, self.tag = out_dir, tag
        d = scene_meta["dir"]
        li = cfg.lidar
        self.mesh = SceneMesh(os.path.join(d, "gt.npz"), device)
        n_hyp = cfg.perception.gsam2.max_objects if cfg.perception.mode != "gt" else None
        self.world = WorldModel(cfg, scene_meta, self.E, device, n_obj=n_hyp)
        targets = [o["target"] for o in scene_meta["objects"]]
        # only target-class objects are perceived as objects (POSE: targets O of known semantic classes); clutter
        # returns are background
        self.target_of = torch.as_tensor(targets + [False], device=device)  # index -1 -> False
        self.surf = SurfaceCoverage(os.path.join(d, "surface.npz"), targets, self.E, device)
        obs_file = os.path.join(d, f"observable_tilt{li.mount_tilt_deg:g}.npz")
        self.observable = torch.as_tensor(np.load(obs_file)["observable"], device=device) if os.path.exists(
            obs_file) else None
        ub_file = os.path.join(d, f"ub_surface_tilt{li.mount_tilt_deg:g}.npz")
        self.ub_mask = None
        if os.path.exists(ub_file):
            u = np.load(ub_file)
            # same key order as SurfaceCoverage (both sort the same keys)
            self.ub_mask = torch.as_tensor(u["posture"], device=device) & self.surf.is_target
        self.term = Termination(cfg.termination, self.E)
        self.dirs = torch.as_tensor(vlp16_dirs(li.channels, li.vfov_deg, li.az_res_deg), dtype=torch.float32,
                                    device=device)
        gdirs = torch.as_tensor(vlp16_dirs(li.channels, li.vfov_deg, cfg.planner.posture.gain_az_res_deg),
                                dtype=torch.float32, device=device)
        self.mount_pos, self.mount_R = mount(li)
        pp = cfg.planner.posture
        self.postures = posture_grid(type(cfg)(pitch_deg=pp.pitch_deg, roll_deg=pp.roll_deg), cfg.robot)
        self.scorer = ViewpointScorer(self.world, cfg, gdirs, self.mount_pos, self.mount_R, self.postures)
        self.perception = perception      # object with .label(session, t, env, o, R, t_hit, obj_gt), or None
        center_fn = None
        self.prior = None
        method = cfg.priors.method
        if method in ("template", "template_vlm"):
            from .priors import TemplatePrior  # noqa: PLC0415

            objs = scene_meta["objects"]
            cls_fn = class_fn or (lambda o: objs[o]["class"])
            self.prior = TemplatePrior(cfg, cls_fn, self.scorer, vlm_hint_fn=vlm_hint_fn)
        elif method == "cad":
            from .priors import CadPrior  # noqa: PLC0415

            self.prior = CadPrior(d, self.scorer, len(scene_meta["objects"]))
        if self.prior is not None:
            region_fn = self.prior.region
            center_fn = self.prior.center if cfg.priors.use_center else None
        self.agents = [InspectionAgent(e, self.world, self.scorer, cfg, scene_meta, region_fn=region_fn,
                                       pruner=pruner_factory(e) if pruner_factory else None, center_fn=center_fn,
                                       selector=selector_factory(e) if selector_factory else None)
                       for e in range(self.E)]
        self.path_len = np.zeros(self.E)
        self.last_xy = None
        self.curves = [[] for _ in range(self.E)]
        self.next_log = 0.0
        self.t_wall0 = time.time()
        self.target_names = [o["name"] for o in scene_meta["objects"] if o["target"]]
        self.traj = [[] for _ in range(self.E)]   # (t, x, y, yaw, pitch, roll) at traj_period_s, for figures / GIFs
        self.next_traj = 0.0
        self.L_at_eff = np.full(self.E, np.nan)
        self.npost_at_eff = np.full(self.E, -1)

    # ---------------------------------------------------------------------------------------------------------------
    def scan(self, t: float, base_xyz: torch.Tensor, R_body: torch.Tensor):
        li, m = self.cfg.lidar, self.cfg.metrics
        active = torch.as_tensor(~self.term.done, device=self.dev)
        if not active.any():
            return
        env = torch.nonzero(active).flatten()
        o, R = sensor_poses(base_xyz.to(self.dev).float()[env], R_body.to(self.dev).float()[env], self.mount_pos,
                            self.mount_R)
        th, _, obj = self.mesh.cast(o, R, self.dirs, li.max_range)
        if self.perception is None:
            labels = torch.where(self.target_of[obj], obj, torch.full_like(obj, -1))
            cov = labels
        else:
            # (object hypothesis per return for the map, GT object id per return where the class is right)
            labels, cov = self.perception.label(self, t, env, o, R, th, obj)
        # the planner's object models use returns up to the discovery range; Co counts only those within d_max
        far = th > self.cfg.perception.discover_range
        self.world.insert_scan(env, o, R, self.dirs, th, torch.where(far, torch.full_like(labels, -1), labels),
                               m.free_ray_stride, m.no_return_free_range)
        # the planner's own record of object surface it has hit closely (no GT): fine grid (priors), coarse counts
        own = (labels >= 0) & ~far & (th <= m.object_range)
        if own.any():
            pts_o = o[:, None, :] + (R @ self.dirs.T).transpose(1, 2) * th[..., None]
            e_o = env[:, None].expand_as(th)[own]
            self.world.mark_close(e_o, pts_o[own])
            if self.world.fine is not None:
                self.world.mark_fine(e_o, pts_o[own])
        labels = torch.where(th <= m.object_range, cov, torch.full_like(cov, -1))
        hit = labels >= 0
        if hit.any():
            pts = o[:, None, :] + (R @ self.dirs.T).transpose(1, 2) * th[..., None]
            e = env[:, None].expand_as(th)
            self.surf.mark(e[hit], pts[hit], labels[hit])

    def control(self, t: float, poses: np.ndarray, postures: np.ndarray) -> list[dict | None]:
        """poses [E, 3] (x, y, yaw), postures [E, 2] (pitch, roll) measured -> command per robot (None when done)."""
        if t + 1e-9 >= self.next_traj:
            self.next_traj = t + self.cfg.metrics.traj_period_s
            for e in range(self.E):
                if not self.term.done[e]:
                    self.traj[e].append([round(float(t), 2), *(round(float(v), 3) for v in poses[e][:3]),
                                         *(round(float(v), 3) for v in postures[e][:2])])
        out = []
        for e, a in enumerate(self.agents):
            out.append(None if self.term.done[e] else a.step(t, poses[e], postures[e]))
        return out

    def cs(self):
        raw = self.world.vg.known_fraction()
        obs = self.world.vg.known_fraction(self.observable) if self.observable is not None else raw
        return raw.cpu().numpy(), obs.cpu().numpy()

    def log(self, t: float, poses: np.ndarray) -> np.ndarray:
        xy = poses[:, :2]
        if self.last_xy is not None:
            step = np.linalg.norm(xy - self.last_xy, axis=1)
            self.path_len += np.where(self.term.done, 0.0, step)
        self.last_xy = xy.copy()
        if t + 1e-9 < self.next_log:
            return self.term.done
        self.next_log = t + self.cfg.metrics.log_period_s
        co = self.surf.co().cpu().numpy()
        co26 = self.surf.co(True).cpu().numpy()
        per = self.surf.per_object().cpu().numpy()
        cs_raw, cs_obs = self.cs()
        if self.ub_mask is not None:
            co_ub = ((self.surf.seen & self.ub_mask).sum(1).float() / self.ub_mask.sum().clamp(min=1)).cpu().numpy()
        else:
            co_ub = np.full(self.E, np.nan)
        was = self.term.done.copy()
        idle = np.array([a.idle_since for a in self.agents], float)
        for e, a in enumerate(self.agents):
            if a.stopped:
                self.term.force(e, t)
        self.term.update(t, cs_obs, idle)
        for e in range(self.E):
            if was[e]:
                continue
            self.curves[e].append([round(t, 2), float(co[e]), float(co26[e]), float(cs_raw[e]), float(cs_obs[e]),
                                   float(co_ub[e]), round(float(self.path_len[e]), 2), self.agents[e].n_post,
                                   [round(float(v), 4) for v in per[e]]])
            if self.term.done[e]:
                t_eff = float(self.term.t_eff[e])
                # path length / postures at the effective terminal (start of the final window)
                row = [c for c in self.curves[e] if c[0] <= t_eff + 1e-6]
                row = row[-1] if row else self.curves[e][-1]
                self.L_at_eff[e], self.npost_at_eff[e] = row[6], row[7]
        return self.term.done

    # ---------------------------------------------------------------------------------------------------------------
    def summary(self, e: int) -> dict:
        c = self.curves[e]
        t_eff = float(self.term.t_eff[e]) if np.isfinite(self.term.t_eff[e]) else (c[-1][0] if c else 0.0)
        row = [r for r in c if r[0] <= t_eff + 1e-6]
        row = row[-1] if row else c[-1]
        names = [o["name"] for o in self.meta["objects"]]
        return {"seed": self.seeds[e], "timeout": bool(self.term.timeout[e]), "stalled": bool(self.term.stalled[e]),
                "finished": bool(self.term.done[e]),
                "t_eff": t_eff, "L_eff": row[6], "N_post": int(row[7]), "Co": row[1], "Co_n26": row[2],
                "Cs_raw": row[3], "Cs": row[4], "Co_vs_ub": row[5],
                "Co_per_object": {n: row[8][i] for i, n in enumerate(names)},
                "semantic_visits": self.agents[e].n_sem_visits}

    def save(self, extra: dict | None = None):
        os.makedirs(self.out_dir, exist_ok=True)
        cols = ["t", "Co", "Co_n26", "Cs_raw", "Cs", "Co_vs_ub", "L", "N_post", "Co_per_object"]
        for e in range(self.E):
            run = {"tag": self.tag, "scene": self.meta["name"], "seed": self.seeds[e], "summary": self.summary(e),
                   "curve_columns": cols, "curve": self.curves[e], "events": self.agents[e].events,
                   "traj_columns": ["t", "x", "y", "yaw", "pitch", "roll"], "traj": self.traj[e],
                   "object_names": [o["name"] for o in self.meta["objects"]],
                   "targets": self.target_names, "wall_s": round(time.time() - self.t_wall0, 1),
                   "config": {k: v for k, v in self.cfg.items() if k != "_source_"}, **(extra or {})}
            with open(os.path.join(self.out_dir, f"seed{self.seeds[e]}.json"), "w") as f:
                json.dump(run, f, default=_json_default)
