"""Per-robot POSE-style inspection planner (engine-agnostic state machine).

Every ``replan_period_s`` (and on arrival / failure) the agent rebuilds its targets from its own map:
  * exploration viewpoints: one per 2D frontier cluster (known-free cell nearest the cluster centroid)
  * semantic viewpoints: for each mapped object, the best candidate (visibility score) of every non-retired sector,
    with its body posture from the gain test (Sec. IV-B); optional geometric pruning (G* < prune_min_gain); a
    ``pruner`` hook (stage-5 VLM) may keep only some of them
and orders all of them as an open TSP from the robot (greedy + 2-opt). The head of the tour is executed:
  goto (path waypoints) -> align (face the object, or aim-aligned yaw) -> tilt (track + hold) -> level -> retire sector
The engine turns the returned command into motion: ``move`` (turn-then-go to ``waypoint``), ``turn`` (to ``yaw``),
``hold``; ``posture`` = (pitch, roll) target for the posture-tracking gait.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .navigation import NavPlanner, tour
from .viewpoints import aim_aligned, sample_candidates


@dataclass
class Target:
    kind: str                       # explore | semantic
    xy: np.ndarray
    yaw: float | None = None
    obj: int = -1
    sector: int = -1
    posture: tuple = (0.0, 0.0)
    gain: float = 0.0
    gain0: float = 0.0
    score: float = 0.0
    standoff: float = 0.0
    info: dict = field(default_factory=dict)

    @property
    def key(self):
        if self.kind == "semantic":
            return ("s", self.obj, self.sector)
        return ("e", int(round(self.xy[0])), int(round(self.xy[1])))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class InspectionAgent:
    def __init__(self, env_id: int, world, scorer, cfg, scene_meta: dict, region_fn=None, pruner=None,
                 center_fn=None):
        self.e, self.w, self.sc, self.cfg = env_id, world, scorer, cfg
        self.p = cfg.planner
        self.meta = scene_meta
        self.region_fn = region_fn          # (agent, obj id, model) -> region for the gain (default: inflated box)
        self.pruner = pruner                # (agent, obj id, model, targets) -> kept targets (stage 5)
        self.center_fn = center_fn          # (agent, obj id, model) -> object centre xy from a shape prior, or None
        self.retired: dict[int, set[int]] = {}
        self.blacklist: list[np.ndarray] = []
        self.mode, self.target, self.path, self.wp_i = "plan", None, None, 0
        self.t_mode, self.t_plan = 0.0, -1e9
        self.progress = (0.0, np.inf)
        self.posture_cmd = (0.0, 0.0)
        self.yaw_cmd = 0.0
        self.n_post, self.n_sem_visits = 0, 0
        self.events: list[dict] = []
        self.models: dict[int, dict] = {}
        self.objects_done: set[int] = set()
        self.last_targets: list[Target] = []
        self.nav = None
        self.idle_since = float("nan")    # time the planner last ran out of targets (nan while it has some)

    # ---------------------------------------------------------------------------------------------------------------
    def _event(self, t, ev, **kw):
        self.events.append({"t": round(float(t), 2), "type": ev, **kw})

    def _set_mode(self, mode, t):
        self.mode, self.t_mode = mode, t

    def _blacklisted(self, xy) -> bool:
        r = self.p.exploration.blacklist_radius
        return any(np.linalg.norm(xy - b) < r for b in self.blacklist)

    def _region(self, o, m):
        if self.region_fn is not None:
            return self.region_fn(self, o, m)
        return self.sc.region_box(m["lo"], m["hi"], self.p.posture.delta_b)

    # ---------------------------------------------------------------------------------------------------------------
    def object_center(self, o: int, m: dict) -> np.ndarray:
        c = self.center_fn(self, o, m) if self.center_fn is not None else None
        return np.asarray(m["centroid"][:2] if c is None else c, float)

    def semantic_targets(self, o: int, m: dict, nav: NavPlanner) -> list[Target]:
        sp = self.p.semantic
        cxy = self.object_center(o, m)
        cands = sample_candidates(cxy, sp.n_sectors, sp.standoffs)
        retired = self.retired.get(o, set())
        keep = [c for c in cands if int(c[3]) not in retired and nav.traversable(c[:2])
                and np.isfinite(nav.path_length(c[:2]))]
        if not keep:
            return []
        keep = np.array(keep)
        S = self.sc.visibility(self.e, keep[:, :3], m["voxels"])
        best: dict[int, tuple[float, np.ndarray]] = {}
        for c, s in zip(keep, S):
            if s >= sp.s_min_deg2 and (int(c[3]) not in best or s > best[int(c[3])][0]):
                best[int(c[3])] = (float(s), c)
        out = []
        region = self._region(o, m) if best else None
        for s, (score, c) in sorted(best.items()):
            tgt = Target("semantic", c[:2].copy(), float(c[2]), o, s, score=score,
                         standoff=float(np.linalg.norm(c[:2] - cxy)))
            if self.p.posture.enabled:
                pt, rl, g, g0 = self.sc.best_posture(self.e, c[:2], float(c[2]), region)
                tgt.posture, tgt.gain, tgt.gain0 = (pt, rl), g, g0
            else:
                g0 = float(self.sc.gains(self.e, [(c[0], c[1], c[2], 0.0, 0.0)], region)[0])
                tgt.gain, tgt.gain0 = g0, g0
            thr = sp.prune_min_gain_fine if isinstance(region, dict) else sp.prune_min_gain
            if sp.prune == "geometric" and tgt.gain < thr:
                # pruned for good: a refitted model may push the gain back over the threshold and the robot
                # would dither between sectors
                self.retired.setdefault(o, set()).add(s)
                self._event(self.t_plan, "prune", obj=o, sector=s, gain=round(float(tgt.gain), 1))
                continue
            out.append(tgt)
        if self.pruner is not None:
            out = self.pruner(self, o, m, out)
        return out

    def exploration_targets(self, nav: NavPlanner, nav_map) -> list[Target]:
        out = []
        for cells, cen in self.w.frontiers(nav_map, self.p.exploration.min_frontier_cells):
            d = np.linalg.norm(cells - cen, axis=1)
            for k in np.argsort(d)[:20]:
                xy = nav.xy_of(*cells[k])
                if nav.traversable(xy) and np.isfinite(nav.path_length(xy)) and not self._blacklisted(xy):
                    out.append(Target("explore", xy, info={"size": int(len(cells))}))
                    break
        return out

    def replan(self, t: float, pose):
        x, y, yaw = pose[:3]
        nav_map = self.w.nav_map(self.e)
        origin, cell = self.w.nav_origin()
        nav = NavPlanner(nav_map, origin, cell, self.p.nav, self.cfg.robot.radius, (x, y))
        self.nav = nav
        targets: list[Target] = []
        if self.p.exploration.enabled:
            targets += self.exploration_targets(nav, nav_map)
        if self.p.semantic.enabled:
            mf = self.cfg.perception.gsam2.min_footprint if self.cfg.perception.mode != "gt" else 0.0
            self.models = self.w.object_models(self.e, self.p.semantic.min_object_voxels, mf)
            for o, m in self.models.items():
                st = self.semantic_targets(o, m, nav)
                if not st:
                    self.objects_done.add(o)
                else:
                    self.objects_done.discard(o)
                targets += st
        self.last_targets = targets
        self.t_plan = t
        if not targets:
            return None
        u, w = self.p.speed, self.p.yaw_rate
        xys = [tg.xy for tg in targets]
        start = np.array([nav.path_length(xy) / u + abs(wrap(math.atan2(xy[1] - y, xy[0] - x) - yaw)) / w
                          for xy in xys])
        pair = nav.pairwise(xys, self.p.nav.coarse) / u
        order = tour(start, pair, self.p.tour.two_opt_iters)
        # keep the current target while it is still offered (receding horizon without dithering)
        if self.target is not None:
            same = [tg for tg in targets if tg.key == self.target.key]
            if same:
                return same[0]
        return targets[order[0]]

    # ---------------------------------------------------------------------------------------------------------------
    def _start_goto(self, t, tgt: Target, pose):
        self.target = tgt
        self.path = self.nav.path(tgt.xy) if self.nav is not None else None
        if self.path is None:
            self.blacklist.append(tgt.xy.copy())
            self.target = None
            self._set_mode("plan", t)
            return
        self.wp_i = 1 if len(self.path) > 1 else 0
        self.progress = (t, self.nav.path_length(tgt.xy))
        self._set_mode("goto", t)
        self._event(t, "goto", kind=tgt.kind, xy=tgt.xy.round(2).tolist(), obj=tgt.obj, sector=tgt.sector)

    def step(self, t: float, pose, posture_meas) -> dict:
        """pose = (x, y, yaw), posture_meas = (pitch, roll) [rad] -> command dict."""
        x, y, yaw = pose[:3]
        p = self.p
        if self.mode == "plan" or (self.mode in ("idle", "goto") and t - self.t_plan >= p.replan_period_s):
            tgt = self.replan(t, pose)
            if tgt is None:
                self.target = None
                if self.mode != "idle":
                    self.idle_since = t
                self._set_mode("idle", t)
            elif self.mode != "goto" or self.target is None or tgt.key != self.target.key:
                self.idle_since = float("nan")
                self._start_goto(t, tgt, pose)
            else:  # same target: refresh its data and the path from here
                self.target = tgt
                path = self.nav.path(tgt.xy)
                if path is not None:
                    self.path, self.wp_i = path, (1 if len(path) > 1 else 0)
        if self.mode == "idle":
            return {"mode": "hold", "posture": (0.0, 0.0)}
        if self.mode == "plan":
            return {"mode": "hold", "posture": (0.0, 0.0)}

        tgt = self.target
        if self.mode == "goto":
            xy = np.array([x, y])
            final = self.wp_i >= len(self.path) - 1
            r = p.final_reach_radius if (final and tgt.kind == "semantic") else p.reach_radius
            while np.linalg.norm(self.path[self.wp_i] - xy) < r and not final:
                self.wp_i += 1
                final = self.wp_i >= len(self.path) - 1
                r = p.final_reach_radius if (final and tgt.kind == "semantic") else p.reach_radius
            if final and np.linalg.norm(self.path[-1] - xy) < r:
                return self._arrive(t, pose)
            # stuck detection on the remaining path length
            remaining = np.linalg.norm(self.path[self.wp_i] - xy) + float(
                np.sum(np.linalg.norm(np.diff(self.path[self.wp_i:], axis=0), axis=1)))
            if remaining < self.progress[1] - 0.5:
                self.progress = (t, remaining)
            elif t - self.progress[0] > p.stuck_s:
                self._event(t, "stuck", key=str(tgt.key))
                self.blacklist.append(tgt.xy.copy())
                if tgt.kind == "semantic":
                    self.retired.setdefault(tgt.obj, set()).add(tgt.sector)
                self.target = None
                self._set_mode("plan", t)
                return {"mode": "hold", "posture": (0.0, 0.0)}
            return {"mode": "move", "waypoint": self.path[self.wp_i], "posture": (0.0, 0.0)}

        if self.mode == "align":
            if abs(wrap(self.yaw_cmd - yaw)) < math.radians(5.0) or t - self.t_mode > 6.0:
                self._begin_scan(t, pose)
            return {"mode": "turn", "yaw": self.yaw_cmd, "posture": (0.0, 0.0)}

        if self.mode == "tilt":
            err = math.degrees(max(abs(posture_meas[0] - self.posture_cmd[0]), abs(posture_meas[1] - self.posture_cmd[1])))
            tracked = err < p.posture.settle_tol_deg or t - self.t_mode > p.posture.settle_timeout_s
            if tracked and "hold_from" not in tgt.info:
                tgt.info["hold_from"] = t
                self._event(t, "tracked", obj=tgt.obj, sector=tgt.sector, err_deg=round(err, 1),
                            cmd_deg=[round(math.degrees(v), 1) for v in self.posture_cmd],
                            meas_deg=[round(math.degrees(float(v)), 1) for v in posture_meas])
            if "hold_from" in tgt.info and t - tgt.info["hold_from"] >= p.posture.hold_s:
                self._set_mode("level", t)
            return {"mode": "turn", "yaw": self.yaw_cmd, "posture": self.posture_cmd}

        if self.mode == "level":
            err = math.degrees(max(abs(posture_meas[0]), abs(posture_meas[1])))
            if err < p.posture.settle_tol_deg or t - self.t_mode > p.posture.settle_timeout_s:
                self._finish_semantic(t)
            return {"mode": "turn", "yaw": self.yaw_cmd, "posture": (0.0, 0.0)}

        if self.mode == "scan_level":
            if t - self.t_mode >= p.semantic.level_hold_s:
                self._finish_semantic(t)
            return {"mode": "turn", "yaw": self.yaw_cmd, "posture": (0.0, 0.0)}
        return {"mode": "hold", "posture": (0.0, 0.0)}

    # ---------------------------------------------------------------------------------------------------------------
    def _arrive(self, t, pose):
        tgt = self.target
        self._event(t, "arrive", kind=tgt.kind, key=str(tgt.key))
        if tgt.kind == "explore":
            # still a frontier here next time -> do not come back
            self.blacklist.append(tgt.xy.copy())
            self.target = None
            self._set_mode("plan", t)
            return {"mode": "hold", "posture": (0.0, 0.0)}
        x, y, yaw = pose[:3]
        m = self.models.get(tgt.obj)
        cxy = self.object_center(tgt.obj, m) if m is not None else None
        face = math.atan2(cxy[1] - y, cxy[0] - x) if m is not None else tgt.yaw
        self.yaw_cmd = face
        self.posture_cmd = (0.0, 0.0)
        if self.p.posture.enabled and m is not None:
            # re-select the posture from the arrival pose on the current map
            pt, rl, g, g0 = self.sc.best_posture(self.e, (x, y), face, self._region(tgt.obj, m))
            tgt.posture, tgt.gain, tgt.gain0 = (pt, rl), g, g0
            if abs(pt) + abs(rl) > 1e-6:
                if self.p.posture.aim_aligned:
                    lim = self.cfg.robot
                    psi, th, ph = aim_aligned(face, pt, rl, yaw, math.radians(self.p.posture.aim_step_deg),
                                              math.radians(lim.pitch_limit_deg), math.radians(lim.roll_limit_deg))
                    self.yaw_cmd, self.posture_cmd = psi, (th, ph)
                else:
                    self.posture_cmd = (pt, rl)
        self._set_mode("align", t)
        return {"mode": "turn", "yaw": self.yaw_cmd, "posture": (0.0, 0.0)}

    def _begin_scan(self, t, pose):
        tgt = self.target
        if abs(self.posture_cmd[0]) + abs(self.posture_cmd[1]) > 1e-6:
            self.n_post += 1
            self._event(t, "posture", obj=tgt.obj, sector=tgt.sector, xy=[round(pose[0], 2), round(pose[1], 2)],
                        yaw=round(float(self.yaw_cmd), 3),
                        pitch_deg=round(math.degrees(self.posture_cmd[0]), 1),
                        roll_deg=round(math.degrees(self.posture_cmd[1]), 1), gain=tgt.gain, gain0=tgt.gain0)
            self._set_mode("tilt", t)
        else:
            self._set_mode("scan_level", t)

    def _finish_semantic(self, t):
        tgt = self.target
        self.retired.setdefault(tgt.obj, set()).add(tgt.sector)
        self.n_sem_visits += 1
        self._event(t, "retire", obj=tgt.obj, sector=tgt.sector)
        self.target = None
        self.posture_cmd = (0.0, 0.0)
        self._set_mode("plan", t)
