"""Controlled experiments around the target selector (engine-agnostic, picked up by ``Session`` from the config).

* ``perception.false_detections``: synthetic false detections on top of the ground-truth labels. The real zero-shot
  pipeline (stage 4) reports rack loads / clutter as machines; here the same failure is injected with a known amount,
  so every selector sees exactly the same false hypotheses:

      perception:
        false_detections:
          clutter: all        # non-target scene objects reported as machines: all | k | [lo, hi] (random per scene)
          boxes: 0            # extra boxes on background geometry (racks, walls) reported as machines: k | [lo, hi]
          seed: 0

  Co still counts the real targets only. Also usable in training (rl_train.py): the policy then sees false
  hypotheses whose visits cost time and return no coverage.

* ``selector``: hand-written selectors on the same hook as the RL head (the agent commits to the chosen target, no
  tour): ``nearest`` (smallest travel time) and ``gain_rate`` (unknown cells revealed per second: G* of a semantic
  viewpoint, frontier size of an exploration viewpoint). Neither stops early; combine with
  ``planner.semantic.prune: geometric`` for a selective rule.
"""

from __future__ import annotations

import os
import zlib

import numpy as np
import torch


def _count(v, n_max: int | None, rng) -> int:
    if v in (None, 0, False):
        return 0
    if v == "all":
        return int(n_max or 0)
    if isinstance(v, (list, tuple)):
        v = int(rng.integers(int(v[0]), int(v[1]) + 1))
    return int(v) if n_max is None else min(int(v), n_max)


class FalseDetections:
    """Perception hook: GT labels + false object hypotheses (see the module docstring)."""

    def __init__(self, cfg, scene_meta: dict, device: str):
        fd = cfg.perception.false_detections
        objs = scene_meta["objects"]
        n = len(objs)
        rng = np.random.default_rng([int(fd.get("seed", 0)), zlib.crc32(str(scene_meta["name"]).encode())])
        clutter = [i for i, o in enumerate(objs) if not o["target"]]
        k = _count(fd.get("clutter", 0), len(clutter), rng)
        fake = sorted(rng.choice(clutter, k, replace=False).tolist()) if k else []
        is_fake = np.zeros(n + 1, bool)
        is_fake[fake] = True
        self.is_fake = torch.as_tensor(is_fake, device=device)
        self.is_target = torch.as_tensor([bool(o["target"]) for o in objs] + [False], device=device)
        boxes = self._boxes(scene_meta, _count(fd.get("boxes", 0), None, rng), fd, rng)
        self.boxes = torch.as_tensor(boxes, dtype=torch.float32, device=device).reshape(-1, 6)
        self.n_scene, self.n_obj = n, n + len(boxes)
        self.info = {"clutter": [objs[i]["name"] for i in fake], "boxes": [[round(float(v), 2) for v in b]
                                                                           for b in boxes]}

    @staticmethod
    def _boxes(meta: dict, k: int, fd, rng) -> list:
        """k boxes (xmin, ymin, zmin, xmax, ymax, zmax) centred on occupied cells away from every scene object."""
        if k <= 0:
            return []
        g = np.load(os.path.join(meta["dir"], "grid.npz"))
        occ, origin, cell = g["occupied"].astype(bool), g["origin"], float(g["cell"])
        ys, xs = np.nonzero(occ)
        xy = np.c_[origin[0] + (xs + 0.5) * cell, origin[1] + (ys + 0.5) * cell]
        b = meta["bounds"]   # cells outside the scene bounds are padding, not geometry
        keep = (xy[:, 0] >= b[0] - 0.1) & (xy[:, 0] <= b[2] + 0.1) & (xy[:, 1] >= b[1] - 0.1) & (xy[:, 1] <= b[3] + 0.1)
        for o in meta["objects"]:
            lo, hi = np.asarray(o["bbox_min"][:2]) - 1.0, np.asarray(o["bbox_max"][:2]) + 1.0
            keep &= ~((xy >= lo).all(1) & (xy <= hi).all(1))
        xy = xy[keep]
        side, top, sep = fd.get("side", [1.2, 2.5]), fd.get("top", [1.5, 2.5]), float(fd.get("min_sep", 3.0))
        out, centres = [], []
        for i in rng.permutation(len(xy)):
            c = xy[i]
            if any(np.linalg.norm(c - q) < sep for q in centres):
                continue
            sx, sy = rng.uniform(side[0], side[1], 2) / 2
            out.append([c[0] - sx, c[1] - sy, 0.15, c[0] + sx, c[1] + sy, float(rng.uniform(top[0], top[1]))])
            centres.append(c)
            if len(out) == k:
                break
        return out

    def label(self, session, t, env, origins, rots, th, obj):
        """-> (hypothesis per return for the robot's map, GT object per return of a real target for Co)."""
        real = self.is_target[obj]
        none = torch.full_like(obj, -1)
        labels = torch.where(real | self.is_fake[obj], obj, none)
        if len(self.boxes):
            pts = origins[:, None, :] + (rots @ session.dirs.T).transpose(1, 2) * th[..., None]
            free = (labels < 0) & (th < 1.0e8)
            for j, b in enumerate(self.boxes):
                inside = free & (pts >= b[:3]).all(-1) & (pts <= b[3:]).all(-1)
                labels[inside] = self.n_scene + j
        return labels, torch.where(real, obj, none)


def heuristic_selector(name: str):
    """(agent, targets, travel times [s], t) -> index of the next target (same hook as the RL head)."""

    def nearest(agent, targets, times, t):
        return int(np.argmin(times))

    def gain_rate(agent, targets, times, t):
        gain = np.array([tg.gain if tg.kind == "semantic" else tg.info.get("size", 0) for tg in targets], float)
        return int(np.argmax(gain / (np.asarray(times, float) + agent.p.posture.hold_s)))

    return {"nearest": nearest, "gain_rate": gain_rate}[name]
