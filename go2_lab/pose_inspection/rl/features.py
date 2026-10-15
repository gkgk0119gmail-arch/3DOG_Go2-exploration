"""Observation of the RL head: one feature row per candidate target + one global row.

Everything here comes from the robot's own map and planner state (no ground truth): the same information the rule-based
POSE planner uses to score and order its targets.

candidate row (N_CAND_FEAT):
  0-1   kind one-hot (explore, semantic)
  2     travel time to it (path / speed + turn) / 60 s
  3-4   bearing from the robot heading (cos, sin)
  5     straight-line distance / 20 m
  6     explore: frontier cluster size / 200 cells
  7     semantic: best-posture gain G* / 300 (unknown voxels the tilted scan would reach)
  8     semantic: level gain G0 / 300
  9     semantic: tilt chosen (G* > G0)
  10-11 semantic: |pitch| / 25 deg, |roll| / 15 deg
  12    semantic: visibility score (log1p(S) / 10)
  13    semantic: object closely scanned fraction (voxels with >= tint_min_hits returns within d_max / mapped)
  14    semantic: object size (log1p(mapped voxels) / 8)
  15    semantic: object mapped height / 5 m
  16    semantic: sectors of this object already visited / 8
  17    semantic: standoff / 5 m
global row (N_GLOBAL_FEAT):
  0     elapsed time / 600 s
  1     known fraction of the robot's own voxel map
  2     objects found / 10
  3     mean closely scanned fraction over found objects
  4     candidates / 40
  5     explore candidates / 10
"""

from __future__ import annotations

import math

import numpy as np
import torch

N_CAND_FEAT = 18
N_GLOBAL_FEAT = 6


def object_scanned_fraction(agent, m) -> float:
    hits = m.get("close_hits")
    if hits is None or m["count"] == 0:
        return 0.0
    return float((hits >= agent.cfg.planner.vlm.tint_min_hits).float().mean())


def observe(agent, targets, times, t: float, pose=None):
    """-> (cand [N, N_CAND_FEAT], glob [N_GLOBAL_FEAT]) float32 numpy."""
    x, y, yaw = agent.last_pose if pose is None else pose[:3]
    frac = {o: object_scanned_fraction(agent, m) for o, m in agent.models.items()}
    C = np.zeros((len(targets), N_CAND_FEAT), np.float32)
    for i, (tg, tt) in enumerate(zip(targets, times)):
        dx, dy = tg.xy[0] - x, tg.xy[1] - y
        b = math.atan2(dy, dx) - yaw
        C[i, 2] = min(tt / 60.0, 5.0)
        C[i, 3], C[i, 4] = math.cos(b), math.sin(b)
        C[i, 5] = math.hypot(dx, dy) / 20.0
        if tg.kind == "explore":
            C[i, 0] = 1.0
            C[i, 6] = min(tg.info.get("size", 0) / 200.0, 5.0)
        else:
            C[i, 1] = 1.0
            C[i, 7] = min(tg.gain / 300.0, 5.0)
            C[i, 8] = min(tg.gain0 / 300.0, 5.0)
            C[i, 9] = float(tg.gain > tg.gain0)
            C[i, 10] = abs(math.degrees(tg.posture[0])) / 25.0
            C[i, 11] = abs(math.degrees(tg.posture[1])) / 15.0
            C[i, 12] = math.log1p(max(tg.score, 0.0)) / 10.0
            m = agent.models.get(tg.obj)
            if m is not None:
                C[i, 13] = frac.get(tg.obj, 0.0)
                C[i, 14] = math.log1p(m["count"]) / 8.0
                C[i, 15] = float(m["hi"][2]) / 5.0
            C[i, 16] = len(agent.retired.get(tg.obj, ())) / 8.0
            C[i, 17] = tg.standoff / 5.0
    G = np.zeros(N_GLOBAL_FEAT, np.float32)
    G[0] = t / 600.0
    G[1] = float(agent.w.vg.known_fraction()[agent.e])
    G[2] = len(agent.models) / 10.0
    G[3] = float(np.mean(list(frac.values()))) if frac else 0.0
    G[4] = len(targets) / 40.0
    G[5] = sum(tg.kind == "explore" for tg in targets) / 10.0
    return C, G


def to_batch(obs_list, device):
    """Pad a list of (cand, glob) to tensors: cand [B, Nmax, F], mask [B, Nmax] (True = real), glob [B, G]."""
    n = max(len(c) for c, _ in obs_list)
    B = len(obs_list)
    cand = np.zeros((B, max(n, 1), N_CAND_FEAT), np.float32)
    mask = np.zeros((B, max(n, 1)), bool)
    glob = np.zeros((B, N_GLOBAL_FEAT), np.float32)
    for i, (c, g) in enumerate(obs_list):
        cand[i, :len(c)] = c
        mask[i, :len(c)] = True
        glob[i] = g
    return (torch.as_tensor(cand, device=device), torch.as_tensor(mask, device=device),
            torch.as_tensor(glob, device=device))
