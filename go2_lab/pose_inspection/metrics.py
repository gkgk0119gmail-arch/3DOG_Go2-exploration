"""POSE evaluation metrics (Sec. V-B, Eq. 8) plus the observable-surface upper bound.

  Co(t)   object-surface coverage: observed GT surface voxels of the target objects / all of them. A voxel of object
          o is observed when a LiDAR return labelled o (GT segmentation, or the perception module's label) lands in
          it. Also kept per object, and a tolerant variant (``co_n26``: a return within the 26-neighbourhood).
  Cs(t)   scene coverage: (free + occupied) voxels / all voxels of the scene volume; reported raw (POSE) and over the
          observable volume (upper_bound.py), which the termination test uses.
  AUCo    area under Co(t) on [0, T] / T, T = longest run among the compared runs of a scene; shorter runs hold their
          last value.
  teff, Leff, Npost   at the effective terminal: start of the final window for finished runs, the limit otherwise.
"""

from __future__ import annotations

import numpy as np
import torch

OFFS27 = torch.tensor([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])


def voxel_keys_t(ijk: torch.Tensor) -> torch.Tensor:
    ijk = ijk.long()
    return (ijk[..., 0] << 42) + (ijk[..., 1] << 21) + ijk[..., 2]


class SurfaceCoverage:
    """Observed GT surface voxels per environment (batched, GPU)."""

    def __init__(self, surface_npz: str, target_mask, n_envs: int, device: str = "cuda:0"):
        s = np.load(surface_npz)
        self.device, self.n_envs = device, n_envs
        self.voxel = float(s["voxel"])
        self.origin = torch.as_tensor(s["origin"], dtype=torch.float32, device=device)
        ijk = torch.as_tensor(s["vox_ijk"].astype(np.int64), device=device)
        obj = torch.as_tensor(s["vox_obj"].astype(np.int64), device=device)
        keys = voxel_keys_t(ijk) * 64 + obj
        order = torch.argsort(keys)
        self.keys, self.obj = keys[order], obj[order]
        self.n_obj = int(s["n_obj"])
        tm = torch.as_tensor(np.asarray(target_mask, bool), device=device)
        self.is_target = tm[self.obj]
        self.n_target = int(self.is_target.sum())
        self.per_obj_total = torch.bincount(self.obj, minlength=self.n_obj).float()
        self.seen = torch.zeros(n_envs, len(self.keys), dtype=torch.bool, device=device)
        self.seen_n26 = torch.zeros_like(self.seen)
        self._offs = OFFS27.to(device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.seen.zero_()
            self.seen_n26.zero_()
        else:
            self.seen[env_ids] = False
            self.seen_n26[env_ids] = False

    def _lookup(self, k: torch.Tensor):
        pos = torch.searchsorted(self.keys, k).clamp(max=len(self.keys) - 1)
        return pos, self.keys[pos] == k

    def mark(self, env: torch.Tensor, points: torch.Tensor, obj: torch.Tensor):
        """Returns ``points`` [N, 3] labelled ``obj`` [N] (-1 = not an object) seen by environments ``env`` [N]."""
        ok = obj >= 0
        if not ok.any():
            return
        env, p, o = env[ok].long(), points[ok], obj[ok].long()
        base = torch.floor((p - self.origin) / self.voxel).long()
        pos, hit = self._lookup(voxel_keys_t(base) * 64 + o)
        self.seen[env[hit], pos[hit]] = True
        nb = (base[:, None, :] + self._offs[None]).reshape(-1, 3)
        pos, hit = self._lookup(voxel_keys_t(nb) * 64 + o.repeat_interleave(27))
        e27 = env.repeat_interleave(27)
        self.seen_n26[e27[hit], pos[hit]] = True

    def co(self, tolerant: bool = False) -> torch.Tensor:
        s = self.seen_n26 if tolerant else self.seen
        return (s & self.is_target).sum(1).float() / max(1, self.n_target)

    def per_object(self) -> torch.Tensor:
        """[n_envs, n_obj] fraction of each object's surface voxels observed."""
        out = torch.zeros(self.n_envs, self.n_obj, device=self.device)
        for e in range(self.n_envs):
            out[e] = torch.bincount(self.obj[self.seen[e]], minlength=self.n_obj).float()
        return out / self.per_obj_total.clamp(min=1)


class Termination:
    """POSE termination: Cs > threshold and growth < min_gain over the last window, or the time limit."""

    def __init__(self, cfg, n_envs: int):
        self.cfg = cfg
        self.hist: list[list[tuple[float, float]]] = [[] for _ in range(n_envs)]
        self.done = np.zeros(n_envs, bool)
        self.t_eff = np.full(n_envs, np.nan)
        self.timeout = np.zeros(n_envs, bool)

    def update(self, t: float, cs: np.ndarray):
        c = self.cfg
        for e in range(len(cs)):
            if self.done[e]:
                continue
            h = self.hist[e]
            h.append((t, float(cs[e])))
            while h and h[0][0] < t - c.window_s - 1e-6:
                h.pop(0)
            full = h[0][0] <= t - c.window_s + 1e-6 and len(h) > 1
            if cs[e] > c.cs_threshold and full and (h[-1][1] - h[0][1]) < c.min_gain:
                self.done[e], self.t_eff[e] = True, t - c.window_s
            elif t >= c.time_limit_s:
                self.done[e], self.t_eff[e], self.timeout[e] = True, c.time_limit_s, True
        return self.done


# --------------------------------------------------------------------------------------------------------------------
# offline aggregation
# --------------------------------------------------------------------------------------------------------------------
def value_at(curve, t: float) -> float:
    """Step-hold value of a (t, v) curve at time t."""
    ts = np.asarray([c[0] for c in curve])
    i = np.searchsorted(ts, t, side="right") - 1
    return float(curve[max(i, 0)][1]) if len(curve) else 0.0


def auc(curve, T: float, t_end: float | None = None) -> float:
    """Area under a step-hold curve on [0, T] / T; values after ``t_end`` hold the value at ``t_end``."""
    if not curve:
        return 0.0
    ts = np.asarray([c[0] for c in curve], float)
    vs = np.asarray([c[1] for c in curve], float)
    if t_end is not None:
        keep = ts <= t_end
        ts, vs = ts[keep], vs[keep]
    ts = np.append(ts, T)
    return float(np.sum(np.diff(ts).clip(min=0) * vs) / T) if T > 0 else 0.0


def mean_std(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], float)
    return (float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else 0.0) if len(x) else (float("nan"), float("nan"))
