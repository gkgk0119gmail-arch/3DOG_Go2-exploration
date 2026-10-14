"""Online world model of every robot (batched over environments, GPU): what the planner is allowed to know.

* 3D voxel map (unknown / free / occupied) at ``metrics.scene_voxel`` -- also the Cs map
* object label grid: the object id of the last labelled return in each occupied voxel (-1 none). An object's model
  is its labelled voxels: centroid, box, size (POSE: c_o, s_o, dense object cloud)
* 2D navigation map derived from the 3D map: a column is occupied when an occupied voxel lies in the obstacle band
  (above the floor), known free when the band was traversed by rays, unknown otherwise
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage

from .raycast import VoxelGrid

UNKNOWN2D, FREE2D, OCC2D = -1, 0, 100


class WorldModel:
    def __init__(self, cfg, scene_meta: dict, n_envs: int, device: str = "cuda:0", n_obj: int | None = None):
        self.cfg, self.n_envs, self.device = cfg, n_envs, device
        m = cfg.metrics
        self.vg = VoxelGrid(scene_meta["bounds"], scene_meta["z_range"], m.scene_voxel, n_envs, device)
        self.labels = torch.full((n_envs, *self.vg.shape), -1, dtype=torch.int16, device=device)
        self.n_obj = n_obj or len(scene_meta["objects"])  # GT ids, or perception hypotheses
        v = self.vg.voxel
        band = cfg.planner.nav.obstacle_band
        self.band = (int(np.floor(band[0] / v)), int(np.ceil(band[1] / v)))
        self.lo = torch.as_tensor(self.vg.lo, device=device)
        self.centers_xy = None

    # ---------------------------------------------------------------------------------------------------------------
    def reset(self, env_ids=None):
        self.vg.reset(env_ids)
        if env_ids is None:
            self.labels.fill_(-1)
        else:
            self.labels[env_ids] = -1

    def voxel_index(self, p: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ijk = torch.floor((p - self.lo) / self.vg.voxel).long()
        shape = torch.tensor(self.vg.shape, device=p.device)
        ok = ((ijk >= 0) & (ijk < shape)).all(-1)
        return ijk, ok

    def insert_scan(self, env: torch.Tensor, origins, rots, dirs, t, labels, stride: int, free_range: float):
        """One sweep per pose: free space (every ``stride``-th ray column) + occupancy + object labels.

        env [P], origins [P, 3], rots [P, 3, 3], dirs [D, 3], t [P, D] (1e9 = no return), labels [P, D] (-1 none).
        """
        D = len(dirs)
        cols = torch.arange(D, device=self.device) % stride == 0
        self.vg.insert(origins, rots, dirs[cols], t[:, cols], env.to(torch.int32), free_range)
        # every return marks its voxel occupied (the strided free pass skipped some)
        hit = t < 1.0e8
        pts = origins[:, None, :] + (rots @ dirs.T).transpose(1, 2) * t[..., None]
        e = env[:, None].expand_as(t)
        ijk, ok = self.voxel_index(pts)
        ok = ok & hit
        i, j, k = ijk[..., 0][ok], ijk[..., 1][ok], ijk[..., 2][ok]
        self.vg.grid[e[ok], i, j, k] = 2
        lab = labels[ok]
        sel = lab >= 0
        self.labels[e[ok][sel], i[sel], j[sel], k[sel]] = lab[sel].to(torch.int16)

    # ---------------------------------------------------------------------------------------------------------------
    def nav_map(self, e: int) -> np.ndarray:
        """2D map [ny, nx] (rows = y): OCC2D / FREE2D / UNKNOWN2D."""
        g = self.vg.grid[e, :, :, self.band[0]:self.band[1]]  # [nx, ny, nb]
        occ = (g == 2).any(-1)
        free = (g == 1).any(-1)
        out = torch.full(occ.shape, UNKNOWN2D, dtype=torch.int16, device=self.device)
        out[free] = FREE2D
        out[occ] = OCC2D
        return out.T.cpu().numpy()

    def nav_origin(self):
        return np.asarray(self.vg.lo[:2], float), self.vg.voxel

    def frontiers(self, nav: np.ndarray, min_cells: int):
        """Clusters of known-free cells next to unknown cells -> list of (cells [K, 2] (iy, ix), centroid (iy, ix))."""
        free = nav == FREE2D
        unk = nav == UNKNOWN2D
        near_unk = ndimage.binary_dilation(unk, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool))
        fr = free & near_unk
        lab, n = ndimage.label(fr, structure=np.ones((3, 3), bool))
        out = []
        for c in range(1, n + 1):
            cells = np.argwhere(lab == c)
            if len(cells) >= min_cells:
                out.append((cells, cells.mean(0)))
        return out

    # ---------------------------------------------------------------------------------------------------------------
    def object_models(self, e: int, min_voxels: int) -> dict[int, dict]:
        """Objects with at least ``min_voxels`` labelled voxels: centroid, box (lo, hi), voxel centres."""
        lab = self.labels[e]
        idx = torch.nonzero(lab >= 0)
        if len(idx) == 0:
            return {}
        ids = lab[idx[:, 0], idx[:, 1], idx[:, 2]].long()
        cnt = torch.bincount(ids, minlength=self.n_obj)
        out = {}
        v = self.vg.voxel
        for o in torch.nonzero(cnt >= min_voxels).flatten().tolist():
            c = (idx[ids == o].float() + 0.5) * v + self.lo
            lo, hi = c.min(0).values - v / 2, c.max(0).values + v / 2
            out[o] = {"centroid": ((lo + hi) / 2).cpu().numpy(), "lo": lo.cpu().numpy(), "hi": hi.cpu().numpy(),
                      "voxels": c, "count": int(len(c))}
        return out
