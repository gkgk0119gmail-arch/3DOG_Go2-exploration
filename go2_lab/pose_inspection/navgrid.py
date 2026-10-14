"""Ground-truth 2D traversability of a built scene (grid.npz): clearance, reachable set, start poses.

Grid convention (scene_builder.traversability): ``occupied[iy, ix]``, cell centre = origin + (i + 0.5) * cell.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


class NavGrid:
    def __init__(self, grid_npz: str, robot_radius: float):
        g = np.load(grid_npz)
        self.occ = g["occupied"].astype(bool)
        self.origin = g["origin"].astype(np.float64)
        self.cell = float(g["cell"])
        self.ny, self.nx = self.occ.shape
        self.dist = ndimage.distance_transform_edt(~self.occ) * self.cell  # to the nearest occupied cell [m]
        self.clear = self.dist > robot_radius
        self.robot_radius = robot_radius
        self.reachable = self.clear.copy()

    # ---------------------------------------------------------------------------------------------------------------
    def ij(self, xy):
        xy = np.atleast_2d(np.asarray(xy, float))
        ix = np.floor((xy[:, 0] - self.origin[0]) / self.cell).astype(int)
        iy = np.floor((xy[:, 1] - self.origin[1]) / self.cell).astype(int)
        return iy, ix

    def xy(self, iy, ix):
        return np.stack([self.origin[0] + (np.asarray(ix) + 0.5) * self.cell,
                         self.origin[1] + (np.asarray(iy) + 0.5) * self.cell], -1)

    def inside(self, iy, ix):
        return (iy >= 0) & (iy < self.ny) & (ix >= 0) & (ix < self.nx)

    def clearance(self, xy) -> np.ndarray:
        iy, ix = self.ij(xy)
        ok = self.inside(iy, ix)
        out = np.zeros(len(iy))
        out[ok] = self.dist[iy[ok], ix[ok]]
        return out

    def is_reachable(self, xy) -> np.ndarray:
        iy, ix = self.ij(xy)
        ok = self.inside(iy, ix)
        out = np.zeros(len(iy), bool)
        out[ok] = self.reachable[iy[ok], ix[ok]]
        return out

    def restrict_to_region(self, region):
        """Reachable = clear cells 4-connected to the clear cells inside ``region`` [x0, y0, x1, y1]."""
        lab, _ = ndimage.label(self.clear)
        iy0, ix0 = self.ij([region[0], region[1]])
        iy1, ix1 = self.ij([region[2], region[3]])
        seeds = np.unique(lab[iy0[0]:iy1[0] + 1, ix0[0]:ix1[0] + 1])
        seeds = seeds[seeds > 0]
        self.reachable = np.isin(lab, seeds)
        return self

    def lattice(self, step: float) -> np.ndarray:
        """Reachable cell centres on a ``step`` lattice [N, 2]."""
        k = max(1, int(round(step / self.cell)))
        iy, ix = np.nonzero(self.reachable[k // 2::k, k // 2::k])
        return self.xy(iy * k + k // 2, ix * k + k // 2)

    def sample_starts(self, region, n: int, seed: int, min_clearance: float) -> np.ndarray:
        """``n`` start poses (x, y, yaw) in ``region`` on reachable cells with ``min_clearance``; reproducible."""
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(100_000):
            xy = np.array([rng.uniform(region[0], region[2]), rng.uniform(region[1], region[3])])
            if self.is_reachable(xy)[0] and self.clearance(xy)[0] >= min_clearance:
                out.append([xy[0], xy[1], rng.uniform(-np.pi, np.pi)])
                if len(out) == n:
                    return np.array(out)
        raise RuntimeError("could not sample start poses")
