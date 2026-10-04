"""Map-quality metrics against a ground-truth occupancy map (tools/usd_occupancy_map.py output)."""

from __future__ import annotations

import numpy as np

FREE, OCCUPIED, UNKNOWN = 0, 100, -1


def gt_on_grid(gt_path: str, origin, cell: float, shape):
    """Resample the GT map (0.1 m, indexed [x, y], 0 = occupied, 255 = free, 127 = unknown) onto a run grid.

    Returns (gt_occupied, gt_free, cell centre xs, cell centre ys) for a grid of ``shape`` = (ny, nx).
    A run cell is GT-occupied when >= 2 of its sub-cells are (ignores single-sample speckle).
    """
    d = np.load(gt_path)
    g, go, gr = d["grid"], d["origin"], float(d["resolution"])
    ny, nx = shape
    ys = origin[1] + (np.arange(ny) + 0.5) * cell
    xs = origin[0] + (np.arange(nx) + 0.5) * cell
    k = int(round(cell / gr))
    occ_count = np.zeros(shape, int)
    free = np.zeros(shape, bool)
    for dy in range(k):
        for dx in range(k):
            ix = np.floor((xs - cell / 2 + (dx + 0.5) * gr - go[0]) / gr).astype(int)
            iy = np.floor((ys - cell / 2 + (dy + 0.5) * gr - go[1]) / gr).astype(int)
            okx = (ix >= 0) & (ix < g.shape[0])
            oky = (iy >= 0) & (iy < g.shape[1])
            sub = np.full(shape, 127, np.uint8)
            sub[np.ix_(oky, okx)] = g[np.ix_(ix[okx], iy[oky])].T
            occ_count += sub == 0
            free |= sub == 255
    occ = occ_count >= 2
    return occ, free & ~occ, xs, ys


class MapScorer:
    """Precomputes the GT on the run grid once; scores many maps of the same grid quickly."""

    def __init__(self, gt_path: str, origin, cell: float, shape, bounds=None):
        from scipy import ndimage

        self._nd = ndimage
        self.gt_occ, self.gt_free, xs, ys = gt_on_grid(gt_path, origin, cell, shape)
        if bounds is None:
            self.inside = np.ones(shape, bool)
        else:
            self.inside = ((ys >= bounds[1]) & (ys <= bounds[3]))[:, None] & ((xs >= bounds[0]) & (xs <= bounds[2]))[None, :]
        self.gt_occ_near = ndimage.binary_dilation(self.gt_occ, iterations=1)
        self.gt_occ_core = ndimage.binary_erosion(self.gt_occ)

    def __call__(self, grid: np.ndarray) -> dict[str, float]:
        m_occ, m_free = grid == OCCUPIED, grid == FREE
        ins = self.inside
        return {
            "precision": float((m_occ & self.gt_occ_near & ins).sum() / max((m_occ & ins).sum(), 1)),
            "recall": float((self.gt_occ & self._nd.binary_dilation(m_occ) & ins).sum() / max((self.gt_occ & ins).sum(), 1)),
            "false_free": float((m_free & self.gt_occ_core & ins).sum() / max((m_free & ins).sum(), 1)),
            "coverage": float(((grid != UNKNOWN) & self.gt_free & ins).sum() / max((self.gt_free & ins).sum(), 1)),
        }
