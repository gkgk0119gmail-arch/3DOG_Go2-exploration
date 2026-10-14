"""Path costs and tours on the online 2D map (rows = y).

* cost map: occupied cells inflated by the robot radius are blocked, cells near obstacles cost more (stay mid-aisle),
  unknown cells are traversable at ``unknown_cost`` (POSE plans into unknown space and re-validates)
* Dijkstra (scipy csgraph, 8-connected) from the robot gives path lengths + predecessors; a coarser copy gives the
  pairwise lengths between targets for the tour
* tour: open TSP from the robot -- greedy nearest neighbour then 2-opt; cost = path length / u + turning time
"""

from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from .mapping import FREE2D, OCC2D, UNKNOWN2D

_N8 = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)),
       (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]


def cost_map(nav: np.ndarray, cell: float, nav_cfg, robot_radius: float) -> np.ndarray:
    occ = nav == OCC2D
    dist = ndimage.distance_transform_edt(~occ) * cell
    c = 1.0 + nav_cfg.clearance_weight * np.clip((nav_cfg.clearance - dist) /
                                                  max(nav_cfg.clearance - robot_radius, 1e-6), 0, 1) ** 2
    c[nav == UNKNOWN2D] *= nav_cfg.unknown_cost
    c[dist <= robot_radius] = np.inf
    return c


def _graph(cost: np.ndarray):
    ny, nx = cost.shape
    idx = np.arange(ny * nx).reshape(ny, nx)
    rows, cols, w = [], [], []
    fin = np.isfinite(cost)
    for dy, dx, s in _N8:
        ys, ye = max(0, -dy), ny - max(0, dy)
        xs, xe = max(0, -dx), nx - max(0, dx)
        a = (slice(ys, ye), slice(xs, xe))
        b = (slice(ys + dy, ye + dy), slice(xs + dx, xe + dx))
        ok = fin[a] & fin[b]
        if dy and dx:  # no corner cutting
            ok &= fin[(slice(ys + dy, ye + dy), slice(xs, xe))] & fin[(slice(ys, ye), slice(xs + dx, xe + dx))]
        rows.append(idx[a][ok])
        cols.append(idx[b][ok])
        w.append((s * 0.5 * (cost[a] + cost[b]))[ok])
    return coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(ny * nx, ny * nx)).tocsr()


class NavPlanner:
    """One planning snapshot: cost map, Dijkstra from the robot, coarse pairwise distances."""

    def __init__(self, nav: np.ndarray, origin, cell: float, nav_cfg, robot_radius: float, robot_xy):
        self.nav, self.origin, self.cell, self.cfg = nav, np.asarray(origin, float), cell, nav_cfg
        self.cost = cost_map(nav, cell, nav_cfg, robot_radius)
        self.ny, self.nx = nav.shape
        s = self.nearest_finite(self.cell_of(robot_xy))
        self.start = s
        self.G = _graph(self.cost)
        if s is None:
            self.dist = np.full(self.ny * self.nx, np.inf)
            self.pred = np.full(self.ny * self.nx, -9999)
        else:
            self.dist, self.pred = dijkstra(self.G, indices=s[0] * self.nx + s[1], return_predecessors=True)
        self.dist = self.dist * cell  # [m] (cost-weighted)

    def cell_of(self, xy):
        return (int(math.floor((xy[1] - self.origin[1]) / self.cell)), int(math.floor((xy[0] - self.origin[0]) / self.cell)))

    def xy_of(self, iy, ix):
        return np.array([self.origin[0] + (ix + 0.5) * self.cell, self.origin[1] + (iy + 0.5) * self.cell])

    def inside(self, c):
        return 0 <= c[0] < self.ny and 0 <= c[1] < self.nx

    def nearest_finite(self, c, radius: int = 6):
        if self.inside(c) and np.isfinite(self.cost[c]):
            return c
        best, bd = None, np.inf
        for y in range(max(0, c[0] - radius), min(self.ny, c[0] + radius + 1)):
            for x in range(max(0, c[1] - radius), min(self.nx, c[1] + radius + 1)):
                if np.isfinite(self.cost[y, x]):
                    d = (y - c[0]) ** 2 + (x - c[1]) ** 2
                    if d < bd:
                        best, bd = (y, x), d
        return best

    def traversable(self, xy) -> bool:
        c = self.cell_of(xy)
        return self.inside(c) and bool(np.isfinite(self.cost[c]))

    def path_length(self, xy) -> float:
        c = self.cell_of(xy)
        if not self.inside(c):
            return np.inf
        return float(self.dist[c[0] * self.nx + c[1]])

    def path(self, xy) -> np.ndarray | None:
        """Waypoints [N, 2] from the robot to ``xy`` (line-of-sight shortcut), or None."""
        c = self.cell_of(xy)
        if not self.inside(c) or not np.isfinite(self.dist[c[0] * self.nx + c[1]]):
            return None
        k = c[0] * self.nx + c[1]
        cells = []
        while k >= 0:
            cells.append((k // self.nx, k % self.nx))
            k = self.pred[k]
        cells = cells[::-1]
        out = [cells[0]]
        i = 0
        while i < len(cells) - 1:
            j = len(cells) - 1
            while j > i + 1 and not self._line_ok(cells[i], cells[j]):
                j -= 1
            out.append(cells[j])
            i = j
        return np.array([self.xy_of(y, x) for y, x in out])

    def _line_ok(self, a, b) -> bool:
        n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) * 2 + 1
        rr = np.rint(np.linspace(a[0], b[0], n)).astype(int)
        cc = np.rint(np.linspace(a[1], b[1], n)).astype(int)
        c = self.cost[rr, cc]
        return bool(np.all(np.isfinite(c)) and np.all(c <= self.cfg.shortcut_max_cost))

    def pairwise(self, xys: list, coarse: int) -> np.ndarray:
        """Approximate path lengths between targets on a ``coarse``-times coarser grid [K, K] (m)."""
        if len(xys) < 2:
            return np.zeros((len(xys), len(xys)))
        k = max(1, coarse)
        ny, nx = self.ny // k, self.nx // k
        c = self.cost[: ny * k, : nx * k].reshape(ny, k, nx, k)
        fin = np.isfinite(c)
        n = fin.sum(axis=(1, 3))
        cc = np.where(n > 0, np.where(fin, c, 0.0).sum(axis=(1, 3)) / np.maximum(n, 1), np.inf)
        G = _graph(cc)
        nodes = []
        for p in xys:
            y = int(min(ny - 1, max(0, math.floor((p[1] - self.origin[1]) / (self.cell * k)))))
            x = int(min(nx - 1, max(0, math.floor((p[0] - self.origin[0]) / (self.cell * k)))))
            nodes.append(y * nx + x)
        D = dijkstra(G, indices=nodes)[:, nodes] * self.cell * k
        eu = np.linalg.norm(np.asarray(xys)[:, None] - np.asarray(xys)[None], axis=-1)
        return np.where(np.isfinite(D), np.maximum(D, eu), eu * self.cfg.unreachable_penalty)


def tour(start_cost: np.ndarray, pair: np.ndarray, two_opt_iters: int) -> list[int]:
    """Open TSP from the robot: greedy nearest neighbour + 2-opt. start_cost [K], pair [K, K] (time units)."""
    K = len(start_cost)
    if K == 0:
        return []
    left = set(range(K))
    cur = int(np.argmin(start_cost))
    order = [cur]
    left.remove(cur)
    while left:
        nxt = min(left, key=lambda j: pair[cur, j])
        order.append(nxt)
        left.remove(nxt)
        cur = nxt

    def edge(a, b):  # a = -1 is the robot
        return start_cost[b] if a < 0 else pair[a, b]

    # 2-opt on the open path robot -> order[0] -> ... (reverse order[i..j]; pair costs taken as symmetric)
    for _ in range(two_opt_iters):
        improved = False
        for i in range(0, K - 1):
            a = order[i - 1] if i > 0 else -1
            for j in range(i + 1, K):
                b, c = order[i], order[j]
                d = order[j + 1] if j + 1 < K else None
                old = edge(a, b) + (pair[c, d] if d is not None else 0.0)
                new = edge(a, c) + (pair[b, d] if d is not None else 0.0)
                if new < old - 1e-9:
                    order[i:j + 1] = order[i:j + 1][::-1]
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return order
