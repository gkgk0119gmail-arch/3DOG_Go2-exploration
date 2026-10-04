"""Grid A* for following ARiADNE waypoints on the live occupancy map.

The cost map keeps the robot in the middle of aisles: cells within ``robot_radius`` of an obstacle are
blocked, and cells within ``clearance`` get an extra cost that decays with distance. Unknown cells are
blocked (exploration waypoints always lie in known free space). The resulting 8-connected path is shortcut
with line-of-sight checks so the follower walks straight segments instead of a staircase.
"""

from __future__ import annotations

import heapq
import math

import numpy as np

FREE, OCCUPIED, UNKNOWN = 0, 100, -1
_SQRT2 = math.sqrt(2.0)
_NEIGHBORS = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
              (-1, -1, _SQRT2), (-1, 1, _SQRT2), (1, -1, _SQRT2), (1, 1, _SQRT2)]


def build_cost_map(grid: np.ndarray, cell: float, robot_radius: float = 0.3, clearance: float = 1.0,
                   clearance_weight: float = 3.0) -> np.ndarray:
    """Traversal cost per cell ([rows=y, cols=x]); ``inf`` where blocked."""
    from scipy import ndimage  # lazy: scipy/OpenBLAS must not load before Kit starts

    obstacle = grid == OCCUPIED
    dist = ndimage.distance_transform_edt(~obstacle) * cell  # [m] to the nearest obstacle
    cost = 1.0 + clearance_weight * np.clip((clearance - dist) / max(clearance - robot_radius, 1e-6), 0.0, 1.0) ** 2
    cost[dist <= robot_radius] = np.inf
    cost[grid == UNKNOWN] = np.inf
    return cost


def astar(cost: np.ndarray, start: tuple[int, int], goal: tuple[int, int], max_expansions: int = 200_000):
    """8-connected A* on ``cost`` ([row, col] indices). Returns the list of cells or None."""
    rows, cols = cost.shape
    if not (0 <= goal[0] < rows and 0 <= goal[1] < cols) or not np.isfinite(cost[goal]):
        return None
    g = np.full(cost.shape, np.inf)
    parent = -np.ones(cost.shape + (2,), dtype=np.int32)
    g[start] = 0.0
    h = lambda r, c: math.hypot(r - goal[0], c - goal[1])  # noqa: E731
    open_heap = [(h(*start), 0.0, start)]
    expansions = 0
    while open_heap:
        _, gc, (r, c) = heapq.heappop(open_heap)
        if (r, c) == goal:
            path = [(r, c)]
            while (r, c) != start:
                r, c = parent[r, c]
                path.append((int(r), int(c)))
            return path[::-1]
        if gc > g[r, c]:
            continue
        expansions += 1
        if expansions > max_expansions:
            return None
        for dr, dc, step in _NEIGHBORS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            w = cost[nr, nc]
            if not np.isfinite(w):
                continue
            if dr and dc and not (np.isfinite(cost[r + dr, c]) and np.isfinite(cost[r, c + dc])):
                continue  # no corner cutting
            ng = gc + step * w
            if ng < g[nr, nc]:
                g[nr, nc] = ng
                parent[nr, nc] = (r, c)
                heapq.heappush(open_heap, (ng + h(nr, nc), ng, (nr, nc)))
    return None


def _line_free(cost: np.ndarray, a, b, max_cost: float) -> bool:
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) * 2 + 1
    rr = np.rint(np.linspace(a[0], b[0], n)).astype(int)
    cc = np.rint(np.linspace(a[1], b[1], n)).astype(int)
    c = cost[rr, cc]
    return bool(np.all(np.isfinite(c)) and np.all(c <= max_cost))


def shortcut(cost: np.ndarray, path: list[tuple[int, int]], max_cost: float = 1.5) -> list[tuple[int, int]]:
    """Greedy line-of-sight smoothing; only shortcuts through low-cost (well-clear) cells."""
    if len(path) <= 2:
        return path
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not _line_free(cost, path[i], path[j], max_cost):
            j -= 1
        out.append(path[j])
        i = j
    return out


def nearest_free(cost: np.ndarray, cell: tuple[int, int], radius: int = 4):
    """Closest traversable cell to ``cell`` within ``radius`` cells (the robot may stand inside the margin)."""
    r0, c0 = cell
    best, best_d = None, np.inf
    for r in range(max(r0 - radius, 0), min(r0 + radius + 1, cost.shape[0])):
        for c in range(max(c0 - radius, 0), min(c0 + radius + 1, cost.shape[1])):
            if np.isfinite(cost[r, c]):
                d = (r - r0) ** 2 + (c - c0) ** 2
                if d < best_d:
                    best, best_d = (r, c), d
    return best


def plan_path(grid: np.ndarray, origin: np.ndarray, cell: float, start_xy, goal_xy, **cost_kwargs):
    """A* from ``start_xy`` to ``goal_xy`` [m]. Returns (waypoints [N, 2] in metres, cost map) or (None, cost)."""
    cost = build_cost_map(grid, cell, **cost_kwargs)
    to_cell = lambda p: (int((p[1] - origin[1]) // cell), int((p[0] - origin[0]) // cell))  # noqa: E731
    s = nearest_free(cost, to_cell(start_xy))
    g = nearest_free(cost, to_cell(goal_xy))
    if s is None or g is None:
        return None, cost
    path = astar(cost, s, g)
    if path is None:
        return None, cost
    path = shortcut(cost, path)
    xy = np.array([[origin[0] + (c + 0.5) * cell, origin[1] + (r + 0.5) * cell] for r, c in path])
    return xy, cost
