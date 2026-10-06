"""BIM-based optimal scan plan on a 2.5D world: the reference path a policy is trained to follow (step 2).

It sees the ground truth (the BIM), so it is an upper bound, not something the robot can run in the field.
  1. candidates  free cells on a grid, >= 0.4 m from obstacles
  2. visibility  wall / ceiling elements the tilted VLP-16 sees from each candidate (8 headings)
  3. plan        greedy on coverage rate: from where the robot is, go to the candidate with the most new elements
                 per second of Go2 time (turn-then-go), until 98 % of what any candidate sees is covered
                 (select() + order() is the set-cover + tour alternative: fewer stops, slower to reach coverage)
The plan is then executed with the same time model and 3D sweeps as env3d.Env3D.step, so its coverage-vs-time
curve is directly comparable with a policy's.
"""

import heapq

import numpy as np
from scipy.ndimage import distance_transform_edt

from lidar3d import Belief3D, VLP16
from parameter import OMEGA, SCAN_STEP, SCAN_TURN, TILT_DEG, TURN_SETTLE, TURN_THR, V_MAX


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def line_free(free, a, b):
    n = int(np.ceil(np.abs(np.subtract(b, a)).max() * 2)) + 1
    xs = np.rint(np.linspace(a[0], b[0], n)).astype(int)
    ys = np.rint(np.linspace(a[1], b[1], n)).astype(int)
    return bool(free[ys, xs].all())


def grid_path(free, a, b):
    """Shortest 8-connected path between cells a, b ([x, y]), string-pulled to straight segments."""
    ny, nx = free.shape
    start, goal = (int(a[1]), int(a[0])), (int(b[1]), int(b[0]))
    dist, prev, pq = {start: 0.0}, {}, [(0.0, start)]
    steps = [(dy, dx, np.hypot(dy, dx)) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
    while pq:
        d, u = heapq.heappop(pq)
        if u == goal:
            break
        if d > dist[u]:
            continue
        for dy, dx, c in steps:
            v = (u[0] + dy, u[1] + dx)
            if 0 <= v[0] < ny and 0 <= v[1] < nx and free[v] and d + c < dist.get(v, 1e18):
                dist[v], prev[v] = d + c, u
                heapq.heappush(pq, (d + c, v))
    if goal not in dist:
        return None
    cells = [goal]
    while cells[-1] != start:
        cells.append(prev[cells[-1]])
    cells = [np.array([x, y], float) for y, x in reversed(cells)]
    out = [cells[0]]
    i = 0
    while i < len(cells) - 1:
        j = len(cells) - 1
        while j > i + 1 and not line_free(free, cells[i], cells[j]):
            j -= 1
        out.append(cells[j])
        i = j
    return out


def dist_field(free, cell_xy):
    """8-connected path length [cells] from cell_xy to every free cell (inf where unreachable)."""
    ny, nx = free.shape
    d = np.full(free.shape, np.inf)
    s = (int(cell_xy[1]), int(cell_xy[0]))
    d[s] = 0.0
    pq = [(0.0, s)]
    steps = [(dy, dx, np.hypot(dy, dx)) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx]
    while pq:
        du, u = heapq.heappop(pq)
        if du > d[u]:
            continue
        for dy, dx, c in steps:
            v = (u[0] + dy, u[1] + dx)
            if 0 <= v[0] < ny and 0 <= v[1] < nx and free[v] and du + c < d[v]:
                d[v] = du + c
                heapq.heappush(pq, (du + c, v))
    return d


class Plan:
    def __init__(self, scene, start, cell=0.4, spacing=2, tilt_deg=TILT_DEG, min_clear=1.0):
        self.scene, self.cell = scene, cell
        self.free = scene.free
        self.start = np.asarray(start, float)
        self.sensor = VLP16(cell, tilt_deg=tilt_deg)
        clear = distance_transform_edt(self.free)
        ys, xs = np.nonzero((clear >= min_clear) & (np.add.outer(np.arange(self.free.shape[0]) % spacing,
                                                          np.arange(self.free.shape[1]) % spacing) == 0))
        self.cands = np.stack([xs, ys], 1).astype(float)

    def _seen(self, xy):
        """Ids of true wall bins and ceiling cells seen from xy over 8 headings."""
        sc, ny, nx = self.scene, *self.scene.free.shape
        ids = set()
        for h in np.arange(8) * np.pi / 4:
            walls, ceil, _, _ = self.sensor.scan(sc, xy, h)
            if len(walls):
                walls = walls[sc.wall_true[walls[:, 0], walls[:, 1], walls[:, 2]]]
                ids.update(((walls[:, 0] * nx + walls[:, 1]) * 100 + walls[:, 2]).tolist())
            if len(ceil):
                ceil = ceil[sc.ceil_true[ceil[:, 0], ceil[:, 1]]]
                ids.update((-(ceil[:, 0] * nx + ceil[:, 1]) - 1).tolist())
        return ids

    def select(self, target=0.98):
        seen = [self._seen(c) for c in self.cands]
        universe = set().union(*seen)
        covered, chosen = set(self._seen(self.start)), []
        while len(covered) < target * len(universe):
            gains = [len(s - covered) for s in seen]
            k = int(np.argmax(gains))
            if gains[k] == 0:
                break
            chosen.append(k)
            covered |= seen[k]
        self.points = self.cands[chosen]
        return self.points

    def visibility(self):
        """What each candidate sees per heading, compact (Python sets of all of it took ~1 GB per big storey):
        uni [U] sorted element ids, ids [N] uint32 indices into uni, off [8 n_cand + 1];
        candidate k, heading h -> ids[off[8 k + h]:off[8 k + h + 1]]. Independent of the start."""
        raw = [np.fromiter(self._seen_dir(c, h), np.int64) for c in self.cands for h in np.arange(8) * np.pi / 4]
        flat = np.concatenate(raw)
        uni = np.unique(flat)
        off = np.zeros(len(raw) + 1, np.int64)
        off[1:] = np.cumsum([len(r) for r in raw])
        return uni, np.searchsorted(uni, flat).astype(np.uint32), off

    def greedy(self, target=0.98, overhead=2.0, heading=0.0):
        """Coverage-rate greedy tour; sets self.seq / self.paths for execute()."""
        if getattr(self, "vis", None) is None:  # kept for the next start (bim_env.py also loads it from disk)
            self.vis = self.visibility()
        uni, ids, off = self.vis
        covered = np.zeros(len(uni), bool)

        def cover(xy, h):  # elements outside what any candidate sees do not count
            raw = np.fromiter(self._seen_dir(xy, h), np.int64)
            i = np.minimum(np.searchsorted(uni, raw), len(uni) - 1)
            covered[i[uni[i] == raw]] = True

        cover(self.start, heading)
        pos, pts = self.start, [self.start]
        visited = set()
        while covered.sum() < target * len(uni) and len(pts) < 400:
            d = dist_field(self.free, pos)
            best, k_best = 0.0, None
            for k, c in enumerate(self.cands):
                dk = d[int(c[1]), int(c[0])]
                if k in visited or not np.isfinite(dk) or dk == 0:
                    continue
                v = c - pos
                psi = np.arctan2(v[1], v[0])  # arrival heading (straight-line estimate): what the tilted lidar sees
                j = 8 * k + int(np.round(wrap(psi) / (np.pi / 4))) % 8
                seen = ids[off[j]:off[j + 1]]
                g = len(seen) - int(covered[seen].sum())
                if g == 0:
                    continue
                turn = abs(wrap(psi - heading))
                t = dk * self.cell / V_MAX + turn / OMEGA + (TURN_SETTLE if turn > TURN_THR else 0.0)
                if g / (t + overhead) > best:
                    best, k_best = g / (t + overhead), k
            if k_best is None:
                break
            path = grid_path(self.free, pos, self.cands[k_best])
            for a, b in zip(path[:-1], path[1:]):  # what is seen on the way counts too
                for f in np.linspace(0, 1, max(2, int(np.linalg.norm(b - a) * self.cell / SCAN_STEP) + 1)):
                    cover(a + (b - a) * f, np.arctan2(*(b - a)[::-1]))
            heading = float(np.arctan2(*(path[-1] - path[-2])[::-1])) if len(path) > 1 else heading
            cover(path[-1], heading)
            visited.add(k_best)  # each stop once: its predicted gain may not all show up from the real arrival heading
            pos = self.cands[k_best]
            pts.append(pos)
        self.points = np.array(pts[1:])
        self.seq = list(range(len(pts)))
        self.paths = {(i, i + 1): grid_path(self.free, pts[i], pts[i + 1]) for i in range(len(pts) - 1)}
        return self.points

    def _seen_dir(self, xy, h):
        sc, nx = self.scene, self.scene.free.shape[1]
        walls, ceil, _, _ = self.sensor.scan(sc, xy, h)
        ids = set()
        if len(walls):
            walls = walls[sc.wall_true[walls[:, 0], walls[:, 1], walls[:, 2]]]
            ids.update(((walls[:, 0] * nx + walls[:, 1]) * 100 + walls[:, 2]).tolist())
        if len(ceil):
            ceil = ceil[sc.ceil_true[ceil[:, 0], ceil[:, 1]]]
            ids.update((-(ceil[:, 0] * nx + ceil[:, 1]) - 1).tolist())
        return ids

    def _time(self, path, heading):
        """Go2 time along a polyline from a heading; returns (time, final heading)."""
        t = 0.0
        for a, b in zip(path[:-1], path[1:]):
            seg = (b - a) * self.cell
            psi = float(np.arctan2(seg[1], seg[0]))
            d = abs(wrap(psi - heading))
            t += d / OMEGA + (TURN_SETTLE if d > TURN_THR else 0.0) + np.linalg.norm(seg) / V_MAX
            heading = psi
        return t, heading

    def order(self):
        pts = [self.start] + list(self.points)
        n = len(pts)
        paths = {}
        for i in range(n):
            for j in range(n):
                if i != j and (j, i) not in paths:
                    p = grid_path(self.free, pts[i], pts[j])
                    paths[(i, j)] = p
                    paths[(j, i)] = None if p is None else p[::-1]
        ok = [j for j in range(1, n) if paths[(0, j)] is not None]
        length = lambda i, j: sum(np.linalg.norm(b - a) for a, b in zip(paths[(i, j)][:-1], paths[(i, j)][1:]))

        def cost(seq):
            t, h = 0.0, 0.0
            for i, j in zip(seq[:-1], seq[1:]):
                dt, h = self._time(paths[(i, j)], h)
                t += dt
            return t

        seq = [0]
        rest = set(ok)
        while rest:  # nearest neighbour by distance
            j = min(rest, key=lambda k: length(seq[-1], k))
            seq.append(j)
            rest.remove(j)
        best, improved = cost(seq), True
        while improved:  # 2-opt on Go2 time (open tour, start fixed)
            improved = False
            for i in range(1, len(seq) - 1):
                for j in range(i + 1, len(seq)):
                    cand = seq[:i] + seq[i:j + 1][::-1] + seq[j + 1:]
                    c = cost(cand)
                    if c < best - 1e-6:
                        seq, best, improved = cand, c, True
        self.seq, self.paths = seq, paths
        return seq

    def execute(self, heading=0.0):
        """Run the plan with env3d's time model and sweeps; returns [(time s, 3D coverage)] and the polyline."""
        b = Belief3D(self.scene, self.sensor)
        b.observe(self.start, heading)
        t, curve, poly = 0.0, [(0.0, b.coverage())], [self.start]
        for i, j in zip(self.seq[:-1], self.seq[1:]):
            path = self.paths[(i, j)]
            for a, c in zip(path[:-1], path[1:]):
                seg = c - a
                dist = float(np.linalg.norm(seg)) * self.cell
                psi = float(np.arctan2(seg[1], seg[0]))
                dpsi = wrap(psi - heading)
                for k in range(1, int(abs(dpsi) // SCAN_TURN) + 1):
                    b.observe(a, heading + np.sign(dpsi) * k * SCAN_TURN)
                t += abs(dpsi) / OMEGA + (TURN_SETTLE if abs(dpsi) > TURN_THR else 0.0)
                n = max(1, int(np.ceil(dist / SCAN_STEP)))
                for k in range(1, n + 1):
                    b.observe(a + seg * k / n, psi)
                    curve.append((t + dist * k / n / V_MAX, b.coverage()))
                t += dist / V_MAX
                heading = psi
                poly.append(c)
        return np.array(curve), np.array(poly)
