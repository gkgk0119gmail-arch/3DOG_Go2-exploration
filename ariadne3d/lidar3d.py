"""2.5D world, tilted VLP-16 and 3D surface belief for training ARiADNE on 3D coverage.

The ARiADNE training maps are 2D (free / occupied, 0.4 m cells). ``Scene25D`` lifts one into a building:
every obstacle component gets a height (outer walls and anything touching the map border reach the ceiling;
free-standing components become low boxes, partitions or tall shelves), and the map gets a ceiling height.

Surface elements (what "3D coverage" counts):
  wall    : (obstacle cell next to free space, 0.5 m height bin below the obstacle top)   0.4 x 0.5 m
  ceiling : every cell whose column is not full height                                    0.4 x 0.4 m
  floor   : free cells (tracked for the metric only; the 2D frontier reward already covers it)

``VLP16`` ray-marches the 16 beams (+-15 deg, tilted nose-up by ``tilt_deg``) at 1 deg azimuth against the scene.
``Belief3D`` keeps what has been seen (walls / ceiling / floor + wall bins a beam passed over = known empty) and
turns the unseen part into a per-location 3D utility: unseen elements whose height lies inside the vertical field
of view from that distance, summed with annulus convolutions (no line of sight; cheap, smooth).
"""

import numpy as np
from scipy.ndimage import binary_dilation
from scipy.signal import fftconvolve
from skimage.measure import label

FREE, OCCUPIED = 255, 1
DZ = 0.5  # wall height bin [m]
H_MAX = 10.0  # highest ceiling the belief assumes before it has seen one [m]
NB = int(round(H_MAX / DZ))
WALL_AREA, CEIL_AREA = 0.4 * DZ, 0.4 * 0.4
CLASSES = [(0.0, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, H_MAX)]  # wall height classes for the utility
FOUR = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], bool)


class Scene25D:
    def __init__(self, ground_truth, seed):
        rng = np.random.default_rng(seed)
        occ = ground_truth == OCCUPIED
        self.free = ground_truth == FREE
        self.H = float(rng.uniform(6.0, H_MAX))
        lab, n = label(occ, connectivity=2, return_num=True)
        border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
        h_of = np.zeros(n + 1, np.float32)
        for comp in range(1, n + 1):
            if comp in border:
                h_of[comp] = self.H
                continue
            kind = rng.random()
            if kind < 0.3:
                h_of[comp] = rng.uniform(0.3, 1.0)  # boxes, pallets
            elif kind < 0.6:
                h_of[comp] = rng.uniform(1.0, 3.0)  # partitions, machines
            elif kind < 0.9:
                h_of[comp] = rng.uniform(3.0, min(8.0, self.H - 0.5))  # racks
            else:
                h_of[comp] = self.H  # full-height walls inside
        self.height = h_of[lab]  # [y, x] metres, 0 on free cells
        boundary = occ & binary_dilation(self.free, FOUR)
        z = (np.arange(NB) + 0.5) * DZ
        self.wall_true = boundary[..., None] & (z[None, None, :] < self.height[..., None])
        self.ceil_true = self.height < self.H - 1e-3
        self.floor_true = self.free
        self.total_area = self.wall_true.sum() * WALL_AREA + (self.ceil_true.sum() + self.floor_true.sum()) * CEIL_AREA


class VLP16:
    def __init__(self, cell_size, tilt_deg=15.0, z_sensor=0.44, max_range=30.0, az_res_deg=1.0):
        self.cell = cell_size
        self.tilt = np.deg2rad(tilt_deg)
        self.zs = z_sensor
        self.az = np.deg2rad(np.arange(0.0, 360.0, az_res_deg))
        self.el = np.deg2rad(np.linspace(-15.0, 15.0, 16))
        self.r = np.arange(1, int(max_range / cell_size) + 1) * cell_size
        self.dx = np.cos(self.az)[:, None] * self.r[None, :] / cell_size  # [A, S] in cells
        self.dy = np.sin(self.az)[:, None] * self.r[None, :] / cell_size

    def scan(self, scene, cell_xy, heading):
        """One sweep from map cell (x, y) (float) with the body facing ``heading``.

        Returns wall hits (y, x, bin), ceiling hits (y, x), floor hits (y, x), passed-over wall bins (y, x, bin).
        """
        ny, nx = scene.height.shape
        ix = np.rint(cell_xy[0] + self.dx).astype(np.int64)
        iy = np.rint(cell_xy[1] + self.dy).astype(np.int64)
        inside = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
        ixc, iyc = np.clip(ix, 0, nx - 1), np.clip(iy, 0, ny - 1)
        hgt = np.where(inside, scene.height[iyc, ixc], scene.H)  # [A, S]
        # nose-up tilt raises the beams ahead and lowers the ones behind
        e = self.el[:, None] + self.tilt * np.cos(self.az - heading)[None, :]  # [K, A]
        zb = self.zs + np.tan(e)[:, :, None] * self.r[None, None, :]  # [K, A, S]
        event = (zb < hgt[None]) | (zb > scene.H)
        hit = event.any(axis=2)
        first = np.argmax(event, axis=2)  # [K, A]
        k, a = np.nonzero(hit)
        s = first[k, a]
        z_e = zb[k, a, s]
        cx, cy = ixc[a, s], iyc[a, s]
        is_ceil = z_e > scene.H
        is_floor = (z_e < 0) & (hgt[a, s] <= 0)
        is_wall = ~is_ceil & ~is_floor & (hgt[a, s] > 0)
        bins = np.clip((z_e / DZ).astype(np.int64), 0, NB - 1)
        walls = np.stack([cy[is_wall], cx[is_wall], bins[is_wall]], 1)
        ceil = np.stack([cy[is_ceil], cx[is_ceil]], 1)
        floor = np.stack([cy[is_floor], cx[is_floor]], 1)
        # beams that cross over an obstacle column before their event: that height bin is empty there
        before = np.arange(self.r.size)[None, None, :] < np.where(hit, first, self.r.size)[..., None]
        over = before & (hgt[None] > 0) & (zb >= 0) & (zb < scene.H)
        kk, aa, ss = np.nonzero(over)
        empty = np.stack([iyc[aa, ss], ixc[aa, ss], np.clip((zb[kk, aa, ss] / DZ).astype(np.int64), 0, NB - 1)], 1)
        return walls, ceil, floor, empty


def annulus(r_min, r_max, cell):
    n = int(np.ceil(r_max / cell))
    y, x = np.mgrid[-n:n + 1, -n:n + 1] * cell
    d = np.hypot(x, y)
    return ((d >= r_min) & (d <= r_max)).astype(np.float32)


class Belief3D:
    """What the robot has seen in 3D, the reward bookkeeping and the 3D utility maps (belief and ground truth)."""

    def __init__(self, scene, sensor, utility_range=20.0, pool=2):
        self.scene, self.sensor = scene, sensor
        shape = scene.height.shape
        self.wall = np.zeros(shape + (NB,), np.uint8)  # 0 unknown, 1 seen surface, 2 seen empty
        self.ceil = np.zeros(shape, bool)
        self.floor = np.zeros(shape, bool)
        self.ceiling_known = False
        self.pool = pool
        self.ucell = sensor.cell * pool
        up = np.tan(np.deg2rad(15.0) + sensor.tilt)  # most optimistic upward reach (straight ahead)
        down = np.tan(np.deg2rad(15.0))

        def r_min(z):
            return (z - sensor.zs) / up if z > sensor.zs else (sensor.zs - z) / down

        self._r_min = r_min
        self.kernels = [annulus(min(r_min(lo), utility_range), utility_range, self.ucell) for lo, _ in CLASSES]
        self.utility_range = utility_range
        self._ceil_kernel_for = {}

    # -- sensing ---------------------------------------------------------------------------------------
    def observe(self, cell_xy, heading):
        """Apply one sweep; return newly seen true area [m^2] (walls, ceiling, floor)."""
        sc = self.scene
        walls, ceil, floor, empty = self.sensor.scan(sc, cell_xy, heading)
        if len(empty):
            w = self.wall[empty[:, 0], empty[:, 1], empty[:, 2]]
            keep = w == 0
            self.wall[empty[keep, 0], empty[keep, 1], empty[keep, 2]] = 2
        new_w = new_c = new_f = 0
        if len(walls):
            walls = np.unique(walls, axis=0)
            true = sc.wall_true[walls[:, 0], walls[:, 1], walls[:, 2]]
            walls = walls[true]
            new_w = int((self.wall[walls[:, 0], walls[:, 1], walls[:, 2]] != 1).sum())
            self.wall[walls[:, 0], walls[:, 1], walls[:, 2]] = 1
        if len(ceil):
            ceil = np.unique(ceil, axis=0)
            ceil = ceil[sc.ceil_true[ceil[:, 0], ceil[:, 1]]]
            new_c = int((~self.ceil[ceil[:, 0], ceil[:, 1]]).sum())
            self.ceil[ceil[:, 0], ceil[:, 1]] = True
            self.ceiling_known = self.ceiling_known or len(ceil) > 0
        if len(floor):
            floor = np.unique(floor, axis=0)
            new_f = int((~self.floor[floor[:, 0], floor[:, 1]]).sum())
            self.floor[floor[:, 0], floor[:, 1]] = True
        return new_w * WALL_AREA, new_c * CEIL_AREA, new_f * CEIL_AREA

    def coverage(self):
        sc = self.scene
        seen = ((self.wall == 1) & sc.wall_true).sum() * WALL_AREA + \
               ((self.ceil & sc.ceil_true).sum() + (self.floor & sc.floor_true).sum()) * CEIL_AREA
        return float(seen / sc.total_area)

    def coverage_parts(self):
        sc = self.scene
        return {"wall": float(((self.wall == 1) & sc.wall_true).sum() / max(1, sc.wall_true.sum())),
                "ceiling": float((self.ceil & sc.ceil_true).sum() / max(1, sc.ceil_true.sum())),
                "floor": float((self.floor & sc.floor_true).sum() / max(1, sc.floor_true.sum()))}

    # -- utility ---------------------------------------------------------------------------------------
    def _pool(self, a):
        p = self.pool
        ny, nx = a.shape
        a = a[: ny // p * p, : nx // p * p]
        return a.reshape(ny // p, p, nx // p, p).sum(axis=(1, 3))

    def _utility(self, wall_unknown, ceil_unknown, H):
        """wall_unknown [y, x, NB] bool, ceil_unknown [y, x] bool -> utility grid (unseen elements in view)."""
        u = np.zeros((wall_unknown.shape[0] // self.pool, wall_unknown.shape[1] // self.pool), np.float32)
        nb_h = int(np.ceil(H / DZ))
        for (lo, hi), ker in zip(CLASSES, self.kernels):
            b0, b1 = int(lo / DZ), min(int(hi / DZ), nb_h)
            if b1 <= b0:
                continue
            cnt = self._pool(wall_unknown[:, :, b0:b1].sum(axis=2).astype(np.float32))
            if cnt.any():
                u += fftconvolve(cnt, ker, mode="same")
        key = round(H, 1)
        if key not in self._ceil_kernel_for:
            r0 = self._r_min(H)
            self._ceil_kernel_for[key] = annulus(r0, self.utility_range, self.ucell) if r0 < self.utility_range else None
        ker = self._ceil_kernel_for[key]
        if ker is not None:
            cnt = self._pool(ceil_unknown.astype(np.float32))
            if cnt.any():
                u += fftconvolve(cnt, ker, mode="same")
        return np.maximum(u, 0.0)

    def belief_utility(self, belief_map):
        """Utility from what the robot knows: unknown wall bins on known obstacle edges + unseen ceiling over known free."""
        H = self.scene.H if self.ceiling_known else H_MAX
        known_free = belief_map == FREE
        edge = (belief_map == OCCUPIED) & binary_dilation(known_free, FOUR)
        nb_h = int(np.ceil(H / DZ))
        wall_unknown = (self.wall == 0) & edge[..., None]
        wall_unknown[:, :, nb_h:] = False
        return self._utility(wall_unknown, known_free & ~self.ceil, H)

    def true_utility(self):
        """Ground-truth version for the critic: true surfaces not seen yet."""
        sc = self.scene
        return self._utility(sc.wall_true & (self.wall != 1), sc.ceil_true & ~self.ceil, sc.H)

    def sample(self, grid, cell_xy):
        """Utility grid value at map cell positions [N, 2] (x, y)."""
        cx = np.clip((np.asarray(cell_xy)[:, 0] // self.pool).astype(int), 0, grid.shape[1] - 1)
        cy = np.clip((np.asarray(cell_xy)[:, 1] // self.pool).astype(int), 0, grid.shape[0] - 1)
        return grid[cy, cx]
