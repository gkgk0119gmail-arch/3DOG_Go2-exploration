"""BIM storeys (tools/bim_to_25d.py output) as 2.5D training-env worlds.

    ARIADNE_BIM_DIR=maps_train python driver3d.py --world bim      # or eval3d.py ... --world bim

Episode index -> one of the .npz worlds in ARIADNE_BIM_DIR (sorted, cycled) and a seeded random start
>= 1.2 m from obstacles. Heights and ceiling come from the BIM, as in warehouse_env.py for the Isaac warehouse.

Guide term (step 2, W_GUIDE > 0, set by driver3d.py --w_guide): at the start of an episode the BIM-based scan plan
(bim_expert.Plan.greedy, which sees the ground truth) is computed from the robot's start. Its stops are passed in
order; the target is the first stop not yet passed (the robot came within GUIDE_REACH x node spacing, in line of
sight, anywhere along its walk). Per step
    progress = (path distance to the target before - after) / step length      in [-1, 1]   (1 = straight at it)
    r       += W_GUIDE * (progress - 1) / 2                                      in [-W_GUIDE, 0]
added to the Env3D reward (or replacing it, GUIDE_ONLY). Path distances go around walls (8-connected, free cells).
Only the reward uses the plan: the policy still sees only its own belief.
"""

import functools
import glob
import os

import numpy as np
from scipy.ndimage import distance_transform_edt

from bim_expert import Plan, dist_field, line_free
from env3d import Env3D
from parameter import GUIDE_ONLY, GUIDE_REACH, NODE_RESOLUTION, TILT_DEG, W_GUIDE
from procwarehouse import ProcScene

BIM_DIR = os.environ.get("ARIADNE_BIM_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "maps_bim"))


@functools.lru_cache(maxsize=None)
def bim_worlds(bim_dir=BIM_DIR):
    files = sorted(glob.glob(os.path.join(bim_dir, "*.npz")))
    if not files:
        raise FileNotFoundError(f"no BIM worlds (*.npz) in {bim_dir}; run tools/bim_to_25d.py first")
    return files


@functools.lru_cache(maxsize=64)
def load_world(path):
    d = np.load(path)
    return d["gt"].astype(int), d["height"].astype(np.float32), float(d["H"])


def _vis_files(path, tilt_deg):
    base = os.path.join(os.path.dirname(path), ".vis", f"{os.path.basename(path)[:-4]}_t{tilt_deg:g}")
    return [f"{base}_{k}.npy" for k in ("uni", "ids", "off")]


def precompute_visibility(tilt_deg=TILT_DEG):
    """Plan visibility of every world to disk, once, before the workers start (driver3d.py): the workers then
    memory-map the same files instead of each holding ~1 GB per big storey."""
    for path in bim_worlds():
        files = _vis_files(path, tilt_deg)
        if all(os.path.exists(f) and os.path.getmtime(f) >= os.path.getmtime(path) for f in files):
            continue
        gt, height, H = load_world(path)
        vis = Plan(ProcScene(gt, height, H), (0, 0), tilt_deg=tilt_deg).visibility()
        os.makedirs(os.path.dirname(files[0]), exist_ok=True)
        for f, a in zip(files, vis):
            np.save(f + ".tmp.npy", a)
            os.replace(f + ".tmp.npy", f)
        print(f"[bim_env] visibility {os.path.basename(path)}: {len(vis[1]) / 1e6:.1f}M entries", flush=True)


@functools.lru_cache(maxsize=16)
def bim_plan(path, tilt_deg):
    """Plan on the ground-truth world; its per-candidate visibility (the slow part) is reused for every start."""
    gt, height, H = load_world(path)
    plan = Plan(ProcScene(gt, height, H), (0, 0), tilt_deg=tilt_deg)
    files = _vis_files(path, tilt_deg)
    if all(os.path.exists(f) for f in files):
        plan.vis = tuple(np.load(f, mmap_mode="r") for f in files)
    return plan


class BimEnv3D(Env3D):
    def __init__(self, episode_index, plot=False, tilt_deg=TILT_DEG, seed=None):
        super().__init__(episode_index, plot, tilt_deg=tilt_deg, seed=seed)
        self.guide_pts = np.zeros((0, 2))
        self.last_guide = 0.0
        if W_GUIDE > 0:
            self._plan_guide(tilt_deg)

    def import_ground_truth(self, episode_index):
        files = bim_worlds()
        self.world_file = files[episode_index % len(files)]
        gt, _, _ = load_world(self.world_file)
        dist = distance_transform_edt(gt == 255)
        cand = np.argwhere(dist >= 3)  # >= 1.2 m from obstacles
        if not len(cand):
            cand = np.argwhere(dist >= 2)
        y, x = cand[np.random.default_rng(2000 + episode_index).integers(len(cand))]
        return gt.copy(), np.array([x, y])

    def make_scene(self, rng):
        _, height, H = load_world(self.world_file)
        return ProcScene(self.ground_truth, height, H)

    # ---- guide term -------------------------------------------------------------------------------------
    def _plan_guide(self, tilt_deg):
        plan = bim_plan(self.world_file, tilt_deg)
        self.guide_free = plan.free
        plan.start = self._free_cell(self._cell(self.robot_location))
        plan.greedy(heading=self.heading)
        self.guide_pts = np.asarray(plan.points, float).reshape(-1, 2)
        self.guide_done = np.zeros(len(self.guide_pts), bool)
        self.guide_fields = {}
        self.guide_reach = GUIDE_REACH * NODE_RESOLUTION / self.cell_size  # [cells]
        here = self._cell(self.robot_location)
        self._pass_guide(here, here)

    def _free_cell(self, xy):
        c = np.rint(xy).astype(int)
        ny, nx = self.guide_free.shape
        if 0 <= c[1] < ny and 0 <= c[0] < nx and self.guide_free[c[1], c[0]]:
            return c.astype(float)
        ys, xs = np.nonzero(self.guide_free)
        k = int(np.argmin((xs - xy[0]) ** 2 + (ys - xy[1]) ** 2))
        return np.array([xs[k], ys[k]], float)

    def _path_dist(self, field, xy):
        """Path distance [cells] from the field's target to xy (nearest finite cell in the 3 x 3 around it), or nan."""
        c = np.rint(xy).astype(int)
        ny, nx = field.shape
        win = field[max(c[1] - 1, 0):min(c[1] + 2, ny), max(c[0] - 1, 0):min(c[0] + 2, nx)]
        if 0 <= c[1] < ny and 0 <= c[0] < nx and np.isfinite(field[c[1], c[0]]):
            return float(field[c[1], c[0]])
        return float(win.min()) if win.size and np.isfinite(win.min()) else np.nan

    def _target(self):
        """Index of the first plan stop not yet passed and reachable from the robot, or None."""
        here = self._cell(self.robot_location)
        for k in np.flatnonzero(~self.guide_done):
            if k not in self.guide_fields:
                self.guide_fields[k] = dist_field(self.guide_free, self.guide_pts[k])
            if not np.isnan(self._path_dist(self.guide_fields[k], here)):
                return k
            self.guide_done[k] = True  # unreachable from here: skip it
        return None

    def _pass_guide(self, a, b):
        """Mark plan stops within reach (in line of sight) of the walk a -> b as passed."""
        if not len(self.guide_pts):
            return
        n = max(1, int(np.ceil(np.linalg.norm(b - a) * self.cell_size)))  # every ~1 m
        for f in np.linspace(0, 1, n + 1):
            p = a + (b - a) * f
            near = np.flatnonzero(~self.guide_done & (np.linalg.norm(self.guide_pts - p, axis=1) <= self.guide_reach))
            for k in near:
                if line_free(self.guide_free, self._free_cell(p), self.guide_pts[k]):
                    self.guide_done[k] = True

    def step(self, next_waypoint):
        if W_GUIDE <= 0:
            return super().step(next_waypoint)
        a = self._cell(self.robot_location)
        k = self._target()
        reward = super().step(next_waypoint)
        b = self._cell(self.robot_location)
        r = 0.0
        if k is not None:
            field = self.guide_fields[k]
            da, db, step = self._path_dist(field, a), self._path_dist(field, b), float(np.linalg.norm(b - a))
            progress = 0.0 if step < 1e-6 or np.isnan(da) or np.isnan(db) else float(np.clip((da - db) / step, -1, 1))
            r = W_GUIDE * (progress - 1) / 2
        self._pass_guide(a, b)
        self.last_guide = r
        return (0.0 if GUIDE_ONLY else reward) + r

    def metrics(self):
        m = super().metrics()
        if W_GUIDE > 0:
            m["guide_frac"] = float(self.guide_done.mean()) if len(self.guide_done) else 1.0  # plan stops passed
        return m
