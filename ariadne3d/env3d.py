"""ARiADNE training env + Go2 turning cost + tilted VLP-16 3D surface coverage.

Per decision step (move to a neighbouring graph node):
  time  t = |dpsi| / OMEGA + TURN_SETTLE * [|dpsi| > TURN_THR]  +  dist / V          (Go2 turn-then-go)
  scans every SCAN_TURN rad while turning in place and every SCAN_STEP m while walking (the VLP-16 runs at 10 Hz)
  r     = frontier term (original ARiADNE)  -  t / 16  (= original -dist/16 when there is no turn)
          + W3D * (new wall + ceiling area) / A_REF
          + 20 once, when the 2D exploration is complete (original bonus)
  done  = 2D complete and (3D utility gone or STALL_STEPS steps without 3D gain), or MAX_EPISODE_STEP
"""

import numpy as np

from env import Env
from lidar3d import Belief3D, Scene25D, VLP16
from parameter import (A_REF, MAX_EPISODE_STEP, OMEGA, SCAN_STEP, SCAN_TURN, STALL_AREA, STALL_STEPS, TILT_DEG,
                       TURN_SETTLE, TURN_THR, U3D_DONE, V_MAX, W3D)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Env3D(Env):
    def __init__(self, episode_index, plot=False, tilt_deg=TILT_DEG, seed=None):
        super().__init__(episode_index, plot)
        rng = np.random.default_rng(episode_index if seed is None else seed)
        self.scene = self.make_scene(rng)
        self.belief3d = Belief3D(self.scene, VLP16(self.cell_size, tilt_deg=tilt_deg))
        self.heading = float(rng.uniform(-np.pi, np.pi))
        self.time_s = 0.0
        self.turn_total = 0.0
        self.turn_time = 0.0
        self.done_2d = False
        self.stall = 0
        self.steps = 0
        self.belief3d.observe(self.robot_cell.astype(float), self.heading)
        self.last_gain = 0.0

    def make_scene(self, rng):
        return Scene25D(self.ground_truth, seed=int(rng.integers(1 << 31)))

    def _cell(self, xy):
        return np.array([(xy[0] - self.belief_origin_x) / self.cell_size, (xy[1] - self.belief_origin_y) / self.cell_size])

    def step(self, next_waypoint):
        start = self.robot_location.copy()
        seg = np.asarray(next_waypoint, float) - start
        dist = float(np.linalg.norm(seg))
        psi = float(np.arctan2(seg[1], seg[0]))
        dpsi = wrap(psi - self.heading)
        t_turn = abs(dpsi) / OMEGA + (TURN_SETTLE if abs(dpsi) > TURN_THR else 0.0)
        t_step = t_turn + dist / V_MAX

        # 3D sweeps along the motion: turning in place, then walking
        gain = np.zeros(3)
        n_turn = int(abs(dpsi) // SCAN_TURN)
        for k in range(1, n_turn + 1):
            gain += self.belief3d.observe(self._cell(start), self.heading + np.sign(dpsi) * k * SCAN_TURN)
        n_walk = max(1, int(np.ceil(dist / SCAN_STEP)))
        for k in range(1, n_walk + 1):
            gain += self.belief3d.observe(self._cell(start + seg * k / n_walk), psi)
        self.heading = psi
        self.turn_total += abs(dpsi)
        self.turn_time += t_turn
        self.time_s += t_step
        self.steps += 1

        # 2D: original ARiADNE update (sensing at the waypoint) and frontier reward
        self.update_robot_location(np.asarray(next_waypoint))
        self.update_robot_belief()
        self.travel_dist += dist
        self.evaluate_exploration_rate()
        reward = self.calculate_reward(dist)  # frontier term - dist / 16
        reward += dist / 16.0 - t_step / 16.0  # replace the distance cost by the time cost (turning included)
        self.last_gain = float(gain[0] + gain[1])  # walls + ceiling [m^2]
        reward += W3D * self.last_gain / A_REF
        return reward

    def check_done(self, utility_2d_sum, util3d_max):
        """Bonus and termination after a step. Returns (bonus, done)."""
        bonus = 0.0
        if not self.done_2d and utility_2d_sum == 0:
            self.done_2d = True
            self.time_2d, self.cov3d_at_2d = self.time_s, self.belief3d.coverage()
            bonus = 20.0
        if self.done_2d:
            self.stall = self.stall + 1 if self.last_gain < STALL_AREA else 0
        done = self.done_2d and (util3d_max < U3D_DONE or self.stall >= STALL_STEPS)
        return bonus, done or self.steps >= MAX_EPISODE_STEP

    def metrics(self):
        parts = self.belief3d.coverage_parts()
        return {"coverage_3d": self.belief3d.coverage(), "wall_cov": parts["wall"], "ceiling_cov": parts["ceiling"],
                "time_s": self.time_s, "turn_total_rad": self.turn_total, "turn_time_s": self.turn_time,
                "steps": self.steps, "done_2d": float(self.done_2d),
                "time_2d_s": getattr(self, "time_2d", np.nan), "cov3d_at_2d": getattr(self, "cov3d_at_2d", np.nan)}
