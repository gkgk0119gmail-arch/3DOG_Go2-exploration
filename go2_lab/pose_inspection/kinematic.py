"""Kinematic engine for the inspection session (no Isaac): turn-then-go base motion at the planner's nominal speeds,
rate-limited body posture, instantaneous 10 Hz VLP-16 sweeps. POSE evaluates with a kinematic Spot; we use this for
development and quick sweeps, and the Isaac Lab Go2 (posture-walk policy) for the reported experiments.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from .raycast import rot_ypr
from .session import Session


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class KinematicEngine:
    def __init__(self, session: Session, starts: np.ndarray):
        self.s = session
        self.cfg = session.cfg
        self.pose = starts.astype(float).copy()       # [E, 3] x, y, yaw
        self.post = np.zeros((len(starts), 2))         # measured pitch, roll
        self.turning = np.ones(len(starts), bool)

    def _apply(self, cmds, dt):
        k, p = self.cfg.kinematic, self.cfg.planner
        rate = math.radians(k.posture_rate_deg_s) * dt
        for e, c in enumerate(cmds):
            if c is None:
                continue
            tgt = np.asarray(c.get("posture", (0.0, 0.0)), float)
            self.post[e] += np.clip(tgt - self.post[e], -rate, rate)
            x, y, yaw = self.pose[e]
            if c["mode"] == "move":
                d = np.asarray(c["waypoint"]) - self.pose[e, :2]
                err = wrap(math.atan2(d[1], d[0]) - yaw)
                self.turning[e] = abs(err) > (k.go_threshold if self.turning[e] else k.turn_threshold)
                w = np.clip(1.5 * err, -p.yaw_rate, p.yaw_rate)
                v = 0.0 if self.turning[e] else min(p.speed, max(0.3, 0.8 * np.linalg.norm(d)))
                yaw = yaw + w * dt
                x, y = x + v * math.cos(yaw) * dt, y + v * math.sin(yaw) * dt
            elif c["mode"] == "turn":
                err = wrap(c["yaw"] - yaw)
                yaw = yaw + np.clip(1.5 * err, -p.yaw_rate, p.yaw_rate) * dt
            self.pose[e] = (x, y, wrap(yaw))

    def body(self):
        E = len(self.pose)
        base = torch.tensor(np.c_[self.pose[:, :2], np.full(E, self.cfg.robot.base_height)], dtype=torch.float32)
        R = rot_ypr(torch.tensor(self.pose[:, 2]), torch.tensor(self.post[:, 0]), torch.tensor(self.post[:, 1]))
        return base, R

    def run(self, log=print):
        dt = self.cfg.kinematic.dt
        scan_every = max(1, int(round(1.0 / (self.cfg.lidar.rate_hz * dt))))
        t, k = 0.0, 0
        while True:
            if k % scan_every == 0:
                self.s.scan(t, *self.body())
            cmds = self.s.control(t, self.pose, self.post)
            self._apply(cmds, dt)
            done = self.s.log(t, self.pose)
            if done.all():
                break
            if k % int(60.0 / dt) == 0:
                co = self.s.surf.co().cpu().numpy()
                _, cs = self.s.cs()
                log(f"  t={t:6.0f}s  Co {np.round(co, 3)}  Cs {np.round(cs, 3)}  done {done.sum()}/{len(done)}")
            t += dt
            k += 1
        return t
