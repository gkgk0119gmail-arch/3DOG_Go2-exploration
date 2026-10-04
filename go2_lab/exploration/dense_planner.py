"""The ARiADNE training-code graph + policy as a sim planner (same interface as AriadnePlanner).

Runs exactly what the policy saw in training (sim/ariadne3d): a dense node lattice (NODE_RESOLUTION) over the
known free space, frontier utility, guidepost, and, for 6-input checkpoints, [heading cost, 3D utility].
The robot hops node to node: a new decision is made only when it reaches the current target (or the target
becomes invalid), as in training.

Termination (both 4- and 6-input checkpoints, so they are compared on the same rule): 2D exploration complete and
(3D utility at every node < U3D_DONE or STALL_STEPS decisions in a row that added < STALL_AREA of 3D surface).

Must be imported in its own process (planner_worker): ariadne3d's module names (parameter, agent, ...) clash with
the ROS planner's.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch

ARIADNE3D = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "ariadne3d")
DEFAULT_CKPT = os.path.join(ARIADNE3D, "model", "go2_turn_3d", "checkpoint.pth")


class DensePlanner:
    def __init__(self, checkpoint: str | None = None, node_res: float | None = None, reach: float = 1.2, **_):
        if node_res:
            os.environ["ARIADNE_NODE_RES"] = str(node_res)
        os.environ.setdefault("ARIADNE_NODE_PAD", "1500")
        if ARIADNE3D not in sys.path:
            sys.path.insert(0, ARIADNE3D)
        import parameter as P
        from agent import Agent
        from agent3d import Agent3D
        from lidar3d import Belief3D
        from model import PolicyNet
        from utils import MapInfo

        self.P, self._MapInfo, self._sample = P, MapInfo, Belief3D.sample
        ck = torch.load(checkpoint or DEFAULT_CKPT, map_location="cpu", weights_only=False)
        in_dim = ck["policy_model"]["initial_embedding.weight"].shape[1]
        net = PolicyNet(in_dim, P.EMBEDDING_DIM)
        net.load_state_dict(ck["policy_model"])
        net.eval()
        self.uses_3d = in_dim > 4
        self.robot = (Agent3D if self.uses_3d else Agent)(net, "cpu", False)
        self.pool = 2  # utility grid = occupancy grid pooled 2x2 (Live3D.utility)
        self.reach = reach
        self.map_info = None
        self.pos = None
        self.location = None  # current graph node
        self.target = None
        self.heading = 0.0
        self.util_grid = None
        self.seen_area = 0.0
        self._area_at_decision = 0.0
        self.done = self.done_2d = False
        self.stall = 0
        self.decisions = 0
        self.node_u3d = np.zeros(0)

    # -- inputs -------------------------------------------------------------------------------------
    def update_map(self, grid: np.ndarray, origin_xy):
        """ROS-layout grid (0 free / 100 occupied / -1 unknown) -> training layout (255 / 1 / 127)."""
        m = np.full(grid.shape, 127, dtype=int)
        m[grid == 0] = 255
        m[grid == 100] = 1
        self.map_info = self._MapInfo(m, float(origin_xy[0]), float(origin_xy[1]), self.P.CELL_SIZE)

    def update_location(self, xy):
        self.pos = np.asarray(xy, dtype=np.float64)

    def update_3d(self, heading: float, util_grid: np.ndarray | None, seen_area: float):
        self.heading, self.util_grid, self.seen_area = float(heading), util_grid, float(seen_area)

    # -- helpers --------------------------------------------------------------------------------------
    def _free(self, xy):
        mi = self.map_info
        cx, cy = int(round((xy[0] - mi.map_origin_x) / mi.cell_size)), int(round((xy[1] - mi.map_origin_y) / mi.cell_size))
        return 0 <= cy < mi.map.shape[0] and 0 <= cx < mi.map.shape[1] and mi.map[cy, cx] == 255

    def _snap(self, xy):
        """Nearest free lattice point (training nodes sit on multiples of NODE_RESOLUTION)."""
        r = self.P.NODE_RESOLUTION
        base = np.floor(np.asarray(xy) / r) * r
        cand = np.array([base + np.array([i, j]) * r for i in range(-1, 3) for j in range(-1, 3)])
        cand = np.around(cand[np.argsort(np.linalg.norm(cand - xy, axis=1))], 1)
        for c in cand:
            if self._free(c) and (len(self.robot.node_manager.nodes_dict) == 0
                                  or self.robot.node_manager.nodes_dict.find(c.tolist()) is not None):
                return c
        return None

    def _util3d(self, coords):
        if self.util_grid is None:
            return np.zeros(len(coords))
        mi = self.map_info
        cells = np.stack([(coords[:, 0] - mi.map_origin_x) / mi.cell_size, (coords[:, 1] - mi.map_origin_y) / mi.cell_size], 1)
        return self._sample(self, self.util_grid, cells)

    # -- one replanning tick ----------------------------------------------------------------------------
    def plan(self):
        if self.done or self.map_info is None or self.pos is None:
            return None if self.done else self.target
        if self.target is not None:
            if np.linalg.norm(self.pos - self.target) > self.reach and self._free(self.target):
                return self.target  # still walking to the chosen node
            self.location = self.target if self._free(self.target) else None
            self.target = None
        if self.location is None or self.robot.node_manager.nodes_dict.find(self.location.tolist()) is None:
            self.location = self._snap(self.pos)
            if self.location is None:
                return None
        self.robot.update_planning_state(self.map_info, self.location)
        self.node_u3d = self._util3d(self.robot.node_coords)
        # termination (same rule as training)
        gain = self.seen_area - self._area_at_decision
        self._area_at_decision = self.seen_area
        if self.robot.utility.sum() == 0:
            if self.done_2d:
                self.stall = self.stall + 1 if gain < self.P.STALL_AREA else 0
            self.done_2d = True
            if float(self.node_u3d.max(initial=0)) < self.P.U3D_DONE or self.stall >= self.P.STALL_STEPS:
                self.done = True
                return None
        if self.uses_3d:
            self.robot.set_3d(self.heading, self.util_grid if self.util_grid is not None else
                              np.zeros((self.map_info.map.shape[0] // 2, self.map_info.map.shape[1] // 2), np.float32), self)
        obs = self.robot.get_observation()
        _, _, _, _, current_edge, _ = obs
        with torch.no_grad():
            logp = self.robot.policy_net(*obs)
        a = int(torch.argmax(logp, dim=1))
        self.target = np.asarray(self.robot.node_coords[current_edge[0, a, 0].item()], dtype=np.float64)
        self.decisions += 1
        return self.target

    def sample(self, grid, cells):  # Agent3D calls belief3d.sample(grid, cells)
        return self._sample(self, grid, cells)

    def graph(self):
        coords = np.asarray(self.robot.node_coords) if self.robot.node_coords is not None else np.zeros((0, 2))
        util = np.asarray(self.robot.utility) if self.robot.utility is not None else np.zeros(0)
        frontier = np.array(list(self.robot.frontier)) if self.robot.frontier else np.zeros((0, 2))
        return coords, util, [], frontier
