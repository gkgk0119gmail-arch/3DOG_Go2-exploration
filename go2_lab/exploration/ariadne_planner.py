"""ROS-free wrapper around the ARiADNE planner (marmotlab/ARiADNE-ROS-Planner, RA-L 2024 checkpoint).

Mirrors ``src/scripts/rl_planner.py`` (Runner.run) with the ROS I/O removed:
    - ``update_map(grid, origin_xy)`` replaces the ``/projected_map`` callback
      (ROS OccupancyGrid layout: rows = y, cols = x, values FREE=0 / OCCUPIED=100 / UNKNOWN=-1)
    - ``update_location(xy)`` replaces the ``/state_estimation`` callback
    - ``plan()`` is one replanning tick and returns the next waypoint (or None once exploration is done)
Parameters follow ``src/launch/rl_planner.launch``.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import torch

PLANNER_DIR = Path(__file__).resolve().parents[2] / "third_party" / "ARiADNE-ROS-Planner" / "src" / "scripts"

# the planner core only uses rospy for log messages
sys.modules.setdefault(
    "rospy", types.SimpleNamespace(loginfo=lambda *a: None, logdebug=lambda *a: None, logwarn=print)
)
if str(PLANNER_DIR) not in sys.path:
    sys.path.insert(0, str(PLANNER_DIR))
# NumPy 2 removed np.lib.pad, which utils.get_frontier_in_map still calls
if not hasattr(np.lib, "pad"):
    np.lib.pad = np.pad

import parameter  # noqa: E402
from agent import Agent  # noqa: E402
from model import PolicyNet  # noqa: E402
from node_manager import NodeManager  # noqa: E402
from utils import MapInfo, check_collision, is_free  # noqa: E402

LAUNCH_PARAMS = {
    "CELL_SIZE": 0.4,
    "NODE_RESOLUTION": 2.0,
    "SENSOR_RANGE": 20.0,
    "MIN_UTILITY": 3,
    "CLUSTER_RANGE": 10.0,
    "THR_NEXT_WAYPOINT": 4.0,
    "THR_GRAPH_HARD_UPDATE": 10.0,
    "THR_TO_WAYPOINT": 2.0,
    "AVOID_OSCILLATION": True,
    "ENABLE_SAVE_MODE": False,
    "ENABLE_DSTARLITE": False,
}


class AriadnePlanner:
    def __init__(self, checkpoint: str | None = None, device: str = "cpu", **overrides):
        params = {**LAUNCH_PARAMS, **overrides}
        for k, v in params.items():
            setattr(parameter, k, v)
        parameter.UTILITY_RANGE = 0.5 * parameter.SENSOR_RANGE
        parameter.FRONTIER_CELL_SIZE = parameter.CELL_SIZE
        parameter.UPDATING_MAP_SIZE = 4 * parameter.SENSOR_RANGE + 4 * parameter.NODE_RESOLUTION

        self.device = device
        net = PolicyNet(parameter.NODE_INPUT_DIM, parameter.EMBEDDING_DIM).to(device)
        ckpt = checkpoint or str(PLANNER_DIR / "model" / "checkpoint.pth")
        net.load_state_dict(torch.load(ckpt, map_location=device)["policy_model"])
        net.eval()
        self.robot = Agent(net, device, False)

        self.map_info: MapInfo | None = None
        self.robot_location: np.ndarray | None = None
        self.start: np.ndarray | None = None
        self.next_waypoint: np.ndarray | None = None
        self.next_waypoint_list: list = []
        self.history_waypoint_list: list = []
        self.done = False

    # -- inputs ---------------------------------------------------------------------------------

    def update_map(self, grid: np.ndarray, origin_xy: tuple[float, float]):
        """Occupancy grid in ROS layout (rows = y), cell size = parameter.CELL_SIZE."""
        pad = int(parameter.NODE_RESOLUTION // parameter.CELL_SIZE + 1)
        padded = np.pad(grid.astype(np.int8), pad, "constant", constant_values=parameter.UNKNOWN)
        ox = origin_xy[0] - parameter.CELL_SIZE * pad
        oy = origin_xy[1] - parameter.CELL_SIZE * pad
        self.map_info = MapInfo(padded, ox, oy, parameter.CELL_SIZE)

    def update_location(self, xy):
        self.robot_location = np.around(np.asarray(xy, dtype=np.float64), 1)
        if self.start is None and self.map_info is not None:
            r = parameter.NODE_RESOLUTION
            x = np.array([(self.robot_location[0] // r) * r, (self.robot_location[0] // r + 1) * r])
            y = np.array([(self.robot_location[1] // r) * r, (self.robot_location[1] // r + 1) * r])
            t1, t2 = np.meshgrid(x, y)
            candidates = np.vstack([t1.T.ravel(), t2.T.ravel()]).T
            for start in candidates[np.argsort(np.linalg.norm(candidates - self.robot_location, axis=1))]:
                if is_free(start, self.map_info):
                    self.start = np.around(start, 1)
                    break
            if self.start is not None:
                self.robot.node_manager = NodeManager(self.start)

    # -- one replanning tick (rl_planner.Runner.run without save mode) ----------------------------

    def plan(self) -> np.ndarray | None:
        if self.done or self.start is None or self.map_info is None:
            return self.next_waypoint

        if parameter.AVOID_OSCILLATION and len(self.history_waypoint_list) > 4:
            h = self.history_waypoint_list
            if h[-1] == h[-3] and h[-2] == h[-4]:
                self.next_waypoint_list = []
                if np.linalg.norm(self.next_waypoint - self.robot_location) > parameter.THR_TO_WAYPOINT:
                    return self.next_waypoint

        if len(self.next_waypoint_list) > 0:
            if np.linalg.norm(self.next_waypoint - self.robot_location) <= parameter.THR_TO_WAYPOINT:
                self.robot_location = self.next_waypoint
                self.next_waypoint = self.next_waypoint_list.pop(0)
        self.next_waypoint_list = []

        self.robot.node_manager.check_valid_node(self.robot_location, self.map_info)
        self._reanchor_origin()

        robot_node_location = self.robot_location
        if self.robot_location[0] != self.start[0] or self.robot_location[1] != self.start[1]:
            if len(self.robot.node_manager.nodes_dict) == 0:
                robot_node_location = self.start
            else:
                nearest = self.robot.node_manager.nodes_dict.nearest_neighbors(self.robot_location.tolist(), 1)[0]
                robot_node_location = nearest.data.coords

        self.robot.update_planning_state(self.map_info, robot_node_location)

        if sum(self.robot.key_utility) == 0:
            self.done = True
            return None

        observation = self.robot.get_observation(self.robot_location)
        next_location, next_node_index = self.robot.select_next_waypoint(observation)
        self.next_waypoint_list.append(next_location)
        key = (next_location[0], next_location[1])
        if not self.history_waypoint_list or key != self.history_waypoint_list[-1]:
            self.history_waypoint_list.append(key)

        # plan one more step if the chosen node has no utility
        if self.robot.node_manager.nodes_dict.find(next_location.tolist()).data.utility == 0:
            next_observation = self.robot.get_next_observation(next_node_index, observation)
            next_next_location, _ = self.robot.select_next_waypoint(next_observation)
            if np.linalg.norm(next_location - self.robot_location) < parameter.NODE_RESOLUTION:
                self.next_waypoint_list = []
            self.next_waypoint_list.append(next_next_location)

        self.next_waypoint = self.next_waypoint_list.pop(0)
        return self.next_waypoint

    def _reanchor_origin(self):
        """Robustness fix: if the graph origin node was removed (its cell became occupied, e.g. sensor noise or an
        object placed there), NodeManager.remove_unconnected_nodes crashes on ``find(start)``; move the origin to
        the existing node nearest to the robot instead."""
        nm = self.robot.node_manager
        if len(nm.nodes_dict) == 0 or nm.nodes_dict.find(nm.start.tolist()) is not None:
            return
        nearest = nm.nodes_dict.nearest_neighbors(self.robot_location.tolist(), 1)[0].data
        nm.start = np.asarray(nearest.coords)
        self.start = nm.start
        self.reanchors = getattr(self, "reanchors", 0) + 1

    # -- visualization helpers ----------------------------------------------------------------------

    def graph(self):
        """(key node coords [N, 2], utilities [N], edges [(a, b)], frontiers [M, 2])."""
        if self.robot.key_node_coords is None:
            return np.zeros((0, 2)), np.zeros(0), [], np.zeros((0, 2))
        edges = []
        for c in self.robot.key_node_coords:
            node = self.robot.node_manager.key_node_dict[(c[0], c[1])]
            edges += [(c, n) for n in node.neighbor_set]
        frontier = np.array(list(self.robot.frontier)) if self.robot.frontier else np.zeros((0, 2))
        return np.asarray(self.robot.key_node_coords), np.asarray(self.robot.key_utility), edges, frontier


__all__ = ["AriadnePlanner", "check_collision"]
