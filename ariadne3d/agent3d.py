"""Observation = original ARiADNE node features + [heading cost, 3D utility] appended per node.

heading cost : |bearing from the robot to the node - robot heading| / pi   (0 = straight ahead, 1 = behind)
3D utility   : unseen surface elements in view from the node (Belief3D), / U3D_NORM
               policy: from the robot's belief; critic: ground truth (privileged, like ARiADNE's ground-truth critic)
The new columns come last, so the pretrained first layer keeps its meaning for the original four (five) inputs.
"""

import numpy as np
import torch

from agent import Agent
from ground_truth_node_manager import GroundTruthNodeManager
from parameter import NODE_PADDING_SIZE, U3D_NORM


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def extra_features(coords, location, heading, util_grid, belief3d, map_info):
    d = coords - np.asarray(location)[None, :]
    bearing = np.arctan2(d[:, 1], d[:, 0])
    hc = np.abs(wrap(bearing - heading)) / np.pi
    hc[np.linalg.norm(d, axis=1) < 1e-6] = 0.0
    cells = np.stack([(coords[:, 0] - map_info.map_origin_x) / map_info.cell_size,
                      (coords[:, 1] - map_info.map_origin_y) / map_info.cell_size], 1)
    u = belief3d.sample(util_grid, cells) / U3D_NORM
    f = np.stack([hc, np.minimum(u, 4.0)], 1).astype(np.float32)
    return torch.nn.functional.pad(torch.from_numpy(f), (0, 0, 0, NODE_PADDING_SIZE - len(f))).unsqueeze(0)


class Agent3D(Agent):
    heading = 0.0
    util_grid = None
    belief3d = None

    def set_3d(self, heading, util_grid, belief3d):
        self.heading, self.util_grid, self.belief3d = heading, util_grid, belief3d

    def get_observation(self):
        obs = super().get_observation()
        f = extra_features(self.node_coords, self.location, self.heading, self.util_grid, self.belief3d, self.map_info)
        obs[0] = torch.cat([obs[0], f.to(obs[0].device)], dim=-1)
        return obs


class GroundTruthNodeManager3D(GroundTruthNodeManager):
    def get_ground_truth_observation_3d(self, robot_location, heading, util_grid, belief3d):
        obs = self.get_ground_truth_observation(robot_location)
        f = extra_features(self.ground_truth_node_coords, robot_location, heading, util_grid, belief3d,
                           self.ground_truth_map_info)
        obs[0] = torch.cat([obs[0], f.to(obs[0].device)], dim=-1)
        return obs
