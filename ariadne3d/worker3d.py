"""One training / evaluation episode in Env3D (same 27-slot episode buffer as the original worker)."""

import numpy as np
import torch

from agent3d import Agent3D, GroundTruthNodeManager3D
from env3d import Env3D
from lidar3d import ViewGain
from parameter import MAX_EPISODE_STEP, U3D_DONE, U3D_DONE_VIEW, UTIL3D
from worker import Worker


class Worker3D(Worker):
    def __init__(self, meta_agent_id, policy_net, global_step, device='cpu', greedy=False, tilt_deg=None, env_cls=Env3D,
                 record=True):
        self.meta_agent_id = meta_agent_id
        self.global_step = global_step
        self.save_image = False
        self.device = device
        self.greedy = greedy
        self.record = record  # False (evaluation): no critic observations, no replay data
        self.env = env_cls(global_step) if tilt_deg is None else env_cls(global_step, tilt_deg=tilt_deg)
        self.robot = Agent3D(policy_net, self.device, False)
        self.ground_truth_node_manager = GroundTruthNodeManager3D(self.robot.node_manager, self.env.ground_truth_info,
                                                                  device=self.device, plot=False)
        self.episode_buffer = [[] for _ in range(27)]
        self.perf_metrics = dict()
        self.view = ViewGain(self.env.belief3d) if UTIL3D == "view" else None

    def _observe(self):
        if self.view is not None:
            return self._observe_view()
        return self._observe_grid()

    def _observe_view(self):
        e = self.env
        origin, loc = (e.belief_origin_x, e.belief_origin_y), e.robot_location
        fn_b = lambda c: self.view.node_gain(c, loc, origin, "belief", e.robot_belief)  # noqa: E731
        self.robot.set_3d(e.heading, node_u_fn=fn_b)
        obs = self.robot.get_observation()
        gt_obs = None
        if self.record:
            fn_t = lambda c: self.view.node_gain(c, loc, origin, "truth")  # noqa: E731
            gt_obs = self.ground_truth_node_manager.get_ground_truth_observation_3d(loc, e.heading, None, None, fn_t)
        u = fn_b(self.robot.node_coords)  # cached: free
        # done threshold in the units of this utility (Env3D.check_done compares with U3D_DONE)
        return obs, gt_obs, float(u.max()) * U3D_DONE / U3D_DONE_VIEW if len(u) else 0.0

    def _observe_grid(self):
        b3 = self.env.belief3d
        u_belief = b3.belief_utility(self.env.robot_belief)
        self.robot.set_3d(self.env.heading, u_belief, b3)
        obs = self.robot.get_observation()
        gt_obs = None
        if self.record:
            gt_obs = self.ground_truth_node_manager.get_ground_truth_observation_3d(
                self.env.robot_location, self.env.heading, b3.true_utility(), b3)
        node_u3d = b3.sample(u_belief, np.stack([
            (self.robot.node_coords[:, 0] - self.env.belief_info.map_origin_x) / self.env.cell_size,
            (self.robot.node_coords[:, 1] - self.env.belief_info.map_origin_y) / self.env.cell_size], 1))
        return obs, gt_obs, float(node_u3d.max()) if len(node_u3d) else 0.0

    def _act(self, observation):
        if not self.greedy:
            return self.robot.select_next_waypoint(observation)
        _, _, _, _, current_edge, _ = observation
        with torch.no_grad():
            logp = self.robot.policy_net(*observation)
        action_index = torch.argmax(logp, dim=1).long()
        return self.robot.node_coords[current_edge[0, action_index.item(), 0].item()], action_index

    def run_episode(self):
        done = False
        self.robot.update_planning_state(self.env.belief_info, self.env.robot_location)
        observation, gt_observation, _ = self._observe()
        total_reward = 0.0
        for _ in range(MAX_EPISODE_STEP):
            if self.record:
                self.save_observation(observation, gt_observation)
            next_location, action_index = self._act(observation)
            if self.record:
                self.save_action(action_index)
            reward = self.env.step(next_location)
            self.robot.update_planning_state(self.env.belief_info, self.env.robot_location)
            observation, gt_observation, u3d_max = self._observe()
            bonus, done = self.env.check_done(self.robot.utility.sum(), u3d_max)
            reward += bonus
            total_reward += reward
            if self.record:
                self.save_reward_done(reward, done)
                self.save_next_observations(observation, gt_observation)
            if done:
                break
        self.perf_metrics = {'travel_dist': self.env.travel_dist, 'explored_rate': self.env.explored_rate,
                             'success_rate': float(self.env.done_2d), 'episode_reward': total_reward,
                             **self.env.metrics()}
