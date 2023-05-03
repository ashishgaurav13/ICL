import numpy as np
import os
from gym.envs.mujoco.ant_v3 import AntEnv

ABS_PATH = os.path.abspath(os.path.dirname(__file__))

class AntWall(AntEnv):
    def __init__(
            self,
            healthy_reward=1.0,             # default: 1.0
            terminate_when_unhealthy=False, # default: True
            xml_file=ABS_PATH+"/xmls/ant_circle.xml",
            reset_noise_scale=0.1,
            exclude_current_positions_from_observation=False
    ):
       super(AntWall, self).__init__(
                xml_file=xml_file,
                healthy_reward=healthy_reward,
                terminate_when_unhealthy=terminate_when_unhealthy,
                reset_noise_scale=reset_noise_scale,
                exclude_current_positions_from_observation=exclude_current_positions_from_observation
        )
    def step(self, action):
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = abs(xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = self.control_cost(action)
        contact_cost = self.contact_cost

        forward_reward = x_velocity
        healthy_reward = self.healthy_reward

#        rewards = forward_reward + healthy_reward
        distance_from_origin = np.linalg.norm(xy_position_after, ord=2)
        rewards = distance_from_origin + healthy_reward
        costs = ctrl_cost + contact_cost

        reward = rewards - costs
        done = self.terminated
        observation = self._get_obs()
        info = {
            # 'reward_forward': forward_reward,
            # 'reward_ctrl': -ctrl_cost,
            # 'reward_contact': -contact_cost,
            # 'reward_survive': healthy_reward,

            # 'x_position': xy_position_after[0],
            # 'y_position': xy_position_after[1],
            # 'distance_from_origin': np.linalg.norm(xy_position_after, ord=2),

            # 'x_velocity': x_velocity,
            # 'y_velocity': y_velocity,
            # 'forward_reward': forward_reward,
        }
        return observation, reward, done, info
