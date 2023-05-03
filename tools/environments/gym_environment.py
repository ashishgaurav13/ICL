from tools.base import Environment
import numpy as np
import gym
import math
import numpy as np
import pygame
from pygame import gfxdraw
from gym import spaces, logger
import random
import os
from gym.envs.mujoco.ant_v3 import AntEnv
from gym.envs.mujoco.half_cheetah import HalfCheetahEnv


class BrokenJoint(gym.Wrapper):
    """Wrapper that disables one coordinate of the action, setting it to 0."""
    def __init__(self, env, broken_joint=None):
        super(BrokenJoint, self).__init__(env)
        # Change dtype of observation to be float32.
        self.observation_space = gym.spaces.Box(
                low=self.observation_space.low,
                high=self.observation_space.high,
                dtype=np.float32,
        )
        if broken_joint is not None:
            assert 0 <= broken_joint <= len(self.action_space.low) - 1
        self.broken_joint = broken_joint

    def step(self, action):
        action = action.copy()
        if self.broken_joint is not None:
            action[self.broken_joint] = 0

        return super(BrokenJoint, self).step(action)

class AntWall(AntEnv):
    def __init__(
            self,
            healthy_reward=1.0,             # default: 1.0
            terminate_when_unhealthy=False, # default: True
            xml_file="assets/mujoco/ant_circle.xml",
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

class HalfCheetahWithPos(HalfCheetahEnv):
    """Also returns the `global' position in HalfCheetah."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float64)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.random(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def old_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_run = abs(xposafter - xposbefore) / self.dt
        reward = reward_ctrl + reward_run

        info = dict(
                # reward_run=reward_run,
                # reward_ctrl=reward_ctrl,
                # xpos=xposafter
                )

        return reward, info

    def new_reward(self, xposbefore, xposafter, action):
        reward_ctrl = -0.1 * np.square(action).sum()
        reward_dist = abs(xposafter)
        reward_run  = reward_dist / self.dt

        reward = reward_dist + reward_ctrl
        info = dict(
                # reward_run=reward_run,
                # reward_ctrl=reward_ctrl,
                # reward_dist=reward_dist,
                # xpos=xposafter
                )

        return reward, info

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_type = 'old'         # Which reward to use, traditional or new one?
        if reward_type == 'new':
            reward, info = self.new_reward(xposbefore,
                                           xposafter,
                                           action)
        elif reward_type == 'old':
            reward, info = self.old_reward(xposbefore,
                                           xposafter,
                                           action)
        done = False
        return ob, reward, done, info


# Taken from https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/normalized_actions.py
class NormalizedActions(Environment):

    def __init__(self, env):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space

    def action(self, action):
        action = (action + 1) / 2  # [-1, 1] => [0, 1]
        action *= (self.action_space.high - self.action_space.low)
        action += self.action_space.low
        return action

    def reverse_action(self, action):
        action -= self.action_space.low
        action /= (self.action_space.high - self.action_space.low)
        action = action * 2 - 1
        return action
    
    def step(self, a):
        return self.env.step(self.action(a))

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def seed(self, s=None):
        return self.env.seed(s=s)
    
    @property
    def state(self):
        return self.env.state

    def render(self, **kwargs):
        return self.env.render(**kwargs)

# https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo2_continuous_action.py
class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean([x], axis=0)
        batch_var = np.var([x], axis=0)
        batch_count = 1
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

class NormalizedStates(Environment):
    def __init__(self, env, ob=True, ret=True, clipob=1., gamma=0.99, epsilon=1e-8):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.ob_rms = RunningMeanStd(shape=self.observation_space.shape) if ob else None
        self.clipob = clipob
        self.gamma = gamma
        self.epsilon = epsilon

    def step(self, action):
        step_data = self.env.step(action)
        if type(step_data) == dict:
            obs, rews, dones, infos = step_data["next_state"], step_data["reward"],\
                step_data["done"], step_data["info"]
        else:
            obs, rews, dones, infos = step_data
        obs = self._obfilt(obs)
        if type(step_data) == dict:
            return {
                "next_state": obs,
                "reward": rews,
                "done": dones,
                "info": infos,
            }
        else:
            return obs, rews, dones, infos

    def _obfilt(self, obs):
        if self.ob_rms:
            self.ob_rms.update(obs)
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs

    def seed(self, s=None):
        return self.env.seed(s=s)

    @property
    def state(self):
        return self.env.state

    def reset(self, **kwargs):
        self.ret = np.zeros(())
        obs = self.env.reset(**kwargs)
        return self._obfilt(obs)
    
    def render(self, **kwargs):
        return self.env.render(**kwargs)

# github.com/openai/gym/blob/master/gym/wrappers/time_limit.py
class TimeLimit(Environment):
    def __init__(self, env, max_episode_steps=None):
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert (
            self._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        step_data = self.env.step(action)
        if type(step_data) == dict:
            observation, reward, done, info = step_data["next_state"], step_data["reward"],\
                step_data["done"], step_data["info"]
        else:
            observation, reward, done, info = step_data
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            done = True
        if type(step_data) == dict:
            return {
                "next_state": observation, 
                "reward": reward, 
                "done": done, 
                "info": info
            }
        else:
            return observation, reward, done, info

    def seed(self, s=None):
        return self.env.seed(s=s)

    @property
    def state(self):
        return self.env.state

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)
    
    def render(self, **kwargs):
        return self.env.render(**kwargs)

"""
github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

class CustomCartPole(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, noise=0., start_pos=None):
        self.noise = noise
        self.start_pos = start_pos
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # actually half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.kinematics_integrator = "euler"

        # Angle at which to fail the episode
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds.
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_done = None

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        action = int(action)
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # For the interested reader:
        # https://coneural.org/florian/papers/05_cart_pole.pdf
        temp = (
            force + self.polemass_length * theta_dot ** 2 * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.masspole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == "euler":
            x = x + self.tau * x_dot + np.random.uniform(-self.noise, self.noise)
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:  # semi-implicit euler
            x_dot = x_dot + self.tau * xacc 
            x = x + self.tau * x_dot + np.random.uniform(-self.noise, self.noise)
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)

        done = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(self, **kwargs):
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        if self.start_pos != None:
            interval = random.choice(self.start_pos)
            assert(interval[0] <= interval[1])
            self.state[0] = np.random.uniform(low=interval[0], high=interval[1])
        self.steps_beyond_done = None
        return np.array(self.state, dtype=np.float32)
        
    def render(self, **kwargs):
        screen_width = 600
        screen_height = 400

        world_width = self.x_threshold * 2
        scale = screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2 * self.length)
        cartwidth = 50.0
        cartheight = 30.0

        if self.state is None:
            return None

        x = self.state

        if self.screen is None:
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((screen_width, screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        self.surf = pygame.Surface((screen_width, screen_height))
        self.surf.fill((255, 255, 255))

        l, r, t, b = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2
        axleoffset = cartheight / 4.0
        cartx = x[0] * scale + screen_width / 2.0  # MIDDLE OF CART
        carty = 100  # TOP OF CART
        cart_coords = [(l, b), (l, t), (r, t), (r, b)]
        cart_coords = [(c[0] + cartx, c[1] + carty) for c in cart_coords]
        gfxdraw.aapolygon(self.surf, cart_coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, cart_coords, (0, 0, 0))

        l, r, t, b = (
            -polewidth / 2,
            polewidth / 2,
            polelen - polewidth / 2,
            -polewidth / 2,
        )

        pole_coords = []
        for coord in [(l, b), (l, t), (r, t), (r, b)]:
            coord = pygame.math.Vector2(coord).rotate_rad(-x[2])
            coord = (coord[0] + cartx, coord[1] + carty + axleoffset)
            pole_coords.append(coord)
        gfxdraw.aapolygon(self.surf, pole_coords, (202, 152, 101))
        gfxdraw.filled_polygon(self.surf, pole_coords, (202, 152, 101))

        gfxdraw.aacircle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )
        gfxdraw.filled_circle(
            self.surf,
            int(cartx),
            int(carty + axleoffset),
            int(polewidth / 2),
            (129, 132, 203),
        )

        gfxdraw.hline(self.surf, 0, screen_width, carty, (0, 0, 0))

        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if kwargs["mode"] == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        if kwargs["mode"] == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
        else:
            return self.isopen

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.isopen = False

class GymEnvironment(Environment):
    """
    Gym environment wrapper, with more functionality.
    """

    def __init__(self, name, horizon=None, **kwargs):
        """
        Instantiate wrapper by creating the gym environment object.
        """
        if name=="CustomCartPole":
            self.obj = CustomCartPole(**kwargs)
        else:
            self.obj = gym.make(name)
        self.observation_space = self.obj.observation_space
        self.action_space = self.obj.action_space
        self.curr_state = None
    
    def seed(self, s=None):
        """
        Seed this environment.
        """
        self.obj.seed(s)

    @property
    def state(self):
        """
        Get the current state.
        """
        return self.curr_state

    def reset(self, **kwargs):
        """
        Reset the environment.
        """
        self.curr_state = self.obj.reset(**kwargs)
        return self.curr_state
    
    def step(self, action=None):
        """
        Step the environment.
        """
        if type(self.action_space) == gym.spaces.Box:
            action = np.clip(action, self.action_space.low, self.action_space.high)
        next_state, reward, done, info = self.obj.step(action)
        self.curr_state = next_state
        return {
            "next_state": next_state,
            "reward": reward,
            "done": done,
            "info": info
        }

    def render(self, **kwargs):
        """
        Render the environment.
        """
        return self.obj.render(**kwargs)
