import numpy as np
import time, datetime
from inspect import isfunction
import json
import gym
from tools.base import Environment
from tools.graphics import Canvas2D
from tools.data import Bits
from tools.utils import dict_to_numpy, combine_dicts
from tools.logic import RewardStructure
from tools.math import Direction2D
import random
import pyglet, PIL

class DrivingEnvironment(Environment):
    """
    Driving environment.
    Taken from github.com/ashishgaurav13/wm2

    Canvas cannot have more than 32 agents (32-bit int).
    zero_pad zero_pads by a certain number of features.

    Eg.

    canvas = ...
    default_policy = lambda c: c.aggressive_driving()
    env = Environment(canvas, default_policy, zero_pad = 3)

    Freeze/unfreeze agents which you want to be updated. If
    frozen, these agents won't move, i.e. their policies
    will not be called. If agent.ego is True, then 
    default_policy doesn't apply to it.

    env.agents_unfrozen() => all by default
    env.agent_freeze(...)
    env.agent_unfreeze(...)
    env.agent_freeze_all()
    env.agent_unfreeze_all()

    Debugging functionalities can be set to True. By default,
    there is no debugging. (TODO: list of debug functionalities)

    env.debug['intersection_enter'] = True

    ...
    env.reset() => return env.state
    env.state => get numpy representation of state (mostly trimmed)
    env.f[a]['b'] => get agent (id:a)'s feature 'b'
    ...
    next_obs, reward, done, info = env.step()
    ...
    env.close()
    """

    def __init__(self, canvas, default_policy = None, zero_pad = 0,
        discrete = False):
        """
        Initialize DrivingEnvironment.
        """

        assert(type(canvas) == Canvas2D)
        self.canvas = canvas
        self.rendering = False
        assert(hasattr(self.canvas, 'agents'))
        assert(type(self.canvas.agents) == list)
        self.agents = self.canvas.agents        
        num_egos = sum([agent.ego for agent in self.agents])
        self.ego_id = None
        for aid, agent in enumerate(self.agents):
            if agent.ego:
                self.ego_id = aid
        assert(num_egos <= 1)
        self.num_agents = len(self.agents)
        self.reward_specified = False
        self.state_specified = False
        self.init_time = time.time()

        # Which agents to actually draw (or update)
        self.agents_drawn = Bits()
        for ai in range(self.num_agents):
            self.agents_drawn[ai] = True

        # Feature-sets for all agents
        self.f = [agent.f for agent in self.agents]

        # Policies for non ego
        self.policies = [default_policy for agent in self.agents]
        if self.ego_id != None: self.policies[self.ego_id] = None

        # Debugging
        self.debug_fns = {
            'intersection_enter': self.debug_intersection_enter,
            'state_inspect': None,
            'kill_after_state_inspect': None,
            'show_elapsed': None,
            'show_steps': None,
            'show_reasons': None,
            'record_reward_trajectory': None,
            'action_buckets': None,
        }
        self.debug = {k: False for k in self.debug_fns.keys()}
        self.debug_variables = {}
        self.debug_variables['buckets'] = set([])

        # Call make_ready to set ready to true
        self.ready = False

        # Zero pad features
        self.zero_pad = zero_pad

        # Discrete actions
        self.discrete = discrete

    def seed(self, s=None):
        """
        Seed this environment.
        """
        random.seed(s)
        np.random.seed(s)

    def make_ready(self):
        """
        Create observation space and action space after everything is set.
        """
        self.ready = True

        # Observation space (TODO: can be better defined)
        s = self.state
        self.observation_space = gym.spaces.Box(
            low = -np.inf,
            high = np.inf,
            shape = s.shape,
        )

        # Action space
        self.action_space = None
        if self.ego_id != None:
            ego = self.agents[self.ego_id]
            amax = ego.MAX_ACCELERATION
            psidotmax = ego.MAX_STEERING_ANGLE_RATE
            if self.discrete:
                self.action_mapping = {
                    0: [amax, psidotmax],
                    1: [amax, 0.75 * psidotmax],
                    2: [amax, 0.5 * psidotmax],
                    3: [amax, 0.25 * psidotmax],
                    4: [amax, 0],
                    5: [amax, -0.25 * psidotmax],
                    6: [amax, -0.5 * psidotmax],
                    7: [amax, -0.75 * psidotmax],
                    8: [amax, -psidotmax],
                    9: [amax/2, 0],
                    10: [0, 0],
                    11: [-amax/2, 0],
                    12: [-amax, 0]
                }
                self.action_space = gym.spaces.Discrete(len(self.action_mapping))
            else:
                self.action_space = gym.spaces.Box(
                    low = np.array([-amax, -psidotmax]),
                    high = np.array([amax, psidotmax]),
                )

    def agents_unfrozen(self):
        """
        Which agents are unfrozen?
        """
        return [ai for ai in range(self.num_agents) \
            if self.agents_drawn[ai] == True]

    def agent_unfreeze(self, i):
        """
        Unfreeze agent i.
        """
        self.agents_drawn[i] = True

    def agent_unfreeze_all(self):
        """
        Unfreeze all agents.
        """
        for ai in range(self.num_agents):
            self.agents_drawn[ai] = True

    def agent_freeze(self, i):
        """
        Freeze agent i.
        """
        self.agents_drawn[i] = False

    def agent_freeze_all(self):
        """
        Freeze all agents.
        """
        for ai in range(self.num_agents):
            self.agents_drawn[ai] = False

    def specify_state(self, ego_fn, other_fn):
        """
        Specify how the state is constructed.
        """
        assert(not self.state_specified)
        assert(isfunction(ego_fn))
        assert(isfunction(other_fn))
        self.state_specified = True
        self.ego_fn = ego_fn
        self.other_fn = other_fn
        # self.debug['state_inspect'] = True
        # self.debug['kill_after_state_inspect'] = True

    def specify_action_multipliers(self, amul):
        """
        Specify action multipliers.
        """
        self.amul = amul

    @property
    def state(self):
        """
        Returns current state.
        """
        assert(hasattr(self, 'f'))

        ret = None
        if self.state_specified:
            ret = {}
            for aid, agent in enumerate(self.agents):
                if aid == self.ego_id:
                    ret[aid] = self.ego_fn(agent, self.reward_structure)
                else:
                    ret[aid] = self.other_fn(agent, self.reward_structure)
            
            if self.zero_pad > 0:
                ret["null"] = {("null_feature_%d" % ki): 0.0 for ki in range(self.zero_pad)}

            if self.debug['state_inspect']: 
                print('Dict:')
                print(json.dumps(ret, indent = 2))
                print('Flattened Numpy:')
                print(dict_to_numpy(ret))
                print('')
                if self.debug['kill_after_state_inspect']:
                    print('Killing ...')
                    exit(1)
            ret = dict_to_numpy(ret)

        return ret

    def new_debug_variable(self, key, value):
        """
        Create a debug variable.
        """
        if key not in self.debug_variables:
            self.debug_variables[key] = value

    def debug_intersection_enter(self):
        """
        (Debug function)
        Prints when inside the intersection.
        """
        self.new_debug_variable('order', [])
        ii = self.canvas.intersections[0]
        inside = []
        for aid, agent in enumerate(self.agents):
            if ii.x1 <= agent.f['x'] <= ii.x2 and \
                ii.y1 <= agent.f['y'] <= ii.y2:
                if aid not in self.debug_variables['order']:
                    inside += [aid]
                    self.debug_variables['order'] += [aid]
        if len(inside) > 0:
            print("Inside the intersection: %s" % inside)
    
    def is_agent_in_bounds(self, agent):
        """
        (Debug function)
        Is agent within bounds?
        """
        return self.canvas.is_agent_in_bounds(agent)

    def reset(self, **kwargs):
        """
        Reset the environment.
        """
        assert(self.ready)
        self.init_time = time.time()
        for agent in self.agents:
            agent.reset()

        if self.reward_specified:
            self.total_reward = 0.0
            self.reward_structure.reset()

        if self.debug['record_reward_trajectory']:
            self.trajectory = []
        
        if hasattr(self.canvas, "text"):
            self.canvas.text.items[0].text = "t=0"

        return self.state
    
    def reward_structure(self, d, p, r, t, s, round_to = 3, clip_to = None,
            combine_rewards = lambda a, b: a + b):
        """
        Defines structure using definitions, properties, rewards, terminations, 
        and successes. Round reward to some decimal places.
        """

        assert(not self.reward_specified)
        assert(self.ego_id != None)
        objs = {
            'ego': self.agents[self.ego_id],
            'v': self.agents,
        }
        self.reward_structure = RewardStructure(d, p, r, t, s, objs,
            combine_rewards = combine_rewards)
        self.reward_specified = True
        self.round_to = round_to
        self.clip_to = [-np.inf, np.inf]
        if clip_to != None:
            assert(len(clip_to) == 2)
            assert(clip_to[0] <= clip_to[1])
            self.clip_to = clip_to

    def step(self, action):
        """
        Step the environment.
        """
        assert(self.ready)
        agents_in_bounds = 0
        reward = 0
        done = False
        info = {}

        if hasattr(self, "amul"):
            for ai in range(len(action)):
                action[ai] *= self.amul[ai]

        if hasattr(self.canvas, "text"):
            curr = self.canvas.text.items[0].text
            self.canvas.text.items[0].text = "t=%d" % (int(curr.split("=")[-1])+1)

        if self.discrete:
            if type(action) == int:
                action = self.action_mapping[action]
            else:
                action = self.action_mapping[action[0]]

        # update ego
        if self.ego_id != None:
            if self.agents_drawn[self.ego_id]:
                self.agents[self.ego_id].step(action)
            agents_in_bounds += \
                self.is_agent_in_bounds(self.agents[self.ego_id])

        # update non-egos
        for aid, agent in enumerate(self.agents):
            if not agent.ego and self.agents_drawn[aid]:
                control_inputs = self.policies[aid](agent).act()
                agent.step(control_inputs)
            agents_in_bounds += self.is_agent_in_bounds(agent)

        # ask the reward structure: what is the reward?
        if self.reward_specified:
            reward, info, _ = self.reward_structure.step()
            if self.total_reward + reward > self.clip_to[1]:
                if self.clip_to[1] != np.inf:
                    reward = self.clip_to[1]-self.total_reward
                else:
                    reward = 0
            if self.total_reward + reward < self.clip_to[0]:
                if self.clip_to[0] != -np.inf:
                    reward = self.clip_to[0]-self.total_reward
                else:
                    reward = 0
            reward = float(reward)
            reward = round(reward, self.round_to)
            self.total_reward += reward

        # if terminated or succeeded
        if info != {}:
            if info['mode'] == 'success': done = True
            if info['mode'] == 'termination': done = True

        # terminate if nothing is within bounds
        if agents_in_bounds == 0: 
            done = True
        
        info['theta_Mean_Std'] = round(((2*np.pi-self.agents[self.ego_id].f['theta'])-np.pi)/np.pi*180., 2)
        info['x_Mean_Std'] = round(self.agents[self.ego_id].f['x'], 2)
        info['y_Mean_Std'] = round(self.agents[self.ego_id].f['y'], 2)
        info['u0_Mean_Std'] = round(action[0], 2)
        info['u1_Mean_Std'] = round(action[1], 2)

        # debugging
        if self.debug['intersection_enter']:
            self.debug_fns['intersection_enter']()
        if self.debug['show_steps']:
            T = self.reward_structure._p.t
            print(T, reward, info, done)
        if self.debug['show_elapsed'] and done:
            assert(self.init_time)
            diff = int(time.time() - self.init_time)
            print('Execution: %s' % str(datetime.timedelta(seconds = diff)))
        if self.debug['record_reward_trajectory']:
            self.trajectory += [reward]
            if done: info['traj'] = self.trajectory[:]
        if self.debug['show_reasons'] and done:
            print(info)
        if self.debug['action_buckets']: # bucketize actions
            self.debug_variables['buckets'].add(tuple(action))

        return {
            "next_state": self.state, 
            "reward": reward, 
            "done": done,
            "info": info
        }

    def render(self, **kwargs):
        """
        Render the environment.
        """
        if not self.rendering:
            self.rendering = True
            self.canvas.set_visible(True)

        self.canvas.clear()
        self.canvas.switch_to()
        self.canvas.dispatch_events()
        drew_agents = self.canvas.on_draw()
        self.canvas.flip()

        # turn off rendering if nothing is drawn
        if drew_agents == 0:
            self.rendering = False
            self.canvas.set_visible(False)

        if "mode" in kwargs.keys() and kwargs["mode"] == "rgb_array":
            # Capture image from the OpenGL buffer
            buffer = ( pyglet.gl.GLubyte * (3*self.canvas.width*self.canvas.height) )(0)
            pyglet.gl.glReadPixels(0, 0, self.canvas.width, self.canvas.height,
                pyglet.gl.GL_RGB, pyglet.gl.GL_UNSIGNED_BYTE, buffer)

            # Use PIL to convert raw RGB buffer and flip the right way up
            image = PIL.Image.frombytes(mode="RGB", 
                size=(self.canvas.width, self.canvas.height), data=buffer)     
            image = image.transpose(PIL.Image.FLIP_TOP_BOTTOM)
            return np.asarray(image)
