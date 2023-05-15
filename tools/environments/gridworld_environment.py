from tools.base import Environment
from tools.graphics import Plot2D
import numpy as np
import gym
import random

class GridworldEnvironment(Environment):
    """
    nxm Gridworld. Discrete states and actions (up/down/left/right/stay).
    Agent starts randomly.
    Goal is to reach the reward.
    Inspired from following work:
    github.com/yrlu/irl-imitation/blob/master/mdp/gridworld.py
    """

    def __init__(self, r, t, transition_prob=1., stay_action=True, unsafe_states=[],
        start_states=None):
        """
        Construct the environment.
        Reward matrix is a 2D numpy matrix or list of lists.
        Terminal cells is a list/set of (i, j) values.
        Transition probability is the probability to execute an action and
        end up in the right next cell.
        """
        self.reward_mat = np.array(r)
        assert(len(self.reward_mat.shape) == 2)
        self.h, self.w = len(self.reward_mat), len(self.reward_mat[0])
        self.n = self.h*self.w
        self.terminals = t
        if stay_action:
            self.neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1), (0, 0)]
            self.actions = [0, 1, 2, 3, 4, 5, 6, 7, 8]
            self.n_actions = len(self.actions)
            self.dirs = {0: 'r', 1: 'l', 2: 'd', 3: 'u', 4: 'rd', 5: 'ru', 6: 'ld', 7: 'lu', 8: 's'}
            self.action_space = gym.spaces.Discrete(9)
        else:
            self.neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (1, -1), (-1, -1)]
            self.actions = [0, 1, 2, 3, 4, 5, 6, 7]
            self.n_actions = len(self.actions)
            self.dirs = {0: 'r', 1: 'l', 2: 'd', 3: 'u', 4: 'rd', 5: 'ru', 6: 'ld', 7: 'lu'}
            self.action_space = gym.spaces.Discrete(8)
        self.transition_prob = transition_prob
        self.terminated = True
        self.observation_space = gym.spaces.Box(low=np.array([0, 0]), 
            high=np.array([self.h, self.w]), dtype=np.int32)
        self.unsafe_states = unsafe_states
        self.start_states = start_states
    
    def get_states(self):
        """
        Returns list of all states.
        """
        return filter(
            lambda x: self.reward_mat[x[0]][x[1]] not in \
                [-np.inf, float('inf'), np.nan, float('nan')],
            [(i, j) for i in range(self.h) for j in range(self.w)]
        )
    
    def get_actions(self, state):
        """
        Returns list of actions that can be taken from the given state.
        """
        if self.reward_mat[state[0]][state[1]] in \
            [-np.inf, float('inf'), np.nan, float('nan')]:
            return [4]
        actions = []
        for i in range(len(self.actions) - 1):
            inc = self.neighbors[i]
            a = self.actions[i]
            nei_s = (state[0] + inc[0], state[1] + inc[1])
            if nei_s[0] >= 0 and nei_s[0] < self.h and \
                nei_s[1] >= 0 and nei_s[1] < self.w and \
                self.reward_mat[nei_s[0]][nei_s[1]] not in \
                [-np.inf, float('inf'), np.nan, float('nan')]:
                actions.append(a)
        return actions
    
    def terminal(self, state):
        """
        Check if the state is terminal.
        """
        for terminal_state in self.terminals:
            if state[0] == terminal_state[0] and state[1] == terminal_state[1]:
                return True
        return False

    def get_next_states_and_probs(self, state, action):
        """
        Given a state and action, return list of (next_state, probability) pairs.
        """
        if self.terminal(state):
            return [((state[0], state[1]), 1)]
        if self.transition_prob == 1:
            inc = self.neighbors[action]
            nei_s = (state[0] + inc[0], state[1] + inc[1])
            if nei_s[0] >= 0 and nei_s[0] < self.h and \
                nei_s[1] >= 0 and nei_s[1] < self.w and \
                self.reward_mat[nei_s[0]][nei_s[1]] not in \
                [-np.inf, float('inf'), np.nan, float('nan')]:
                return [(nei_s, 1)]
            else:
                return [((state[0], state[1]), 1)] # state invalid
        else:
            mov_probs = np.zeros([self.n_actions])
            mov_probs[action] = self.trans_prob
            mov_probs += (1-self.trans_prob)/self.n_actions
            for a in range(self.n_actions):
                inc = self.neighbors[a]
                nei_s = (state[0] + inc[0], state[1] + inc[1])
                if nei_s[0] < 0 or nei_s[0] >= self.h or \
                    nei_s[1] < 0 or nei_s[1] >= self.w or \
                    self.reward_mat[nei_s[0]][nei_s[1]] in \
                    [-np.inf, float('inf'), np.nan, float('nan')]:
                    mov_probs[-1] += mov_probs[a]
                    mov_probs[a] = 0
            res = []
            for a in range(self.n_actions):
                if mov_probs[a] != 0:
                    inc = self.neighbors[a]
                    nei_s = (state[0] + inc[0], state[1] + inc[1])
                    res.append((nei_s, mov_probs[a]))
            return res
    
    @property
    def state(self):
        """
        Return the current state.
        """
        return self.curr_state
    
    def pos2idx(self, pos):
        """
        Convert column-major 2d position to 1d index.
        """
        return pos[0] + pos[1] * self.h

    def idx2pos(self, idx):
        """
        Convert 1d index to 2d column-major position.
        """
        return (idx % self.h, idx // self.h)

    def reset(self, **kwargs):
        """
        Reset the environment.
        """
        if self.start_states != None:
            random_state = random.choice(self.start_states)
            self.curr_state = random_state
        else:
            random_state = np.random.randint(self.h * self.w)
            self.curr_state = self.idx2pos(random_state)
        while self.curr_state in self.terminals or self.curr_state in self.unsafe_states:
            if self.start_states != None:
                random_state = random.choice(self.start_states)
                self.curr_state = random_state
            else:
                random_state = np.random.randint(self.h * self.w)
                self.curr_state = self.idx2pos(random_state)
        self.terminated = False
        return list(self.state)
    
    def step(self, action):
        """
        Step the environment.
        """
        action = int(action)
        if self.terminal(self.state):
            self.terminated = True
            return {
                "next_state": list(self.state), 
                "reward": self.reward_mat[self.state[0], self.state[1]],
                "done": True, 
                "info": {}
            }
        self.terminated = False
        st_prob = self.get_next_states_and_probs(self.state, action)
        sampled_idx = np.random.choice(np.arange(0, len(st_prob)), \
            p=[prob for st, prob in st_prob])
        last_state = self.state
        next_state = st_prob[sampled_idx][0]
        reward = self.reward_mat[last_state[0]][last_state[1]]
        self.curr_state = next_state
        return {
            "next_state": list(self.state),
            "reward": reward,
            "done": False, 
            "info": {}
        }
    
    def seed(self, s=None):
        """
        Seed this environment.
        """
        random.seed(s)
        np.random.seed(s)

    def render(self,**kwargs):
        """
        Render the environment.
        """
        self.state_mat = np.zeros([self.h, self.w, 3])
        self.state_mat[self.state[0], self.state[1], :] = 1.
        if not hasattr(self, "plot"):
            self.plot = Plot2D({
                "env": lambda p, l, t: self,
            }, [
                [
                    lambda p, l, t: not l["env"].terminated,
                    lambda p, l, o, t: p.imshow(l["env"].state_mat, o=o)
                ],
            ], mode="dynamic", interval=1)
        self.plot.show(block=False)
        if "mode" in kwargs.keys() and kwargs["mode"] == "rgb_array":
            self.plot.fig.canvas.draw()
            img = np.frombuffer(self.plot.fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(self.plot.fig.canvas.get_width_height()[::-1] + (3,))
            return img
