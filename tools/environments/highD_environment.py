from tools.base import Environment
from tools.environments import DrivingEnvironment
from tools.graphics import Canvas2D
from tools.logic import RandomPolicy
from tools.math import Direction2D, Box2D
from tools.data import HighDSampleReader
from tools.utils import combine_dicts
import gym, tqdm
import numpy as np
import random
from copy import deepcopy
import gc

class HighDSampleEnvironment(DrivingEnvironment):
    """
    Environment based on HighD dataset sample.
    """

    def __init__(self, static_elements, agents, timelimit, boxes, discrete=False):
        """
        Initialize InDSampleEnvironment.
        """
        self.canvas = Canvas2D(1000, 100,
            static_elements, agents,
            ox = 500, oy = 50, scale = 100/100, agentscale=2.
        )
        # double lane
        self.canvas.set_lane_width(boxes[-1].x1, boxes[-1].x2, boxes[-1].y1, boxes[-1].y2, factor=2.)
        act_space = gym.spaces.Box(
                    low = np.array([-2, -1]),
                    high = np.array([2, 1]),
                )
        default_policy = lambda agent: RandomPolicy(act_space)
        super().__init__(self.canvas, discrete=discrete, default_policy=default_policy)
        # Specify reward structure
        euclidean = lambda x1, y1, x2, y2: np.sqrt((x1-x2)**2+(y1-y2)**2)
        norm = euclidean(boxes[0].center[0], boxes[0].center[1],
                         boxes[1].center[0], boxes[1].center[1])
        d = {
            "caf_dist": lambda p, t: p['ego'].closest_agent_forward(
                 p['ego'].agents_in_front_behind())['how_far'],
            "caf_dist_parsed": lambda p, t: p["caf_dist"] if p["caf_dist"] >= 0. else 1000.
        }
        p = {
            "stopped": lambda p, t: t > 0 and p['ego'].f['v'] == 0,
            "near_goal": lambda p, t: boxes[1].inside(p['ego'].f['x'], p['ego'].f['y']),
            "time_limit": lambda p, t: t >= timelimit,
            "out_of_canvas": lambda p, t: p['ego'].ego_appeared and not (-500 <= p['ego'].f['x'] <= 500 and \
                -50 <= p['ego'].f['y'] <= 50),
            "collision": lambda p, t: p['ego'].collided(gap=4.),
        }
        r = [
            ["stopped", -0.1, 'satisfaction'],
        ]
        t = [
            ["time_limit", \
                lambda p, t: -p['ego'].Lp(boxes[1].center[0], boxes[1].center[1])/norm, 'satisfaction'],
            ["collision", \
                lambda p, t: -p['ego'].Lp(boxes[1].center[0], boxes[1].center[1])/norm, 'satisfaction'],
            ["out_of_canvas", \
                lambda p, t: -p['ego'].Lp(boxes[1].center[0], boxes[1].center[1])/norm, 'satisfaction'],
        ]
        s = [
            ['near_goal', 1, 'satisfaction'],
        ]
        self.reward_structure(d, p, r, t, s, round_to = 3)
        # Empty state
        ego_fn = lambda agent, rs: combine_dicts(agent.f.get_dict(), rs._p.get_dict(), rs._d.get_dict())
        other_fn = lambda agent, rs: {}
        self.specify_state(ego_fn, other_fn)
        self.specify_action_multipliers([1, 0])
        # Make ready
        self.make_ready()
        # Change car constants
        for agent in self.agents:
            agent.MAX_STEERING_ANGLE = np.pi/2
            agent.THETA_DEVIATION_ALLOWED = np.pi

class HighDSampleEnvironmentWrapper(Environment):
    """
    Wrapper around HighDSampleEnvironment.
    """
    def __init__(self, discrete=False, timelimit=1000, sequential=False):
        """
        Initialize HighDSampleEnvironmentWrapper.
        """
        self.timelimit = timelimit
        self.discrete = discrete
        self.reader = HighDSampleReader(dim=[1000, 100])
        self.reader.read_data()
        self.static_elements = [
            ['StretchBackground', 'assets/highD/17_highway.png'],
            ['Rectangle', -500, -400, -30, 0, (1, 1, 1, 0.4)],
            ['Rectangle', 400, 500, -30, 0, (1, 1, 1, 0.4)],
            ['Text', 't=0', -450, -45, (0, 0, 0, 1)], # Special text
        ]
        self.agents = []
        self.boxes = [
            Box2D(-500, -400, -30, 0, name="box0"),
            Box2D(400, 500, -30, 0, name="box1"),
            Box2D(-500, 500, -30, 0, name="box3"),
        ]
        self.possibleegoids = []
        self.possibleego = lambda c, f: c[-1][0] > c[0][0] and\
                    self.boxes[0].inside(c[0][0], c[0][1]) and\
                    self.boxes[1].inside(c[-1][0], c[-1][1])
        self.start_loc = self.reader.get_best_start(gap=self.timelimit, frames_len_max=self.timelimit)
        assert(self.start_loc != None)
        self.sequential = sequential
        if sequential:
            self.eid = 0
        for i in range(len(self.reader.bboxes)):
            centerpts, angles, frames, speedxs, speedys,\
                accxs, accys, h, w = self.reader.get_track(i)
            if len(frames) > self.timelimit: continue
            if frames[0] < self.start_loc or frames[-1] > self.start_loc+self.timelimit: continue
            frames = np.array(frames)
            if (not (np.isnan(centerpts).any() or np.isnan(angles).any() or \
                np.isnan(frames).any())) and self.boxes[0].inside(centerpts[0][0], centerpts[0][1]):
                frames -= self.start_loc
                frames = frames.astype(float)
                frames = np.floor(frames).astype(int)
                if self.possibleego(centerpts, frames):
                    self.possibleegoids += [len(self.agents)]
                self.agents += [['VehicleDataHighD', centerpts, angles, frames, speedxs, \
                    speedys, accxs, accys, None, None]]
                # print(np.max(np.sqrt(speedxs**2+speedys**2)), 
                #     np.max(np.sqrt(accxs**2+accys**2)))
        print("%d agents, %d possible egos" % (len(self.agents), len(self.possibleegoids)))
        self.choose_ego()
        # self.env = HighDSampleEnvironment(self.static_elements, new_agents, timelimit=self.timelimit,
        #     boxes=self.boxes, discrete=self.discrete)
        # self.observation_space = self.env.observation_space
        # self.action_space = self.env.action_space
        self.env.reset()
        for i in range(self.play_until):
            ret = self.env.step(np.array([0, 0]))
        # print("Played until", self.play_until)

    def choose_ego(self):
        """
        Choose an ego id from list of ego ids.
        """
        if not hasattr(self, "envs"):
            self.envs = {}
            self.play_untils = {}
            for egoid in self.possibleegoids:
                direction = '+x' if self.agents[egoid][1][-1][0] > self.agents[egoid][1][0][0] else '-x'
                ix = float(self.agents[egoid][1][0][0])
                iy = float(self.agents[egoid][1][0][1])
                iv = float(np.sqrt(self.agents[egoid][4][0]**2+self.agents[egoid][5][0]**2))
                ia = float(np.sqrt(self.agents[egoid][6][0]**2+self.agents[egoid][7][0]**2))
                # print(np.sqrt(agents[egoid][4]**2+agents[egoid][5]**2))
                # print(np.sqrt(agents[egoid][6]**2+agents[egoid][7]**2))
                new_agents = deepcopy(self.agents)
                self.play_untils[egoid] = self.agents[egoid][3][0]
                new_agents[egoid] = ['Ego', ix, iy, iv, Direction2D(mode = direction), self.agents[egoid][3][0], ia]
                # print(egoid, new_agents[egoid])
                env = HighDSampleEnvironment(self.static_elements, new_agents, timelimit=self.timelimit,
                    boxes=self.boxes, discrete=self.discrete)
                self.observation_space = env.observation_space
                self.action_space = env.action_space
                self.envs[egoid] = env
        if self.sequential:
            egoid = self.possibleegoids[self.eid]
            self.eid = (self.eid+1)%len(self.possibleegoids)
        else:
            egoid = random.choice(self.possibleegoids)
        self.env = self.envs[egoid]
        self.play_until = self.play_untils[egoid]

    def step(self, action):
        """
        Step through the environment.
        """
        return self.env.step(action)

    def seed(self, s=None):
        """
        Seed the environment.
        """
        return self.env.seed(s=s)

    @property
    def state(self):
        """
        Return the environment state.
        """
        return self.env.state

    def reset(self, **kwargs):
        """
        Choose an ego id, recreate the environment and play until ego is starting.
        """
        self.choose_ego()
        # self.env.canvas.close()
        # del self.env
        # gc.collect()
        # self.env = HighDSampleEnvironment(self.static_elements, new_agents, timelimit=self.timelimit,
        #     boxes=self.boxes, discrete=self.discrete)
        # self.observation_space = self.env.observation_space
        # self.action_space = self.env.action_space
        ret = {"next_state": self.env.reset(**kwargs)}
        for i in range(self.play_until):
            ret = self.env.step(np.array([0, 0]))
        # print("Played until", self.play_until)
        return ret["next_state"]
    
    def render(self, **kwargs):
        """
        Render the environment.
        """
        return self.env.render(**kwargs)