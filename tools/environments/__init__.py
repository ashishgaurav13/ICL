from .gym_environment import GymEnvironment, NormalizedStates, \
    NormalizedActions, TimeLimit
from .driving_environment import DrivingEnvironment
from .highD_environment import HighDSampleEnvironmentWrapper
from .gridworld_environment import GridworldEnvironment
from .exiD_environment import ExiDSampleEnvironmentLateral
import gym
import numpy as np

def create(env_type, env_id, normalize_states=True, normalize_actions=True,
    time_limit=200, **kwargs):
    """
    Create an environment based on the env_type, env_id
    """
    assert(env_type in ["gym", "driving", "gridworld"])
    ret = None
    if env_type == "gym":
        if env_id == "CartPole-MoveRight":
            ret = GymEnvironment("CustomCartPole")
        elif env_id == "CartPole-Middle":
            ret = GymEnvironment("CustomCartPole", start_pos=[[-2.4, -1.15], [1.15, 2.4]])
        else:
            ret = GymEnvironment(env_id)
    elif env_type == "driving":
        assert(env_id in ["highD", "exiD"])
        if env_id == "highD":
            ret = HighDSampleEnvironmentWrapper(discrete=False)
        if env_id == "exiD":
            ret = ExiDSampleEnvironmentLateral()
    elif env_type == "gridworld":
        if env_id == "custom1":
            if time_limit is not None:
                time_limit = 50
            r = np.zeros((7, 7)); r[6, 0] = 1.
            t = [(6, 0)]
            u = [(ui, uj) for ui in [3] for uj in [0,1,2,3]]
            s = [(ui, uj) for ui in [0,1,2] for uj in [0,1]]
            ret = GridworldEnvironment(r=r, t=t, stay_action=False, unsafe_states=u,
                start_states=s)
        elif env_id == "custom2":
            if time_limit is not None:
                time_limit = 50
            r = np.zeros((7, 7)); r[6, 6] = 1.
            t = [(6, 6)]
            u = [(ui, uj) for ui in [2,3,4] for uj in [2,3,4]]
            s = [(ui, uj) for ui in [0,1] for uj in [0,1]]
            s += [(ui, uj) for ui in [2,3,4,5,6] for uj in [0,1]]
            s += [(ui, uj) for ui in [0,1] for uj in [2,3,4,5,6]]
            ret = GridworldEnvironment(r=r, t=t, stay_action=False, unsafe_states=u,
                start_states=s)
    if ret is not None and time_limit != None:
        ret = TimeLimit(ret, time_limit)
    if ret is not None and (type(ret.action_space) == gym.spaces.Box) \
        and normalize_actions:
        ret = NormalizedActions(ret)
    if ret is not None and normalize_states:
        ret = NormalizedStates(ret)
    ret.early_exit_metrics = {}
    if ret is not None:
        ret.time_limit = time_limit
    return ret

def get_state_action_space(env_type, env_id, state_only=False):
    """
    Get a smaller state-action space to compute cost maps, accruals etc.
    """
    if env_type == "gym":
        if "CartPole" in env_id:
            return [
                [[([x], 0) for x in np.arange(-2.4, 2.4+0.1, 0.1)], 
                    {"type":"line", "label":"a=0", "x":np.arange(-2.4, 2.4+0.1, 0.1),
                    "left_state_gap": 0.05-1e-3, "right_state_gap": 0.05}],
                [[([x], 1) for x in np.arange(-2.4, 2.4+0.1, 0.1)], 
                    {"type":"line", "label":"a=1", "x":np.arange(-2.4, 2.4+0.1, 0.1),
                    "left_state_gap": 0.05-1e-3, "right_state_gap": 0.05}],
            ]
        if "AntWall" in env_id or "HCWithPos" in env_id:
            return [
                [[([x], np.zeros(8)) for x in np.arange(-5, 5+0.1, 0.1)],
                    {"type":"line", "x": np.arange(-5, 5+0.1, 0.1),
                    "left_state_gap": 0.05-1e-3, "right_state_gap": 0.05,
                    "left_action_gap":1e12-1e-3, "right_action_gap": 1e12}]
            ]
    if env_type == "driving":
        if env_id == "highD":
            return [
                [
                    [[([0, 0, i, j], [0, 0]) for j in np.arange(0, 1010, 10)] for i in np.arange(0, 101, 1)],
                    {"type": "imshow", "process": lambda o: o.T, "scale": [1, 0.1],
                    "kwargs": {"origin": "lower", "extent": [0, 100, 0, 1000], "aspect":0.1}, "fast": 0}
                ]
            ]
        if env_id == "exiD":
            return [
                [
                    [[([i], [j]) for j in np.arange(-2.5, 2.6, 0.1)] for i in np.arange(-10, 10.5, 0.5)],
                    {"type": "imshow", "process": lambda o: np.flip(o, axis=0), "scale": [1, 0.25],
                    "kwargs": {"aspect":0.25, "extent": [-2.5, 2.5, -10, 10]},
                    "left_state_gap": 0.25-1e-3, "right_state_gap": 0.25,
                    "left_action_gap": 0.05-1e-3, "right_action_gap": 0.05}
                ]
            ]
    if env_type == "gridworld":
        return [
            [[[[([x, y], a) for a in np.arange(5)] for y in np.arange(7)] \
                for x in np.arange(7)], {"type":"imshow", "fast": 0}]
        ]
