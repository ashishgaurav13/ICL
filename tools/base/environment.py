import abc
from tools.utils import combine_dicts, rewards_to_returns
from .dataset import TrajectoryDataset
import time
import pickle
import numpy as np
from tools.safe_rl.utils.load_utils import load_policy

class Environment(abc.ABC):
    """
    Abstract environment class. Any subclass must implement `state` (which
    gets current state), `reset` and `step`.
    """

    @abc.abstractmethod
    def seed(self, s=None):
        """
        Seed this environment.
        """
        pass

    @property
    @abc.abstractmethod
    def state(self):
        """
        Get the current state.
        """
        pass

    @abc.abstractmethod
    def reset(self, **kwargs):
        """
        Resets the environment.
        """
        pass

    @abc.abstractmethod
    def step(self, action=None):
        """
        Steps the environment with action (or None if no action).
        """
        pass

    @abc.abstractmethod
    def render(self, **kwargs):
        """
        Renders the environment.
        """
        pass

    def play_episode(self, policy, render=False, buf=None, info=False,
        sleep=None, frames=False, cost=None, deterministic=False):
        """
        Play an episode using the given policy.
        If buffer is given, add data to it.
        If info is True, return combined dict info of entire episode.
        If sleep is True, sleep by that amount at every step
        If frames is True, return rgb_array renderings
        Returns S, A, R, {Info}, {Frames}
        """
        S, A, R = [], [], []
        S.append(self.reset())
        done = False
        Info = {}
        Frames = []
        Costs = []
        Ret = []
        kwargs = {"deterministic": True} if deterministic else {}
        if render:
            if frames:
                Frames += [self.render(mode="rgb_array")]
            else:
                self.render()
            if sleep != None:
                time.sleep(sleep)
        while not done:
            action = policy.act(S[-1], **kwargs)
            A.append(action)
            step_data = self.step(action)
            if render:
                if frames:
                    Frames += [self.render(mode="rgb_array")]
                else:
                    self.render()
                if sleep != None:
                    time.sleep(sleep)
            if cost is not None:
                Costs += [cost((S[-1], action))]
            S.append(step_data["next_state"])
            R.append(step_data["reward"])
            if "info" in step_data.keys():
                Info = combine_dicts(Info, step_data["info"])
            done = step_data["done"]
            if buf != None:
                buf.add((S[-2], A[-1], R[-1], S[-1], done))
        Info["max_cost_reached"] = 0.
        if cost is not None and \
            rewards_to_returns(Costs, cost.discount_factor)[0] >= cost.beta:
            # done = True
            # print(Costs)
            Info["max_cost_reached"] = 1.
        if info:
            Ret += [Info]
        if frames:
            Ret += [Frames]
        if cost is not None:
            Ret += [Costs]
        return S, A, R, *Ret
    
    def trajectory_dataset(self, policy, N, cost=None, deterministic=False,
        weights=None, p=None, config=None, only_success=False, is_torch_policy=True):
        """
        Collect N episodes worth of state-action data, and return the data.
        """
        Data = []
        Gc0 = []
        n = 0
        for n in range(N):
            if weights != None and len(weights) > 0:
                if p != None and len(p) > 0:
                    p2 = np.array(p)/np.sum(p)
                else:
                    p2 = np.ones((len(weights)))/len(weights)
                policy_weights = weights[\
                    np.random.choice(len(weights), p=p2)]
                if is_torch_policy:
                    policy.Pi.load_state_dict(policy_weights)
                else:
                    policy = load_policy(policy_weights, policy, config)
            if cost is None:
                S, A, R = self.play_episode(policy, deterministic=deterministic)
            else:
                S, A, R, C = self.play_episode(policy, cost=cost, deterministic=deterministic)
                if config != None:
                    Gc0 += [rewards_to_returns(C, config["discount_factor"])[0]]
            Data += [[S[:-1], A]]
        if config != None:
            EGc0 = sum(Gc0)/N
            if only_success and EGc0 >= config["beta"] and cost is not None:
                while True:
                    max_index = -1
                    max_val = -float('inf')
                    for n in range(N):
                        if Gc0[n] > max_val:
                            max_val = Gc0[n]
                            max_index = n
                    S, A, R, C = self.play_episode(policy, cost=cost, deterministic=deterministic)
                    Gc0[max_index] = rewards_to_returns(C, config["discount_factor"])[0]
                    oldEGc0 = EGc0
                    EGc0 = sum(Gc0)/N
                    print("Resampling to get expected costs below beta: %g -> %g" % (oldEGc0, EGc0))
                    Data[max_index] = [S[:-1], A]
                    if EGc0 < config["beta"]:
                        break
        return TrajectoryDataset(Data)
