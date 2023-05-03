from tools.base import RLAlgorithm, Policy
from tools.data import Configuration, ReplayBufferPPO
from tools.utils import combine_dicts, rewards_to_returns, FixedNormal
import numpy as np
import torch
import gym

class PPOPolicy(Policy):
    """
    Policy based on V, Pi networks.
    Supports both discrete/continuous action spaces.
    """

    def __init__(self, config):
        """
        Create V, Pi networks.
        """
        self.config = config
        self.network_names = ["V", "Pi"]
        self.obs_n = config["env"].observation_space.shape
        assert(len(self.obs_n) == 1)
        self.obs_n = self.obs_n[0]
        self.h_n = config["hidden_size"]
        self.discrete = type(config["env"].action_space) == gym.spaces.Discrete
        if self.discrete:
            self.act_n = config["env"].action_space.n
        else:
            assert(len(config["env"].action_space.shape) == 1)
            self.act_n = config["env"].action_space.shape[0]
        self.reset_parameters()
        self.Normal = FixedNormal(0., 1.)

    def reset_parameters(self):
        """
        Reset all networks.
        """
        self.V = \
            self.config["t"].nn([
                [self.obs_n, self.h_n], 'r', 
                [self.h_n, self.h_n], 'r', 
                [self.h_n, 1]
            ])
        self.OptV = \
            self.config["t"].adam(self.V, self.config["learning_rate"])
        if self.discrete:
            self.Pi = \
                self.config["t"].nn([
                    [self.obs_n, self.h_n], 'r',
                    [self.h_n, self.h_n], 'r',
                    [self.h_n, self.act_n]
                ])
        else:
            self.Pi = \
                self.config["t"].nn([
                    [self.obs_n, self.h_n], 't',
                    [self.h_n, self.h_n], 't',
                    ['g', self.h_n, self.act_n] # Gaussian policy
                ])
        self.OptPi = \
            self.config["t"].adam(self.Pi, self.config["learning_rate"])

    def act(self, s, deterministic=False):
        """
        Returns action to be taken in state s.
        """
        s = self.config["t"].f([s])
        if self.discrete:
            probs = torch.nn.Softmax(dim=-1)(self.Pi(s)).view(-1)
            action = np.random.choice(self.act_n, p = probs.cpu().detach().numpy())
        else:
            self.Normal.update(*self.Pi(s))
            if deterministic:
                action = self.Normal.mode().view(-1).detach().cpu().numpy()
            else:
                action = self.Normal.sample().view(-1).detach().cpu().numpy()
        return action

class PPO(RLAlgorithm):
    """
    Proximal Policy Optimization. Supports discrete/continuous action space.
    
    Based loosely on Slide 14
    cs.uwaterloo.ca/~ppoupart/teaching/cs885-fall21/slides/cs885-module1.pdf
    """

    def __init__(self, config):
        """
        Initialize PPO (replay buffer, policy etc.)
        """
        self.config = config
        self.policy = PPOPolicy(config)
        self.discrete = self.policy.discrete
        self.buffer = ReplayBufferPPO(self.policy.obs_n, self.policy.act_n,
            discrete=self.discrete, config=config)
        self.epoch = 0 # epochs completed
 
    def collect_single_episode(self, metrics, **kwargs):
        """
        Collect a single episode.
        """
        collected_data = {}
        if self.config["debug"]:
            S, A, R, Info = self.config["env"].play_episode(self.policy,
                info=True)
            if "env_info" not in metrics.keys(): metrics["env_info"] = {}
            metrics["env_info"] = combine_dicts(metrics["env_info"], Info) 
        else:
            S, A, R = self.config["env"].play_episode(self.policy)
        collected_data["S"] = S
        collected_data["A"] = A
        collected_data["R"] = R
        return collected_data, metrics

    def collect_episodes(self, metrics, **kwargs):
        """
        Collect some episodes.
        """
        all_S, all_A, all_returns = [], [], []
        episode_L, episode_R = [], []
        for episode in range(self.config["episodes_per_epoch"]):
            collected_data, metrics = self.collect_single_episode(metrics, **kwargs)
            all_S += collected_data["S"][:-1] # ignore last state
            all_A += collected_data["A"]
            all_returns += [self.config["t"].f(rewards_to_returns(collected_data["R"], 
                self.config["discount_factor"]))]
            episode_L += [len(collected_data["R"])]
            episode_R += [sum(collected_data["R"])]
        S = self.config["t"].f(np.array(all_S))
        returns = torch.cat(all_returns, dim=0).flatten()
        if self.discrete:
            A = self.config["t"].l(np.array(all_A))
            log_probs = torch.nn.LogSoftmax(dim=-1)(self.policy.Pi(S)).\
                gather(1, A.view(-1, 1)).view(-1)
        else:
            A = self.config["t"].f(np.array(all_A))
            self.policy.Normal.update(*self.policy.Pi(S))
            log_probs = self.policy.Normal.log_probs(A).view(-1)
        self.buffer.add((S, A, returns, log_probs.detach()))
        metrics["avg_env_ep_len"] = np.mean(episode_L)
        metrics["avg_env_reward"] = np.mean(episode_R)
        return metrics

    def train_minibatches(self, metrics, **kwargs):
        """
        Train over minibatches.
        """
        avg_loss_v, avg_loss_pi = [], []
        for subepoch in range(self.config["subepochs"]):
            S, A, returns, old_log_probs = self.buffer.sample(self.config["minibatch_size"])
            if self.discrete:
                log_probs = torch.nn.LogSoftmax(dim=-1)(self.policy.Pi(S)).\
                    gather(1, A.view(-1, 1)).view(-1)
            else:
                self.policy.Normal.update(*self.policy.Pi(S))
                log_probs = self.policy.Normal.log_probs(A).view(-1)
            self.policy.OptV.zero_grad()
            objective1 = (returns - self.policy.V(S)).pow(2).mean()
            objective1.backward()
            self.policy.OptV.step()
            self.policy.OptPi.zero_grad()
            advantages = returns - self.policy.V(S)
            ratio = torch.exp(log_probs - old_log_probs)
            clipped_ratio = torch.clamp(ratio, 1-self.config["ppo_clip_param"], 
                1+self.config["ppo_clip_param"])
            objective2 = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
            if self.discrete:
                softmax = torch.nn.Softmax(dim=-1)(self.policy.Pi(S))
                logsoftmax = torch.nn.LogSoftmax(dim=-1)(self.policy.Pi(S))
                H = -(softmax * logsoftmax).sum(1).mean()
            else:
                H = self.policy.Normal.entropy().mean()
            objective2 -= self.config["ppo_entropy_coef"] * H
            objective2.backward()
            self.policy.OptPi.step()
            avg_loss_v += [objective1.item()]
            avg_loss_pi += [objective2.item()]
        metrics["avg_loss_v"] = np.mean(avg_loss_v)
        metrics["avg_loss_pi"] = np.mean(avg_loss_pi)
        return metrics