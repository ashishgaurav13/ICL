from tools.algorithms import PPO
from tools.utils import combine_dicts, FixedNormal
import torch
import numpy as np
from tools.functions import create_flow
from tools.utils import rewards_to_returns
from copy import deepcopy

class CPPO(PPO):
    """
    Constrained PPO.
    """

    def __init__(self, config):
        """
        Initialize CPPO.
        """
        super().__init__(config)
        self.D = self.config["t"].f([config["discount_factor"]**i \
            for i in range(config["env"].time_limit)])
        self.M = self.config["t"].f([
            [0,]*i+list(self.D)[:config["env"].time_limit-i] \
                for i in range(config["env"].time_limit)])


    def collect_single_episode(self, metrics, use_flow_not_cost=False):
        """
        Collect a single episode, but with cost.
        """
        assert("cost" in self.config.data.keys())
        if use_flow_not_cost:
            assert("flow" in self.config.data.keys())
            if not hasattr(self, "novelty"):
                def novelty_fn(sa):
                    (s, a) = sa
                    s = self.config["state_reduction"](s)
                    a = self.config["action_reduction"](a)
                    sa = self.config["input_format"](s, a)
                    sa = self.config["t"].f([sa])
                    if -self.config["flow"].log_probs(sa).item() > \
                        self.config["expert_nll"][0]+self.config["expert_nll"][1]: # sa
                        return 1.
                    else:
                        return 0.
                self.novelty = novelty_fn
        collected_data = {}
        if self.config["debug"]:
            if use_flow_not_cost:
                S, A, R, Info = self.config["env"].play_episode(self.policy,
                    info=True, novelty=self.novelty)
            else:
                S, A, R, Info, Costs = self.config["env"].play_episode(self.policy,
                    info=True, cost=self.config["cost"], trunc=False)
            if "env_info" not in metrics.keys(): metrics["env_info"] = {}
            metrics["env_info"] = combine_dicts(metrics["env_info"], Info)
            if not use_flow_not_cost:
                self.max_cost_reached_data += [Info["max_cost_reached"]]
        else:
            if use_flow_not_cost:
                S, A, R = self.config["env"].play_episode(self.policy,
                    novelty=self.novelty)
            else:
                S, A, R, Costs = self.config["env"].play_episode(self.policy,
                    cost=self.config["cost"], trunc=False)
        collected_data["S"] = S
        collected_data["A"] = A
        collected_data["R"] = R
        if not use_flow_not_cost:
            collected_data["C"] = Costs
            self.state_action_cost_data += [[S[:-1], A, Costs]]
        else:
            self.state_action_cost_data += [[S[:-1], A, None]]
        return collected_data, metrics
    
    def train(self, evaluator=None, no_mix=False, **kwargs):
        """
        Train one epoch of PPO, and evaluate if needed.
        """
        # Collect some episodes
        metrics = {}
        self.state_action_cost_data = []
        self.max_cost_reached_data = []
        forward_only = False
        if "forward_only" in kwargs.keys():
            forward_only = kwargs["forward_only"]
            del kwargs["forward_only"]
        metrics = self.collect_episodes(metrics, **kwargs)
        TotalCost = []
        for S, A, Costs in self.state_action_cost_data:
            TotalCost += [rewards_to_returns(Costs, self.config["discount_factor"])[0]]
        metrics["avg_env_edcv"] = sum(TotalCost)/len(TotalCost)
        metrics["max_cost_reached"] = np.mean(self.max_cost_reached_data)
        for key in metrics["env_info"].keys():
            if "Min_Max" in key and hasattr(metrics["env_info"][key], "__len__"):
                metrics["env_info"][key] = (np.min(list(metrics["env_info"][key])), np.max(list(metrics["env_info"][key])))
            elif "Mean_Std" in key and hasattr(metrics["env_info"][key], "__len__"):
                metrics["env_info"][key] = (np.mean(list(metrics["env_info"][key])), np.std(list(metrics["env_info"][key])))
        if forward_only:
            return metrics
        feasibility_losses = []
        OptFeasibility = self.config["t"].adam(self.policy.Pi, \
            self.config["learning_rate_feasibility"])
        for S, A, Costs in self.state_action_cost_data:
            DiscountedCosts = (self.M[:len(Costs), :len(Costs)] \
                @ self.config["t"].f(Costs).view(-1, 1)).view(-1)
            if self.discrete:
                LogProbs = torch.nn.LogSoftmax(dim=-1)(\
                    self.policy.Pi(self.config["t"].f(S))).\
                        gather(1, self.config["t"].l(A).view(-1, 1)).view(-1)
            else:
                self.policy.Normal.update(*self.policy.Pi(self.config["t"].f(S)))
                LogProbs = self.policy.Normal.log_probs(self.config["t"].f(A)).view(-1)
            Diff = (DiscountedCosts * LogProbs).sum()
            Loss = (DiscountedCosts[0] >= self.config["cost"].beta) * Diff
            OptFeasibility.zero_grad()
            Loss.backward()
            OptFeasibility.step()
            feasibility_losses += [Loss.item()]
        metrics["avg_feasibility_loss"] = np.mean(feasibility_losses)
        metrics = self.train_minibatches(metrics)
        # Evaluate
        if evaluator != None:
            eval_metrics = evaluator.evaluate(self.policy)
            metrics = combine_dicts(metrics, eval_metrics)
        self.epoch += 1
        if not no_mix and \
            "mix_save_epoch" in self.config.data.keys() and \
            self.epoch % self.config["mix_save_epoch"] == 0:
            self.config["past_pi_weights"] = self.config["past_pi_weights"] + \
                [deepcopy(self.policy.Pi.state_dict())] 
            print("Intermediate save at epoch=%d" % self.epoch)
            if "flow" in self.config.data.keys():
                dataset = self.config["env"].trajectory_dataset(self.policy, 
                    self.config["expert_episodes"], weights=None)
                aS = self.config["vector_state_reduction"](dataset.S)
                aA = self.config["vector_action_reduction"](dataset.A)
                aSA = self.config["vector_input_format"](aS, aA)
                sims = []
                for i in range(aSA.shape[0]):
                    tSA = aSA[i].view(-1, self.config["i"])[torch.nonzero(
                        dataset.M[i].view(-1)).view(-1)]
                    tp = -self.config["flow"].log_probs(tSA)
                    tpf = (tp > self.config["expert_nll"][0]+\
                        self.config["expert_nll"][1]).float().mean()
                    sims += [tpf.item()]
                am = np.mean(sims)
                self.config["past_pi_dissimilarities"] = \
                    self.config["past_pi_dissimilarities"] +\
                    [am]
        return metrics

    def train_novelty(self, evaluator=None, no_mix=False, **kwargs):
        """
        Train one epoch of PPO, and evaluate if needed. Use flow.
        """
        assert("flow" in self.config.data.keys())
        # Collect some episodes
        metrics = {}
        self.state_action_cost_data = []
        self.max_cost_reached_data = []
        metrics = self.collect_episodes(metrics, use_flow_not_cost=True)
        for key in metrics["env_info"].keys():
            if "Min_Max" in key and hasattr(metrics["env_info"][key], "__len__"):
                metrics["env_info"][key] = (np.min(list(metrics["env_info"][key])), np.max(list(metrics["env_info"][key])))
            elif "Mean_Std" in key and hasattr(metrics["env_info"][key], "__len__"):
                metrics["env_info"][key] = (np.mean(list(metrics["env_info"][key])), np.std(list(metrics["env_info"][key])))
        metrics = self.train_minibatches(metrics)
        # Evaluate
        if evaluator != None:
            eval_metrics = evaluator.evaluate(self.policy)
            metrics = combine_dicts(metrics, eval_metrics)
        self.epoch += 1
        if not no_mix and "mix_save_epoch" in self.config.data.keys() and \
            self.epoch % self.config["mix_save_epoch"] == 0:
            self.config["past_pi_weights"] = self.config["past_pi_weights"] + \
                [deepcopy(self.policy.Pi.state_dict())] 
            print("Intermediate save at epoch=%d" % self.epoch)
        return metrics