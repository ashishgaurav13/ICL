from tools.base import Algorithm, TrajectoryDataset
import numpy as np
import torch
from safe_rl.utils.load_utils import load_policy

class CostAdjustment(Algorithm):
    """
    Adjust cost function so that agent policy cost is increased, while
    expert cost stays within cost threshold.
    """
    
    def __init__(self, config):
        """
        Initialize CostAdjustment.
        """
        self.config = config
        self.past_pi_weights = config["past_pi_weights"]
        self.policy = config["policy_class"](config)
        self.pi_episodes = config["pi_episodes"]
        self.updates_per_epoch = config["updates_per_epoch"]
        self.alpha = config["alpha"]
        self.expert_dataset = TrajectoryDataset.load()
        self.D = self.config["t"].f([config["discount_factor"]**i \
            for i in range(self.expert_dataset.max_trajectory_length+10)])        

    @property
    def hyperparameters(self):
        """
        Return configuration.
        """
        return self.config
    
    def Jc(self, S, A, M, s=None):
        """
        Compute expected episodic cost.
        """
        S = self.config["cost"].vector_state_reduction(S)
        A = self.config["cost"].vector_action_reduction(A)
        SA = self.config["cost"].vector_input_format(S, A)
        ret = ((self.config["cost"].Cost(SA).squeeze() * M) @ self.D[:M.shape[-1]])
        if s is not None:
            ret *= s
        return ret

    def train(self, evaluator=None, compute_sims=True, **kwargs):
        """
        Train one epoch.
        """
        metrics = {}
        if "agent_dataset" not in self.config.data.keys():
            AgentDataset = []
            if len(self.config["past_pi_weights"]) > 0:
                print("we have past_pi_weights")
                if "past_pi_dissimilarities" in self.config.data and \
                    len(self.config["past_pi_dissimilarities"]) > 0:
                    p2 = np.array(self.config["past_pi_dissimilarities"])/\
                        np.sum(self.config["past_pi_dissimilarities"])
                else:
                    p2 = np.ones((len(self.config["past_pi_weights"])))\
                        /len(self.config["past_pi_weights"])
            for ei in range(self.pi_episodes):
                if len(self.config["past_pi_weights"]) > 0:
                    policy_weights = self.past_pi_weights[\
                        np.random.choice(len(self.past_pi_weights), p=p2)]
                    if self.config["is_torch_policy"]:
                        self.policy.Pi.load_state_dict(policy_weights)
                    else:
                        self.policy = load_policy(policy_weights, self.policy, self.config)
                S, A, R = self.config["env"].play_episode(self.policy) # (no cost)
                AgentDataset += [[S[:-1], A]]
            agent_dataset = TrajectoryDataset(AgentDataset)
        else:
            agent_dataset = self.config["agent_dataset"]
        sims = None
        if compute_sims and "flow" in self.config.data.keys():
            print("computing sims")
            aS = self.config["vector_state_reduction"](agent_dataset.S)
            aA = self.config["vector_action_reduction"](agent_dataset.A)
            aSA = self.config["vector_input_format"](aS, aA)
            sims = []
            for i in range(aSA.shape[0]):
                tSA = aSA[i].view(-1, self.config["i"])[torch.nonzero(
                    agent_dataset.M[i].view(-1)).view(-1)]
                tp = -self.config["flow"].log_probs(tSA)
                tpf = (tp > self.config["expert_nll"][0]+\
                    self.config["expert_nll"][1]).float().mean()
                sims += [tpf.item()]
            sims = self.config["t"].f(sims).view(-1)
            sims /= sims.sum()
        if agent_dataset.max_trajectory_length > self.expert_dataset.max_trajectory_length:
            self.D = self.config["t"].f([self.config["discount_factor"]**i \
                for i in range(agent_dataset.max_trajectory_length+10)])        
        AgentCosts = [*self.Jc(agent_dataset.S, agent_dataset.A,
            agent_dataset.M).detach().cpu()]
        ExpertSatisfaction = np.mean((self.Jc(self.expert_dataset.S,
            self.expert_dataset.A, self.expert_dataset.M).detach().cpu() \
            < self.config["cost"].beta).float().numpy())
        metrics["avg_agent_cost"] = np.mean(AgentCosts)
        metrics["expert_satisfaction"] = ExpertSatisfaction
        objective1s, objective2s = [], []
        for updatei in range(self.updates_per_epoch):
            self.config["cost"].Opt.zero_grad()
            if sims != None:
                objective1 = -self.Jc(agent_dataset.S, agent_dataset.A,
                    agent_dataset.M, s=sims).sum()
            else:
                objective1 = -self.Jc(agent_dataset.S, agent_dataset.A,
                    agent_dataset.M).mean()
            objective2 = self.Jc(self.expert_dataset.S, self.expert_dataset.A,
                self.expert_dataset.M)
            objective2 = (objective2 >= self.config["cost"].beta).float() * \
                (objective2 - self.config["cost"].beta)
            objective2 = objective2.mean()
            (objective1+objective2*self.alpha).backward()
            objective1s += [objective1.item()]
            objective2s += [objective2.item()]
            self.config["cost"].Opt.step()
        metrics["avg_objective1"] = np.mean(objective1s)
        metrics["avg_objective2"] = np.mean(objective2s)
        return metrics
