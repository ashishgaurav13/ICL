"""
Different machine learning and reinforcement learning algorithms.
"""
from .ppo import PPO, PPOPolicy
from .cost_adjustment import CostAdjustment
from .constrained_ppo import CPPO
from .ppo_lag import PPOLag, PPOPolicyWithCost

def create(name):
    """
    Create an algorithm based on the name.
    """
    return {
        "ppo": PPO,
        "cost_adjustment": CostAdjustment,
        "constrained_ppo": CPPO,
        "ppo_lagrange": PPOLag,
    }[name]