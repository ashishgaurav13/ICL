"""
Different machine learning and reinforcement learning algorithms.
"""
from .ppo import PPO, PPOPolicy

def create(name):
    """
    Create an algorithm based on the name.
    """
    return {
        "ppo": PPO,
    }[name]