import abc
from typing import Any
from tools.utils import combine_dicts

class Algorithm(abc.ABC):
    """
    Learning algorithm. Consists of a `Hyperparameters` and a training
    method. Testing is not mandatory.
    """

    @property
    @abc.abstractmethod
    def hyperparameters(self):
        """
        Returns a `Hyperparameters` object.
        """
        pass

    @abc.abstractmethod
    def train(self, evaluator=None, **kwargs):
        """
        Details for training are provided in the hyperparameters.
        This method will train for one epoch.
        Returns the training metrics in a dictionary.
        """
        pass

class RLAlgorithm(Algorithm):
    """
    Reinforcement learning algorithm.
    """

    # Using Any to avoid circular import
    config: Any
    policy: Any

    @property
    def hyperparameters(self):
        """
        Return hyperparameters.
        """
        return self.config

    @abc.abstractmethod
    def collect_single_episode(self, metrics, **kwargs):
        """
        Collect a single episode.
        """
        pass

    @abc.abstractmethod
    def collect_episodes(self, metrics, **kwargs):
        """
        Collect a single episode.
        """
        pass

    @abc.abstractmethod
    def train_minibatches(self, metrics, **kwargs):
        """
        Train over minibatches.
        """
        pass

    def train(self, evaluator=None, **kwargs):
        """
        Train one epoch, and evaluate if needed.
        """
        # Collect one episode
        metrics = {}
        metrics = self.collect_episodes(metrics, **kwargs)
        # Train
        metrics = self.train_minibatches(metrics, **kwargs)
        # Evaluate
        if evaluator != None:
            eval_metrics = evaluator.evaluate(self.policy, **kwargs)
            metrics = combine_dicts(metrics, eval_metrics)
        self.epoch += 1
        return metrics