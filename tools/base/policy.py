import abc
from tools.base import Parameterized

class Policy(abc.ABC, Parameterized):
    """
    A policy (deterministic or stochastic) describes how to act in a state.
    """

    @abc.abstractmethod
    def act(self, s):
        """
        Returns action to be taken in the state s.
        """
        pass