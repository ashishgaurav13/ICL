from tools.base import Parameterized, Function
import torch
from inspect import isfunction
from dill.source import getsource

class CostFunction(Parameterized, Function):
    """
    Maps a (state, action) datapoint (or a subset of its features) to a real
    number in [0, 1].
    """

    network_names = ["Cost"]
    constants = ["i", "h", "o"]
    default_attrs = {
        "input_format": "lambda s, a: [*s, a]",
        "vector_input_format": "lambda S, A: torch.cat((S, A), dim=-1)",
        "state_reduction": "lambda s: s",
        "vector_state_reduction": "lambda S: S",
        "action_reduction": "lambda a: a",
        "vector_action_reduction": "lambda A: A",
    }

    def __init__(self, config, **kwargs):
        """
        Initialize the CostFunction.
        """
        self.config = config
        assert("beta" in config.data.keys())
        assert("t" in config.data.keys())
        self.attrs = {}
        for attr in self.default_attrs.keys():
            if attr in kwargs.keys():
                assert(type(kwargs[attr]) == str)
                setattr(self, attr, eval(kwargs[attr]))
                self.attrs[attr] = kwargs[attr]
            elif attr in config.data.keys():
                assert(isfunction(config[attr]))
                setattr(self, attr, config[attr])
                self.attrs[attr] = getsource(config[attr]).strip()
            else:
                setattr(self, attr, eval(self.default_attrs[attr]))
                self.attrs[attr] = self.default_attrs[attr]
        for constant in self.constants:
            assert(constant in kwargs.keys())
        self.Cost = self.config["t"].nn([
            [kwargs["i"], kwargs["h"]], "r",
            [kwargs["h"], kwargs["h"]], "r",
            [kwargs["h"], kwargs["o"]], "s"
        ])
        self.Opt = self.config["t"].adam(self.Cost, 
            self.config["learning_rate"])
    
    @property
    def beta(self):
        """
        Get cost threshold.
        """
        return self.config["beta"]
    
    @property
    def discount_factor(self):
        """
        Get discount factor.
        """
        if "discount_factor" in self.config.data.keys():
            return self.config["discount_factor"]
        return 1.0

    def __call__(self, sa, invert=False):
        """
        Forward pass with the cost network.
        """
        s, a = sa
        s = self.state_reduction(s)
        a = self.action_reduction(a)
        sa = self.config["t"].f(self.input_format(s, a))
        if invert:
            return 1.-self.Cost(sa)
        return self.Cost(sa)