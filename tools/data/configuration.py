from tools.utils import TorchHelper, convert_lambdas
from tools.base import Environment
from inspect import isfunction, getfullargspec
from copy import deepcopy
import pprint
import json

class Configuration(Environment):
    """
    Holds configurable key/value data. 
    
    Configuration can be dynamic, i.e. changing with time. To update the
    `Configuration` object, call `Configuration.step()`. `Configuration[key]` 
    can be directly used to access the current value of any key.
    """

    def __init__(self, data):
        """
        Given data (key/value) pairs, instantiate the `Configuration` object.
        """
        assert(type(data) == dict)
        self.data = data
        self.callables = set([])
        for key in self.data.keys():
            if isfunction(self.data[key]) and getfullargspec(self.data[key])[-1] == ['t']:
                self.callables.add(key)
        if "t" not in self.data.keys():
            self.data["t"] = TorchHelper()
        if "seed" in self.data.keys():
            print("Seed=%d" % self.data["seed"])
            self.seed(self.data["seed"])
        self.t = 0
    
    def seed(self, s=None):
        """
        Seed torch helper, environments.
        """
        if "t" in self.data.keys() and hasattr(self.data["t"], "seed"):
            self.data["t"].seed(s)
        if "env" in self.data.keys() and hasattr(self.data["env"], "seed"):
            self.data["env"].seed(s)
        if "test_env" in self.data.keys() and hasattr(self.data["test_env"], "seed"):
            self.data["test_env"].seed(100+s)
        if "sampling_env" in self.data.keys() and hasattr(self.data["sampling_env"], "seed"):
            self.data["sampling_env"].seed(200+s)

    @property
    def state(self):
        """
        Get the current configuration as a dict.
        """
        return_data = {}
        for key in self.data.keys():
            if key in self.callables:
                return_data[key] = self.data[key](self.t)
            else:
                return_data[key] = self.data[key]
        return return_data
    
    def show(self, **kwargs):
        """
        Show existing configuration.
        """
        self.render(**kwargs)
    
    def render(self, **kwargs):
        """
        Render this configuration environment.
        """
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(self.state)

    def update(self, data):
        """
        Update Configuration object with new data.
        """
        for key in data.keys():
            self[key] = data[key]
            if isfunction(self.data[key]) and getfullargspec(self.data[key])[-1] == ['t']:
                self.callables.add(key)
            else:
                if key in self.callables:
                    self.callables.remove(key)

    def at(self, t):
        """
        Return the configuration at time t.
        """
        old_t = deepcopy(self.t)
        self.t = t
        return_data = deepcopy(self.state)
        self.t = old_t
        return return_data

    def reset(self, **kwargs):
        """
        Reset the configuration to t=0.
        """
        self.t = 0
        return self.state
    
    def step(self, action=None):
        """
        Increment time by 1, and return the (NextState-Reward-Done-Info) dict.
        """
        self.t += 1
        return {
            "next_state": self.state,
            "reward": None,
            "done": None,
            "info": None,
        }
    
    def __getitem__(self, key):
        """
        Get the key from the current configuration.
        """
        return self.state[key]
    
    def __setitem__(self, key, value):
        """
        Set key/value in current configuration.

        Note: if a key is changed to a time function, time is not reset and
        will need to be reset manually.
        """
        self.data[key] = value
        if isfunction(self.data[key]) and getfullargspec(self.data[key])[-1] == ['t']:
            self.callables.add(key)
        else:
            if key in self.callables:
                self.callables.remove(key)
    
    def from_json(f, update_params={}):
        """
        Create a Configuration from JSON file.
        """
        d = json.load(open(f, "r"))
        for key, value in update_params.items():
            d[key] = value
        return Configuration(convert_lambdas(d))