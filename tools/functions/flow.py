from tools.base import Function, Parameterized, Algorithm
import torch
import numpy as np

class Flow(Function, Parameterized, Algorithm):
    """
    Normalizing flow.
    """

    network_names = ["Fnn"]

    def __init__(self, config, data, spec):
        """
        Initialize Flow.
        """
        self.config = config
        assert("t" in config.data.keys())
        self.normalize = True if "normalize_flow_inputs" in config.data.keys()\
            and config["normalize_flow_inputs"] else False
        self.Fnn = config["t"].fnn(spec)
        self.Opt = config["t"].adam(self.Fnn, config["learning_rate"])
        self.data = data
        assert(len(data.shape) == 2)
        self.data_min = torch.min(data, dim=0).values
        self.data_max = torch.max(data, dim=0).values
        self.dataset = torch.utils.data.DataLoader(data, 
            batch_size=self.config["minibatch_size"], shuffle=True)
    
    @property
    def hyperparameters(self):
        return self.config
    
    def update_data(self, new_data):
        assert(len(new_data.shape) == 2)
        self.data = torch.cat([self.data, new_data], dim=0)
        self.data_min = torch.min(self.data, dim=0).values
        self.data_max = torch.max(self.data, dim=0).values
        self.dataset = torch.utils.data.DataLoader(self.data, 
            batch_size=self.config["minibatch_size"], shuffle=True)

    def __call__(self, x, inverse=False, log_probs=False):
        """
        Inference (x->z) or generation (z->x).
        """
        if type(x) in [tuple, list]:
            x = self.config["t"].f(x)
        if not inverse and self.normalize:
            x = (x-self.data_min)/(self.data_max-self.data_min)
            x = x*2-1.
        if log_probs:
            assert(not inverse)
            return self.Fnn.log_probs(x)
        ret = self.Fnn(x, inverse=inverse)
        if inverse and self.normalize:
            x = (x+1.)/2
            ret = ret*(self.data_max-self.data_min)+self.data_min
        return ret
    
    def log_probs(self, x):
        """
        Likelihood of data point x.
        """
        if type(x) in [tuple, list]:
            x = self.config["t"].f(x)
        if self.normalize:
            x = (x-self.data_min)/(self.data_max-self.data_min)
            x = x*2-1.
        return self.Fnn.log_probs(x)
    
    def train(self):
        """
        Train to maximize negative log likelihood for one epoch.
        """
        metrics = {}
        nlls = []
        for batch in iter(self.dataset):
            self.Opt.zero_grad()
            nll = -self.log_probs(batch).mean()
            nlls += [nll.item()]
            nll.backward()
            self.Opt.step()
        metrics["avg_nll"] = np.mean(nlls)
        metrics["std_nll"] = np.std(nlls)
        return metrics
