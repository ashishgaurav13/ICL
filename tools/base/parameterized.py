import torch

class Parameterized:
    """
    Anything with parameters.
    """

    network_names: str

    def save(self, filename="model.pt", **kwargs):
        """
        Save networks.
        """
        ret = {}
        for attr in self.network_names:
            ret[attr] = getattr(self, attr).state_dict()
            print("Saved: %s" % attr)
        torch.save(ret, filename, **kwargs)
    
    def load(self, filename="model.pt", **kwargs):
        """
        Load networks.
        """
        ret = torch.load(filename, **kwargs)
        for attr in self.network_names:
            getattr(self, attr).load_state_dict(ret[attr])
            print("Loaded: %s" % attr)
