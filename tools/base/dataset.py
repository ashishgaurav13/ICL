import torch
import abc
from tools.utils import TorchHelper

class Dataset(abc.ABC):
    """
    Anything with data.
    """

    @property
    @abc.abstractmethod
    def data(self):
        """
        Return data.
        """
        pass

    @abc.abstractmethod
    def __len__(self):
        """
        Return dataset size.
        """
        pass

    def save(self, filename="data.pt"):
        """
        Save dataset.
        """
        torch.save(self.data, filename)
    
    def load(filename="data.pt"):
        """
        Load dataset.
        """
        return torch.load(filename)

class TrajectoryDataset(Dataset):
    """
    Dataset of state-action pairs.
    """

    def __init__(self, d):
        """
        Initialize TrajectoryDataset.
        """
        self.d = d
        t = TorchHelper()
        self.S = torch.zeros(len(self), self.max_trajectory_length, self.obs_n,
            dtype=torch.float, device=t.device)
        self.A = torch.zeros(len(self), self.max_trajectory_length, self.act_dim,
            dtype=torch.float, device=t.device)
        self.M = torch.zeros(len(self), self.max_trajectory_length,
            dtype=torch.bool, device=t.device)
        for i, (S, A) in enumerate(self.d):
            self.S[i, :len(S)] = t.f(S).view(-1, self.obs_n)
            self.A[i, :len(A)] = t.f(A).view(-1, self.act_dim)
            self.M[i, :len(A)] = torch.ones(len(A), dtype=torch.bool, device=t.device)

    @property
    def data(self):
        """
        Return expert dataset.
        """
        return self.d

    def load(filename="data.pt"):
        """
        Load dataset.
        """
        return TrajectoryDataset(torch.load(filename))

    def __len__(self):
        """
        Length of the dataset.
        """
        return len(self.data)
    
    @property
    def max_trajectory_length(self):
        """
        Return max trajectory length.
        """
        return max([len(traj[0]) for traj in self.d])

    @property   
    def obs_n(self):
        """
        Observation dimensionality.
        """
        ret = None
        for [S, A] in self.d:
            for s in S:
                if ret is None:
                    if hasattr(s, "__len__"):
                        ret = len(s)
                    else:
                        ret = 1
                else:
                    if hasattr(s, "__len__"):
                        assert(len(s) == ret)
        return ret
    
    @property
    def act_dim(self):
        """
        Action space dimensionality.
        """
        ret = None
        for [S, A] in self.d:
            for a in A:
                if ret is None:
                    if hasattr(a, "__len__"):
                        ret = len(a)
                    else:
                        ret = 1
                else:
                    if hasattr(a, "__len__"):
                        assert(len(a) == ret)
        return ret