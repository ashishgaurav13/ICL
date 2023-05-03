from tools.base import Buffer
import collections
import numpy as np
import random
import torch

class ReplayBuffer(Buffer):
    """
    Simple replay buffer. Can be used for DQN.
    """
    
    def __init__(self, obs_n, act_dim, discrete, config):
        """
        Create a deque to hold the experiences.
        """
        self.discrete = discrete
        self.config = config
        self.buf = collections.deque(maxlen = config["replay_buffer_size"])
    
    def clear(self):
        """
        Empty buffer.
        """
        self.buf.clear()

    def __len__(self):
        """
        Return current size of replay buffer.
        """
        return len(self.buf)

    def add(self, data):
        """
        Add data to the replay buffer.
        Data should be a 5-tuple (s, a, r, s', d)
        """
        assert(len(data) == 5)
        self.buf.append(data)
    
    def sample(self, n):
        """
        Return minibatch of size n.
        """
        minibatch = random.sample(self.buf, min(n, len(self.buf)))
        S, A, R, S2, D = [], [], [], [], []
        
        for mb in minibatch:
            s, a, r, s2, d = mb
            S += [s]; A += [a]; R += [r]; S2 += [s2]; D += [d]

        t = self.config["t"]
        if type(A[0]) == int:
            return t.f(S), t.l(A), t.f(R), t.f(S2), t.i(D)
        elif type(A[0]) in [float, np.ndarray]:
            return t.f(S), t.f(A), t.f(R), t.f(S2), t.i(D)
        else:
            return t.f(S), torch.stack(A).to(t.device), t.f(R), t.f(S2), t.i(D)


class ReplayBufferPPO(Buffer):
    """
    Replay buffer for PPO.
    """
    
    def __init__(self, obs_n, act_dim, discrete, config):
        """
        Create a deque to hold the experiences.
        """
        self.config = config
        self.N = config["replay_buffer_size"]
        self.discrete = discrete
        self.S = config["t"].f(torch.zeros((self.N, obs_n)))
        if discrete:
            self.A = config["t"].l(torch.zeros((self.N)))
        else:
            self.A = config["t"].f(torch.zeros((self.N, act_dim)))
        self.returns = config["t"].f(torch.zeros((self.N)))
        self.log_probs = config["t"].f(torch.zeros((self.N)))
        self.i = 0
        self.t = config["t"]
        self.filled = 0
    
    def __len__(self):
        """
        Return current size of replay buffer.
        """
        return self.filled

    def clear(self):
        """
        Empty buffer.
        """
        self.S.zero_()
        self.A.zero_()
        self.returns.zero_()
        self.log_probs.zero_()
        self.i = 0
        self.filled = 0

    def add(self, data):
        """
        Add data to the replay buffer.
        Data should be (S, A, returns, log_probs), i.e. for multiple steps.
        """
        assert(len(data) == 4)
        S, A, returns, log_probs = data
        M = S.shape[0]
        self.filled = min(self.filled+M, self.N)
        assert(M <= self.N)
        for j in range(M):
            self.S[self.i] = self.t.f(S[j, :])
            if self.discrete:
                self.A[self.i] = self.t.l(A[j])
            else:
                self.A[self.i] = self.t.f(A[j])
            self.returns[self.i] = self.t.f(returns[j])
            self.log_probs[self.i] = self.t.f(log_probs[j])
            self.i = (self.i + 1) % self.N
    
    def sample(self, n):
        """
        Return minibatch of size n.
        """
        minibatch = random.sample(range(self.filled), min(n, self.filled))
        S, A, returns, log_probs = [], [], [], []
        
        for mbi in minibatch:
            s, a, ret, lp = self.S[mbi], self.A[mbi], self.returns[mbi], \
                self.log_probs[mbi]
            S += [s]; A += [a]; returns += [ret]; log_probs += [lp]

        t = self.config["t"]
        return torch.stack(S).to(t.device), torch.stack(A).to(t.device), \
            torch.stack(returns).to(t.device), \
            torch.stack(log_probs).to(t.device)