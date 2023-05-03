from tools.data import Bits
from inspect import isfunction
from tools.utils import combine_dicts
from tools.logic import SequentialPredicates

class AtomicPropositions:
    """
    Atomic Propositions. Pass in propositions as
    lambda functions dependent on the timestep t.
    Ensure these lambdas return boolean.

    propositions = {
        "A": lambda t: t % 2 == 0,
        "B": lambda t: t == 3,
    }
    ap = AtomicPropositions(propositions)
    ap.t => 0
    ap[0,1] => True, False
    ap.step()
    ap.t => 1
    ap[0,1] => False, False
    ...

    Taken from github.com/ashishgaurav13/wm2
    """

    def __init__(self, propositions = {}):
        """
        Initialize AtomicPropositions.
        """
        assert(type(propositions) == dict)
        self._d = Bits()
        self._p = propositions
        self._n = len(self._p)
        self._k = list(propositions.keys())
        self.t = 0
        self.update_data() # initial values

    def update_data(self):
        """
        Update _d for all propositions.
        """
        for i, func in enumerate(self._p.values()):
            if not isfunction(func):
                self._d[i] = func
            else:
                self._d[i] = func(self.t)

    def reset(self):
        """
        Initialize t=0.
        """
        self.t = 0

    def step(self):
        """
        Increment t and update data.
        """
        self.t += 1
        self.update_data() # initial values
    
    def __getitem__(self, index):
        """
        Return data by proposition index.
        """
        return self._d[index]
    
    def __iter__(self):
        """
        Yield data for each proposition, one by one.
        """
        for i in range(self._n):
            yield self._d[i]
    
    def __int__(self):
        """
        Convert _d to integer.
        """
        return int(self._d)
    
    def get_dict(self):
        """
        Get the dictionary associated with the propositions.
        """
        ret = {}
        for i, k in enumerate(self._k):
            ret[k] = self._d[i]
        return ret

class SequentialAtomicPropositions:
    """
    Sequential Atomic Propositions. Pass in propositions as
    lambda functions dependent on previous evaluations, timestep t.
    Ensure these lambdas return boolean.

    Additionally,
    
    (1) you can also pass extra objects in objs and use them through p.
    (2) you can pass a SequentialPredicates object in pre and use it through p.

    class Z: pass
    z = Z()
    z.zz = 20
    objs = {
        "z": z,
    }
    seq_propositions = {
        "A": lambda p, t: t % 2 == 0,
        "B": lambda p, t: not p['A'],
        "C": lambda p, t: p['z'].zz == 20
    }
    ap = SequentialAtomicPropositions(seq_propositions, objs)
    ap.t => 0
    ap[0,1,2] => True, False, True
    ap.step()
    ap.t => 1
    ap[0,1,2] => False, True, True
    ap.step()
    ap.t => 2
    ap[0,1,2] => True, False, True
    ...

    Taken from github.com/ashishgaurav13/wm2
    """

    def __init__(self, propositions = {}, objs = {}, pre = None):
        """
        Initialize SequentialAtomicPropositions.
        """
        assert(type(propositions) == dict)
        assert(type(objs) == dict)
        self._d = Bits()
        self._p = propositions
        self._n = len(self._p)
        self._k = list(self._p.keys())
        self.objs = objs
        if pre != None: assert(type(pre) == SequentialPredicates)
        self.pre = pre
        self.t = 0
        self.update_data() # initial values

    def update_data(self):
        """
        Update _d for each proposition.
        """
        evaluated = combine_dicts({}, self.objs)
        if self.pre != None:
            assert(self.pre.t == self.t)
            evaluated = combine_dicts(evaluated, self.pre.get_dict())
        for i, func in enumerate(self._p.values()):
            if not isfunction(func):
                evaluated[self._k[i]] = func
            else:   
                evaluated[self._k[i]] = func(evaluated, self.t)
            self._d[i] = evaluated[self._k[i]]          

    def reset(self):
        """
        Initialize t=0.
        """
        self.t = 0

    def step(self):
        """
        Increment t and update data.
        """
        self.t += 1
        self.update_data() # initial values
    
    def __getitem__(self, index):
        """
        Get proposition data by index.
        """
        return self._d[index]
    
    def __iter__(self):
        """
        Yield proposition data one by one.
        """
        for i in range(self._n):
            yield self._d[i]
    
    def __int__(self):
        """
        Convert _d to int.
        """
        return int(self._d)
    
    def get_dict(self, keys = None):
        """
        Get dictionary associated with propositions.
        """
        ret = {}
        for i, k in enumerate(self._k):
            if keys != None and k in keys:
                ret[k] = self._d[i]
            elif keys == None:
                ret[k] = self._d[i]
        return ret