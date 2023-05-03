from tools.logic import Scanner, Parser, Errors, \
    AtomicPropositions, SequentialAtomicPropositions
from tools.data import Bits

class LTLProperty(object):
    """
    Holds an LTL property.

    propositions = {
        "A": func1(...),
        "B": func2(...),
    }
    ltlp = LTLProperty("A U B", 0, propositions)
    ltlp.x => "A U B"
    ltlp.reward => 0
    ltlp.active => True
    ltlp.mode => "violation"
    ltlp([some_value_A, some_value_B]) => True/False, {...}
    ltlp.reset()

    Taken from github.com/ashishgaurav13/wm2
    """

    def __init__(self, x, reward = 0, propositions = {}, mode = 'violation', active = True):
        """
        Initialize an LTLProperty.
        """
        assert(mode in ['violation', 'satisfaction'])
        assert(type(propositions) == dict)
        self.parser = Parser()
        self.x = x
        self.reward = reward
        self.active = active
        self.mode = mode

        #: initialise the Errors class to give meaningful messages when calling parser.SetProperty
        Errors.Init(self.x, "", False, self.parser.getParsingPos,
                    self.parser.errorMessages)

        self._p = propositions

        # If propositions is {"A": func1(...), "B": func2(...)},
        # parser.APdict = {"A": 0, "B": 1}
        self.parser.APdict = {key: i for i, key in enumerate(self._p.keys())}
        self.parser.SetProperty(Scanner(self.x))
        self.mc_status = Parser.UNDECIDED
        self.status = False

    def activate(self):
        """
        Set to active.
        """
        self.active = True

    def deactivate(self):
        """
        Set to inactive.
        """
        self.active = False

    def update_status(self):
        """
        Check mc_status and update status.
        """
        self.status = {
            'violation': self.mc_status == Parser.FALSE,
            'satisfaction': self.mc_status == Parser.TRUE,
        }[self.mode]        
        return self.status

    def reset(self):
        """
        Reset LTLProperty.
        """
        self.parser.ResetProperty()
        self.mc_status = Parser.UNDECIDED
        self.status = False

    def check(self, trace):
        """
        Check property against complete trace.
        """
        if self.active:
            self.mc_status = self.parser.Check(trace)
        return self.mc_status

    def check_incremental(self, next_state):
        """
        Add next_state to current trace and re-check.
        """
        if self.active:
            self.mc_status = self.parser.CheckIncremental(next_state)
        return self.mc_status

    def __call__(self, next_state = None):
        """
        Returns status, info.
        """
        if next_state != None:
            self.check_incremental(next_state)
        self.update_status()
        ret_info = {}
        if self.status == True:
            ret_info[self.mode] = self.x
        return self.status, ret_info

class LTLProperties:
    """
    Holds a set of LTL properties.

    propositions = {
        "a": lambda t: t >= 2,
        "b": lambda t: t >= 4,
        "c": lambda t: t == 3,
        "d": lambda t: 0 <= t <= 3,
    }
    # for 0 <= t <= 5
    ltl = LTLProperties(propositions, [
        ["a => b", -1, "violation"], # 0,0,-1,-1,0,0
        ["b or c", -1, "violation"], # -1,-1,-1,0,0,0
        ["a U b", 1, "satisfaction"], # 0,0,0,0,0,0
        ["d U b", 1, "satisfaction"], # 1,1,1,1,1,1
    ])
    ltl.reset() => 0+(-1)+0, {"violation": ["b or c"]}, (T=) 0
    ltl.step() => 0+(-1)+0, {"violation", ["b or c"]}, (T=) 1
    ltl.step() => (-1)+(-1)+1, {...}, ...
    ...

    Taken from github.com/ashishgaurav13/wm2
    """

    def __init__(self, propositions, properties, 
        sequential = False, objs = {}):
        """
        Initialize LTLProperties.
        If sequential, then objs can be passed.
        """

        assert(type(propositions) == dict)
        assert(type(properties) == list)
        for p in properties: assert(type(p) == list and len(p) == 3)
        self.sequential = sequential
        # Create AP
        if sequential:
            self.ap = SequentialAtomicPropositions(propositions, objs)
        else:
            self.ap = AtomicPropositions(propositions)
        # Create properties
        self.properties = []
        for p in properties:
            property_str, reward, mode = p
            self.properties += [
                LTLProperty(property_str, reward, propositions, mode)
            ]
            self.properties[-1].reset()

    def check(self):
        """
        Returns total reward, info and time.
        """
        total_reward = 0
        violations = []
        satisfactions = []
        for p in self.properties:
            status, info = p(int(self.ap)) # list(self.ap) to see the APs
            if status:
                total_reward += p.reward
                if "violation" in info: violations += [info["violation"]]
                if "satisfaction" in info: satisfactions += [info["satisfaction"]]
        info = {}
        if len(violations) > 0: info["violations"] = violations
        if len(satisfactions) > 0: info["satisfactions"] = satisfactions
        return total_reward, info, self.ap.t

    def reset(self):
        """
        Reset LTLProperties.
        """
        for p in self.properties:
            p.reset()
        return self.check()
    
    def step(self):
        """
        Step through time.
        """
        self.ap.step()
        return self.check()