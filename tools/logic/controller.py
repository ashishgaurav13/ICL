from tools.base import Policy
from tools.utils import combine_dicts

class Controller(Policy):
    """
    A type of rule based policy.
    Taken from github.com/ashishgaurav13/wm2 
    """

    def __init__(self, precondition_fn, rule_fn, name):
        """
        Initialize Controller.
        Every controller has a precondition_fn and a rule_fn to produce
        control inputs.
        """
        assert(precondition_fn)
        assert(rule_fn)
        self.precondition_fn = precondition_fn
        self.rule_fn = rule_fn
        self.name = name

    def active(self, p):
        """
        Checks if precondition is met.
        p is a dict of predicates.
        """
        return self.precondition_fn(p)
    
    def act(self, i):
        """
        i = (p, m)
        p is a dict of predicates
        m is a dict of multipliers
        """
        p, m = i
        assert(self.active(p))
        return self.rule_fn(p, m)

class DefaultController(Policy):
    """
    Always active default controller.
    Taken from github.com/ashishgaurav13/wm2
    """

    def __init__(self, rule_fn, name):
        """
        Initialize DefaultController.
        """
        assert(rule_fn)
        self.rule_fn = rule_fn
        self.name = name

    def act(self, i):
        """
        i = (p, m)
        p is a dict of predicates
        m is a dict of multipliers
        """
        p, m = i
        return self.rule_fn(p, m)


class ComplexController(Policy):
    """
    Combination of controllers.
    Taken from github.com/ashishgaurav13/wm2
    """

    def __init__(self, predicates, multipliers, controllers, debug = False):
        """
        Initialize ComplexController.
        """
        self.debug = debug
        self.predicates = predicates
        self.multipliers = multipliers
        assert(type(controllers) == list)
        assert(type(controllers[-1]) == DefaultController)
        names_of_controllers = []
        self.default_controller = controllers[-1]
        names_of_controllers += [self.default_controller.name]
        self.controllers = controllers[:-1]
        for controller in self.controllers:
            assert(controller.name not in names_of_controllers)
            names_of_controllers += [controller.name]

    def sequential_evaluate(self, x, ex = {}):
        """
        Sequentially evaluate the dict x.
        x has key, value pairs such that the key is the name
        of the attribute and value is either of:
        (i) a lambda function that takes in previously evaluated predicates,
        (ii) a list of 2 lambda functions, first being condition and second
        being the function to set key's value to, if condition is true.
        """
        ret = {}
        total = combine_dicts(ex, ret)
        for key, fn in x.items():
            assert(key not in total.keys())
            if type(fn) == list:
                assert(len(fn) == 2)
                if fn[0](total):
                    new_value = fn[1](total)
                    ret[key] = new_value
                    total[key] = new_value
            else:
                new_value = fn(total)
                ret[key] = new_value
                total[key] = new_value
        return ret

    def act(self, i = None):
        """
        Acting doesn't need an input.
        """
        ep = self.sequential_evaluate(self.predicates)
        em = self.sequential_evaluate(self.multipliers, ep)
        for controller in self.controllers:
            if controller.active(ep):
                if self.debug: print(controller.name)
                return controller.act([ep, em])
        if self.debug: print(self.default_controller.name)
        return self.default_controller.act([ep, em])