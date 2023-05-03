from tools.base import Function, TrajectoryDataset
from tools.graphics import Plot2D
import numpy as np
import tqdm
from functools import partial

class NormalizedAccrual(Function):
    """
    Takes in an expert dataset of trajectories, each represented by states and
    actions, and produces a normalized accrual over an input space.
    """

    def __call__(self, config, give_plot=True, silent=False):
        """
        Call this function.
        """
        assert("dataset" in config.keys())
        assert("spaces" in config.keys())
        dataset = config["dataset"]
        spaces = config["spaces"]
        self.config = config
        self.f = "sum"
        if "normalize_func" in config.keys():
            self.f = config["normalize_func"]
        self.fast = False
        assert(type(dataset) == TrajectoryDataset)
        ret_spaces = []
        accrual = []
        for space in spaces:
            assert(len(space) <= 2)
            ret_spaces += [space[0]]
            if len(space) == 2 and "left_state_gap" in space[1].keys():
                if hasattr(self, "left_state_gap"):
                    assert(self.left_state_gap == space[1]["left_state_gap"])
                else:
                    self.left_state_gap = space[1]["left_state_gap"]
            else:
                if hasattr(self, "left_state_gap"):
                    assert(self.left_state_gap == 0.5-1e-6)
                else:
                    self.left_state_gap = 0.5-1e-6
            if len(space) == 2 and "right_state_gap" in space[1].keys():
                if hasattr(self, "right_state_gap"):
                    assert(self.right_state_gap == space[1]["right_state_gap"])
                else:
                    self.right_state_gap = space[1]["right_state_gap"]
            else:
                if hasattr(self, "right_state_gap"):
                    assert(self.right_state_gap == 0.5)
                else:
                    self.right_state_gap = 0.5
            if len(space) == 2 and "left_action_gap" in space[1].keys():
                if hasattr(self, "left_action_gap"):
                    assert(self.left_action_gap == space[1]["left_action_gap"])
                else:
                    self.left_action_gap = space[1]["left_action_gap"]
            else:
                if hasattr(self, "left_action_gap"):
                    assert(self.left_action_gap == 0)
                else:
                    self.left_action_gap = 0
            if len(space) == 2 and "right_action_gap" in space[1].keys():
                if hasattr(self, "right_action_gap"):
                    assert(self.right_action_gap == space[1]["right_action_gap"])
                else:
                    self.right_action_gap = space[1]["right_action_gap"]
            else:
                if hasattr(self, "right_action_gap"):
                    assert(self.right_action_gap == 0)
                else:
                    self.right_action_gap = 0
            if len(space) == 2 and "fast" in space[1].keys():
                self.fast = True
                self.fast_min = space[1]["fast"]
                self.scale = space[1]["scale"] if "scale" in space[1].keys() else [1, 1]
            accrual += [self.get_zeros(space[0])]
        if self.fast:
            print("Fast mode on.")
        if silent:
            iterator = dataset.d
        else:
            iterator = tqdm.tqdm(dataset.d)
        for (S, A) in iterator:
            for (s, a) in zip(S, A):
                accrual = self.increment_accrual(ret_spaces, accrual, (s, a))
        accrual = self.normalize_accrual(accrual, self.f)
        cmds = []
        legend = False
        for i, space in enumerate(spaces):
            if len(space) == 2:
                plot_dict = space[1]
                if "kwargs" in plot_dict.keys():
                    kwargs = plot_dict["kwargs"]
                else:
                    kwargs = {}
                if plot_dict['type'] == 'imshow':
                    assert(len(accrual[i].shape) <= 3)
                    if len(accrual[i].shape) == 3:
                        img_data = np.sum(accrual[i], axis=-1)
                    else:
                        img_data = accrual[i]
                    if "process" in plot_dict.keys():
                        img_data = plot_dict["process"](img_data)
                    cmds += [partial(
                        lambda p,l,o,t,d,k: p.imshow(d, cmap='gray', **k),
                        d=img_data, k=kwargs)]
                if plot_dict['type'] == 'line':
                    assert('x' in plot_dict.keys())
                    assert(len(accrual[i].shape) == 1)
                    if "process" in plot_dict.keys():
                        accrual[i] = plot_dict["process"](accrual[i])
                    if 'label' in plot_dict.keys():
                        cmds += [partial(lambda p,l,o,t,x,d,lbl,k: p.line(x, 
                            d, label=lbl, **k), 
                            x=plot_dict['x'], d=accrual[i], lbl=plot_dict['label'],
                            k=kwargs)]
                        legend = True
                    else:
                        cmds += [partial(lambda p,l,o,t,x,d,k: p.line(x,
                            d, **k), x=plot_dict['x'], d=accrual[i], k=kwargs)]
        plot = None
        if cmds != [] and give_plot:
                plot = Plot2D(l={}, cmds=cmds, legend=legend)
        else:
            return accrual
        if "flatten" in config.keys() and config["flatten"]:
            return np.array(accrual).flatten(), plot
        return accrual, plot
    
    def get_zeros(self, space):
        """
        Get np.zeros for a given input space.
        """
        if type(space) == list:
            ret_space = []
            for item in space:
                ret_space += [self.get_zeros(item)]
            return np.array(ret_space)
        else:
            return 0
    
    def increment_accrual(self, spaces, accrual, sa):
        """
        Increment the corresponding accrual.
        """
        if self.fast:
            assert(len(accrual) == 1)
            space = spaces[0]
            if len(accrual[0].shape) == 3:
                accrual[0] = np.zeros((accrual[0].shape[0], accrual[0].shape[1]))
            assert(len(accrual[0].shape) <= 2)
            s, _ = sa
            s = self.config["state_reduction"](s)
            s = np.array(s) - self.fast_min
            if len(s) == 1:
                accrual[0][int(s[0]*self.scale[0])] += 1
            else:
                accrual[0][int(s[0]*self.scale[0])][int(s[1]*self.scale[1])] += 1
            return accrual
        if hasattr(accrual, "__len__"):
            ret = []
            for space, aitem in zip(spaces, accrual):
                ret += [self.increment_accrual(space, aitem, sa)]
            return ret
        else:
            s, a = sa
            cmp_s, cmp_a = spaces
            if self.compare(s, cmp_s, self.left_state_gap, self.right_state_gap) and \
               self.compare(a, cmp_a, self.left_action_gap, self.right_action_gap):
               return accrual+1
            return accrual

    def compare(self, v, cv, l, r):
        """
        Compare a value to another value. Values can be list.
        """
        if hasattr(v, "__len__") and hasattr(cv, "__len__"):
            s = min(len(v), len(cv))
            for item, item2 in zip(v[:s], cv[:s]):
                if not self.compare(item, item2, l, r):
                    return False
            return True
        else:
            return cv-l <= v <= cv+r

    def normalize_accrual(self, accrual, f="sum"):
        """
        Normalize the accrual.
        """
        assert(f in ["sum", "max", "bin"])
        if f=="sum":
            s = 0
            for item in accrual:
                s += np.sum(item)
        else:
            s = -np.inf
            for item in accrual:
                s = max(s, np.max(item))
            assert(s != -np.inf)
        ret = []
        for item in accrual:
            if f=="bin":
                ret += [(item > 0).astype(np.float)]
            else:
                ret += [item/(s+1e-6)]
        return ret