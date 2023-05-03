import abc
from tools.graphics import Plot2D
import numpy as np
from functools import partial
import torch

class Function(abc.ABC):
    """
    Base Function class. A function is anything that can accept an input
    and produce an output.
    """

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        """
        Call function.
        """
        pass

    def outputs(self, input_spaces, evaluate=False, no_plot=False, **kwargs):
        """
        Input spaces are a list of nested input spaces, the lowest level
        of which is a non-list. The non-lists are passed to the function, and
        outputs are constructed.
        """
        if not evaluate:
            ret = []
            cmds = []
            legend = False
            for i, input_space_spec in enumerate(input_spaces):
                input_space = input_space_spec[0]
                ret += [np.array(self.outputs(input_space, evaluate=True, **kwargs))]
                if len(input_space_spec) == 2:
                    plot_dict = input_space_spec[1]
                    if "kwargs" in plot_dict.keys():
                        plot_kwargs = plot_dict["kwargs"]
                    else:
                        plot_kwargs = {}
                    if plot_dict['type'] == 'imshow':
                        assert(len(ret[i].shape) <= 3)
                        if len(ret[i].shape) == 3:
                            img_data = np.mean(ret[i], axis=-1)
                        else:
                            img_data = ret[i]
                        if "process" in plot_dict.keys():
                            img_data = plot_dict["process"](img_data)
                        cmds += [partial(
                            lambda p,l,o,t,d,k: p.imshow(d, cmap='gray', **k),
                            d=img_data, k=plot_kwargs)]
                    if plot_dict['type'] == 'line':
                        assert('x' in plot_dict.keys())
                        assert(len(ret[i].shape) == 1)
                        if "process" in plot_dict.keys():
                            line_data = plot_dict["process"](ret[i])
                        else:
                            line_data = ret[i]
                        if 'label' in plot_dict.keys():
                            cmds += [partial(lambda p,l,o,t,x,d,lbl,k: p.line(x, 
                                d, label=lbl, **k), 
                                x=plot_dict['x'], d=line_data, lbl=plot_dict['label'],
                                k=plot_kwargs)]
                            legend = True
                        else:
                            cmds += [partial(lambda p,l,o,t,x,d,k: p.line(x,
                                d, **k), x=plot_dict['x'], d=line_data, k=plot_kwargs)]
            if cmds != [] and not no_plot:
                plot = Plot2D(l={}, cmds=cmds, legend=legend)
                return ret, plot
            return ret, None
        else:
            if type(input_spaces) != list:
                ret = self(input_spaces, **kwargs)
                if type(ret) == torch.Tensor:
                    ret = ret.detach().cpu().squeeze().numpy()
                return ret
            else:
                ret = []
                for item in input_spaces:
                    ret += [self.outputs(item, evaluate=True, **kwargs)]
                return ret