import pprint
import matplotlib
import numpy as np
from tools.utils.misc import combine_dicts
import time
import os
from torch.utils.tensorboard import SummaryWriter
import wandb
import sys

class Logger:
    """
    Progress bar based logger.
    """

    def __init__(self, n=None, window=None, logdir=None, project=None,
        config=None, silent=False):
        """
        Progress is just printed to stdout.
        If window is provided, we compute and show window metrics too.
        """
        if logdir != None:
            logdir = os.path.join("runs", logdir)
        self.pbar = None
        self.n = n
        self.pp = pprint.PrettyPrinter(indent=4, stream=sys.stdout)
        self.window = window
        self.all_metrics = []
        self.t = 0
        self.local_t = {}
        self.silent = silent
        self.start_time = time.time()
        if logdir != None:
            self.writer = SummaryWriter(logdir)
        if logdir != None and project != None:
            wandb.init(project=project, name=logdir, config=config)
            self.wandb = True
    
    def recompute_metrics(self):
        """
        Compute window metrics
        """
        if self.window != None:
            new_dict = {}
            for key in self.all_metrics[-1].keys():
                if "window" in key: continue
                try:
                    values = []
                    for metrics in self.all_metrics[-self.window:]:
                        values += [metrics[key]]
                    new_dict["%s_window" % key] = np.mean(values)
                except Exception as e:
                    # print('error')
                    # print(e)
                    pass
            self.all_metrics[-1] = combine_dicts(self.all_metrics[-1], new_dict)

    def write_scalars(self, to_log):
        n = len(to_log)
        i = 0
        for key in to_log.keys():
            try:
                if key not in self.local_t.keys():
                    self.local_t[key] = 0
                if hasattr(self, 'writer'):
                    if type(to_log[key]) == matplotlib.figure.Figure:
                        self.writer.add_figure(key, to_log[key], self.local_t[key])
                    else:
                        self.writer.add_scalar(key, to_log[key], self.local_t[key])
                if hasattr(self, 'wandb'):
                    wandb.log({key: to_log[key], 't': self.local_t[key]}, commit=(i==n-1))
                self.local_t[key] += 1
            except Exception as e:
                # print(e)
                pass
            i += 1
        if hasattr(self, 'writer'):
            self.writer.flush()

    def update(self, metrics, early_exit_metrics = None):
        """
        Given a `metrics` dictionary with values, print it accordingly.
        """
        self.all_metrics += [metrics]
        self.recompute_metrics()
        to_log = combine_dicts({
            't': self.t, 'elapsed': time.time()-self.start_time,
        }, self.all_metrics[-1])
        if not self.silent:
            self.pp.pprint(to_log)
            sys.stdout.flush()
        self.write_scalars(self.all_metrics[-1])
        if not self.silent:
            print("\n")
        self.t += 1
        if early_exit_metrics is not None and len(early_exit_metrics) > 0:
            satisfied = True
            for metric, value in early_exit_metrics.items():
                if to_log[metric] < value:
                    satisfied = False
            return satisfied, to_log
        else:
            return False, to_log
