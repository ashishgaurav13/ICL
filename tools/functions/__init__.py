"""
Contains functions (input to output mappings), including neural networks.
"""

from .cost_function import CostFunction
from .normalized_accrual import NormalizedAccrual
from .flow import Flow

import torch

def create_flow(config, data, flow_type, i, h=64, n_blocks=5, 
    normalize_input=False):
    """
    Create a Flow.
    """
    assert(flow_type in ["realnvp", "made", "residual", "nsf-ar", "nsf-c"])
    if flow_type == "realnvp":
        mask = torch.arange(i) % 2
        components = []
        for block in range(n_blocks):
            components += [["an", i], ["1x1conv", i], ["realnvp", i, h, mask]]
            mask = 1-mask
    if flow_type in ["made", "residual", "nsf-ar", "nsf-c"]:
        components = []
        for block in range(n_blocks):
            components += [["an", i], ["1x1conv", i], [flow_type, i, h]]
    if normalize_input:
        components = [["bn", i]]+components
    return Flow(config, data, components)