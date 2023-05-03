from ctypes import cast
import torch
import random
import numpy as np
from numbers import Number
import math
from typing import List, Optional
import torch.nn.functional as F
import tensorflow as tf
import os

class TorchHelper:
    """
    Helper class for PyTorch functions.
    """

    def __init__(self):
        """
        Sets `device` to `cpu/cuda` automatically.
        """
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
    
    def seed(self, seed):
        """
        Fix all seeds.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        tf.compat.v1.disable_eager_execution()
        tf.random.set_seed(seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        tf.config.threading.set_inter_op_parallelism_threads(1)
        tf.config.threading.set_intra_op_parallelism_threads(1)

    def f(self, x):
        """
        Convert x to float tensor.
        """
        return torch.tensor(x, dtype=torch.float, device=self.device)
    
    def i(self, x):
        """
        Convert x to int tensor.
        """
        return torch.tensor(x, dtype=torch.int, device=self.device)
    
    def l(self, x):
        """
        Convert x to long tensor.
        """
        return torch.tensor(x, dtype=torch.long, device=self.device)
    
    def init_weights(self, m):
        """
        Initialize weights.
        """
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def nn(self, layers):
        """
        Create sequential neural network based on list specification.
        Eg. [[1, 10], 'r', [10, 10], 'r', [10, 1]] creates a 2 hidden layer
        neural network with ReLU non linearities.
        """
        nnl = []
        for l in layers:
            if type(l) == list:
                if l[0] == 'g':
                    nnl += [Gaussian2(*l[1:])]
                elif l[0] == 'ln':
                    nnl += [torch.nn.LayerNorm(l[1:])]
                else:
                    nnl += [torch.nn.Linear(*l)]
            if l == 'r':
                nnl += [torch.nn.ReLU()]
            if l == 's':
                nnl += [torch.nn.Sigmoid()]
            if l == 'sm':
                nnl += [torch.nn.Softmax(dim=-1)]
            if l == 'lsm':
                nnl += [torch.nn.LogSoftmax(dim=-1)]
            if l == 't':
                nnl += [torch.nn.Tanh()]
        network = torch.nn.Sequential(*nnl).to(self.device)
        network.apply(self.init_weights)
        return torch.jit.script(network)
    
    def fnn(self, layers):
        """
        Create flow sequential neural network.
        """
        fnnl = []
        for l in layers:
            if type(l) == list:
                if l[0] == 'bn':
                    fnnl += [FlowBN(l[1])]
                if l[0] == 'an':
                    fnnl += [FlowAN(l[1])]
                if l[0] == '1x1conv':
                    fnnl += [Flow1x1Conv(l[1])]
                if l[0] == 'made':
                    fnnl += [FlowMADE(*l[1:])]
                if l[0] == 'realnvp':
                    fnnl += [FlowRealNVP(*l[1:])]
                if l[0] == 'residual':
                    fnnl += [FlowResidual(*l[1:])]
                if l[0] == 'nsf-ar':
                    fnnl += [FlowNSFAutoRegressive(*l[1:])]
                if l[0] == 'nsf-c':
                    fnnl += [FlowNSFCoupling(*l[1:])]
        fnetwork = FlowSequential(*fnnl).to(self.device)
        fnetwork.apply(self.init_weights)
        return torch.jit.script(fnetwork)

    def adam(self, net, lr, **kwargs):
        """
        Shorthand for Adam optimizer.
        """
        return torch.optim.Adam(net.parameters(), lr=lr, **kwargs)
    
    def sgd(self, net, lr, **kwargs):
        """
        Shorthand for SGD optimizer.
        """
        return torch.optim.SGD(net.parameters(), lr=lr, **kwargs)
    
    def item(self, x):
        """
        Convert x to numpy equivalent.
        """
        return x.detach().cpu().numpy()

class AddBias(torch.jit.ScriptModule):
    """
    Bias adder.
    """

    def __init__(self, bias):
        """
        Initialize AddBias.
        """
        super(AddBias, self).__init__()
        self._bias = torch.nn.Parameter(bias.unsqueeze(1))

    @torch.jit.script_method
    def forward(self, x):
        """
        Forward pass with AddBias.
        """
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)
        return x + bias

FixedNormal = torch.distributions.Normal
log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(
    self, actions).sum(
        -1, keepdim=True)
normal_entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: normal_entropy(self).sum(-1)
FixedNormal.mode = lambda self: self.mean

def update_loc_scale(obj, loc, scale):
    obj.loc, obj.scale = torch.distributions.utils.broadcast_all(loc, scale)
    if isinstance(loc, Number) and isinstance(scale, Number):
        obj._batch_shape = torch.Size()
    else:
        obj._batch_shape = obj.loc.size()

FixedNormal.update = lambda self, loc, scale: update_loc_scale(self, loc, scale)

class Gaussian(torch.jit.ScriptModule):
    """
    Gaussian policy with variable bias. Outputs are in [-1, 1].
    """

    def __init__(self, num_inputs, num_outputs):
        """
        Initialize Gaussian policy with variable bias.
        """
        super(Gaussian, self).__init__()
        self.fc_mean = torch.nn.Linear(num_inputs, num_outputs)
        self.logstd = AddBias(torch.zeros(num_outputs))
        self.tanh = torch.nn.Tanh()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @torch.jit.script_method
    def forward(self, x):
        """
        Forward pass.
        """
        action_mean = self.tanh(self.fc_mean(x))

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size(), device=self.device)

        action_logstd = self.logstd(zeros)
        return action_mean, action_logstd.exp()

class Gaussian2(torch.jit.ScriptModule):
    def __init__(self, i, o, h=8):
        super().__init__()
        self.E = torch.nn.Sequential(
            torch.nn.Linear(i, h), torch.nn.ReLU(),
            torch.nn.Linear(h, h), torch.nn.ReLU(),            
        )
        self.Mu = torch.nn.Linear(h, o)
        self.LogStd = torch.nn.Linear(h, o)
        self.tanh = torch.nn.Tanh()
    @torch.jit.script_method
    def forward(self, x):
        x = self.E(x)
        return self.tanh(self.Mu(x)), torch.clamp(self.LogStd(x), -20, 0.5).exp()

class FlowMaskedLinear(torch.jit.ScriptModule):
    """
    Masked Linear layer.
    Taken from github.com/ikostrikov/pytorch-flows
    """
    
    def __init__(self, i, o, mask):
        super(FlowMaskedLinear, self).__init__()
        self.linear = torch.nn.Linear(i, o)
        self.register_buffer('mask', mask)

    @torch.jit.script_method
    def forward(self, x):
        output = torch.nn.functional.linear(x, self.linear.weight * self.mask,
            self.linear.bias)
        return output

@torch.jit.script
def get_mask(i:int, o:int, f:int, mask_type:str=""):
    """
    Mask for MADE.
    Taken from github.com/ikostrikov/pytorch-flows    
    """
    if mask_type == 'input':
        id = torch.arange(i) % f
    else:
        id = torch.arange(i) % (f - 1)

    if mask_type == 'output':
        od = torch.arange(o) % f - 1
    else:
        od = torch.arange(o) % (f - 1)

    return (od.unsqueeze(-1) >= id.unsqueeze(0)).float()

class FlowMADE(torch.jit.ScriptModule):
    """
    Masked Autoencoder for Distribution Estimation.
    arxiv.org/abs/1502.03509
    Taken from github.com/ikostrikov/pytorch-flows
    """

    def __init__(self, i, h, act='relu'):
        super(FlowMADE, self).__init__()
        activations = {
            'relu': torch.nn.ReLU, 
            'sigmoid': torch.nn.Sigmoid, 
            'tanh': torch.nn.Tanh
        }
        input_mask = get_mask(i, h, i, mask_type='input')
        hidden_mask = get_mask(h, h, i)
        output_mask = get_mask(h, i*2, i, mask_type='output')
        self.joiner = FlowMaskedLinear(i, h, input_mask)
        self.trunk = torch.nn.Sequential(
            activations[act](), FlowMaskedLinear(h, h, hidden_mask), 
            activations[act](), FlowMaskedLinear(h, i * 2, output_mask),
        )

    @torch.jit.script_method
    def forward(self, x, inverse:bool):
        if not inverse:
            h = self.joiner(x)
            m, a = self.trunk(h).chunk(2, 1)
            u = (x - m) * torch.exp(-a)
            return u, -a.sum(-1, keepdim=True)
        else:
            x = torch.zeros_like(x)
            a = torch.zeros_like(x)
            for i_col in range(x.shape[1]):
                h = self.joiner(x)
                m, a = self.trunk(h).chunk(2, 1)
                x[:, i_col] = x[:, i_col] * torch.exp(a[:, i_col]) + m[:, i_col]
            return x, -a.sum(-1, keepdim=True)

class FlowRealNVP(torch.jit.ScriptModule):
    """
    RealNVP Coupling Layer
    arxiv.org/abs/1605.08803
    Taken from github.com/ikostrikov/pytorch-flows
    """

    def __init__(self, i, h, mask, s_act='tanh', t_act='relu'):
        super(FlowRealNVP, self).__init__()

        self.i = i
        self.mask = mask

        activations = {
            'relu': torch.nn.ReLU, 
            'sigmoid': torch.nn.Sigmoid, 
            'tanh': torch.nn.Tanh
        }
        self.scale_net = torch.nn.Sequential(
            torch.nn.Linear(i, h), activations[s_act](),
            torch.nn.Linear(h, h), activations[s_act](),
            torch.nn.Linear(h, i)
        )
        self.translate_net = torch.nn.Sequential(
            torch.nn.Linear(i, h), activations[t_act](),
            torch.nn.Linear(h, h), activations[t_act](),
            torch.nn.Linear(h, i)
        )

    @torch.jit.script_method
    def forward(self, x, inverse:bool):
        if self.mask.device != x.device:
            self.mask = self.mask.to(x.device)
        masked_x = x * self.mask
        if not inverse:
            log_s = self.scale_net(masked_x) * (1 - self.mask)
            t = self.translate_net(masked_x) * (1 - self.mask)
            s = torch.exp(log_s)
            return x * s + t, log_s.sum(-1, keepdim=True)
        else:
            log_s = self.scale_net(masked_x) * (1 - self.mask)
            t = self.translate_net(masked_x) * (1 - self.mask)
            s = torch.exp(-log_s)
            return (x - t) * s, -log_s.sum(-1, keepdim=True)

class FlowBN(torch.jit.ScriptModule):
    """ 
    Batch normalization layer from RealNVP paper
    arxiv.org/abs/1605.08803
    Taken from github.com/ikostrikov/pytorch-flows
    """

    def __init__(self, num_inputs, momentum=0.0, eps=1e-5):
        super(FlowBN, self).__init__()
        self.log_gamma = torch.nn.Parameter(torch.zeros(num_inputs))
        self.beta = torch.nn.Parameter(torch.zeros(num_inputs))
        self.momentum = momentum
        self.eps = eps
        self.register_buffer('batch_mean', torch.zeros(num_inputs))
        self.register_buffer('batch_var', torch.ones(num_inputs))
        self.register_buffer('running_mean', torch.zeros(num_inputs))
        self.register_buffer('running_var', torch.ones(num_inputs))

    @torch.jit.script_method
    def forward(self, inputs, inverse:bool):
        if not inverse:
            if self.training:
                self.batch_mean = inputs.mean(0)
                self.batch_var = (
                    inputs - self.batch_mean).pow(2).mean(0) + self.eps

                self.running_mean.mul_(self.momentum)
                self.running_var.mul_(self.momentum)

                self.running_mean.add_(self.batch_mean.data *
                                       (1 - self.momentum))
                self.running_var.add_(self.batch_var.data *
                                      (1 - self.momentum))

                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - mean) / var.sqrt()
            y = torch.exp(self.log_gamma) * x_hat + self.beta
            return y, (self.log_gamma - 0.5 * torch.log(var)).sum(
                -1, keepdim=True)
        else:
            if self.training:
                mean = self.batch_mean
                var = self.batch_var
            else:
                mean = self.running_mean
                var = self.running_var

            x_hat = (inputs - self.beta) / torch.exp(self.log_gamma)

            y = x_hat * var.sqrt() + mean

            return y, (-self.log_gamma + 0.5 * torch.log(var)).sum(
                -1, keepdim=True)

class FlowSequential(torch.jit.ScriptModule):
    """
    torch.nn.Sequential equivalent for flows.
    Taken from github.com/ikostrikov/pytorch-flows
    """

    def __init__(self, *args):
        super().__init__()
        self.modulelist = torch.nn.ModuleList(args)

    @torch.jit.script_method
    def forward(self, x, inverse:bool):
        logdets = torch.zeros(x.size(0), 1, device=x.device)
        if not inverse:
            for module in self.modulelist:
                x, logdet = module.forward(x, inverse=inverse)
                logdets += logdet
        else:
            for module in self.modulelist[::-1]:
                x, logdet = module.forward(x, inverse=inverse)
                logdets += logdet
        return x, logdets
    
    @torch.jit.script_method
    def log_probs(self, x):
        u, log_jacob = self.forward(x, inverse=False)
        log_probs = (-0.5 * u.pow(2) - 0.5 * math.log(2 * math.pi)).sum(
            -1, keepdim=True)
        return (log_probs + log_jacob).sum(-1, keepdim=True)

class FlowAN(torch.jit.ScriptModule):
    """
    ActNorm layer (Kingma and Dhariwal, 2018)
    Taken from github.com/tonyduan/normalizing-flows
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.mu = torch.nn.Parameter(torch.zeros(dim, dtype = torch.float).to(self.device))
        self.log_sigma = torch.nn.Parameter(torch.zeros(dim, dtype = torch.float).to(self.device))

    @torch.jit.script_method
    def forward(self, x, inverse:bool):
        if not inverse:
            z = x * torch.exp(self.log_sigma) + self.mu
            log_det = torch.sum(self.log_sigma)
            return z, log_det
        else:
            z = x
            x = (z - self.mu) / torch.exp(self.log_sigma)
            log_det = -torch.sum(self.log_sigma)
            return x, log_det

class Flow1x1Conv(torch.jit.ScriptModule):
    """
    Invertible 1x1 convolution. (Kingma and Dhariwal, 2018)
    Taken from github.com/tonyduan/normalizing-flows
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        for _ in range(2):
            try:
                W, _ = torch.linalg.qr(torch.randn(dim, dim).to(self.device))
            except:
                pass
            else:
                break
        P, L, U = torch.lu_unpack(*W.lu())
        self.P = P.clone().detach().float()
        self.L = torch.nn.Parameter(L.clone().detach().float())
        self.S = torch.nn.Parameter(torch.diag(U).float())
        self.U = torch.nn.Parameter(torch.triu(U.clone().detach().float(),
                              diagonal = 1))
        self.W_inv = torch.zeros_like(W)

    @torch.jit.script_method
    def forward(self, x, inverse:bool):
        if not inverse:
            L = torch.tril(self.L, diagonal = -1) + torch.diag(torch.ones(self.dim).to(self.device))
            U = torch.triu(self.U, diagonal = 1)
            z = x @ self.P @ L @ (U + torch.diag(self.S))
            log_det = torch.sum(torch.log(torch.abs(self.S)))
            return z, log_det
        else:
            z = x
            L = torch.tril(self.L, diagonal = -1) + \
                torch.diag(torch.ones(self.dim).to(self.device))
            U = torch.triu(self.U, diagonal = 1)
            W = self.P @ L @ (U + torch.diag(self.S))
            self.W_inv = torch.inverse(W)
            x = z @ self.W_inv
            log_det = -torch.sum(torch.log(torch.abs(self.S)))
            return x, log_det

@torch.jit.script
def searchsorted(bin_locations, inputs, eps:float=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(
        inputs[..., None] >= bin_locations,
        dim=-1
    ) - 1

@torch.jit.script
def RQS(inputs, unnormalized_widths, unnormalized_heights,
        unnormalized_derivatives, inverse:bool=False, left:float=0., right:float=1.,
        bottom:float=0., top:float=1., min_bin_width:float=1e-3,
        min_bin_height:float=1e-3,
        min_derivative:float=1e-3):
    assert(not (torch.min(inputs) < left or torch.max(inputs) > right))
    num_bins = unnormalized_widths.shape[-1]

    assert(min_bin_width * num_bins <= 1.0)
    assert(min_bin_height * num_bins <= 1.0)

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode='constant', value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode='constant', value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)
    input_derivatives_plus_one = input_derivatives_plus_one[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (((inputs - input_cumheights) * (input_derivatives \
            + input_derivatives_plus_one - 2 * input_delta) \
            + input_heights * (input_delta - input_derivatives)))
        b = (input_heights * input_derivatives - (inputs - input_cumheights) \
            * (input_derivatives + input_derivatives_plus_one \
            - 2 * input_delta))
        c = - input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all()

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta \
                      + ((input_derivatives + input_derivatives_plus_one \
                      - 2 * input_delta) * theta_one_minus_theta)
        derivative_numerator = input_delta.pow(2) \
                               * (input_derivatives_plus_one * root.pow(2) \
                                + 2 * input_delta * theta_one_minus_theta \
                                + input_derivatives * (1 - root).pow(2))
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (input_delta * theta.pow(2) \
                    + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + ((input_derivatives \
                      + input_derivatives_plus_one - 2 * input_delta) \
                      * theta_one_minus_theta)
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) \
                               * (input_derivatives_plus_one * theta.pow(2) \
                                + 2 * input_delta * theta_one_minus_theta \
                                + input_derivatives * (1 - theta).pow(2))
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
        return outputs, logabsdet

@torch.jit.script
def unconstrained_RQS(inputs, unnormalized_widths, unnormalized_heights,
                      unnormalized_derivatives, inverse:bool=False,
                      tail_bound:float=1., min_bin_width:float=1e-3,
                      min_bin_height:float=1e-3,
                      min_derivative:float=1e-3):
    inside_intvl_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_intvl_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
    constant = math.log(math.exp(1 - min_derivative) - 1)
    unnormalized_derivatives[..., 0] = constant
    unnormalized_derivatives[..., -1] = constant

    outputs[outside_interval_mask] = inputs[outside_interval_mask]
    logabsdet[outside_interval_mask] = 0

    outputs[inside_intvl_mask], logabsdet[inside_intvl_mask] = RQS(
        inputs=inputs[inside_intvl_mask],
        unnormalized_widths=unnormalized_widths[inside_intvl_mask, :],
        unnormalized_heights=unnormalized_heights[inside_intvl_mask, :],
        unnormalized_derivatives=unnormalized_derivatives[inside_intvl_mask, :],
        inverse=inverse,
        left=-tail_bound*1., right=tail_bound*1., bottom=-tail_bound*1., top=tail_bound*1.,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative
    )
    return outputs, logabsdet

class FlowNSFAutoRegressive(torch.jit.ScriptModule):
    """
    Neural spline flow, auto-regressive. (Durkan et al. 2019)
    Taken from github.com/tonyduan/normalizing-flows
    """
    def __init__(self, dim, hidden_dim = 8, K = 5, B = 3):
        super().__init__()
        self.dim = dim
        self.K = K
        self.B = B
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.layers = torch.nn.ModuleList()
        self.init_param = torch.nn.Parameter(torch.Tensor(3 * K - 1))
        for i in range(1, dim):
            self.layers += [torch.nn.Sequential(
                torch.nn.Linear(i, hidden_dim), torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, 3 * K - 1)
            )]

    @torch.jit.script_method
    def forward(self, x, inverse:bool):
        if not inverse:
            z = torch.zeros_like(x)
            log_det = torch.zeros(z.shape[0]).to(self.device)
            for i in range(self.dim):
                if i == 0:
                    init_param = self.init_param.expand(x.shape[0], 3 * self.K - 1)
                    W, H, D = torch.split(init_param, self.K, dim = 1)
                else:
                    layer:torch.nn.Sequential = self.layers[i-1]
                    out = layer.forward(x[:, :i])
                    W, H, D = torch.split(out, self.K, dim = 1)
                W, H = torch.softmax(W, dim = 1), torch.softmax(H, dim = 1)
                W, H = 2 * self.B * W, 2 * self.B * H
                D = F.softplus(D)
                z[:, i], ld = unconstrained_RQS(
                    x[:, i], W*1., H*1., D*1., inverse=False, tail_bound=self.B*1.)
                log_det += ld
            return z, log_det.view(-1, 1)
        else:
            z = x
            x = torch.zeros_like(z)
            log_det = torch.zeros(x.shape[0]).to(self.device)
            for i in range(self.dim):
                if i == 0:
                    init_param = self.init_param.expand(x.shape[0], 3 * self.K - 1)
                    W, H, D = torch.split(init_param, self.K, dim = 1)
                else:
                    layer:torch.nn.Sequential = self.layers[i-1]
                    out = layer.forward(x[:, :i])
                    W, H, D = torch.split(out, self.K, dim = 1)
                W, H = torch.softmax(W, dim = 1), torch.softmax(H, dim = 1)
                W, H = 2 * self.B * W, 2 * self.B * H
                D = F.softplus(D)
                x[:, i], ld = unconstrained_RQS(
                    z[:, i], W*1., H*1., D*1., inverse = True, tail_bound = self.B*1.)
                log_det += ld
            return x, log_det.view(-1, 1)

class FlowNSFCoupling(torch.jit.ScriptModule):
    """
    Neural spline flow, coupling layer. (Durkan et al. 2019)
    Taken from github.com/tonyduan/normalizing-flows
    """
    def __init__(self, dim, hidden_dim = 8, K = 5, B = 3):
        super().__init__()
        self.dim = dim
        self.K = K
        self.B = B
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.f1 = torch.nn.Sequential(
            torch.nn.Linear(dim // 2, hidden_dim), torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, (3 * K - 1) * dim // 2)
        )
        self.f2 = torch.nn.Sequential(
            torch.nn.Linear(dim // 2, hidden_dim), torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, (3 * K - 1) * dim // 2)
        )

    @torch.jit.script_method
    def forward(self, x, inverse:bool):
        if not inverse:
            log_det = torch.zeros(x.shape[0]).to(self.device)
            lower, upper = x[:, :self.dim // 2], x[:, self.dim // 2:]
            out = self.f1(lower).reshape(-1, self.dim // 2, 3 * self.K - 1)
            W, H, D = torch.split(out, self.K, dim = 2)
            W, H = torch.softmax(W, dim = 2), torch.softmax(H, dim = 2)
            W, H = 2 * self.B * W, 2 * self.B * H
            D = F.softplus(D)
            upper, ld = unconstrained_RQS(
                upper, W*1., H*1., D*1., inverse=False, tail_bound=self.B*1.)
            log_det += torch.sum(ld, dim = 1)
            out = self.f2(upper).reshape(-1, self.dim // 2, 3 * self.K - 1)
            W, H, D = torch.split(out, self.K, dim = 2)
            W, H = torch.softmax(W, dim = 2), torch.softmax(H, dim = 2)
            W, H = 2 * self.B * W, 2 * self.B * H
            D = F.softplus(D)
            lower, ld = unconstrained_RQS(
                lower, W*1., H*1., D*1., inverse=False, tail_bound=self.B*1.)
            log_det += torch.sum(ld, dim = 1)
            return torch.cat([lower, upper], dim = 1), log_det.view(-1, 1)
        else:
            z = x
            log_det = torch.zeros(z.shape[0]).to(self.device)
            lower, upper = z[:, :self.dim // 2], z[:, self.dim // 2:]
            out = self.f2(upper).reshape(-1, self.dim // 2, 3 * self.K - 1)
            W, H, D = torch.split(out, self.K, dim = 2)
            W, H = torch.softmax(W, dim = 2), torch.softmax(H, dim = 2)
            W, H = 2 * self.B * W, 2 * self.B * H
            D = F.softplus(D)
            lower, ld = unconstrained_RQS(
                lower, W*1., H*1., D*1., inverse=True, tail_bound=self.B*1.)
            log_det += torch.sum(ld, dim = 1)
            out = self.f1(lower).reshape(-1, self.dim // 2, 3 * self.K - 1)
            W, H, D = torch.split(out, self.K, dim = 2)
            W, H = torch.softmax(W, dim = 2), torch.softmax(H, dim = 2)
            W, H = 2 * self.B * W, 2 * self.B * H
            D = F.softplus(D)
            upper, ld = unconstrained_RQS(
                upper, W*1., H*1., D*1., inverse = True, tail_bound = self.B*1.)
            log_det += torch.sum(ld, dim = 1)
            return torch.cat([lower, upper], dim = 1), log_det.view(-1, 1)

class FlowResidual(torch.jit.ScriptModule):
    """
    Taken from github.com/VincentStimper/normalizing-flows/
    Which is based on github.com/rtqichen/residual-flows
    """
    def __init__(self, i, h, n_exact_terms=2, n_samples=1):
        super().__init__()
        self.iresblock = iResBlock(i, h, n_samples=n_samples,
                                   n_exact_terms=n_exact_terms)

    def forward(self, x, inverse:bool):
        if inverse:
            x, log_det = self.iresblock.inverse(x, 0.)
        else:
            x, log_det = self.iresblock.forward(x, 0.)
        return x, -log_det.view(-1, 1)


class iResBlock(torch.jit.ScriptModule):
    """
    Taken from github.com/VincentStimper/residual-flows
    """

    def __init__(
        self,
        i, h,
        geom_p=0.5,
        lamb=2.,
        n_samples=1,
        n_exact_terms=2,
    ):
        super().__init__()
        self.nnet = torch.nn.Sequential(
            torch.nn.Linear(i, h), torch.nn.ReLU(),
            torch.nn.Linear(h, h), torch.nn.ReLU(),
            torch.nn.Linear(h, i)
        )
        self.geom_p = torch.nn.Parameter(
            torch.tensor(np.log(geom_p) - np.log(1. - geom_p)))
        self.lamb = torch.nn.Parameter(torch.tensor(lamb))
        self.n_samples = n_samples
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.samples = torch.zeros(n_samples).to(self.device)
        self.n_exact_terms = n_exact_terms

    def forward(self, x, logpx:float):
        g, logdetgrad = self._logdetgrad(x)
        return x + g, logpx - logdetgrad

    def inverse(self, y, logpy:float):
        x = self._inverse_fixed_point(y)
        return x, logpy + self._logdetgrad(x)[1]

    def _inverse_fixed_point(self, y, atol:float=1e-5, rtol:float=1e-5):
        x, x_prev = y - self.nnet(y), y
        i = 0
        tol = atol + y.abs() * rtol
        while not torch.all((x - x_prev)**2 / tol < 1):
            x, x_prev = y - self.nnet(x), x
            i += 1
            if i > 1000:
                print('Iterations exceeded 1000 for inverse.')
                break
        return x

    def _logdetgrad(self, x):
        """Returns g(x) and logdet|d(x+g(x))/dx|."""

        geom_p = torch.sigmoid(self.geom_p)
        lamb = self.lamb.item()
        self.samples = geometric_sample(geom_p.detach(), self.n_samples).to(self.device)
        n_power_series = torch.max(self.samples).int() + self.n_exact_terms
        vareps = torch.randn_like(x)
        x = x.requires_grad_(True)
        g = self.nnet(x)
        logdetgrad = basic_logdet_estimator(g, x, n_power_series, vareps, self.training, self)

        return g, logdetgrad.view(-1, 1)

def basic_logdet_estimator(g, x, n_power_series:Optional[int], vareps, training:bool, obj:iResBlock):
    vjp:Optional[torch.Tensor] = vareps
    logdetgrad = torch.tensor(0.).to(x)
    assert(n_power_series is not None)
    for k in range(1, n_power_series + 1):
        vjp_pass:List[Optional[torch.Tensor]] = [vjp]
        vjp = torch.autograd.grad([g], [x], vjp_pass, create_graph=training, retain_graph=True)[0]
        assert(vjp is not None)
        tr = torch.sum(vjp.view(x.shape[0], -1) * vareps.view(x.shape[0], -1), 1)
        ret = 1 / geometric_1mcdf(obj.geom_p.detach(), k, obj.n_exact_terms) * \
            sum(obj.samples >= k - obj.n_exact_terms) / len(obj.samples)
        delta = (-1)**(k + 1) / k * ret * tr
        logdetgrad = logdetgrad + delta
    return logdetgrad

def geometric_sample(p:float, n_samples:int):
    return torch.zeros(n_samples).geometric_(p)

def geometric_1mcdf(p:float, k:int, offset:int):
    if k <= offset:
        return 1.
    else:
        k = k - offset
    """P(n >= k)"""
    return float((1 - p)**max(k - 1, 0))
