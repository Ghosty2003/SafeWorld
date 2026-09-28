"""Nonnegative state-only candidate families; none implies certificate validity."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

FAMILIES = ('linear', 'quadratic', 'rbf', 'tanh', 'residual', 'fourier')
EXTRA_FAMILIES = ('elu', 'gated')


class ValueFamily(nn.Module):
    """Shared learned accepting scalar; alternative waiting-state functions.

    All outputs are pointwise in (normalized state, automaton state). No time,
    trajectory identity, endpoint, or future observation is an input.
    """
    def __init__(self, family, input_dim, width=128, centers=None):
        super().__init__()
        if family not in FAMILIES + EXTRA_FAMILIES:
            raise ValueError(f'Unknown family {family}')
        self.family, self.input_dim, self.width = family, input_dim, width
        self.accepting_raw = nn.Parameter(torch.tensor(-4.))
        if family == 'linear':
            self.linear = nn.Linear(input_dim, 1)
        elif family == 'quadratic':
            self.linear = nn.Linear(input_dim, 1)
            self.projection = nn.Linear(input_dim, 16, bias=False)
        elif family == 'rbf':
            if centers is None: centers = torch.zeros(width, input_dim)
            if tuple(centers.shape) != (width, input_dim):
                raise ValueError('Expected width training-only centers')
            self.register_buffer('centers', centers.detach().clone())
            self.log_bandwidth = nn.Parameter(torch.tensor(math.log(.5)))
            self.linear = nn.Linear(input_dim, 1)
            self.kernel_weights = nn.Linear(width, 1, bias=False)
        elif family == 'tanh':
            self.net = nn.Sequential(nn.Linear(input_dim,width), nn.Tanh(),
                                     nn.Linear(width,width), nn.Tanh(), nn.Linear(width,1))
        elif family == 'residual':
            self.input = nn.Linear(input_dim,width)
            self.blocks = nn.ModuleList([
                nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,width))
                for _ in range(2)])
            self.output = nn.Linear(width,1)
        elif family == 'elu':
            self.net = nn.Sequential(nn.Linear(input_dim,width),nn.ELU(),
                                     nn.Linear(width,width),nn.ELU(),nn.Linear(width,1))
        elif family == 'gated':
            self.experts = nn.Linear(input_dim,4)
            self.gate = nn.Sequential(nn.Linear(input_dim,width),nn.Tanh(),nn.Linear(width,4))
        else:
            self.register_buffer('frequencies',torch.randn(input_dim,width)/math.sqrt(input_dim))
            self.register_buffer('phases',2*math.pi*torch.rand(width))
            self.linear = nn.Linear(input_dim,1)
            self.spectral_weights = nn.Linear(width,1,bias=False)

    def forward(self, x, done):
        if x.shape[:-1] != done.shape or x.shape[-1] != self.input_dim:
            raise ValueError('State and automaton labels must align')
        if self.family == 'linear':
            raw = self.linear(x)
        elif self.family == 'quadratic':
            raw = self.linear(x) + self.projection(x).square().mean(-1,keepdim=True)
        elif self.family == 'rbf':
            dist = (x.square().sum(-1,keepdim=True) + self.centers.square().sum(-1)
                    - 2*(x @ self.centers.T)).clamp_min(0)/self.input_dim
            bandwidth = self.log_bandwidth.exp().clamp(.05,10.)
            kernel = torch.exp(-dist/(2*bandwidth.square()))
            raw = self.linear(x) + self.kernel_weights(kernel)
        elif self.family == 'tanh':
            raw = self.net(x)
        elif self.family == 'residual':
            h = F.silu(self.input(x))
            for block in self.blocks: h = h + .5*block(h)
            raw = self.output(F.silu(h))
        elif self.family == 'elu':
            raw = self.net(x)
        elif self.family == 'gated':
            raw = (self.gate(x).softmax(-1)*self.experts(x)).sum(-1,keepdim=True)
        else:
            features = torch.cos(x @ self.frequencies + self.phases)
            raw = self.linear(x) + self.spectral_weights(features)
        waiting = F.softplus(raw.squeeze(-1))
        return torch.where(done.bool(), F.softplus(self.accepting_raw), waiting)


def predict_family(model, z, done):
    net = ValueFamily(model['family'],len(model['mean']),model['width'])
    net.load_state_dict(model['weights'])
    net.eval()
    x = torch.as_tensor((z-model['mean'])/model['scale'],dtype=torch.float32)
    q = torch.as_tensor(done,dtype=torch.float32)
    with torch.no_grad():
        return net(x,q).numpy().astype('float64')
