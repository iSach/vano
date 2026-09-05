"""Dense/conv primitives with the initialisation used by the reference VANO code.

The reference implementation is written in Flax, whose defaults differ from
PyTorch's.  Reproducing the paper's numbers requires reproducing those defaults,
so the initialisers below mirror ``flax.linen.initializers`` exactly:

* ``nn.Dense``/``nn.Conv`` draw kernels from a *truncated* normal
  (``lecun_normal``), not PyTorch's Kaiming-uniform.
* Layers with random weight factorisation (RWF, Wang et al. 2022) factor the
  kernel as ``W = g * V`` with a per-output-unit scale ``g``; both factors are
  trained.  The base kernel is ``glorot_normal`` for dense layers and
  ``lecun_normal`` for convolutions, matching ``layers.py`` in the release.

RWF is enabled per benchmark exactly as in the release: on for the 2D Gaussian
densities, Cahn-Hilliard and InSAR models, off for the 1D GRF model and for the
discretise-first VAE baseline (their ``archs.py`` shadows the factorised layers
with plain Flax ones).
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

# Flax rescales the truncated-normal standard deviation by the standard
# deviation of a standard normal truncated to [-2, 2].
_TRUNC_STDDEV = 0.8796256610342398


def variance_scaling_(tensor, fan_in, fan_out, scale=1.0, mode="fan_in"):
    """In-place ``jax.nn.initializers.variance_scaling`` (truncated normal)."""
    denominator = {
        "fan_in": fan_in,
        "fan_out": fan_out,
        "fan_avg": 0.5 * (fan_in + fan_out),
    }[mode]
    std = math.sqrt(scale / denominator) / _TRUNC_STDDEV
    return nn.init.trunc_normal_(tensor, 0.0, std, -2.0 * std, 2.0 * std)


def lecun_normal_(tensor, fan_in, fan_out):
    return variance_scaling_(tensor, fan_in, fan_out, 1.0, "fan_in")


def glorot_normal_(tensor, fan_in, fan_out):
    return variance_scaling_(tensor, fan_in, fan_out, 1.0, "fan_avg")


class _Factorised(nn.Module):
    """Mixin holding either a plain kernel or its ``g * v`` factorisation."""

    def _init_kernel(self, kernel, weight_fact, scale_shape, mean=1.0, stddev=0.01):
        if weight_fact:
            g = torch.exp(mean + stddev * torch.randn(scale_shape[0]))
            self.g = nn.Parameter(g.view(scale_shape))
            self.v = nn.Parameter(kernel / self.g.detach())
        else:
            self.weight = nn.Parameter(kernel)

    @property
    def kernel(self):
        return self.g * self.v if hasattr(self, "g") else self.weight


class Dense(_Factorised):
    """``flax.linen.Dense`` with optional random weight factorisation."""

    def __init__(self, in_features, out_features, weight_fact=False, bias=True):
        super().__init__()
        kernel = torch.empty(out_features, in_features)
        # glorot_normal is the base initialiser of the factorised dense layer;
        # the unfactorised path is plain flax nn.Dense, i.e. lecun_normal.
        init = glorot_normal_ if weight_fact else lecun_normal_
        init(kernel, in_features, out_features)
        self._init_kernel(kernel, weight_fact, (out_features, 1))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        return F.linear(x, self.kernel, self.bias)


class Conv2d(_Factorised):
    """2x2 stride-2 ``flax.linen.Conv`` with ``padding='SAME'`` semantics."""

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2,
                 weight_fact=False):
        super().__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        kernel = torch.empty(out_channels, in_channels, kernel_size, kernel_size)
        receptive = kernel_size * kernel_size
        lecun_normal_(kernel, receptive * in_channels, receptive * out_channels)
        self._init_kernel(kernel, weight_fact, (out_channels, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def _same_padding(self, size):
        out = -(-size // self.stride)  # ceil
        total = max((out - 1) * self.stride + self.kernel_size - size, 0)
        return total // 2, total - total // 2  # lax pads the high side first

    def forward(self, x):
        pad_h = self._same_padding(x.shape[-2])
        pad_w = self._same_padding(x.shape[-1])
        if any(pad_h + pad_w):
            x = F.pad(x, (*pad_w, *pad_h))
        return F.conv2d(x, self.kernel, self.bias, stride=self.stride)


class ConvTranspose2d(_Factorised):
    """2x2 stride-2 ``flax.linen.ConvTranspose`` (doubles the spatial size)."""

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2,
                 weight_fact=False):
        super().__init__()
        self.stride = stride
        # torch expects (in, out, kh, kw); fans follow the flax kernel layout.
        kernel = torch.empty(in_channels, out_channels, kernel_size, kernel_size)
        receptive = kernel_size * kernel_size
        lecun_normal_(kernel, receptive * in_channels, receptive * out_channels)
        self._init_kernel(kernel, weight_fact, (out_channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        return F.conv_transpose2d(x, self.kernel, self.bias, stride=self.stride)


class MLP(nn.Module):
    """``num_layers`` hidden layers of ``hidden_dim`` units, then a linear head."""

    def __init__(self, in_dim, num_layers, hidden_dim, out_dim,
                 activation=nn.GELU, weight_fact=False):
        super().__init__()
        layers, width = [], in_dim
        for _ in range(num_layers):
            layers += [Dense(width, hidden_dim, weight_fact), activation()]
            width = hidden_dim
        layers.append(Dense(width, out_dim, weight_fact))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
