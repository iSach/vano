"""2D isotropic Gaussian densities (paper section 6.2).

Each function is the density of ``N(mu, sigma^2 I)`` on ``[0, 1]^2`` with
``mu ~ U([0, 1]^2)``.  The paper writes ``sigma ~ U(0, 0.1) + 0.01``; the
released code draws the *variance* from ``U(0, 0.01) + 0.001`` and passes it as
the covariance scale.  We follow the code -- the resulting standard deviations,
``[0.032, 0.105]``, are the ones that produce the published figures.

The family is a smooth two-parameter manifold that is badly approximated by any
low-dimensional linear subspace, which is exactly the point of the benchmark.
"""

import math

import torch

from .base import FunctionData, unit_grid

RESOLUTION = 48
VAR_RANGE = (0.0, 0.01)
VAR_OFFSET = 0.001


def load(num_samples=2048, resolution=RESOLUTION, seed=0,
         var_range=VAR_RANGE, var_offset=VAR_OFFSET):
    generator = torch.Generator().manual_seed(seed)
    y = unit_grid(resolution, dim=2)
    mu = torch.rand(num_samples, 1, 2, generator=generator)
    var = torch.rand(num_samples, 1, 1, generator=generator)
    var = var * (var_range[1] - var_range[0]) + var_range[0] + var_offset
    sq_dist = (y.unsqueeze(0) - mu).square().sum(-1, keepdim=True)
    s = torch.exp(-0.5 * sq_dist / var) / (2.0 * math.pi * var)
    w = 1.0 / s.squeeze(-1).norm(dim=1).square()
    u = s.reshape(num_samples, resolution, resolution, 1)
    return FunctionData(u=u, y=y, s=s, w=w,
                        grid_shape=(resolution, resolution))
