"""1D Gaussian random field (paper section 6.1).

Samples are truncated Karhunen-Loeve expansions of a Gaussian measure on
``[0, 1]`` with covariance ``(-Delta + tau^2 I)^(-alpha)`` and periodic
eigenfunctions::

    u(x) = sum_{k=1..K} xi_k sqrt(lambda_k) phi_k(x),   xi_k ~ N(0, 1)
    lambda_k = ((2 pi k)^2 + tau^2)^(-alpha),   phi_k(x) = sqrt(2) sin(2 pi k x)

Because the eigenpairs are known in closed form, this benchmark lets us check
directly whether the linear decoder recovers the optimal (KL) basis.
"""

import numpy as np
import torch

from .base import FunctionData

ALPHA = 2.0
TAU = 0.1
NUM_EIGENPAIRS = 32
NUM_POINTS = 128


def legendre_nodes(num_points, bounds=(0.0, 1.0), dtype=torch.float32):
    """Gauss-Legendre nodes and weights rescaled to ``bounds``.

    The reference evaluates the functions at quadrature nodes rather than on a
    uniform grid, so the sampling grid is reproduced here even though the
    covariance metric later uses uniform weights.
    """
    lower, upper = bounds
    nodes, weights = np.polynomial.legendre.leggauss(num_points)
    nodes = 0.5 * (upper - lower) * (nodes + 1.0) + lower
    weights = weights * 0.5 * (upper - lower)
    return (torch.tensor(nodes, dtype=dtype).unsqueeze(-1),
            torch.tensor(weights, dtype=dtype))


def eigenpairs(x, num_eigenpairs=NUM_EIGENPAIRS, alpha=ALPHA, tau=TAU):
    """Exact KL eigenvalues ``(K,)`` and eigenfunctions ``(P, K)`` at ``x``."""
    k = torch.arange(1, num_eigenpairs + 1, dtype=x.dtype, device=x.device)
    evals = ((2.0 * torch.pi * k) ** 2 + tau**2) ** (-alpha)
    efuns = np.sqrt(2.0) * torch.sin(2.0 * torch.pi * k * x)
    return evals, efuns


def load(num_samples=2048, num_points=NUM_POINTS,
         num_eigenpairs=NUM_EIGENPAIRS, seed=0, alpha=ALPHA, tau=TAU):
    x, _ = legendre_nodes(num_points)
    evals, efuns = eigenpairs(x, num_eigenpairs, alpha, tau)
    generator = torch.Generator().manual_seed(seed)
    xi = torch.randn(num_samples, num_eigenpairs, generator=generator)
    u = (xi * evals.sqrt()) @ efuns.T                       # (N, P)
    w = 1.0 / u.norm(dim=1).square()
    return FunctionData(u=u, y=x, s=u.unsqueeze(-1), w=w,
                        grid_shape=(num_points,))
