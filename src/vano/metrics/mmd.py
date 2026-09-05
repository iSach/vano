"""Generalised maximum mean discrepancy between two sets of functions.

The paper reports "generalised MMD": the supremum, over a family of Gaussian
kernels, of the unbiased MMD between ground-truth and generated samples.  The
kernel is normalised by the number of degrees of freedom of a function so that
the value is comparable across resolutions::

    k_sigma(u, v) = exp(-||u - v||^2 / (2 sigma^2 D)),   D = number of points

and ``sigma^2`` sweeps ``logspace(-2, 2, 100)``, as in the released notebooks.
"""

import torch

SIGMA_SQ = torch.logspace(-2, 2, 100)


def mmd_curve(x, y, sigma_sq=None):
    """Unbiased MMD^2 for every bandwidth, given ``(N, ...)`` sample tensors."""
    x = x.reshape(x.shape[0], -1).double()
    y = y.reshape(y.shape[0], -1).double()
    dim = x.shape[1]
    sigma_sq = (SIGMA_SQ if sigma_sq is None else sigma_sq).to(x).reshape(-1, 1, 1)

    def kernel(a, b):
        return torch.exp(-torch.cdist(a, b).square() / (2.0 * sigma_sq * dim))

    n, m = x.shape[0], y.shape[0]
    # The released code sums the full Gram matrices, diagonal included, while
    # dividing by n(n-1); we reproduce that normalisation exactly.
    xx = kernel(x, x).sum((-2, -1)) / (n * (n - 1))
    yy = kernel(y, y).sum((-2, -1)) / (m * (m - 1))
    xy = kernel(x, y).sum((-2, -1)) / (n * m)
    return xx + yy - 2.0 * xy


def generalised_mmd(x, y, sigma_sq=None):
    """The scalar reported in the paper: the largest MMD over the family."""
    return mmd_curve(x, y, sigma_sq).max().item()
