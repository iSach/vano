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


def mmd_curve(x, y, sigma_sq=None, unbiased=False):
    """MMD^2 for every bandwidth, given ``(N, ...)`` sample tensors.

    Defaults to the released estimator so the values match the paper's; see the
    note below and ``docs/REPLICATION.md`` for why that is not the estimator you
    would choose.
    """
    x = x.reshape(x.shape[0], -1).double()
    y = y.reshape(y.shape[0], -1).double()
    dim = x.shape[1]
    sigma_sq = (SIGMA_SQ if sigma_sq is None else sigma_sq).to(x).reshape(-1, 1, 1)

    def kernel(a, b):
        return torch.exp(-torch.cdist(a, b).square() / (2.0 * sigma_sq * dim))

    n, m = x.shape[0], y.shape[0]
    # The released code sums the full Gram matrices, diagonal included, while
    # dividing by n(n-1) -- neither the biased nor the unbiased estimator.  Since
    # k(x, x) = 1 this adds exactly 2/(n-1), which is over half of the values the
    # paper reports and which changes with the sample count.  We reproduce it so
    # our numbers are comparable to the published ones; pass
    # ``unbiased=True`` for the estimator you would actually want.
    # See docs/REPLICATION.md.
    kxx, kyy = kernel(x, x), kernel(y, y)
    if unbiased:
        kxx = kxx - torch.diag_embed(torch.diagonal(kxx, dim1=-2, dim2=-1))
        kyy = kyy - torch.diag_embed(torch.diagonal(kyy, dim1=-2, dim2=-1))
    xx = kxx.sum((-2, -1)) / (n * (n - 1))
    yy = kyy.sum((-2, -1)) / (m * (m - 1))
    xy = kernel(x, y).sum((-2, -1)) / (n * m)
    return xx + yy - 2.0 * xy


def generalised_mmd(x, y, sigma_sq=None, unbiased=False):
    """The scalar reported in the paper: the largest MMD over the family."""
    return mmd_curve(x, y, sigma_sq, unbiased).max().item()
