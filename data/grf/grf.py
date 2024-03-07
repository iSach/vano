import torch
import matplotlib.pyplot as plt


def gen_grfs(nb_fcts, alpha=2, tau=1, trunc_sum_elems=32, a=0, b=1, steps=128):
    """
    Generates a set of Gaussian random fields (GRFs) using the spectral representation.

    The resulting GRFs follow a distribution with
        mean = 0
        covariance = (∆ - τ^2 * I)^(-α)

        where ∆ is the Laplacian operator, τ is the smoothness parameter, and α is the decay parameter.

    Args:
        nb_fcts (int): Number of GRFs to generate.
        alpha (float): Decay parameter.
        tau (float): Smoothness parameter.
        trunc_sum_elems (int): Number of terms in the sum used to approximate the GRF.
        a (float): Lower bound of the domain.
        b (float): Upper bound of the domain.
        steps (int): Number of points in the domain.
    """
    sum_range = torch.arange(1, trunc_sum_elems + 1).float()

    X = torch.linspace(a, b, steps)
    Xi = torch.tensordot(X, sum_range, dims=0)

    ksi = torch.randn(nb_fcts, trunc_sum_elems)
    sqrt_λ = (
        torch.sqrt(((2 * torch.pi * sum_range) ** 2 + tau**2) ** (-alpha))
        .unsqueeze(0)
        .expand(nb_fcts, -1)
    )
    φ = torch.sqrt(torch.tensor(2.0)) * torch.sin(2 * torch.pi * Xi) \
                .unsqueeze(1) \
                .expand(-1, nb_fcts, trunc_sum_elems)
    u = torch.sum(ksi * sqrt_λ * φ, dim=2)

    X = X.unsqueeze(0).expand(nb_fcts, -1)
    u = u.T

    return X, u
