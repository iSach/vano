"""Covariance-operator diagnostics for the GRF benchmark (paper section 6.1).

With a linear decoder ``u(x) = sum_j z_j tau_j(x)`` and ``z ~ N(0, I)``, the
push-forward of the prior is a Gaussian measure whose covariance operator is
``C_hat = sum_j tau_j (x) tau_j``.  The paper measures how close that is to the
true covariance in the normalised Hilbert-Schmidt (Frobenius) norm.
"""

import torch


def covariance_from_basis(basis, evals=None, quad_weight=None):
    """``sum_j lambda_j phi_j phi_j^T`` for a ``(P, K)`` basis, as a ``(P, P)``."""
    if evals is None:
        evals = torch.ones(basis.shape[1], dtype=basis.dtype, device=basis.device)
    if quad_weight is None:
        quad_weight = 1.0 / basis.shape[0]
    return quad_weight * (basis * evals) @ basis.T


def hilbert_schmidt_error(learned_basis, evals, efuns, num_components=None):
    """Normalised HS distance ``||C - C_hat||_F / ||C||_F``.

    ``num_components`` truncates the reference covariance.  The released code
    compares against the rank-``min(n, K)`` truncation of the true covariance,
    which isolates *basis* error from truncation error; passing ``None`` uses
    the full covariance, whose floor is the optimal KL truncation error.
    """
    if num_components is not None:
        num_components = min(num_components, efuns.shape[1])
        evals, efuns = evals[:num_components], efuns[:, :num_components]
    target = covariance_from_basis(efuns, evals)
    learned = covariance_from_basis(learned_basis)
    return (torch.linalg.norm(target - learned) /
            torch.linalg.norm(target)).item()


def optimal_truncation_error(evals, efuns, rank):
    """HS error of the best rank-``r`` approximation: the theoretical floor."""
    full = covariance_from_basis(efuns, evals)
    truncated = covariance_from_basis(efuns[:, :rank], evals[:rank])
    return (torch.linalg.norm(full - truncated) /
            torch.linalg.norm(full)).item()


def basis_eigenfunctions(learned_basis, num=8):
    """Leading eigenpairs of ``C_hat``, sign-fixed, for plotting against the KL basis."""
    cov = covariance_from_basis(learned_basis)
    evals, evecs = torch.linalg.eigh(cov.double())
    evals, evecs = evals.flip(0)[:num], evecs.flip(1)[:, :num]
    # eigh returns unit-norm discrete vectors; rescale to the L^2 normalisation
    # of the analytic eigenfunctions, and fix the sign by the first extremum.
    evecs = evecs * learned_basis.shape[0] ** 0.5
    peak = evecs.abs().argmax(dim=0)
    sign = torch.sign(evecs[peak, torch.arange(num, device=evecs.device)])
    return evals.to(learned_basis.dtype), (evecs * sign).to(learned_basis.dtype)
