import torch

from vano.data import grf
from vano.metrics import (
    circular_skewness,
    circular_variance,
    generalised_mmd,
    hilbert_schmidt_error,
    mmd_curve,
    optimal_truncation_error,
)


def test_mmd_is_small_between_two_draws_of_the_same_law():
    torch.manual_seed(0)
    x, y = torch.randn(256, 64), torch.randn(256, 64)
    same = generalised_mmd(x, y)
    shifted = generalised_mmd(x, y + 1.0)
    assert same < 0.02
    assert shifted > 10 * same


def test_mmd_curve_covers_the_bandwidth_family():
    torch.manual_seed(0)
    curve = mmd_curve(torch.randn(64, 8), torch.randn(64, 8) + 2.0)
    assert curve.shape == (100,)
    assert curve.max() > curve.min()


def test_hilbert_schmidt_error_vanishes_on_the_exact_basis():
    x, _ = grf.legendre_nodes(128)
    evals, efuns = grf.eigenpairs(x)
    exact = efuns * evals.sqrt()          # sum_k tau_k tau_k^T == C
    assert hilbert_schmidt_error(exact, evals, efuns) < 1e-5


def test_optimal_truncation_error_decreases_with_rank():
    x, _ = grf.legendre_nodes(128)
    evals, efuns = grf.eigenpairs(x)
    errors = [optimal_truncation_error(evals, efuns, r) for r in (1, 2, 4, 8)]
    assert errors == sorted(errors, reverse=True)
    assert optimal_truncation_error(evals, efuns, 32) < 1e-6


def test_circular_statistics_on_known_fields():
    constant = torch.full((3, 8, 8), 0.7)
    assert torch.allclose(circular_variance(constant), torch.zeros(3), atol=1e-6)
    torch.manual_seed(0)
    uniform = (torch.rand(4, 128, 128) * 2 - 1) * torch.pi
    assert (circular_variance(uniform) > 0.98).all()
    assert torch.isfinite(circular_skewness(uniform)).all()
