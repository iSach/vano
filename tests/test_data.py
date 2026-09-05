import torch

from vano.data import bumps, grf, unit_grid


def test_unit_grid_orders_x_first():
    grid = unit_grid(4, dim=2).reshape(4, 4, 2)
    assert torch.allclose(grid[:, 0, 0], torch.linspace(0, 1, 4))
    assert torch.allclose(grid[0, :, 1], torch.linspace(0, 1, 4))


def test_grf_matches_its_analytic_covariance():
    """The empirical covariance of the samples converges to sum_k lambda_k phi_k phi_k."""
    data = grf.load(num_samples=20000, seed=0)
    x, _ = grf.legendre_nodes(grf.NUM_POINTS)
    evals, efuns = grf.eigenpairs(x)
    target = (efuns * evals) @ efuns.T
    empirical = data.u.T @ data.u / len(data)
    error = torch.linalg.norm(target - empirical) / torch.linalg.norm(target)
    assert error < 0.05


def test_grf_weights_normalise_the_residual():
    data = grf.load(num_samples=8, seed=0)
    assert torch.allclose(data.w, 1.0 / data.u.norm(dim=1).square())


def test_bumps_are_normalised_densities():
    """Each function integrates to ~1 over the unit square."""
    data = bumps.load(num_samples=64, resolution=192, seed=0)
    mass = data.s[:, :, 0].mean(dim=1)
    # Bumps whose centre sits near the boundary lose part of their mass.
    assert mass.median() > 0.9
    assert mass.max() < 1.05


def test_bumps_image_and_targets_share_a_layout():
    data = bumps.load(num_samples=4, resolution=16, seed=0)
    assert torch.equal(data.u.reshape(4, -1, 1), data.s)
    assert torch.isfinite(data.w).all()
