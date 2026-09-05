import pytest
import torch

from vano.configs import CONFIGS, get_config
from vano.data import unit_grid
from vano.models import VANO
from vano.models.layers import Dense
from vano.models.vano import elbo_loss, kl_divergence


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_every_config_builds_and_runs(name):
    cfg = get_config(name).evolve(latent_dim=8, seed=0)
    if cfg.decoder.name == "split":
        cfg = cfg.evolve(**{"decoder.num_layers": 4})
    model = VANO.from_config(cfg)
    height = cfg.input_shape[0]
    u = torch.randn(3, *cfg.input_shape)
    y = unit_grid(height, dim=cfg.query_dim)
    s = torch.randn(3, y.shape[0], cfg.out_dim)
    loss, parts = elbo_loss(model, u, y, s, torch.ones(3),
                            torch.randn(2, 3, cfg.latent_dim), cfg.beta)
    assert torch.isfinite(loss)
    assert parts["kl_loss"] >= 0


def test_random_weight_factorisation_reproduces_its_kernel():
    torch.manual_seed(0)
    layer = Dense(5, 7, weight_fact=True)
    assert layer.kernel.shape == (7, 5)
    # g is folded out of v at construction, so the product is the base kernel.
    assert torch.allclose(layer.g * layer.v, layer.kernel)
    x = torch.randn(3, 5)
    assert torch.allclose(layer(x), x @ layer.kernel.T + layer.bias, atol=1e-6)


def test_kl_divergence_vanishes_at_the_prior():
    zeros = torch.zeros(4, 6)
    assert torch.allclose(kl_divergence(zeros, zeros), torch.zeros(4))
    assert (kl_divergence(torch.randn(4, 6), torch.randn(4, 6)) >= 0).all()


def test_decoder_is_discretisation_agnostic():
    """The same latent code decoded on nested grids agrees where they overlap."""
    torch.manual_seed(0)
    cfg = get_config("cahn_hilliard").evolve(latent_dim=8)
    model = VANO.from_config(cfg).eval()
    z = torch.randn(2, 8)
    with torch.no_grad():
        coarse = model.decode(z, unit_grid(9, dim=2)).reshape(2, 9, 9)
        fine = model.decode(z, unit_grid(17, dim=2)).reshape(2, 17, 17)
    assert torch.allclose(coarse, fine[:, ::2, ::2], atol=1e-5)


def test_monte_carlo_axis_matches_a_python_loop():
    torch.manual_seed(0)
    cfg = get_config("bumps_concat").evolve(latent_dim=8)
    model = VANO.from_config(cfg).eval()
    u = torch.randn(3, 48, 48, 1)
    y = unit_grid(48, dim=2)
    eps = torch.randn(4, 3, 8)
    with torch.no_grad():
        batched, _, _ = model(u, y, eps)
        mu, logvar = model.encode(u)
        looped = torch.stack([model.decode(mu + e * (0.5 * logvar).exp(), y)
                              for e in eps])
    assert torch.allclose(batched, looped, atol=1e-5)
