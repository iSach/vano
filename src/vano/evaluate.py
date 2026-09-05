"""Benchmark-specific evaluation, shared by the CLI and the figure scripts."""

import torch

from .data import TEST_SEED, load_dataset, unit_grid
from .data.grf import eigenpairs, legendre_nodes
from .data.insar import phase
from .metrics import (
    circular_skewness,
    circular_variance,
    generalised_mmd,
    hilbert_schmidt_error,
    optimal_truncation_error,
)


@torch.no_grad()
def reconstruct(model, data, batch_size=64, y=None, deterministic=True):
    """Posterior-mean (or sampled) reconstructions of every function in ``data``."""
    y = data.y if y is None else y
    out = []
    for start in range(0, len(data), batch_size):
        u = data.u[start : start + batch_size]
        mu, logvar = model.encode(u)
        z = mu if deterministic else mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        out.append(model.decode(z, y))
    return torch.cat(out)


@torch.no_grad()
def sample(model, num_samples, y=None, batch_size=64, seed=0, device=None):
    """Draw functions from the prior and evaluate them on ``y``."""
    device = device or next(model.parameters()).device
    generator = torch.Generator(device=device).manual_seed(seed)
    out = []
    for start in range(0, num_samples, batch_size):
        n = min(batch_size, num_samples - start)
        z = torch.randn(n, model.latent_dim, device=device, generator=generator)
        out.append(model.decode(z, y))
    return torch.cat(out)


def relative_l2(target, prediction):
    """Mean relative L2 error over a batch of functions."""
    axes = tuple(range(1, target.dim()))
    return ((target - prediction).square().sum(axes).sqrt()
            / target.square().sum(axes).sqrt()).mean().item()


@torch.no_grad()
def covariance_error(model, num_points=128, num_eigenpairs=32, device="cuda"):
    """Normalised Hilbert-Schmidt error of the learned covariance (section 6.1).

    Reported under both conventions: against the rank-``n`` truncation of the
    true covariance (what the release logs) and against the full covariance,
    whose floor is the optimal Karhunen-Loeve truncation error.
    """
    x, _ = legendre_nodes(num_points)
    x = x.to(device)
    evals, efuns = eigenpairs(x, num_eigenpairs)
    tau = model.decoder.trunk(x)
    rank = min(model.latent_dim, num_eigenpairs)
    return {
        "hs_error_truncated": hilbert_schmidt_error(tau, evals, efuns,
                                                    model.latent_dim),
        "hs_error_full": hilbert_schmidt_error(tau, evals, efuns),
        "hs_error_optimal": optimal_truncation_error(evals, efuns, rank),
    }


@torch.no_grad()
def mmd_against(model, data, num_samples=512, num_trials=5, resolution=None,
                seed=0):
    """Generalised MMD between model samples and held-out data, over trials."""
    y = data.y if resolution is None else unit_grid(
        resolution, dim=data.y.shape[-1], device=data.y.device)
    values = []
    for trial in range(num_trials):
        offset = trial * num_samples
        targets = data.s[offset : offset + num_samples]
        if len(targets) < num_samples:
            raise ValueError("not enough held-out functions for this many trials")
        samples = sample(model, num_samples, y, seed=seed + trial)
        values.append(generalised_mmd(samples, targets))
    values = torch.tensor(values)
    return {"mmd_mean": values.mean().item(), "mmd_std": values.std().item(),
            "mmd_trials": values.tolist()}


@torch.no_grad()
def circular_statistics(fields):
    """Circular variance and skewness of ``(N, H, W, 2)`` cosine/sine fields."""
    angles = phase(fields)
    return {"circular_variance": circular_variance(angles),
            "circular_skewness": circular_skewness(angles)}


def test_data(cfg):
    """The held-out set for ``cfg``'s benchmark.

    The two analytic benchmarks reserve a second random stream (the release uses
    ``PRNGKey(1)``); Cahn-Hilliard has a real split; the InSAR release trains on
    all 4096 interferograms and holds nothing out, so we score against the same
    set it was trained on and say so.
    """
    if cfg.dataset in ("grf", "bumps"):
        return load_dataset(cfg.dataset, seed=TEST_SEED, **cfg.dataset_kwargs)
    if cfg.dataset == "cahn_hilliard":
        return load_dataset(cfg.dataset, split="test", **cfg.dataset_kwargs)
    return load_dataset(cfg.dataset, **cfg.dataset_kwargs)


def evaluate(model, cfg, data=None, device="cuda"):
    """Every metric the paper reports for ``cfg``'s benchmark."""
    data = (test_data(cfg) if data is None else data).to(device)
    metrics = {"relative_l2": relative_l2(data.s, reconstruct(model, data))}

    if cfg.dataset == "grf":
        metrics.update(covariance_error(model, device=device))
    elif cfg.dataset == "bumps":
        metrics.update(mmd_against(model, data, num_samples=512, num_trials=4))
    elif cfg.dataset == "cahn_hilliard":
        metrics.update(mmd_against(model, data, num_samples=512, num_trials=5))
    elif cfg.dataset == "insar":
        stats = circular_statistics(data.u)
        samples = sample(model, len(data), data.y).reshape(data.u.shape)
        model_stats = circular_statistics(samples)
        for key, value in model_stats.items():
            metrics[f"{key}_mean"] = value.mean().item()
            metrics[f"{key}_data_mean"] = stats[key].mean().item()
    return metrics
