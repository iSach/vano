"""Paper Figures 3 and 6: does the linear decoder recover the optimal KL basis?

Figure 3 (left)  normalised Hilbert-Schmidt error of the learned covariance
                 operator against the latent dimension, over 10 seeds.
Figure 3 (right) the learned basis next to the analytic Karhunen-Loeve basis.
Figure 6         ground-truth GRF samples next to samples from the prior.

    python experiments/grf_basis.py --workers 8
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from _common import base_parser, load_run, sweep

from vano.configs import get_config
from vano.data import TEST_SEED, grf
from vano.evaluate import covariance_error, sample
from vano.metrics import basis_eigenfunctions, optimal_truncation_error
from vano.plotting import save, use_paper_style

LATENT_DIMS = (2, 4, 8, 16, 32, 64)
SEEDS = tuple(range(10))
REFERENCE = get_config("grf")


def configs():
    return [REFERENCE.evolve(latent_dim=n, seed=s)
            for n in LATENT_DIMS for s in SEEDS]


def collect(device):
    """Hilbert-Schmidt errors for every (latent dimension, seed) pair."""
    rows = []
    for n in LATENT_DIMS:
        for seed in SEEDS:
            cfg = REFERENCE.evolve(latent_dim=n, seed=seed)
            model, _ = load_run(cfg, device)
            rows.append({"latent_dim": n, "seed": seed,
                         **covariance_error(model, device=device)})
    return rows


def figure3(rows, device, out_dir):
    """Left: covariance error vs latent size.  Right: learned vs optimal basis."""
    fig = plt.figure(figsize=(12.5, 5.6))
    spec = fig.add_gridspec(3, 2, width_ratios=[1.25, 1.0], hspace=0.55,
                            wspace=0.28)

    # -- left: error vs latent dimension, under both reference conventions --
    left = fig.add_subplot(spec[:, 0])
    x, evals, efuns = _analytic(device)
    for key, label, style in (
        ("hs_error_truncated", "VANO, vs rank-$n$ truncation", "-o"),
        ("hs_error_full", "VANO, vs full covariance", "--s"),
    ):
        mean, std = _aggregate(rows, key)
        left.errorbar(LATENT_DIMS, mean, yerr=std, fmt=style, capsize=4,
                      label=label)
    floor = [optimal_truncation_error(evals, efuns, min(n, efuns.shape[1]))
             for n in LATENT_DIMS]
    left.plot(LATENT_DIMS, np.maximum(floor, 1e-16), ":k",
              label="optimal rank-$n$ (KL) truncation")
    left.set_xscale("log", base=2)
    left.set_yscale("log")
    left.set_xticks(LATENT_DIMS, [str(n) for n in LATENT_DIMS])
    left.set_xlabel("latent dimension $n$")
    left.set_ylabel(r"$\|\Gamma-\hat{\Gamma}\|_{HS}\,/\,\|\Gamma\|_{HS}$")
    left.set_title("Covariance recovery")
    left.legend(fontsize=10, loc="lower left")
    left.grid(alpha=0.3)
    # The KL term prices each active direction, so the model keeps far fewer
    # than n of them; annotate how many it actually uses.
    rank, _ = _aggregate(rows, "effective_rank")
    mean, _ = _aggregate(rows, "hs_error_truncated")
    for n, value, r in zip(LATENT_DIMS, mean, rank):
        left.annotate(f"rank {r:.1f}", (n, value), textcoords="offset points",
                      xytext=(0, 11), ha="center", fontsize=9, color="0.35")

    # -- right: the two bases, as in the paper --
    model, _ = load_run(REFERENCE, device)
    tau = model.decoder.trunk(x).cpu()
    kl = (efuns * evals.sqrt()).cpu()
    xs = x.squeeze(-1).cpu()

    top = fig.add_subplot(spec[0, 1])
    top.plot(xs, kl, lw=1.8)
    top.set_ylabel(r"$\sqrt{\lambda_i}\,\phi_i(x)$")
    top.set_title(f"Optimal (KL) basis, {kl.shape[1]} terms", fontsize=13)

    middle = fig.add_subplot(spec[1, 1], sharey=top)
    middle.plot(xs, tau, lw=1.8)
    middle.set_xlabel("$x$")
    middle.set_ylabel(r"$\tau_j(x)$")
    middle.set_title(f"Learned basis, $n={REFERENCE.latent_dim}$", fontsize=13)

    # Spectrum: the quantitative version of "the bases agree".
    bottom = fig.add_subplot(spec[2, 1])
    learned_evals, _ = basis_eigenfunctions(tau, num=8)
    bottom.semilogy(np.arange(1, 9), evals[:8].cpu(), "ko-", label="$\\lambda_i$")
    bottom.semilogy(np.arange(1, 9), learned_evals.clamp_min(1e-16), "C3s--",
                    label=r"eigenvalues of $\hat{\Gamma}$")
    bottom.set_xlabel("component $i$")
    bottom.set_ylabel("eigenvalue")
    bottom.legend(fontsize=10)
    bottom.grid(alpha=0.3)

    save(fig, out_dir / "figure3_grf_covariance.png")


def figure6(device, out_dir):
    model, cfg = load_run(REFERENCE, device)
    data = grf.load(2048, seed=TEST_SEED).to(device)
    samples = sample(model, 16, data.y, seed=123).squeeze(-1).cpu()
    xs = data.y.squeeze(-1).cpu()

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), sharey=True)
    for i in range(16):
        axes[0].plot(xs, data.s[i, :, 0].cpu(), lw=1.6)
        axes[1].plot(xs, samples[i], lw=1.6)
    axes[0].set_title("Ground truth")
    axes[1].set_title("VANO samples")
    for ax in axes:
        ax.set_xlabel("$x$")
    axes[0].set_ylabel("$u(x)$")
    fig.tight_layout()
    save(fig, out_dir / "figure6_grf_samples.png")


def _analytic(device):
    x, _ = grf.legendre_nodes(grf.NUM_POINTS)
    x = x.to(device)
    evals, efuns = grf.eigenpairs(x)
    return x, evals, efuns


def _aggregate(rows, key):
    mean, std = [], []
    for n in LATENT_DIMS:
        values = np.array([r[key] for r in rows if r["latent_dim"] == n])
        mean.append(values.mean())
        std.append(values.std())
    return np.array(mean), np.array(std)


def main():
    args = base_parser(__doc__).parse_args()
    sweep(configs(), workers=args.workers, device=args.device, force=args.force)
    use_paper_style()
    with torch.no_grad():
        rows = collect(args.device)
        args.figures.mkdir(parents=True, exist_ok=True)
        (args.figures / "figure3_grf_covariance.json").write_text(
            json.dumps(rows, indent=2))
        figure3(rows, args.device, args.figures)
        figure6(args.device, args.figures)


if __name__ == "__main__":
    main()
