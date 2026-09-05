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
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.2))

    # -- left: error vs latent dimension, against both reference covariances --
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
    left.set_ylabel(r"$\|C-\hat{C}\|_F\,/\,\|C\|_F$")
    left.set_title("Covariance recovery")
    left.legend(fontsize=10)
    left.grid(alpha=0.3)

    # -- right: learned basis vs the analytic KL basis --
    model, _ = load_run(REFERENCE, device)
    tau = model.decoder.trunk(x).cpu()
    # tau spans the learned subspace in an arbitrary basis; diagonalising the
    # induced covariance puts it in the same gauge as the analytic eigenpairs.
    learned_evals, learned = basis_eigenfunctions(tau, num=4)
    kl = (efuns * evals.sqrt()).cpu()
    xs = x.squeeze(-1).cpu()
    for k in range(4):
        colour = f"C{k}"
        right.plot(xs, kl[:, k], colour, lw=2.2,
                   label="Karhunen-Loeve" if k == 0 else None)
        right.plot(xs, learned[:, k] * learned_evals[k].sqrt(), colour,
                   ls="--", lw=2.2, label="VANO (learned)" if k == 0 else None)
    right.set_xlabel("$x$")
    right.set_ylabel(r"$\sqrt{\lambda_k}\,\phi_k(x)$")
    right.set_title("Leading basis functions")
    right.legend(fontsize=10)

    fig.tight_layout()
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
