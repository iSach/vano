"""Paper Figures 4, 7-11: why the decoder has to be nonlinear.

The functions are 2D Gaussian densities.  They form a smooth two-parameter
family, but one that no low-dimensional *linear* subspace captures: translating
a narrow bump is a strongly nonlinear operation in function space.

Figure 4     generalised MMD against the latent dimension, linear vs nonlinear
             decoder, five seeds each, with samples from both models.
Figure 7     PCA spectrum of the training set -- the slow decay that explains
             why a linear decoder needs so many components.
Figures 8-9  test reconstructions for each decoder.
Figures 10-11 samples decoded on a 256x256 grid after training at 48x48.

    python experiments/bumps_decoders.py --workers 8
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from _common import base_parser, load_run, sweep

from vano.configs import get_config
from vano.data import bumps, unit_grid
from vano.evaluate import reconstruct, relative_l2, sample
from vano.metrics import generalised_mmd
from vano.plotting import save, show_field, use_paper_style

LATENT_DIMS = (4, 32, 64, 128, 256, 512)
SEEDS = tuple(range(5))
DECODERS = {"linear": "Linear decoder", "concat": "Nonlinear decoder"}
NUM_MMD_SAMPLES = 512
SUPERRES = 256


def configs():
    return [get_config(f"bumps_{d}").evolve(latent_dim=n, seed=s)
            for d in DECODERS for n in LATENT_DIMS for s in SEEDS]


def collect(device):
    test = bumps.load(2048, seed=1).to(device)
    targets = test.s[:NUM_MMD_SAMPLES]
    rows = []
    for decoder in DECODERS:
        for n in LATENT_DIMS:
            for seed in SEEDS:
                cfg = get_config(f"bumps_{decoder}").evolve(latent_dim=n, seed=seed)
                model, _ = load_run(cfg, device)
                samples = sample(model, NUM_MMD_SAMPLES, test.y, seed=123)
                rows.append({"decoder": decoder, "latent_dim": n, "seed": seed,
                             "mmd": generalised_mmd(samples, targets),
                             "relative_l2": relative_l2(
                                 test.s[:512], reconstruct(model, test[:512]))})
    return rows, test


def figure4(rows, test, device, out_dir):
    fig = plt.figure(figsize=(13.0, 4.6))
    spec = fig.add_gridspec(2, 6, width_ratios=[2.6, 0.25, 1, 1, 1, 1])

    curve = fig.add_subplot(spec[:, 0])
    for i, (decoder, label) in enumerate(DECODERS.items()):
        mean, std = _aggregate(rows, decoder, "mmd")
        curve.errorbar(LATENT_DIMS, mean, yerr=std, fmt=f"-{'os'[i]}",
                       capsize=4, label=label)
    curve.set_xscale("log", base=2)
    curve.set_yscale("log")
    curve.set_xticks(LATENT_DIMS, [str(n) for n in LATENT_DIMS])
    curve.set_xlabel("latent dimension $n$")
    curve.set_ylabel("generalised MMD")
    curve.set_title("Sample quality vs latent dimension")
    curve.legend()
    curve.grid(alpha=0.3)

    # Samples from the best latent dimension of each decoder.
    for row, decoder in enumerate(DECODERS):
        best = _best_latent(rows, decoder)
        cfg = get_config(f"bumps_{decoder}").evolve(latent_dim=best, seed=0)
        model, _ = load_run(cfg, device)
        samples = sample(model, 4, test.y, seed=7).cpu()
        for col in range(4):
            ax = fig.add_subplot(spec[row, col + 2])
            show_field(ax, samples[col, :, 0].reshape(*test.grid_shape))
            if col == 0:
                ax.set_ylabel(f"{DECODERS[decoder].split()[0]}\n$n={best}$",
                              fontsize=11)
            if row == 0 and col == 1:
                ax.set_title("Generated samples", fontsize=14, loc="left")
    fig.tight_layout()
    save(fig, out_dir / "figure4_bumps_mmd.png")


def figure7(train, out_dir):
    """PCA spectrum of the training functions."""
    x = train.s[:, :, 0].double()
    x = x - x.mean(0, keepdim=True)
    spectrum = torch.linalg.svdvals(x).square() / (len(x) - 1)
    spectrum = (spectrum / spectrum.sum()).cpu().numpy()
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.semilogy(np.arange(1, 513), spectrum[:512], lw=2)
    for n in (32, 128, 512):
        ax.axvline(n, color="0.7", ls=":", lw=1.2)
        ax.text(n, spectrum[:512].max(), f" $n={n}$", fontsize=10, va="top")
    ax.set_xlabel("component index")
    ax.set_ylabel("normalised eigenvalue")
    ax.set_title("PCA spectrum of the training set")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, out_dir / "figure7_bumps_pca.png")


def figures8_9(test, device, out_dir):
    idx = torch.arange(6)
    fig, axes = plt.subplots(3, 6, figsize=(12.0, 6.2))
    for col, i in enumerate(idx):
        show_field(axes[0, col], test.s[i, :, 0].reshape(*test.grid_shape).cpu())
    axes[0, 0].set_ylabel("Ground truth")
    for row, decoder in enumerate(DECODERS, start=1):
        cfg = get_config(f"bumps_{decoder}").evolve(latent_dim=32, seed=0)
        model, _ = load_run(cfg, device)
        pred = reconstruct(model, test[idx]).cpu()
        for col in range(len(idx)):
            show_field(axes[row, col],
                       pred[col, :, 0].reshape(*test.grid_shape))
        axes[row, 0].set_ylabel(DECODERS[decoder].split()[0])
    fig.suptitle("Test reconstructions, $n=32$", y=0.98)
    fig.tight_layout()
    save(fig, out_dir / "figures8_9_bumps_reconstructions.png")


def figures10_11(device, out_dir):
    grid = unit_grid(SUPERRES, dim=2, device=device)
    fig, axes = plt.subplots(2, 5, figsize=(11.0, 4.6))
    for row, decoder in enumerate(DECODERS):
        cfg = get_config(f"bumps_{decoder}").evolve(latent_dim=32, seed=0)
        model, _ = load_run(cfg, device)
        samples = sample(model, 5, grid, seed=11).cpu()
        for col in range(5):
            show_field(axes[row, col],
                       samples[col, :, 0].reshape(SUPERRES, SUPERRES))
        axes[row, 0].set_ylabel(DECODERS[decoder].split()[0])
    fig.suptitle(f"Trained at $48\\times48$, decoded at "
                 f"${SUPERRES}\\times{SUPERRES}$", y=0.99)
    fig.tight_layout()
    save(fig, out_dir / "figures10_11_bumps_superresolution.png")


def _aggregate(rows, decoder, key):
    mean, std = [], []
    for n in LATENT_DIMS:
        values = np.array([r[key] for r in rows
                           if r["decoder"] == decoder and r["latent_dim"] == n])
        mean.append(values.mean())
        std.append(values.std())
    return np.array(mean), np.array(std)


def _best_latent(rows, decoder):
    mean, _ = _aggregate(rows, decoder, "mmd")
    return LATENT_DIMS[int(np.argmin(mean))]


def main():
    args = base_parser(__doc__).parse_args()
    sweep(configs(), workers=args.workers, device=args.device, force=args.force)
    use_paper_style()
    args.figures.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        rows, test = collect(args.device)
        (args.figures / "figure4_bumps_mmd.json").write_text(
            json.dumps(rows, indent=2))
        figure4(rows, test, args.device, args.figures)
        figure7(bumps.load(2048, seed=0).to(args.device), args.figures)
        figures8_9(test, args.device, args.figures)
        figures10_11(args.device, args.figures)


if __name__ == "__main__":
    main()
