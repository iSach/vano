"""Paper Figures 5, 14, 15-16: InSAR interferograms of the Long Valley Caldera.

The observations are wrapped phases, so the model regresses ``(cos phi, sin phi)``
and every comparison is made with directional statistics.  The GANO baseline is
not retrained: the reference release ships the samples its generator produced,
and those are what Figure 5 compares against.

Figure 5      distributions of circular variance and circular skewness.
Figure 14     reconstructions and absolute phase error.
Figures 15-16 VANO samples next to GANO samples.

    python experiments/insar_volcano.py

Requires the InSAR dataset; see scripts/download_data.py.
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from _common import base_parser, load_run, sweep

from vano.configs import get_config
from vano.data import insar
from vano.data.insar import phase
from vano.evaluate import reconstruct, sample
from vano.metrics import circular_skewness, circular_variance
from vano.plotting import save, use_paper_style

CMAP = "RdYlBu"
CONFIG = get_config("insar")


def _phase_image(ax, field, **kwargs):
    image = ax.imshow(phase(field).T.cpu(), origin="lower", cmap=CMAP,
                      vmin=-np.pi, vmax=np.pi, interpolation="nearest", **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    return image


def figure5(data, model, device, out_dir):
    """Circular variance and skewness: data vs VANO vs GANO."""
    vano = sample(model, len(data), data.y, seed=0).reshape(data.u.shape)
    gano = insar.load_gano_samples().to(device)
    series = {"Ground truth": data.u, "VANO": vano, "GANO": gano}

    summaries, stats = {}, {}
    for name, fields in series.items():
        angles = phase(fields)
        variance = circular_variance(angles).cpu().numpy()
        skewness = circular_skewness(angles).cpu().numpy()
        # Skewness is undefined for a field of constant phase (R = 1); a few
        # GANO samples are exactly that, so drop them and say how many.
        finite = np.isfinite(skewness)
        summaries[name] = {"circular_variance_mean": float(variance.mean()),
                           "circular_skewness_mean": float(skewness[finite].mean()),
                           "num_samples": int(len(variance)),
                           "num_degenerate": int((~finite).sum())}
        stats[name] = (variance, skewness[finite])

    span = np.abs(np.concatenate([s for _, s in stats.values()])).max()
    bins = {0: np.linspace(0.0, 1.0, 30),
            1: np.linspace(-min(span, 5.0), min(span, 5.0), 40)}
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for name, (variance, skewness) in stats.items():
        axes[0].hist(variance, bins=bins[0], histtype="step", lw=2.2,
                     density=True, label=name)
        axes[1].hist(skewness, bins=bins[1], histtype="step", lw=2.2,
                     density=True, label=name)
    axes[0].set_xlabel("circular variance")
    axes[1].set_xlabel("circular skewness")
    for ax in axes:
        ax.set_ylabel("density")
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
    fig.suptitle("Directional statistics of the interferogram distribution")
    fig.tight_layout()
    save(fig, out_dir / "figure5_insar_circular_statistics.png")
    return summaries


def figure14(data, model, out_dir, num=4):
    """Reconstructions and their absolute phase error."""
    subset = data[:num]
    pred = reconstruct(model, subset).reshape(subset.u.shape)
    fig, axes = plt.subplots(num, 3, figsize=(9.2, 3.0 * num))
    for row in range(num):
        _phase_image(axes[row, 0], subset.u[row])
        image = _phase_image(axes[row, 1], pred[row])
        error = (phase(subset.u[row]) - phase(pred[row])).abs()
        error = torch.minimum(error, 2 * np.pi - error)
        err = axes[row, 2].imshow(error.T.cpu(), origin="lower", cmap="magma",
                                  vmin=0, vmax=np.pi, interpolation="nearest")
        axes[row, 2].set_xticks([])
        axes[row, 2].set_yticks([])
    for ax, title in zip(axes[0], ("Ground truth", "Reconstruction",
                                   "Absolute phase error")):
        ax.set_title(title, fontsize=13)
    fig.colorbar(image, ax=axes[:, :2], shrink=0.6, label=r"$\varphi$")
    fig.colorbar(err, ax=axes[:, 2], shrink=0.6)
    save(fig, out_dir / "figure14_insar_reconstructions.png")


def figures15_16(data, model, device, out_dir):
    """VANO samples next to the GANO samples shipped with the release."""
    vano = sample(model, 8, data.y, seed=3).reshape(-1, *data.grid_shape, 2)
    gano = insar.load_gano_samples().to(device)[:8]
    truth = data.u[:8]
    fig, axes = plt.subplots(3, 8, figsize=(16.0, 6.4))
    for col in range(8):
        _phase_image(axes[0, col], truth[col])
        _phase_image(axes[1, col], vano[col])
        _phase_image(axes[2, col], gano[col])
    for ax, label in zip(axes[:, 0], ("Ground truth", "VANO", "GANO")):
        ax.set_ylabel(label, fontsize=13)
    fig.tight_layout()
    save(fig, out_dir / "figures15_16_insar_samples.png")


def main():
    args = base_parser(__doc__).parse_args()
    sweep([CONFIG], workers=1, device=args.device, force=args.force)
    use_paper_style()
    args.figures.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        model, _ = load_run(CONFIG, args.device)
        data = insar.load().to(args.device)
        stats = figure5(data, model, args.device, args.figures)
        print(json.dumps(stats, indent=2))
        (args.figures / "figure5_insar_circular_statistics.json").write_text(
            json.dumps(stats, indent=2))
        figure14(data, model, args.figures)
        figures15_16(data, model, args.device, args.figures)


if __name__ == "__main__":
    main()
