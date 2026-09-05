"""Paper Table 1 and Figures 12-13: discretisation-agnostic super-resolution.

A single VANO is trained on 64x64 Cahn-Hilliard patterns and then *decoded* at
64x64, 128x128 and 256x256 without retraining.  The baseline is a
discretise-first convolutional VAE, trained separately at each resolution --
three models against one.

    python experiments/cahn_hilliard_superres.py --workers 2

Requires the Cahn-Hilliard dataset; see scripts/download_data.py.
"""

import json

import matplotlib.pyplot as plt
import torch
from _common import base_parser, load_run, sweep

from vano.configs import get_config
from vano.data import cahn_hilliard, unit_grid
from vano.evaluate import reconstruct, relative_l2, sample
from vano.metrics import generalised_mmd
from vano.plotting import save, show_field, use_paper_style

RESOLUTIONS = (64, 128, 256)
NUM_TRIALS = 5
# The 256x256 comparison uses fewer samples: the released notebook does the
# same, and the MMD Gram matrices grow with the number of pixels.
NUM_SAMPLES = {64: 512, 128: 512, 256: 256}


def configs():
    return [get_config("cahn_hilliard")] + \
           [get_config(f"cahn_hilliard_vae{r}") for r in RESOLUTIONS]


def test_sets(device):
    return {r: cahn_hilliard.load("test", resolution=r).to(device)
            for r in RESOLUTIONS}


def table1(tests, device):
    """Generalised MMD (x100) of VANO and the per-resolution VAEs."""
    vano, _ = load_run(get_config("cahn_hilliard"), device)
    rows = []
    for resolution in RESOLUTIONS:
        data = tests[resolution]
        num = NUM_SAMPLES[resolution]
        grid = unit_grid(resolution, dim=2, device=device)
        baseline, _ = load_run(get_config(f"cahn_hilliard_vae{resolution}"), device)
        for trial in range(NUM_TRIALS):
            targets = data.s[trial * num : (trial + 1) * num]
            rows.append({
                "resolution": resolution, "trial": trial,
                "vano": generalised_mmd(
                    sample(vano, num, grid, seed=100 + trial), targets),
                "vae": generalised_mmd(
                    sample(baseline, num, seed=100 + trial), targets),
            })
    return rows


def format_table(rows):
    lines = ["| resolution | VANO (64x64) x100 | VAE (per resolution) x100 |",
             "| --- | --- | --- |"]
    summary = {}
    for resolution in RESOLUTIONS:
        cell = []
        for model in ("vano", "vae"):
            values = torch.tensor([r[model] for r in rows
                                   if r["resolution"] == resolution])
            summary[(resolution, model)] = (values.mean().item(),
                                            values.std().item())
            cell.append(f"{100 * values.mean():.2f} ± {100 * values.std():.2f}")
        lines.append(f"| {resolution}x{resolution} | {cell[0]} | {cell[1]} |")
    return "\n".join(lines), summary


def figure12(tests, device, out_dir):
    """Reconstructions of the same test function at three resolutions."""
    vano, _ = load_run(get_config("cahn_hilliard"), device)
    index = 6
    fig, axes = plt.subplots(3, 3, figsize=(8.4, 8.4))
    source = tests[64][index : index + 1]
    for row, resolution in enumerate(RESOLUTIONS):
        data = tests[resolution]
        grid = unit_grid(resolution, dim=2, device=device)
        baseline, _ = load_run(get_config(f"cahn_hilliard_vae{resolution}"), device)
        # VANO always encodes the 64x64 observation and decodes on the fine grid.
        vano_pred = reconstruct(vano, source, y=grid)[0, :, 0]
        vae_pred = reconstruct(baseline, data[index : index + 1])[0, :, 0]
        shape = (resolution, resolution)
        show_field(axes[row, 0], data.s[index, :, 0].reshape(shape).cpu())
        show_field(axes[row, 1], vano_pred.reshape(shape).cpu())
        show_field(axes[row, 2], vae_pred.reshape(shape).cpu())
        axes[row, 0].set_ylabel(f"${resolution}\\times{resolution}$")
    for ax, title in zip(axes[0], ("Ground truth", r"VANO$_{64\times64}$",
                                   "VAE (this resolution)")):
        ax.set_title(title, fontsize=13)
    fig.tight_layout()
    save(fig, out_dir / "figure12_cahn_hilliard_reconstructions.png")


def figure13(device, out_dir):
    """Prior samples: one VANO decoded everywhere, one VAE per resolution."""
    vano, _ = load_run(get_config("cahn_hilliard"), device)
    fig, axes = plt.subplots(3, 4, figsize=(11.0, 8.4))
    grid256 = unit_grid(256, dim=2, device=device)
    vano_samples = sample(vano, 3, grid256, seed=5).cpu()
    for row in range(3):
        show_field(axes[row, 0], vano_samples[row, :, 0].reshape(256, 256))
        for col, resolution in enumerate(RESOLUTIONS, start=1):
            baseline, _ = load_run(get_config(f"cahn_hilliard_vae{resolution}"),
                                   device)
            s = sample(baseline, 3, seed=5)[row, :, 0].reshape(
                resolution, resolution).cpu()
            show_field(axes[row, col], s)
    for ax, title in zip(axes[0], (r"VANO$_{64\times64}$ @ $256^2$",
                                   r"VAE$_{64}$", r"VAE$_{128}$", r"VAE$_{256}$")):
        ax.set_title(title, fontsize=13)
    fig.tight_layout()
    save(fig, out_dir / "figure13_cahn_hilliard_samples.png")


def main():
    args = base_parser(__doc__).parse_args()
    sweep(configs(), workers=args.workers, device=args.device, force=args.force)
    use_paper_style()
    args.figures.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        tests = test_sets(args.device)
        rows = table1(tests, args.device)
        table, _ = format_table(rows)
        print(table)
        (args.figures / "table1_cahn_hilliard.json").write_text(
            json.dumps(rows, indent=2))
        (args.figures / "table1_cahn_hilliard.md").write_text(table + "\n")
        vano, _ = load_run(get_config("cahn_hilliard"), args.device)
        print("VANO reconstruction relative L2 @64:",
              relative_l2(tests[64].s[:512],
                          reconstruct(vano, tests[64][:512])))
        figure12(tests, args.device, args.figures)
        figure13(args.device, args.figures)


if __name__ == "__main__":
    main()
