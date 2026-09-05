# Experiments

One script per paper figure. Each trains whatever runs it needs — skipping any
run directory that already holds a `model.pt` — and writes both the figure and
the raw numbers (`*.json`) into `figures/`.

```bash
uv run python experiments/grf_basis.py               --workers 8
uv run python experiments/bumps_decoders.py          --workers 8
uv run python experiments/cahn_hilliard_superres.py  --workers 2
uv run python experiments/insar_volcano.py
```

| script | paper output | runs it trains |
| --- | --- | --- |
| `grf_basis.py` | Figure 3, Figure 6 | 6 latent dims x 10 seeds |
| `bumps_decoders.py` | Figure 4, Figures 7–11 | 2 decoders x 6 latent dims x 5 seeds |
| `cahn_hilliard_superres.py` | Table 1, Figures 12–13 | 1 VANO + 3 VAE baselines |
| `insar_volcano.py` | Figure 5, Figures 14–16 | 1 VANO (GANO samples come from the authors' release) |

`--workers N` runs `N` training processes against the same GPU. The sweep models
are small and kernel-launch bound, so this scales close to linearly; use `1` for
the Cahn-Hilliard and InSAR models, which are large enough to saturate the
device on their own.

Common flags: `--force` retrains everything, `--figures DIR` changes the output
directory, `--device` selects the accelerator. `VANO_RUNS` and `VANO_FIGURES`
override the default locations.
