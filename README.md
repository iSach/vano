# VANO

A PyTorch reimplementation of

> **Variational Autoencoding Neural Operators**
> Jacob H. Seidman, Georgios Kissas, George J. Pappas, Paris Perdikaris — ICML 2023
> [arXiv:2302.10351](https://arxiv.org/abs/2302.10351)

VANO turns an encoder–decoder neural operator into a variational autoencoder for
*functional* data. An encoder maps a function, observed on some grid, to a
Gaussian over a finite-dimensional latent space; a **pointwise** decoder turns a
latent sample and a query coordinate into a function value. Because the decoder
never sees the grid, the learned generative model is discretisation agnostic:
train at 64×64, sample at 256×256.

This repository reproduces the paper's four benchmarks from our own runs.
The reference JAX release is
[PredictiveIntelligenceLab/VANO](https://github.com/PredictiveIntelligenceLab/VANO);
every place where it and the paper disagree is listed in
[`docs/REPLICATION.md`](docs/REPLICATION.md).

## Install

```bash
uv sync --extra cu128            # or --extra cpu
```

The two accelerator extras are mutually exclusive and resolve from the explicit
PyTorch indexes recorded in `pyproject.toml`.

## Data

The GRF and 2D Gaussian-density benchmarks are generated analytically — nothing
to download. Cahn-Hilliard and InSAR use the authors' data release:

```bash
uv run python scripts/download_data.py --root /path/to/scratch/vano-repro
export VANO_DATA_ROOT=/path/to/scratch/vano-repro/datasets
```

That fetches a 2.3 GB archive, unpacks ~50 GB and writes one resampled
Cahn-Hilliard cache per resolution.

## Train

```bash
uv run vano-train grf           --out runs/grf/grf_n64_seed2
uv run vano-train cahn_hilliard --out runs/ch --set training.max_steps=20000
uv run vano-evaluate runs/ch
```

`vano-train` takes any configuration in `src/vano/configs.py` and any dotted
override (`--set latent_dim=32 seed=3 optim.learning_rate=5e-4`). Every
hyperparameter the paper reports lives in that one file.

## Reproduce the figures

Each script trains what it needs (skipping runs that already exist) and writes
into `figures/`:

| script | reproduces |
| --- | --- |
| `experiments/grf_basis.py` | Figure 3 (covariance recovery, learned vs KL basis), Figure 6 |
| `experiments/bumps_decoders.py` | Figure 4 (MMD, linear vs nonlinear), Figures 7–11 |
| `experiments/cahn_hilliard_superres.py` | Table 1, Figures 12–13 (super-resolution vs a discretise-first VAE) |
| `experiments/insar_volcano.py` | Figure 5 (circular statistics vs GANO), Figures 14–16 |
| `experiments/cost_table.py` | Appendix Tables 3–4 (parameters and training cost) |

```bash
uv run python experiments/grf_basis.py --workers 8
```

The sweeps are launch-bound rather than FLOP-bound, so `--workers` runs several
training processes against the same GPU and scales close to linearly.

Measured numbers, wall-clock costs and every deviation from the paper are in
[`docs/REPLICATION.md`](docs/REPLICATION.md).

## Layout

```
src/vano/
  configs.py     every benchmark's hyperparameters, one dataclass each
  data/          the four datasets, all exposed as FunctionData(u, y, s, w)
  models/        layers (incl. random weight factorisation), encodings,
                 encoders, decoders (linear/concat/split/conv), the VANO module
  metrics/       generalised MMD, covariance-operator error, circular statistics
  train.py       one training loop
  evaluate.py    per-benchmark metrics
  cli/           vano-train, vano-evaluate
experiments/     one script per paper figure
scripts/         data download and preparation
tests/           pytest
```
