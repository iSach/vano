# VANO-dev — architecture and traps

PyTorch reimplementation of *Variational Autoencoding Neural Operators*
(Seidman et al., ICML 2023), used as a baseline for NeuralMPM work. Layout
mirrors `NeuralMPM-dev`: `src/` package, `experiments/` per-figure scripts,
`scripts/` data tooling, `pyproject.toml` with mutually exclusive `cpu`/`cu128`
torch extras.

## Where things are

* `src/vano/configs.py` — the only place hyperparameters live. One factory per
  benchmark; `Config.evolve(**dotted_overrides)` makes sweep variants.
* `src/vano/models/` — `layers.py` (Flax-faithful initialisers + random weight
  factorisation), `encodings.py`, `encoders.py`, `decoders.py`, `vano.py`.
* `src/vano/data/` — every dataset returns `FunctionData(u, y, s, w)`:
  encoder input, shared query grid, targets, per-function reconstruction weight.
* `src/vano/train.py` — one loop, no second trainer.
* `experiments/*.py` — one script per paper figure; each trains what it needs
  (skipping existing run directories) and writes into `figures/`.

## Traps

* **The Monte-Carlo axis is folded into the decoder batch.** `VANO.forward`
  takes `eps` of shape `(S, B, n)`, encodes once and decodes `S*B` latents in one
  call. The reference loops (`vmap`) over `S`; the maths is identical but the
  loop is ~7x slower here because these models are kernel-launch bound, not
  FLOP bound. `tests/test_models.py` pins the equivalence.
* **The query grid is shared.** Decoders accept `y` of shape `(P, d)` *or*
  `(B, P, d)`. Passing the shared `(P, d)` grid means positional encodings —
  especially the InSAR hash grid — are computed once instead of once per sample.
* **Random weight factorisation is per benchmark, not global.** On for the 2D
  Gaussian densities, Cahn-Hilliard and InSAR models; off for the GRF model and
  the discretise-first VAE. The release encodes this by *shadowing* its
  factorised layers with plain Flax ones at the top of some `archs.py` files —
  easy to miss.
* **Flax initialisers are not PyTorch's.** `lecun_normal`/`glorot_normal` draw
  from a *truncated* normal with a `0.8796` correction. `layers.py` reproduces
  this; do not replace it with `nn.Linear` defaults.
* **`logvar` really is a log-variance.** The reference names it `logsigma` but
  reparameterises with `sqrt(exp(logsigma))` and uses `exp(logsigma)` in the KL.
* **Field layout convention:** arrays are indexed `[i, j]` for the coordinate
  `(x_i, y_j)`; `vano.plotting.show_field` transposes for display. The reference
  flips and transposes in a couple of places; those are symmetries of these
  datasets, not information.
* **Cahn-Hilliard source data is 48 GB of float64** at 400x400.
  `scripts/download_data.py` writes `cahn_hilliard_r{64,128,256}.npy` caches;
  the loader prefers them. Its train/test split is disjoint by default, unlike
  the release's (`split_overlap=True` restores the release's behaviour).
* **GANO is not retrained.** Figure 5 compares against the sample array shipped
  in the authors' release (`GANO_samples.npy`).

## Hardware and data

Local NVIDIA RTX PRO 6000 Blackwell (96 GB), no other users. Launch training as
background local processes; do **not** use Slurm (site policy reserves job
submission for the user). Datasets live on ceph under
`/mnt/ceph/users/$USER/vano-repro/datasets`; reference code is cloned read-only
under `/mnt/home/$USER/neuralmpm/.refs/VANO`.

No datasets, checkpoints, rendered figures or logs in git.
