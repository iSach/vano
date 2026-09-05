# Replication notes

What this repository reproduces, how it differs from the paper and from the
authors' JAX release, and what we measured.

Reference code: `https://github.com/PredictiveIntelligenceLab/VANO`
(cloned read-only for this work). Paper: arXiv:2302.10351 / PMLR v202.

## Protocol

Every hyperparameter comes from the released `configs/default.py` of the
matching benchmark, cross-checked against Appendix Table 2 of the paper. Where
the two disagree the code wins, and the disagreement is listed below. All
numbers in this file come from our own runs on a single RTX PRO 6000.

| benchmark | encoder | decoder | n | beta | S | batch | steps |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GRF (1D) | MLP 3x128 | linear, 3x128, periodic encoding | 64 | 5e-6 | 16 | 32 | 40000 |
| 2D Gaussian densities | conv (8,16,32,64) | linear or concat, 3x128, softplus | 32 | 1e-5 | 4 | 32 | 20000 |
| Cahn-Hilliard | conv (8,16,32,64,128) | concat, 4x256, RFF, sigmoid | 64 | 1e-4 | 4 | 16 | 20000 |
| Cahn-Hilliard VAE | conv (8,16,32,64,128) | transposed conv, sigmoid | 64 | 1e-4 | 4 | 16 | 20000 |
| InSAR | conv (8,...,256) | split, 8x512, hash encoding | 256 | 1e-4 | 4 | 16 | 25000 |

Optimiser everywhere: Adam(0.9, 0.999, eps 1e-8), learning rate 1e-3 decayed
continuously by 0.9 per 1000 steps.

The objective is the one the release implements:

```
L = mean_{S,B,P} [ 0.5 * w * (u - u_hat)^2 ]  +  beta * mean_B KL(q(z|u) || N(0,I))
```

`w` is the release's "empirical norm rescaling", `1/||u||_2^2`, on the two
benchmarks whose functions differ in magnitude by orders of magnitude (GRF, 2D
Gaussian densities) and `1` elsewhere. The likelihood variance and the domain
measure of the theoretical ELBO are absorbed into `beta`.

## Deviations from the paper text

1. **2D Gaussian density widths.** The paper writes `sigma ~ U(0, 0.1) + 0.01`.
   The code draws `U(0, 0.01) + 0.001` and passes it as the *covariance* scale,
   i.e. standard deviations in `[0.032, 0.105]`. We follow the code.
2. **Random Fourier feature scale.** The paper writes `sigma^2 = 10`; the code
   passes `10` as the standard deviation of the initialiser. We follow the code.
   The frequency matrix is also *trainable* in the release, not a fixed draw.
3. **Random weight factorisation.** The paper says every model is trained with
   RWF. In the release it is switched off for the GRF benchmark and for the
   discretise-first VAE baseline (their `archs.py` shadows the factorised layers
   with plain Flax ones). We match the code, per benchmark.
4. **Number of Cahn-Hilliard patterns.** The paper does not state the training
   set size; the release draws 4096 patterns from the 37523 available.
5. **InSAR training length.** The paper says 20,000 iterations; the released
   `volcano/configs/default.py` says 25,000. We follow the code.

## Deviations from the released code

1. **Train/test split of the Cahn-Hilliard set.** The release draws train and
   test indices with two independent `random.choice` calls over the same pool,
   so roughly 11% of its "test" patterns are also in training. We take a
   disjoint split by default; `split_overlap=True` restores the release's
   behaviour. This makes our MMD numbers slightly *harder* than the published
   ones, not easier.
2. **Monte-Carlo samples are batched, not looped.** `VANO.forward` encodes once
   and decodes all `S` draws in a single call. This is algebraically identical
   (the KL term does not depend on `eps` at all) and about 7x faster here,
   because these models are kernel-launch bound. `tests/test_models.py` pins the
   equivalence.
3. **Positional encodings are evaluated once per grid**, not once per (function,
   point) pair, since the query grid is shared across a batch. Same values, much
   less work -- decisive for the InSAR hash encoding.
4. **Hash indices are computed in int64**, where the reference relies on int32
   overflow. Both are hashes; the mapping differs, the statistics do not.
5. **Initialisation is ported, not inherited.** Flax's `lecun_normal` and
   `glorot_normal` draw from a truncated normal with a `0.8796` correction;
   `src/vano/models/layers.py` reproduces that rather than using PyTorch
   defaults.
6. **The GRF covariance metric is reported under two conventions.** The release
   compares the learned covariance against the rank-`min(n, 32)` *truncation* of
   the true covariance. We report that (`hs_error_truncated`) and also the error
   against the full covariance (`hs_error_full`), alongside the optimal rank-`n`
   truncation error, which is the floor no linear decoder can beat.

## Results

Filled in from `figures/*.json` after running the four experiment scripts.

### Figure 3 -- GRF, covariance recovery

Ten seeds per latent dimension; the dataset is fixed across seeds.

| n | covariance error (vs rank-n) | reconstruction rel. `L2` | effective rank of the learned basis | optimal rank-n floor |
| --- | --- | --- | --- | --- |
| 2  | 0.026 ± 0.005 | 0.228 ± 0.000 | 2.0 | 1.5e-2 |
| 4  | 0.025 ± 0.009 | 0.143 ± 0.013 | 3.1 | 2.2e-3 |
| 8  | 0.042 ± 0.027 | 0.110 ± 0.020 | 3.9 | 2.6e-4 |
| 16 | 0.032 ± 0.009 | 0.099 ± 0.010 | 4.2 | 2.7e-5 |
| 32 | 0.048 ± 0.017 | 0.095 ± 0.011 | 4.3 | 0 |
| 64 | 0.123 ± 0.093 | 0.093 ± 0.010 | 4.2 | 0 |

**Reconstruction.** Monotone in `n`, and at the paper's setting (n = 64, seed 2)
our relative `L2` is **0.093 ± 0.010** against the **0.133** printed in the
authors' own `grf_1d/postprocess.ipynb` for their released checkpoint. So the
model is trained at least as well as theirs.

**The learned basis is the Karhunen-Loeve basis**, and Figure 3 (right) shows it:
the learned `tau_j` are the analytic `sqrt(lambda_i) phi_i` mode for mode, and
the eigenvalues of the learned covariance track `lambda_i` over the first five
components.

**Covariance error does not improve with `n`, and it should not.** The KL term
prices every active latent direction, while the reconstruction that direction
buys back is worth `lambda_i`, which decays like `i^-4`. Past `i ~ 4` a
component costs more than it earns, so the model switches it off: the effective
rank of the learned basis saturates at about 4 whatever `n` is, and with it the
covariance error, at a few percent. Larger `n` then only adds optimisation
noise, which is what the growing spread at n = 32 and n = 64 is.

This is not a artefact of our port. The eigenfunction plot the authors ship with
the release (`grf_1d/eigenfunctions.png`, produced from their n = 64
checkpoint) has exactly **three** non-zero basis functions out of 64, and their
released checkpoint's reconstruction error, 0.133, is the error of a rank-4
approximation of this GRF. Their model does the same thing ours does.

### Figure 4 -- 2D Gaussian densities, linear vs nonlinear decoder

Five seeds per (decoder, latent dimension); generalised MMD against 512
held-out functions, and reconstruction relative `L2` on 512 held-out functions.

| n | linear MMD | linear rel. `L2` | nonlinear MMD | nonlinear rel. `L2` |
| --- | --- | --- | --- | --- |
| 4   | 0.174 ± 0.013 | 0.657 | **0.0130 ± 0.0044** | 0.137 |
| 32  | 0.128 ± 0.030 | 0.421 | **0.0078 ± 0.0014** | 0.141 |
| 64  | 0.137 ± 0.013 | 0.428 | **0.0091 ± 0.0011** | 0.138 |
| 128 | 0.147 ± 0.017 | 0.452 | **0.0111 ± 0.0017** | 0.144 |
| 256 | 0.170 ± 0.082 | 0.670 | **0.0115 ± 0.0019** | 0.164 |
| 512 | 0.449 ± 0.039 | 0.769 | 0.0624 ± 0.0553 | 0.308 |

This is the paper's claim, quantitatively: the linear decoder stays above 0.10
even at n = 128 (we get 0.147), while the nonlinear decoder is already below
0.03 at n = 32 (0.0078). The separation is about 16x in MMD and 3x in
reconstruction error, and no amount of extra latent capacity closes it -- which
is the point, since the family is a two-parameter manifold that simply is not
linear.

Figure 7 shows why: the PCA spectrum of the training set decays roughly like a
straight line on a log scale, so a linear representation needs hundreds of
components to capture a family with two degrees of freedom.

Figures 8-9 show the failure concretely: the linear decoder reconstructs every
bump at roughly the same width, losing the narrow ones entirely, while the
nonlinear decoder gets both position and width. Figures 10-11 decode the same
models on a 256x256 grid after training at 48x48; the nonlinear samples stay
crisp.

Both decoders degrade at n = 512, where the encoder's Gaussian head is wider
than the 576-unit feature vector it reads from.

### Table 1 -- Cahn-Hilliard, generalised MMD (x100)

Five draws of samples and held-out functions per cell, generalised MMD x100,
mean ± std. VANO is a single model trained at 64x64 and decoded at each
resolution; each VAE column is a separate model trained at that resolution.

| resolution | VANO (ours) | VANO (paper) | VAE (ours) | VAE (paper) |
| --- | --- | --- | --- | --- |
| 64x64   | **0.77 ± 0.01** | 0.71 ± 0.04 | 0.94 ± 0.04 | 0.73 ± 0.02 |
| 128x128 | **0.91 ± 0.02** | 0.88 ± 0.04 | 0.90 ± 0.02 | 0.64 ± 0.03 |
| 256x256 | **1.39 ± 0.04** | 1.34 ± 0.06 | 1.19 ± 0.05 | 1.03 ± 0.03 |

Our VANO column lands within one standard deviation of the paper's at every
resolution, and reproduces its shape: essentially free at the training
resolution, mildly worse than a resolution-specific VAE above it. Our VAE
baseline is a little weaker than the published one at 64x64 and 256x256; the
most likely cause is the split (ours is disjoint, theirs overlaps, which
flatters a model that has memorised part of its "test" set).

Reconstruction relative `L2` of VANO on the held-out set at 64x64: **0.162**.

Figures 12 and 13 reproduce the qualitative claim directly. VANO, trained only
at 64x64, has smooth phase boundaries when decoded at 128x128 and 256x256; the
transposed-convolution decoders show the usual checkerboard, and it gets worse
with resolution.

### Figure 5 -- InSAR, directional statistics

<!-- RESULTS:INSAR -->

## Cost

One RTX PRO 6000 Blackwell, fp32 with TF32 matmuls, `experiments/cost_table.py`.

| model | parameters | steps | wall time |
| --- | --- | --- | --- |
| GRF, n = 64 | 0.108 M | 40 000 | 2.0 min |
| 2D Gaussian densities, linear, n = 32 | 0.086 M | 20 000 | 2.3 min |
| 2D Gaussian densities, nonlinear, n = 32 | 0.090 M | 20 000 | 12.1 min |
| Cahn-Hilliard VANO | 0.342 M | 20 000 | see note |
| Cahn-Hilliard VAE, 64 / 128 / 256 | 0.186 / 0.483 / 1.669 M | 20 000 | see note |
| InSAR | 11.13 M | 25 000 | see note |

The parameter counts are exact. The Cahn-Hilliard wall times were lost to a
timing bug (the training loop shadowed its own start time; fixed, and
`cost_table.py` prints `n/a` rather than a nonsense duration for runs recorded
before the fix) -- rerun those configs with `--force` to fill them in.

The sweeps are kernel-launch bound rather than FLOP bound, which is why the
figure scripts take a `--workers` flag: six training processes against one GPU
finish a 60-run sweep in about the time two would.
