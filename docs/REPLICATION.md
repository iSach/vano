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

<!-- RESULTS:GRF -->

### Figure 4 -- 2D Gaussian densities, linear vs nonlinear decoder

<!-- RESULTS:BUMPS -->

### Table 1 -- Cahn-Hilliard, generalised MMD (x100)

<!-- RESULTS:CH -->

### Figure 5 -- InSAR, directional statistics

<!-- RESULTS:INSAR -->

## Cost

<!-- RESULTS:COST -->
