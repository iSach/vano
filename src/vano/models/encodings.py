"""Positional encodings applied to the decoder's query coordinates.

One encoding per benchmark, as in the release:

===================  ==========================================================
``periodic``         GRF (1D): ``[cos(2 pi y / L), sin(2 pi y / L)]``.
``fourier``          Cahn-Hilliard: random Fourier features.  The frequency
                     matrix is a *trainable* parameter initialised from
                     ``N(0, sigma^2)`` with ``sigma = 10`` -- the paper writes
                     ``sigma^2 = 10``, the code passes ``10`` as the standard
                     deviation of the initialiser, and we follow the code.
``multires``         InSAR: multi-resolution hash grid (Mueller et al. 2022).
``tile``             2D Gaussian densities: the raw coordinate repeated to the
                     decoder's width (the release's ``pos_enc={'type':'none'}``
                     branch of ``ConcatDecoder``).
``identity``         raw coordinates (the linear decoder's ``'none'`` branch).
===================  ==========================================================
"""

import math

import torch
from torch import nn


class Identity(nn.Module):
    out_dim: int

    def __init__(self, in_dim):
        super().__init__()
        self.out_dim = in_dim

    def forward(self, y):
        return y


class Tile(nn.Module):
    """Repeat the coordinate until it fills ``out_dim`` channels."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        if out_dim % in_dim:
            raise ValueError(f"{out_dim} is not a multiple of {in_dim}")
        self.reps = out_dim // in_dim
        self.out_dim = out_dim

    def forward(self, y):
        return y.repeat(*([1] * (y.dim() - 1)), self.reps)


class Periodic(nn.Module):
    """``[cos(2 pi y / L), sin(2 pi y / L)]``, optionally tiled first."""

    def __init__(self, in_dim, period=1.0, out_dim=None):
        super().__init__()
        self.tile = Tile(in_dim, out_dim // 2) if out_dim else Identity(in_dim)
        self.period = period
        self.out_dim = 2 * self.tile.out_dim

    def forward(self, y):
        y = self.tile(y) * (2.0 * math.pi / self.period)
        return torch.cat([torch.cos(y), torch.sin(y)], dim=-1)


class Fourier(nn.Module):
    """Random Fourier features with a trainable frequency matrix."""

    def __init__(self, in_dim, out_dim, scale=10.0):
        super().__init__()
        if out_dim % 2:
            raise ValueError("fourier encoding needs an even output width")
        self.kernel = nn.Parameter(scale * torch.randn(in_dim, out_dim // 2))
        self.out_dim = out_dim

    def forward(self, y):
        proj = y @ self.kernel
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


# Corner offsets of a unit voxel, in the order the reference interpolates them.
_OFFSETS = {
    1: [[0], [1]],
    2: [[0, 0], [0, 1], [1, 0], [1, 1]],
}
_PRIMES = (1, 19349663, 915850651, 2147483647)


class MultiresHash(nn.Module):
    """Multi-resolution hash-grid encoding, ``num_levels * num_features`` wide.

    Level resolutions follow the reference: ``N_l = floor(min_res * b^l)`` for
    ``l = 1 .. num_levels`` (so the coarsest level is already finer than
    ``min_res``), with ``b`` the usual geometric growth factor.
    """

    def __init__(self, in_dim, num_levels=16, min_res=16, max_res=1024,
                 hash_size=2**16, num_features=8):
        super().__init__()
        if in_dim not in _OFFSETS:
            raise ValueError("hash encoding supports 1D and 2D domains")
        b = math.exp((math.log(max_res) - math.log(min_res)) / (num_levels - 1))
        res = [int(min_res * b ** level) for level in range(1, num_levels + 1)]
        self.register_buffer("res", torch.tensor(res, dtype=torch.float32))
        self.register_buffer("offsets", torch.tensor(_OFFSETS[in_dim]))
        self.register_buffer("primes", torch.tensor(_PRIMES[:in_dim]))
        self.table = nn.Parameter(
            torch.empty(num_levels, hash_size, num_features).uniform_(-1e-4, 1e-4)
        )
        self.hash_size = hash_size
        self.in_dim = in_dim
        self.out_dim = num_levels * num_features

    def forward(self, y):
        *lead, d = y.shape
        y = y.reshape(-1, 1, d)                              # (P, 1, d)
        res = self.res.view(1, -1, 1)                        # (1, L, 1)
        lower = torch.floor(y * res)                         # (P, L, d)
        corners = lower.unsqueeze(2) + self.offsets          # (P, L, 2^d, d)
        # Spatial hash of the integer corner indices.
        index = torch.zeros_like(corners[..., 0], dtype=torch.long)
        for i in range(self.in_dim):
            index = torch.bitwise_xor(
                corners[..., i].long() * self.primes[i], index
            )
        index = index % self.hash_size
        level = torch.arange(self.table.shape[0], device=y.device).view(1, -1, 1)
        feats = self.table[level, index]                     # (P, L, 2^d, F)
        # Multilinear interpolation inside each voxel.
        frac = y * res - lower                               # (P, L, d)
        weight = torch.ones_like(feats[..., 0])              # (P, L, 2^d)
        for i in range(self.in_dim):
            f = frac[..., i : i + 1]
            on = self.offsets[:, i].to(f.dtype)
            weight = weight * (on * f + (1.0 - on) * (1.0 - f))
        out = (weight.unsqueeze(-1) * feats).sum(dim=2)      # (P, L, F)
        return out.reshape(*lead, self.out_dim)


def build_encoding(spec, in_dim, width):
    """Instantiate the encoding described by ``spec`` (a dict with a ``type``)."""
    spec = dict(spec)
    kind = spec.pop("type")
    if kind == "identity":
        return Identity(in_dim)
    if kind == "tile":
        return Tile(in_dim, spec.get("out_dim", width))
    if kind == "periodic":
        return Periodic(in_dim, spec.get("period", 1.0), spec.get("out_dim"))
    if kind == "fourier":
        return Fourier(in_dim, spec.get("out_dim", width), spec.get("scale", 10.0))
    if kind == "multires":
        return MultiresHash(in_dim, **spec)
    raise ValueError(f"unknown positional encoding {kind!r}")
