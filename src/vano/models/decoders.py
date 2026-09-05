"""Decoders mapping a latent code and a query coordinate to a function value.

Every decoder here is *pointwise*: it is evaluated independently at each query
coordinate, which is what makes the model discretisation agnostic -- the same
weights evaluate the reconstruction on any grid, at any resolution.

``ConvDecoder`` is the exception; it is the discretise-first VAE baseline of
section 6.3 and is tied to the resolution it was trained at.
"""

import math

import torch
from torch import nn

from .encodings import build_encoding
from .layers import MLP, ConvTranspose2d, Dense

ACTIVATIONS = {
    "identity": nn.Identity,
    "sigmoid": nn.Sigmoid,
    "softplus": nn.Softplus,
    "gelu": nn.GELU,
}


def _broadcast(z, y_enc):
    """Pair ``z`` (B, n) with ``y_enc`` (P, E) or (B, P, E)."""
    if y_enc.dim() == 2:
        y_enc = y_enc.unsqueeze(0).expand(z.shape[0], -1, -1)
    return z.unsqueeze(1).expand(-1, y_enc.shape[1], -1), y_enc


class LinearDecoder(nn.Module):
    """``u(y) = sum_j z_j tau_j(y)``: a learned basis, linear in the latent code.

    With a Gaussian prior on ``z`` this is exactly a learned Karhunen-Loeve
    expansion, which is what section 6.1 exploits to compare the learned basis
    against the analytically optimal one.
    """

    def __init__(self, latent_dim, query_dim, pos_enc, num_layers=3,
                 hidden_dim=128, out_activation="identity", weight_fact=False):
        super().__init__()
        self.encoding = build_encoding(pos_enc, query_dim, latent_dim)
        self.basis = MLP(self.encoding.out_dim, num_layers, hidden_dim,
                         latent_dim, weight_fact=weight_fact)
        self.out_activation = ACTIVATIONS[out_activation]()

    def trunk(self, y):
        """The learned basis functions ``tau(y)``, shape (..., latent_dim)."""
        return self.basis(self.encoding(y))

    def forward(self, z, y):
        z, tau = _broadcast(z, self.trunk(y))
        return self.out_activation((z * tau).sum(-1, keepdim=True))


class ConcatDecoder(nn.Module):
    """``u(y) = MLP([z, gamma(y)])``: the simplest nonlinear conditioning."""

    def __init__(self, latent_dim, query_dim, pos_enc, num_layers=3,
                 hidden_dim=128, out_dim=1, out_activation="identity",
                 weight_fact=False):
        super().__init__()
        self.encoding = build_encoding(pos_enc, query_dim, latent_dim)
        self.mlp = MLP(latent_dim + self.encoding.out_dim, num_layers,
                       hidden_dim, out_dim, weight_fact=weight_fact)
        self.out_activation = ACTIVATIONS[out_activation]()

    def forward(self, z, y):
        z, y_enc = _broadcast(z, self.encoding(y))
        return self.out_activation(self.mlp(torch.cat([z, y_enc], dim=-1)))


class SplitDecoder(nn.Module):
    """Splits ``z`` into per-layer chunks re-injected at every hidden layer.

    Deep coordinate MLPs otherwise forget the conditioning signal; this is the
    decoder the paper uses for the 8-layer InSAR model.
    """

    def __init__(self, latent_dim, query_dim, pos_enc, num_layers=8,
                 hidden_dim=512, out_dim=1, out_activation="identity",
                 weight_fact=False, activation=nn.GELU):
        super().__init__()
        if latent_dim % num_layers:
            raise ValueError("latent_dim must be divisible by num_layers")
        self.chunk = latent_dim // num_layers
        self.encoding = build_encoding(pos_enc, query_dim, self.chunk)
        layers, width = [], self.encoding.out_dim
        for _ in range(num_layers):
            layers.append(Dense(width + self.chunk, hidden_dim, weight_fact))
            width = hidden_dim
        self.layers = nn.ModuleList(layers)
        self.act = activation()
        self.head = Dense(width, out_dim, weight_fact)
        self.out_activation = ACTIVATIONS[out_activation]()

    def forward(self, z, y):
        z, h = _broadcast(z, self.encoding(y))
        for i, layer in enumerate(self.layers):
            chunk = z[..., i * self.chunk : (i + 1) * self.chunk]
            h = self.act(layer(torch.cat([h, chunk], dim=-1)))
        return self.out_activation(self.head(h))


class ConvDecoder(nn.Module):
    """Discretise-first baseline: transposed convolutions to a fixed grid."""

    def __init__(self, latent_dim, in_shape, channels=(64, 32, 16, 8, 1),
                 out_activation="identity", activation=nn.GELU,
                 weight_fact=False):
        super().__init__()
        self.in_shape = tuple(in_shape)  # (H, W, C)
        self.project = Dense(latent_dim, math.prod(in_shape), weight_fact)
        blocks, c = [], in_shape[-1]
        for i, out_channels in enumerate(channels):
            blocks.append(ConvTranspose2d(c, out_channels,
                                          weight_fact=weight_fact))
            if i < len(channels) - 1:
                blocks.append(activation())
            c = out_channels
        self.blocks = nn.Sequential(*blocks)
        self.out_activation = ACTIVATIONS[out_activation]()

    def forward(self, z, y=None):
        h, w, c = self.in_shape
        x = self.project(z).view(-1, h, w, c).permute(0, 3, 1, 2)
        x = self.blocks(x).permute(0, 2, 3, 1)
        return self.out_activation(x).flatten(1, 2)


def build_decoder(cfg, latent_dim, query_dim, out_dim):
    kwargs = dict(latent_dim=latent_dim, query_dim=query_dim,
                  pos_enc=cfg.pos_enc, num_layers=cfg.num_layers,
                  hidden_dim=cfg.hidden_dim, out_activation=cfg.out_activation,
                  weight_fact=cfg.weight_fact)
    if cfg.name == "linear":
        if out_dim != 1:
            raise ValueError("the linear decoder is single-channel")
        return LinearDecoder(**kwargs)
    if cfg.name == "concat":
        return ConcatDecoder(out_dim=out_dim, **kwargs)
    if cfg.name == "split":
        return SplitDecoder(out_dim=out_dim, **kwargs)
    if cfg.name == "conv":
        return ConvDecoder(latent_dim, cfg.in_shape, tuple(cfg.channels),
                           cfg.out_activation, weight_fact=cfg.weight_fact)
    raise ValueError(f"unknown decoder {cfg.name!r}")
