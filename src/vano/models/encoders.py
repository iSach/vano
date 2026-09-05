"""Amortised encoders: an input function on a fixed grid -> q(z | u) = N(mu, sigma^2)."""

import math

import torch
from torch import nn

from .layers import Conv2d, Dense


class GaussianHead(nn.Module):
    """Two linear heads producing the mean and the log-variance of q(z | u)."""

    def __init__(self, in_dim, latent_dim, weight_fact=False):
        super().__init__()
        self.mu = Dense(in_dim, latent_dim, weight_fact)
        self.logvar = Dense(in_dim, latent_dim, weight_fact)

    def forward(self, h):
        return self.mu(h), self.logvar(h)


class MlpEncoder(nn.Module):
    """Encoder for functions sampled on a fixed 1D grid (GRF benchmark)."""

    def __init__(self, num_points, latent_dim, num_layers=3, hidden_dim=128,
                 activation=nn.GELU, weight_fact=False):
        super().__init__()
        layers, width = [], num_points
        for _ in range(num_layers):
            layers += [Dense(width, hidden_dim, weight_fact), activation()]
            width = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.head = GaussianHead(width, latent_dim, weight_fact)

    def forward(self, u):
        return self.head(self.trunk(u.flatten(1)))


class ConvEncoder(nn.Module):
    """VGG-style encoder: 2x2 stride-2 convolutions, then a Gaussian head."""

    def __init__(self, in_shape, latent_dim, channels=(8, 16, 32, 64),
                 activation=nn.GELU, weight_fact=False):
        super().__init__()
        height, width, in_channels = in_shape
        blocks, c = [], in_channels
        for out_channels in channels:
            blocks += [Conv2d(c, out_channels, weight_fact=weight_fact),
                       activation()]
            c = out_channels
        self.blocks = nn.Sequential(*blocks)
        with torch.no_grad():
            probe = self.blocks(torch.zeros(1, in_channels, height, width))
        self.head = GaussianHead(probe.numel(), latent_dim, weight_fact)

    def forward(self, u):
        # (B, H, W, C) images, channels-last as in the reference data pipeline.
        h = self.blocks(u.permute(0, 3, 1, 2))
        return self.head(h.permute(0, 2, 3, 1).flatten(1))


def build_encoder(cfg, in_shape, latent_dim):
    if cfg.name == "mlp":
        return MlpEncoder(math.prod(in_shape), latent_dim,
                          cfg.num_layers, cfg.hidden_dim,
                          weight_fact=cfg.weight_fact)
    if cfg.name == "conv":
        return ConvEncoder(in_shape, latent_dim, tuple(cfg.channels),
                           weight_fact=cfg.weight_fact)
    raise ValueError(f"unknown encoder {cfg.name!r}")
