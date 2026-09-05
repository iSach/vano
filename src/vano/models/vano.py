"""The VANO model and its (discretisation-agnostic) variational objective."""

import torch
from torch import nn

from .decoders import build_decoder
from .encoders import build_encoder


class VANO(nn.Module):
    """Variational autoencoding neural operator.

    ``encoder`` maps an input function, observed on a fixed grid, to a Gaussian
    over ``R^n``; ``decoder`` maps a latent sample and a query coordinate to a
    function value.  Because the decoder is pointwise, the query grid at
    evaluation time is free: the model can be trained at one resolution and
    sampled at another (sections 6.2 and 6.3 of the paper).
    """

    def __init__(self, encoder, decoder, latent_dim):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim

    @classmethod
    def from_config(cls, cfg):
        encoder = build_encoder(cfg.encoder, cfg.input_shape, cfg.latent_dim)
        decoder = build_decoder(cfg.decoder, cfg.latent_dim, cfg.query_dim,
                                cfg.out_dim)
        return cls(encoder, decoder, cfg.latent_dim)

    def encode(self, u):
        return self.encoder(u)

    def decode(self, z, y=None):
        return self.decoder(z, y)

    def sample_prior(self, num_samples, y=None, device=None, generator=None):
        z = torch.randn(num_samples, self.latent_dim, device=device,
                        generator=generator)
        return self.decode(z, y)

    def forward(self, u, y, eps):
        """Reconstruct ``u`` on ``y`` for every Monte-Carlo draw in ``eps``.

        ``eps`` is ``(S, B, n)``; the reconstruction comes back as
        ``(S, B, P, c)``.  The encoder runs once and the ``S`` draws are folded
        into the decoder's batch axis, which is what makes the ``S = 16`` GRF
        setting affordable.
        """
        mu, logvar = self.encode(u)
        z = mu + eps * torch.exp(0.5 * logvar)
        num_mc, batch, latent = z.shape
        pred = self.decode(z.reshape(num_mc * batch, latent), y)
        return pred.reshape(num_mc, batch, *pred.shape[1:]), mu, logvar


def kl_divergence(mu, logvar):
    """KL(q(z|u) || N(0, I)), summed over latent dimensions."""
    return 0.5 * (logvar.exp() + mu.square() - 1.0 - logvar).sum(-1)


def elbo_loss(model, u, y, s, w, eps, beta):
    """The training objective, exactly as implemented in the reference release.

    ``eps`` has a leading Monte-Carlo axis of size ``S``.  The reconstruction
    term is a *mean* over query points -- the discrete stand-in for the ``L^2``
    norm of the residual -- rescaled per function by ``w``; ``beta`` absorbs the
    likelihood variance and the domain measure, so it is a plain KL weight.
    """
    pred, mu, logvar = model(u, y, eps)
    residual = 0.5 * (s - pred).square().mean(dim=tuple(range(2, pred.dim())))
    recon = (w * residual).mean()
    kl = kl_divergence(mu, logvar).mean()
    return recon + beta * kl, {"recon_loss": recon.detach(), "kl_loss": kl.detach()}
