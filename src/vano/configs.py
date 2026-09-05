"""Every hyperparameter the paper reports, in one place.

The values below are the released ``configs/default.py`` files of
https://github.com/PredictiveIntelligenceLab/VANO, cross-checked against
Appendix Table 2 of the paper.  Where the two disagree, the code wins and the
deviation is recorded in ``docs/REPLICATION.md``.
"""

from dataclasses import dataclass, field, replace
from typing import Sequence


@dataclass
class EncoderConfig:
    name: str                       # "mlp" | "conv"
    num_layers: int = 3
    hidden_dim: int = 128
    channels: Sequence[int] = ()
    weight_fact: bool = False


@dataclass
class DecoderConfig:
    name: str                       # "linear" | "concat" | "split" | "conv"
    num_layers: int = 3
    hidden_dim: int = 128
    pos_enc: dict = field(default_factory=lambda: {"type": "identity"})
    out_activation: str = "identity"
    weight_fact: bool = False
    in_shape: Sequence[int] = ()    # "conv" decoder only
    channels: Sequence[int] = ()    # "conv" decoder only


@dataclass
class OptimConfig:
    learning_rate: float = 1e-3
    decay_rate: float = 0.9
    decay_steps: int = 1000
    betas: tuple = (0.9, 0.999)
    eps: float = 1e-8


@dataclass
class TrainingConfig:
    batch_size: int = 32
    num_mc_samples: int = 4
    max_steps: int = 20000
    log_every: int = 100


@dataclass
class Config:
    name: str
    dataset: str
    input_shape: Sequence[int]      # encoder input, (P,) or (H, W, C)
    query_dim: int
    out_dim: int
    latent_dim: int
    beta: float
    encoder: EncoderConfig
    decoder: DecoderConfig
    seed: int = 0
    dataset_kwargs: dict = field(default_factory=dict)
    optim: OptimConfig = field(default_factory=OptimConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def evolve(self, **overrides):
        """Return a copy with dotted overrides applied, e.g. ``training.max_steps``."""
        nested, flat = {}, {}
        for key, value in overrides.items():
            if "." in key:
                group, attr = key.split(".", 1)
                nested.setdefault(group, {})[attr] = value
            else:
                flat[key] = value
        for group, values in nested.items():
            flat[group] = replace(getattr(self, group), **values)
        return replace(self, **flat)


def _grf():
    return Config(
        name="grf", dataset="grf", input_shape=(128,), query_dim=1, out_dim=1,
        latent_dim=64, beta=5e-6, seed=2,
        encoder=EncoderConfig("mlp", num_layers=3, hidden_dim=128),
        decoder=DecoderConfig(
            "linear", num_layers=3, hidden_dim=128,
            pos_enc={"type": "periodic", "period": 1.0},
        ),
        training=TrainingConfig(batch_size=32, num_mc_samples=16, max_steps=40000),
    )


def _bumps(decoder_name):
    return Config(
        name=f"bumps_{decoder_name}", dataset="bumps", input_shape=(48, 48, 1),
        query_dim=2, out_dim=1, latent_dim=32, beta=1e-5, seed=4,
        encoder=EncoderConfig("conv", channels=(8, 16, 32, 64), weight_fact=True),
        decoder=DecoderConfig(
            decoder_name, num_layers=3, hidden_dim=128,
            pos_enc={"type": "identity" if decoder_name == "linear" else "tile"},
            out_activation="softplus", weight_fact=True,
        ),
        training=TrainingConfig(batch_size=32, num_mc_samples=4, max_steps=20000),
    )


def _cahn_hilliard():
    return Config(
        name="cahn_hilliard", dataset="cahn_hilliard", input_shape=(64, 64, 1),
        query_dim=2, out_dim=1, latent_dim=64, beta=1e-4, seed=0,
        encoder=EncoderConfig("conv", channels=(8, 16, 32, 64, 128),
                              weight_fact=True),
        decoder=DecoderConfig(
            "concat", num_layers=4, hidden_dim=256,
            pos_enc={"type": "fourier", "scale": 10.0},
            out_activation="sigmoid", weight_fact=True,
        ),
        training=TrainingConfig(batch_size=16, num_mc_samples=4, max_steps=20000),
    )


def _cahn_hilliard_vae(resolution):
    """Discretise-first convolutional VAE baseline, one per resolution."""
    return Config(
        name=f"cahn_hilliard_vae{resolution}", dataset="cahn_hilliard",
        input_shape=(resolution, resolution, 1), query_dim=2, out_dim=1,
        latent_dim=64, beta=1e-4, seed=0,
        dataset_kwargs={"resolution": resolution},
        encoder=EncoderConfig("conv", channels=(8, 16, 32, 64, 128)),
        decoder=DecoderConfig(
            "conv", in_shape=(resolution // 32, resolution // 32, 128),
            channels=(64, 32, 16, 8, 1), out_activation="sigmoid",
        ),
        training=TrainingConfig(batch_size=16, num_mc_samples=4, max_steps=20000),
    )


def _insar():
    return Config(
        name="insar", dataset="insar", input_shape=(128, 128, 2), query_dim=2,
        out_dim=2, latent_dim=256, beta=1e-4, seed=0,
        encoder=EncoderConfig("conv", channels=(8, 16, 32, 64, 128, 256),
                              weight_fact=True),
        decoder=DecoderConfig(
            "split", num_layers=8, hidden_dim=512,
            pos_enc={"type": "multires", "num_levels": 16, "min_res": 16,
                     "max_res": 1024, "hash_size": 2**16, "num_features": 8},
            weight_fact=True,
        ),
        training=TrainingConfig(batch_size=16, num_mc_samples=4, max_steps=25000),
    )


CONFIGS = {
    "grf": _grf,
    "bumps_linear": lambda: _bumps("linear"),
    "bumps_concat": lambda: _bumps("concat"),
    "cahn_hilliard": _cahn_hilliard,
    "cahn_hilliard_vae64": lambda: _cahn_hilliard_vae(64),
    "cahn_hilliard_vae128": lambda: _cahn_hilliard_vae(128),
    "cahn_hilliard_vae256": lambda: _cahn_hilliard_vae(256),
    "insar": _insar,
}


def get_config(name):
    if name not in CONFIGS:
        raise ValueError(f"unknown config {name!r}; pick one of {sorted(CONFIGS)}")
    return CONFIGS[name]()
