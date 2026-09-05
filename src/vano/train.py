"""The single training loop shared by every benchmark."""

import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from tqdm.auto import trange

from .configs import Config
from .data import load_dataset
from .models.vano import VANO, elbo_loss


def make_optimizer(model, cfg):
    """Adam with the paper's continuous exponential decay of the learning rate."""
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate,
                                 betas=tuple(cfg.betas), eps=cfg.eps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: cfg.decay_rate ** (step / cfg.decay_steps)
    )
    return optimizer, scheduler


def train(cfg: Config, device="cuda", data=None, out_dir=None, progress=True,
          log_fn=None, matmul_precision="high"):
    """Train one model and return it together with its loss history.

    ``matmul_precision`` defaults to ``"high"`` (TF32), which is also JAX's
    default on this hardware, so the reference runs used it too.  Set it to
    ``"highest"`` for strict fp32; the InSAR model is roughly 3x slower then.
    """
    torch.set_float32_matmul_precision(matmul_precision)
    torch.manual_seed(cfg.seed)
    if data is None:
        data = load_dataset(cfg.dataset, seed=cfg.seed, **cfg.dataset_kwargs)
    data = data.to(device)

    model = VANO.from_config(cfg).to(device)
    optimizer, scheduler = make_optimizer(model, cfg.optim)
    generator = torch.Generator(device=device).manual_seed(cfg.seed + 1)

    tc = cfg.training
    chunk = tc.mc_chunk or tc.num_mc_samples
    history, start = [], time.time()
    steps = trange(tc.max_steps, disable=not progress)
    for step in steps:
        idx = torch.randperm(len(data), device=device,
                             generator=generator)[: tc.batch_size]
        eps = torch.randn(tc.num_mc_samples, tc.batch_size, cfg.latent_dim,
                          device=device, generator=generator)
        optimizer.zero_grad(set_to_none=True)
        totals = {}
        for start in range(0, tc.num_mc_samples, chunk):
            piece = eps[start : start + chunk]
            weight = len(piece) / tc.num_mc_samples
            loss, parts = elbo_loss(model, data.u[idx], data.y, data.s[idx],
                                    data.w[idx], piece, cfg.beta)
            (weight * loss).backward()
            # Kept on device: reading these every step would stall the queue,
            # and these models are launch bound.
            for key, value in {"loss": loss.detach(), **parts}.items():
                totals[key] = totals.get(key, 0.0) + weight * value
        optimizer.step()
        scheduler.step()

        if step % tc.log_every == 0 or step == tc.max_steps - 1:
            record = {"step": step,
                      **{k: v.item() for k, v in totals.items()}}
            history.append(record)
            if progress:
                steps.set_postfix(recon=f"{record['recon_loss']:.2e}",
                                  kl=f"{record['kl_loss']:.2e}")
            if log_fn is not None:
                log_fn(record)

    if out_dir is not None:
        save(model, cfg, history, out_dir, wall_time=time.time() - start)
    return model, history


def save(model, cfg, history, out_dir, wall_time=None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model.pt")
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    (out_dir / "history.json").write_text(json.dumps(
        {"history": history, "wall_time_s": wall_time,
         "num_parameters": sum(p.numel() for p in model.parameters())}, indent=2))
    return out_dir


def load(out_dir, device="cuda"):
    """Rebuild a trained model from a run directory."""
    from .configs import (
        Config,
        DecoderConfig,
        EncoderConfig,
        OptimConfig,
        TrainingConfig,
    )

    out_dir = Path(out_dir)
    raw = json.loads((out_dir / "config.json").read_text())
    cfg = Config(**{**raw,
                    "encoder": EncoderConfig(**raw["encoder"]),
                    "decoder": DecoderConfig(**raw["decoder"]),
                    "optim": OptimConfig(**raw["optim"]),
                    "training": TrainingConfig(**raw["training"])})
    model = VANO.from_config(cfg).to(device)
    model.load_state_dict(torch.load(out_dir / "model.pt", map_location=device))
    model.eval()
    return model, cfg
