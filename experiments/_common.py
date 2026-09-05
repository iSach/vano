"""Helpers shared by the figure scripts: caching runs, sweeping, output paths."""

import argparse
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from vano.train import load, train

ROOT = Path(__file__).resolve().parent.parent
RUNS = Path(os.environ.get("VANO_RUNS", ROOT / "runs"))
FIGURES = Path(os.environ.get("VANO_FIGURES", ROOT / "figures"))


def run_dir(cfg):
    tag = f"{cfg.name}_n{cfg.latent_dim}_seed{cfg.seed}"
    return RUNS / cfg.dataset / tag


def train_one(cfg, device="cuda", force=False):
    """Train ``cfg`` unless its run directory already holds a model."""
    path = run_dir(cfg)
    if not force and (path / "model.pt").exists():
        return path
    train(cfg, device=device, out_dir=path, progress=False)
    return path


def _worker(payload):
    cfg, device, force = payload
    return str(train_one(cfg, device, force))


def sweep(configs, workers=1, device="cuda", force=False):
    """Train a list of configurations, optionally several at a time.

    The models in the sweeps are small and kernel-launch bound, so running a
    handful of processes against the same GPU is close to a linear speed-up.
    """
    payloads = [(cfg, device, force) for cfg in configs]
    if workers <= 1:
        return [_worker(p) for p in payloads]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_worker, payloads))


def load_run(cfg, device="cuda"):
    return load(run_dir(cfg), device=device)



def base_parser(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--workers", type=int, default=1,
                        help="training processes to run concurrently")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true", help="retrain everything")
    parser.add_argument("--figures", type=Path, default=FIGURES)
    return parser
