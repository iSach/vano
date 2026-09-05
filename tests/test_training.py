import json

import pytest
import torch

from vano.cli.train import main as train_main
from vano.cli.train import parse_overrides
from vano.configs import get_config
from vano.data import bumps, grf
from vano.evaluate import reconstruct, relative_l2
from vano.train import load, train

pytestmark = pytest.mark.slow


def test_overrides_are_parsed_and_typed():
    parsed = parse_overrides(["latent_dim=8", "training.max_steps=3", "beta=1e-4"])
    assert parsed == {"latent_dim": 8, "training.max_steps": 3, "beta": 1e-4}
    cfg = get_config("grf").evolve(**parsed)
    assert (cfg.latent_dim, cfg.training.max_steps, cfg.beta) == (8, 3, 1e-4)


@pytest.mark.parametrize("name,loader,factor", [("grf", grf.load, 0.2),
                                                ("bumps_concat", bumps.load, 0.5)])
def test_short_run_reduces_the_reconstruction_loss(name, loader, factor):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = get_config(name).evolve(latent_dim=8, **{"training.max_steps": 1000,
                                                   "training.log_every": 50})
    data = loader(256, seed=0)
    _, history = train(cfg, device=device, data=data, progress=False)
    assert history[-1]["recon_loss"] < factor * history[0]["recon_loss"]


def test_cli_round_trip(tmp_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = tmp_path / "run"
    train_main(["grf", "--out", str(out), "--quiet", "--device", device,
                "--set", "latent_dim=4", "training.max_steps=20"])
    assert json.loads((out / "history.json").read_text())["num_parameters"] > 0
    model, cfg = load(out, device=device)
    assert cfg.latent_dim == 4
    data = grf.load(16, seed=1).to(device)
    with torch.no_grad():
        assert 0.0 < relative_l2(data.s, reconstruct(model, data)) < 100.0
