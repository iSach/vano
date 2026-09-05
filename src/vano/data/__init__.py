"""Datasets, all exposed as :class:`FunctionData`."""

from . import bumps, cahn_hilliard, grf, insar
from .base import FunctionData, unit_grid
from .paths import data_root, dataset_path

# The dataset draw is fixed across a seed sweep: ``Config.seed`` varies model
# initialisation only, as in the reference release.
TRAIN_SEED = 0
TEST_SEED = 1

LOADERS = {
    "grf": grf.load,
    "bumps": bumps.load,
    "cahn_hilliard": cahn_hilliard.load,
    "insar": insar.load,
}


def load_dataset(name, **kwargs):
    if name not in LOADERS:
        raise ValueError(f"unknown dataset {name!r}; pick one of {sorted(LOADERS)}")
    return LOADERS[name](**kwargs)


__all__ = ["FunctionData", "unit_grid", "load_dataset", "LOADERS",
           "TRAIN_SEED", "TEST_SEED",
           "data_root", "dataset_path", "grf", "bumps", "cahn_hilliard", "insar"]
