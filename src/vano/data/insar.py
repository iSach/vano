"""Sentinel-1 interferograms over the Long Valley Caldera (paper section 6.4).

Each observation is a wrapped phase field ``phi(x) in [-pi, pi]``.  Angles do
not live in a vector space, so the model regresses the embedding
``u(x) = (cos phi(x), sin phi(x))`` and the phase is read back with ``atan2``;
this is what the released ``volcano.npy`` stores.
"""

import numpy as np
import torch

from .base import FunctionData, unit_grid
from .paths import dataset_path

FILENAME = "volcano.npy"
GANO_SAMPLES = "GANO_samples.npy"
RESOLUTION = 128


def phase(field):
    """Wrapped phase of a ``(..., 2)`` cosine/sine field."""
    return torch.atan2(field[..., 1], field[..., 0])


def load(num_samples=None, path=None, seed=None):
    """Load the interferogram stack.  ``seed`` is accepted and ignored: the
    release trains on all 4096 fields and holds nothing out."""
    path = path or dataset_path(FILENAME)
    array = np.load(path)
    if num_samples is not None:
        array = array[:num_samples]
    u = torch.from_numpy(np.ascontiguousarray(array)).float()
    resolution = u.shape[1]
    y = unit_grid(resolution, dim=2)
    s = u.reshape(len(u), -1, u.shape[-1])
    w = torch.ones(len(u))
    return FunctionData(u=u, y=y, s=s, w=w,
                        grid_shape=(resolution, resolution))


def load_gano_samples(resolution=RESOLUTION, path=None):
    """Pre-generated GANO samples shipped with the reference release."""
    array = np.load(path or dataset_path(GANO_SAMPLES))
    samples = torch.from_numpy(np.ascontiguousarray(array)).float()
    if samples.shape[1] != resolution:
        from .cahn_hilliard import resample

        samples = resample(samples, resolution)
    return samples
