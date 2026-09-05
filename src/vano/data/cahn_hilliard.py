"""Cahn-Hilliard phase-separation patterns (paper section 6.3).

Source array: ``cahn_hilliard_patterns.npy`` from the authors' data release,
``(N, 400, 400, 1)`` phase fields.  Samples are bilinearly resampled to the
requested resolution, so the same functions can be presented to the model at
64x64, 128x128 or 256x256 -- the comparison that Table 1 and Figures 12-13 are
built on.  ``scripts/download_data.py`` writes one ``cahn_hilliard_r<res>.npy``
cache per resolution; the loader prefers those over the 48 GB source array.

The release draws train and test indices with two independent ``random.choice``
calls over the same pool, so its splits overlap.  We take a disjoint split by
default (``split_overlap=False``); pass ``split_overlap=True`` to reproduce the
release's protocol exactly.
"""

import numpy as np
import torch
import torch.nn.functional as F

from .base import FunctionData, unit_grid
from .paths import dataset_path

FILENAME = "cahn_hilliard_patterns.npy"
CACHE = "cahn_hilliard_r{resolution}.npy"
RESOLUTION = 64
NUM_SAMPLES = 4096


def resample(images, resolution):
    """Antialiased bilinear resize of ``(N, H, W, C)`` fields."""
    x = images.permute(0, 3, 1, 2)
    if x.shape[-1] != resolution or x.shape[-2] != resolution:
        x = F.interpolate(x, size=(resolution, resolution), mode="bilinear",
                          align_corners=False, antialias=True)
    return x.permute(0, 2, 3, 1)


def load(split="train", num_samples=NUM_SAMPLES, resolution=RESOLUTION,
         seed=0, split_overlap=False, path=None, chunk=512):
    if path is None:
        try:
            path, cached = dataset_path(CACHE.format(resolution=resolution)), True
        except FileNotFoundError:
            path, cached = dataset_path(FILENAME), False
    else:
        cached = False
    array = np.load(path, mmap_mode="r")
    generator = torch.Generator().manual_seed(seed)
    if split_overlap:
        # The release's protocol: independent draws for train and test.
        generator.manual_seed(0 if split == "train" else 1)
        idx = torch.randperm(array.shape[0], generator=generator)[:num_samples]
    else:
        order = torch.randperm(array.shape[0], generator=generator)
        offset = 0 if split == "train" else num_samples
        idx = order[offset : offset + num_samples]
        if len(idx) < num_samples:
            raise ValueError(f"{path} holds too few patterns for two splits")
    idx = idx.sort().values

    images = []
    for start in range(0, len(idx), chunk):
        block = np.asarray(array[idx[start : start + chunk].numpy()])
        block = torch.from_numpy(block).float()
        images.append(block if cached else resample(block, resolution))
    u = torch.cat(images)

    y = unit_grid(resolution, dim=2)
    s = u.reshape(len(u), -1, u.shape[-1])
    w = torch.ones(len(u))
    return FunctionData(u=u, y=y, s=s, w=w,
                        grid_shape=(resolution, resolution))
