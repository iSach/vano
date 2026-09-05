#!/usr/bin/env python
"""Fetch and prepare the datasets used by the paper.

The authors publish the Cahn-Hilliard patterns, the processed InSAR stack
(``volcano.npy``), the raw Sentinel-1 interferograms and a set of GANO samples
as one Google Drive archive.  This script downloads it and unpacks what the
benchmarks need.

The Cahn-Hilliard source array is 37523 x 400 x 400 float64 -- 48 GB -- and is
only ever used at 64, 128 and 256 pixels.  Rather than land it on disk we stream
it straight out of the archive and write one resampled float32 cache per
resolution (13 GB in total), which is also what makes training start instantly.

    python scripts/download_data.py --root /mnt/ceph/$USER/vano-repro
    export VANO_DATA_ROOT=/mnt/ceph/$USER/vano-repro/datasets

The GRF and 2D Gaussian-density benchmarks need none of this; they are
generated analytically.
"""

import argparse
import subprocess
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch

ARCHIVE_ID = "1w_b8uHXSl2d_PkMHC2GufR7LnGM6l_xT"
ARCHIVE_NAME = "datasets+checkpoints.tar.gz"
CH_SOURCE = "cahn_hilliard_patterns.npy"
CH_RESOLUTIONS = (64, 128, 256)
PLAIN_MEMBERS = ("volcano.npy", "GANO_samples.npy")


def download(raw_dir):
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / ARCHIVE_NAME
    if archive.exists():
        print(f"{archive} already downloaded")
        return archive
    print(f"downloading {ARCHIVE_NAME} (2.3 GB) from Google Drive ...")
    subprocess.run([sys.executable, "-m", "gdown", ARCHIVE_ID, "-O", str(archive)],
                   check=True)
    return archive


def _read_exactly(handle, size):
    chunks, remaining = [], size
    while remaining:
        block = handle.read(remaining)
        if not block:
            raise EOFError("truncated array in archive")
        chunks.append(block)
        remaining -= len(block)
    return b"".join(chunks)


def stream_cahn_hilliard(handle, datasets, resolutions, device, rows=64):
    """Resample the pattern array as it comes off the tar stream."""
    from vano.data.cahn_hilliard import CACHE, resample

    version = np.lib.format.read_magic(handle)
    shape, _, dtype = np.lib.format._read_array_header(handle, version)
    count, height, width, channels = shape
    print(f"  streaming {shape} {dtype} -> {list(resolutions)}")

    buffers = {
        resolution: np.lib.format.open_memmap(
            datasets / CACHE.format(resolution=resolution), mode="w+",
            dtype=np.float32, shape=(count, resolution, resolution, channels))
        for resolution in resolutions
    }
    row_bytes = height * width * channels * dtype.itemsize
    for start in range(0, count, rows):
        take = min(rows, count - start)
        block = np.frombuffer(_read_exactly(handle, take * row_bytes), dtype=dtype)
        block = torch.from_numpy(block.reshape(take, height, width, channels))
        block = block.to(device=device, dtype=torch.float32)
        for resolution, buffer in buffers.items():
            buffer[start : start + take] = resample(block, resolution).cpu().numpy()
        print(f"\r  {start + take}/{count}", end="", flush=True)
    print()
    for buffer in buffers.values():
        buffer.flush()


def extract(archive, root, resolutions, device, with_raw_insar=False):
    datasets = root / "datasets"
    datasets.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r|gz") as tar:
        for member in tar:
            name = Path(member.name).name
            if not member.isfile():
                continue
            if name == CH_SOURCE:
                if all((datasets / f"cahn_hilliard_r{r}.npy").exists()
                       for r in resolutions):
                    print("  Cahn-Hilliard caches already built, skipping")
                    continue
                stream_cahn_hilliard(tar.extractfile(member), datasets,
                                     resolutions, device)
            elif name in PLAIN_MEMBERS:
                print(f"  writing {name}")
                (datasets / name).write_bytes(tar.extractfile(member).read())
            elif with_raw_insar and "/InSar/" in member.name:
                target = datasets / "InSar" / name
                target.parent.mkdir(exist_ok=True)
                target.write_bytes(tar.extractfile(member).read())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True,
                        help="directory to hold raw/ and datasets/")
    parser.add_argument("--resolutions", type=int, nargs="+",
                        default=list(CH_RESOLUTIONS))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available()
                        else "cpu", help="device used for the resampling")
    parser.add_argument("--with-raw-insar", action="store_true",
                        help="also unpack the 4096 raw .int interferograms")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()

    archive = (args.root / "raw" / ARCHIVE_NAME) if args.skip_download \
        else download(args.root / "raw")
    extract(archive, args.root, tuple(args.resolutions), args.device,
            args.with_raw_insar)
    print(f"\nexport VANO_DATA_ROOT={args.root / 'datasets'}")


if __name__ == "__main__":
    main()
