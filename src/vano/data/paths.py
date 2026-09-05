"""Where the downloaded datasets live."""

import os
from pathlib import Path

ENV_VAR = "VANO_DATA_ROOT"
DEFAULT_ROOT = Path("data")


def data_root():
    return Path(os.environ.get(ENV_VAR, DEFAULT_ROOT)).expanduser()


def dataset_path(name):
    path = data_root() / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` and point "
            f"${ENV_VAR} at the resulting directory."
        )
    return path
