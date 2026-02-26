from __future__ import annotations

from pathlib import Path

import torch

from .compat import install_legacy_pickle_aliases


def load_dataset(path: str | Path) -> list:
    p = Path(path)
    try:
        data = torch.load(p, weights_only=False)
    except ModuleNotFoundError:
        install_legacy_pickle_aliases()
        data = torch.load(p, weights_only=False)
    return list(data)
