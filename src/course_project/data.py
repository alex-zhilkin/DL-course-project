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


def split_dataset(
    path: str | Path,
    *,
    train_count: int,
    val_count: int,
) -> tuple[list, list, list]:
    sims = load_dataset(path)
    train_count = int(train_count)
    val_count = int(val_count)
    train_data = sims[:train_count]
    val_data = sims[train_count : train_count + val_count]
    test_data = sims[train_count + val_count :]
    return train_data, val_data, test_data
