from __future__ import annotations

from pathlib import Path

import torch


def load_dataset(path: str | Path) -> list:
    return list(torch.load(Path(path), weights_only=False))


def split_dataset(path: str | Path, *, train_count: int, val_count: int) -> tuple[list, list, list]:
    sims = load_dataset(path)
    train_count = int(train_count)
    val_count = int(val_count)
    train_data = sims[:train_count]
    val_data = sims[train_count : train_count + val_count]
    test_data = sims[train_count + val_count :]
    return train_data, val_data, test_data
