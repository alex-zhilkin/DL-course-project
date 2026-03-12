from __future__ import annotations

import torch


def resolve_device(device: str) -> str:
    if device == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "mps":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device
