from __future__ import annotations

import torch


def resolve_device(device: str) -> str:
    return "cpu" if device == "cuda" and not torch.cuda.is_available() else device
