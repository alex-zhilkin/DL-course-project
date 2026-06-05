"""Composition object for a latent-space simulator."""

from __future__ import annotations

import torch.nn as nn


class LatentSpaceSimulator(nn.Module):
    """A simulator composed of an autoencoder and a latent propagator."""

    def __init__(self, autoencoder: nn.Module, propagator: nn.Module):
        super().__init__()
        self.autoencoder = autoencoder
        self.propagator = propagator

    def encode(self, *args, **kwargs):
        return self.autoencoder.encode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.autoencoder.decode(*args, **kwargs)

    def propagate(self, z):
        return self.propagator(z)


__all__ = ["LatentSpaceSimulator"]
