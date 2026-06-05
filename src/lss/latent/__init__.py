"""Latent-space autoencoders, propagators, training, and rollout utilities."""

from .models import (
    LatentDynamicsMLP,
    NodeDeltaAttentionAutoEncoder,
    make_latent_propagator,
)
from .simulator import LatentSpaceSimulator
from .training import (
    LatentNormalizer,
    TrainingConfig,
    TrainingResult,
    train_autoencoder,
    train_propagator,
)

__all__ = [
    "LatentDynamicsMLP",
    "LatentNormalizer",
    "LatentSpaceSimulator",
    "NodeDeltaAttentionAutoEncoder",
    "TrainingConfig",
    "TrainingResult",
    "make_latent_propagator",
    "train_autoencoder",
    "train_propagator",
]
