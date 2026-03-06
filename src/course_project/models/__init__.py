"""Standalone simulator models used by course_project."""

from __future__ import annotations

from torch_geometric.data import Data

from .autoencoder import Model as AutoencoderModel
from .base import BaseModelInputs
from .cv_transformer_simulator import Model as CVTransformerModel
from .hybrid_simulator import Model as HybridModel
from .latent_space_simulator import Model as LatentSpaceModel
from .spatial_simulator import Model as SpatialModel
from .spatial_transformer_simulator import Model as SpatialTransformerModel

MODEL_REGISTRY = {
    "autoencoder": AutoencoderModel,
    "spatial": SpatialModel,
    "spatial_transformer": SpatialTransformerModel,
    "cv_transformer": CVTransformerModel,
    "latent_space": LatentSpaceModel,
    "hybrid": HybridModel,
}

MODEL_EXTRAS_REQUIRED = {
    "autoencoder": {
        "num_mlp",
        "K1",
        "CV",
        "transformer_layers",
        "transformer_heads",
        "transformer_dropout",
        "edge_aggr",
        "use_local_skip",
    },
    "spatial": {
        "num_mlp",
    },
    "spatial_transformer": {
        "num_mlp",
        "K1",
        "K2",
        "transformer_layers",
        "transformer_heads",
        "transformer_dropout",
        "edge_aggr",
        "k2_hidden_size",
        "use_local_skip",
    },
    "cv_transformer": {
        "num_mlp",
        "K1",
        "CV",
        "transformer_layers",
        "transformer_heads",
        "transformer_dropout",
        "edge_aggr",
        "use_local_skip",
    },
    "latent_space": {
        "num_mlp",
        "K1",
        "CV",
        "transformer_layers",
        "transformer_heads",
        "transformer_dropout",
        "edge_aggr",
        "use_local_skip",
    },
    "hybrid": {
        "num_mlp",
        "cv_checkpoint_path",
    },
}

MODEL_EXTRAS_OPTIONAL = {
    "autoencoder": {
        "cv_hidden_size",
    },
    "spatial": set(),
    "spatial_transformer": set(),
    "cv_transformer": set(),
    "latent_space": {
        "cv_hidden_size",
    },
    "hybrid": {
        "cv_inject_scale_init",
    },
}


def resolve_model_extras(model_type: str, extras: dict | None) -> dict:
    required = MODEL_EXTRAS_REQUIRED[model_type]
    optional = MODEL_EXTRAS_OPTIONAL.get(model_type, set())

    if extras is None:
        raise ValueError(
            f"model_extras must be provided for model_type='{model_type}'. "
            f"Required keys: {sorted(required)}"
        )
    if not isinstance(extras, dict):
        raise TypeError(f"model_extras must be a dict, got {type(extras).__name__}")

    extras = dict(extras)
    provided = set(extras.keys())
    missing = sorted(required - provided)
    unexpected = sorted(provided - required - optional)
    if missing or unexpected:
        parts = []
        if missing:
            parts.append(f"missing keys={missing}")
        if unexpected:
            parts.append(f"unexpected keys={unexpected}")
        raise ValueError(
            f"Invalid model_extras for model_type='{model_type}': " + "; ".join(parts)
        )
    return dict(extras)


def resolve_model_inputs(model_type: str):
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model_type '{model_type}' for inputs lookup.")
    return BaseModelInputs


def create_model(
    model_type: str,
    init_graph: Data,
    pos_dim: int,
    hidden_size: int,
    n_layers: int,
    extras: dict,
):
    cls = MODEL_REGISTRY[model_type]
    validated_extras = resolve_model_extras(model_type, extras)
    return cls(
        data=init_graph,
        hidden_size=hidden_size,
        n_layers=n_layers,
        pos_dim=pos_dim,
        **validated_extras,
    )


__all__ = [
    "AutoencoderModel",
    "BaseModelInputs",
    "HybridModel",
    "CVTransformerModel",
    "LatentSpaceModel",
    "SpatialModel",
    "SpatialTransformerModel",
    "MODEL_REGISTRY",
    "MODEL_EXTRAS_REQUIRED",
    "resolve_model_extras",
    "resolve_model_inputs",
    "create_model",
]
