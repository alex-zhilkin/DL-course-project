"""Standalone simulator models used by course_project."""

from __future__ import annotations

from torch_geometric.data import Data

from .base import BaseModelInputs
from .cv_transformer_simulator import Model as CVTransformerModel
from .hybrid_simulator import Model as HybridModel
from .spatial_simulator import Model as SpatialModel
from .spatial_transformer_simulator import Model as SpatialTransformerModel

MODEL_REGISTRY = {
    "spatial": SpatialModel,
    "spatial_transformer": SpatialTransformerModel,
    "cv_transformer": CVTransformerModel,
    "hybrid": HybridModel,
}

MODEL_INPUTS_REGISTRY = {
    "spatial": BaseModelInputs,
    "spatial_transformer": BaseModelInputs,
    "cv_transformer": BaseModelInputs,
    "hybrid": BaseModelInputs,
}

MODEL_EXTRAS_REQUIRED = {
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
    "hybrid": {
        "num_mlp",
        "K1",
        "K2",
        "transformer_layers",
        "transformer_heads",
        "transformer_dropout",
        "k2_hidden_size",
    },
}


def resolve_model_extras(model_type: str, extras: dict | None) -> dict:
    try:
        required = MODEL_EXTRAS_REQUIRED[model_type]
    except KeyError as exc:
        raise ValueError(f"Unknown model_type '{model_type}' for extras lookup.") from exc

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
    unexpected = sorted(provided - required)
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
    if model_type in MODEL_INPUTS_REGISTRY:
        return MODEL_INPUTS_REGISTRY[model_type]
    if model_type in MODEL_REGISTRY:
        # All current simulators use BaseModelInputs; keep this as a safe default
        # if a registry entry is missing.
        return BaseModelInputs
    raise ValueError(f"Unknown model_type '{model_type}' for inputs lookup.")


def create_model(
    model_type: str,
    init_graph: Data,
    pos_dim: int,
    hidden_size: int,
    n_layers: int,
    extras: dict,
):
    try:
        cls = MODEL_REGISTRY[model_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported model_type: {model_type}") from exc
    validated_extras = resolve_model_extras(model_type, extras)
    return cls(
        data=init_graph,
        hidden_size=hidden_size,
        n_layers=n_layers,
        pos_dim=pos_dim,
        **validated_extras,
    )


__all__ = [
    "BaseModelInputs",
    "HybridModel",
    "CVTransformerModel",
    "SpatialModel",
    "SpatialTransformerModel",
    "MODEL_REGISTRY",
    "MODEL_INPUTS_REGISTRY",
    "MODEL_EXTRAS_REQUIRED",
    "resolve_model_extras",
    "resolve_model_inputs",
    "create_model",
]
