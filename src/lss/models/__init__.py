"""Full-space and collective-variable simulator models."""

from __future__ import annotations

from torch_geometric.data import Data

from .base import BaseModelInputs
from .cv_transformer_simulator import Model as CVTransformerModel
from .hybrid_simulator import Model as HybridModel
from .inverse_design_simulator import Model as InverseDesignSimulatorModel
from .linear_cv_simulator import Model as LinearCVSimulatorModel
from .spatial_simulator import Model as SpatialModel
from .transformer_simulator import Model as TransformerSimulatorModel

MODEL_REGISTRY = {
    "spatial": SpatialModel,
    "cv_transformer": CVTransformerModel,
    "linear_cv_simulator": LinearCVSimulatorModel,
    "hybrid": HybridModel,
    "inverse_design_simulator": InverseDesignSimulatorModel,
    "transformer_simulator": TransformerSimulatorModel,
}

MODEL_EXTRAS_REQUIRED = {
    "spatial": {
        "num_mlp",
    },
    "cv_transformer": {
        "num_mlp",
        "transformer_layers",
        "transformer_heads",
        "transformer_dropout",
        "token_sizes",
    },
    "linear_cv_simulator": {
        "num_mlp",
        "token_sizes",
    },
    "hybrid": {
        "num_mlp",
        "cv_checkpoint_path",
    },
    "inverse_design_simulator": {
        "num_mlp",
    },
    "transformer_simulator": {
        "num_mlp",
        "transformer_layers",
        "transformer_heads",
        "transformer_dropout",
    },
}

MODEL_EXTRAS_OPTIONAL = {
    "spatial": {
        "use_skip",
        "final_decoder_local_skip",
    },
    "cv_transformer": {
        "use_normalization",
        "prediction_target",
        "global_decoder_max_nodes",
        "global_decoder_layers",
        "global_decoder_local_skip",
    },
    "linear_cv_simulator": {
        "use_normalization",
        "prediction_target",
        "global_decoder_max_nodes",
        "global_decoder_layers",
        "linear_encoder_max_edges",
        "edge_token_dim",
        "node_token_dim",
    },
    "hybrid": {
        "cv_inject_scale_init",
        "cv_consistency_weight",
        "time_lag_steps",
        "time_lag_weight",
    },
    "inverse_design_simulator": set(),
    "transformer_simulator": {
        "use_normalization",
        "prediction_target",
        "K1",
        "K2",
        "K3",
        "K4",
        "cv_dim",
        "k2_hidden_size",
        "use_local_skip",
        "use_lap_pe",
        "lap_pe_k",
        "lap_pe_is_undirected",
        "edge_aggr",
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
    return extras


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
    "BaseModelInputs",
    "HybridModel",
    "InverseDesignSimulatorModel",
    "CVTransformerModel",
    "LinearCVSimulatorModel",
    "SpatialModel",
    "TransformerSimulatorModel",
    "MODEL_REGISTRY",
    "MODEL_EXTRAS_REQUIRED",
    "resolve_model_extras",
    "resolve_model_inputs",
    "create_model",
]
