from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.data import Data


def _box_dim(box: dict[str, float], axis: int) -> float:
    key = ("x", "y", "z")[axis]
    if key in box:
        return float(box[key])
    key1 = f"{key}1"
    key2 = f"{key}2"
    
    return float(box[key2] - box[key1])


def get_correct_edge_vec(original_graph: Data, pos_dim: int | None = None) -> Tensor:
    """Return edge displacement vectors with periodic box wrapping correction.

    This applies a minimum-image style correction so edges crossing a periodic
    boundary use the short wrapped displacement instead of a large box-spanning
    vector.
    """
    if pos_dim is None:
        node_dim = original_graph.x.shape[1]
        if node_dim in (2, 3):
            pos_dim = node_dim
        elif node_dim % 2 == 0 and (node_dim // 2) in (2, 3):
            pos_dim = node_dim // 2
        else:
            pos_dim = 2

    edge_index = original_graph.edge_index
    positions = original_graph.x[:, :pos_dim]
    source_pos = positions[edge_index[0]]
    target_pos = positions[edge_index[1]]

    raw_box = getattr(original_graph, "box")
    if isinstance(raw_box, dict):
        box = raw_box
    else:
        box = {}
        for key in ("x", "y", "z", "x1", "x2", "y1", "y2", "z1", "z2"):
            if hasattr(raw_box, key):
                box[key] = float(getattr(raw_box, key))
    box_dims = torch.tensor(
        [_box_dim(box, 0), _box_dim(box, 1), _box_dim(box, 2)],
        device=original_graph.x.device,
        dtype=original_graph.x.dtype,
    )[:pos_dim]

    naive_edge_vectors = source_pos - target_pos
    half_box = box_dims / 2
    adjust = torch.abs(naive_edge_vectors) > half_box
    # Apply minimum-image periodic wrapping for edges that cross the box boundary.
    correction = torch.sign(source_pos) * (
        half_box - torch.abs(source_pos) + half_box - torch.abs(target_pos)
    )
    wrapped_target_pos = torch.where(
        adjust,
        source_pos + correction,
        target_pos,
    )
    return wrapped_target_pos - source_pos


def _gaussian_init_linear(layer: torch.nn.Linear, fan_in: int) -> torch.nn.Linear:
    std = 1.0 / (fan_in**0.5)
    torch.nn.init.normal_(layer.weight, mean=0.0, std=std)
    torch.nn.init.normal_(layer.bias, mean=0.0, std=std)
    return layer


def init_transformer_style_weights(module: nn.Module) -> None:
    """Xavier/zero-bias init for Linear and MultiheadAttention layers."""
    for submodule in module.modules():
        if isinstance(submodule, nn.Linear):
            nn.init.xavier_uniform_(submodule.weight, gain=1.0)
            if submodule.bias is not None:
                nn.init.zeros_(submodule.bias)
        elif isinstance(submodule, nn.MultiheadAttention):
            if getattr(submodule, "in_proj_weight", None) is not None:
                nn.init.xavier_uniform_(submodule.in_proj_weight, gain=1.0)
            if getattr(submodule, "in_proj_bias", None) is not None:
                nn.init.zeros_(submodule.in_proj_bias)
            if getattr(submodule, "q_proj_weight", None) is not None:
                nn.init.xavier_uniform_(submodule.q_proj_weight, gain=1.0)
            if getattr(submodule, "k_proj_weight", None) is not None:
                nn.init.xavier_uniform_(submodule.k_proj_weight, gain=1.0)
            if getattr(submodule, "v_proj_weight", None) is not None:
                nn.init.xavier_uniform_(submodule.v_proj_weight, gain=1.0)
            nn.init.xavier_uniform_(submodule.out_proj.weight, gain=1.0)
            if submodule.out_proj.bias is not None:
                nn.init.zeros_(submodule.out_proj.bias)


def init_token_query_params(params, *, std: float = 0.01) -> None:
    """Small normal init for learnable token query parameters."""
    for p in params:
        nn.init.normal_(p, mean=0.0, std=std)


def init_scaled_linear_head(layer: nn.Linear, *, scale: float = 0.1) -> None:
    """Modular-style init for final prediction heads."""
    nn.init.xavier_uniform_(layer.weight, gain=1.0)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    layer.weight.data.mul_(scale)


def build_mlp(
    in_size: int,
    hidden_size: int,
    out_size: int,
    num_mlp: int = 3,
    lay_norm: bool = False,
) -> torch.nn.Sequential:
    layers: list[torch.nn.Module] = []

    def _linear(in_dim: int, out_dim: int) -> torch.nn.Linear:
        layer = torch.nn.Linear(in_dim, out_dim, dtype=torch.float32)
        return _gaussian_init_linear(layer, fan_in=in_dim)

    layers.append(_linear(in_size, hidden_size))
    layers.append(torch.nn.ReLU())

    for _ in range(num_mlp - 2):
        layers.append(_linear(hidden_size, hidden_size))
        layers.append(torch.nn.ReLU())

    layers.append(_linear(hidden_size, out_size))

    if lay_norm:
        layers.append(torch.nn.LayerNorm(normalized_shape=out_size, dtype=torch.float32))

    return torch.nn.Sequential(*layers)
