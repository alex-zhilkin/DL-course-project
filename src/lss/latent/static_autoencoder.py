"""Masked autoencoder for reference-graph geometry and edge properties."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class StaticMessageLayer(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        h = int(hidden_size)
        self.message = nn.Sequential(nn.Linear(3 * h, h), nn.GELU(), nn.Linear(h, h))
        self.node_update = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, h))
        self.edge_update = nn.Sequential(nn.Linear(3 * h, h), nn.GELU(), nn.Linear(h, h))

    def forward(self, node_h: Tensor, edge_h: Tensor, edge_index: Tensor) -> tuple[Tensor, Tensor]:
        source, target = edge_index
        message = self.message(torch.cat([node_h[source], node_h[target], edge_h], dim=-1))
        aggregate = torch.zeros_like(node_h)
        count = torch.zeros(node_h.size(0), 1, device=node_h.device, dtype=node_h.dtype)
        aggregate.index_add_(0, target, message)
        count.index_add_(0, target, torch.ones(message.size(0), 1, device=node_h.device))
        aggregate = aggregate / count.clamp_min(1.0)
        node_h = node_h + self.node_update(torch.cat([node_h, aggregate], dim=-1))
        edge_h = edge_h + self.edge_update(
            torch.cat([edge_h, node_h[source], node_h[target]], dim=-1)
        )
        return node_h, edge_h


class StaticGraphAutoEncoder(nn.Module):
    """Denoise static graph features while exposing local and global encodings."""

    def __init__(
        self,
        *,
        node_dim: int = 2,
        edge_dim: int = 4,
        hidden_size: int = 96,
        static_dim: int = 1,
        message_layers: int = 3,
    ):
        super().__init__()
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.hidden_size = int(hidden_size)
        self.static_dim = int(static_dim)
        self.node_in = nn.Sequential(
            nn.Linear(node_dim + 1, hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size)
        )
        self.edge_in = nn.Sequential(
            nn.Linear(edge_dim + 1, hidden_size), nn.GELU(), nn.Linear(hidden_size, hidden_size)
        )
        self.layers = nn.ModuleList(
            StaticMessageLayer(hidden_size) for _ in range(int(message_layers))
        )
        self.to_static = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, static_dim)
        )
        self.node_decoder = nn.Sequential(
            nn.Linear(hidden_size + static_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, node_dim),
        )
        self.edge_decoder = nn.Sequential(
            nn.Linear(3 * hidden_size + static_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, edge_dim),
        )

    def encode(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        edge_index: Tensor,
        node_mask: Tensor,
        edge_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        node_input = torch.cat(
            [node_features.masked_fill(node_mask[:, None], 0.0), node_mask[:, None].to(node_features)],
            dim=-1,
        )
        edge_input = torch.cat(
            [edge_features.masked_fill(edge_mask[:, None], 0.0), edge_mask[:, None].to(edge_features)],
            dim=-1,
        )
        node_h = self.node_in(node_input)
        edge_h = self.edge_in(edge_input)
        for layer in self.layers:
            node_h, edge_h = layer(node_h, edge_h, edge_index)
        static = self.to_static(node_h.mean(dim=0, keepdim=True))
        return node_h, edge_h, static

    def forward(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        edge_index: Tensor,
        node_mask: Tensor,
        edge_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        node_h, edge_h, static = self.encode(
            node_features, edge_features, edge_index, node_mask, edge_mask
        )
        static_nodes = static.expand(node_h.size(0), -1)
        source, target = edge_index
        static_edges = static.expand(edge_h.size(0), -1)
        node_prediction = self.node_decoder(torch.cat([node_h, static_nodes], dim=-1))
        edge_prediction = self.edge_decoder(
            torch.cat([node_h[source], node_h[target], edge_h, static_edges], dim=-1)
        )
        return node_prediction, edge_prediction, node_h, static


__all__ = ["StaticGraphAutoEncoder"]
