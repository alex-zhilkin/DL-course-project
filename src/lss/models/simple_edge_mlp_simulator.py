"""Minimal graph simulator using only MLPs and an additive edge sum."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SimpleUndirectedEdgeMLPSimulator(nn.Module):
    """One edge MLP, one sum aggregation, and one node decoder.

    The caller supplies one canonical ``i < j`` entry for every undirected
    physical edge. The reverse orientation is constructed analytically.
    There are no attention scores, gates, queries, keys, or softmax operations.
    """

    def __init__(
        self,
        *,
        node_dim: int,
        edge_dim: int,
        hidden_size: int = 64,
        output_dim: int = 2,
    ):
        super().__init__()
        self.node_dim = int(node_dim)
        hidden_size = int(hidden_size)
        output_dim = int(output_dim)

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * self.node_dim + int(edge_dim), hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.node_mlp = nn.Sequential(
            nn.Linear(self.node_dim + hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_dim),
        )
        self.node_skip = nn.Linear(self.node_dim, output_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.node_mlp[-1].weight.data.mul_(0.1)
        self.node_skip.weight.data.mul_(0.1)

    def forward(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        edge_index: Tensor,
        *,
        attention_bias: Tensor | None = None,
        return_attention: bool = False,
    ):
        del attention_bias
        first, second = edge_index
        if torch.any(first >= second):
            raise ValueError(
                "Undirected edge_index must contain one canonical i < j entry per pair."
            )

        reverse_edge = edge_features.clone()
        reverse_edge[:, [0, 1, 2, 3, 9, 10]] *= -1
        forward = self.edge_mlp(
            torch.cat(
                [node_features[first], node_features[second], edge_features],
                dim=-1,
            )
        )
        reverse = self.edge_mlp(
            torch.cat(
                [node_features[second], node_features[first], reverse_edge],
                dim=-1,
            )
        )

        aggregate = node_features.new_zeros(
            (node_features.size(0), forward.size(-1))
        )
        aggregate.index_add_(0, second, forward)
        aggregate.index_add_(0, first, reverse)
        output = self.node_skip(node_features) + self.node_mlp(
            torch.cat([node_features, aggregate], dim=-1)
        )
        if return_attention:
            return output, None
        return output


__all__ = ["SimpleUndirectedEdgeMLPSimulator"]
