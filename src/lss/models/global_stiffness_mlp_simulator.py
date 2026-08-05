"""Graph simulator driven by global stiffness and deformation statistics."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class GlobalStiffnessMLPSimulator(nn.Module):
    """Predict node increments without edge-specific messages.

    The undirected physical edge set is summarized by the mean and standard
    deviation of its normalized static stiffness feature. The current graph
    state is summarized by the mean and standard deviation of the node
    features. One shared decoder receives those global statistics together
    with each node's own geometry.
    """

    def __init__(
        self,
        *,
        node_dim: int,
        edge_dim: int,
        hidden_size: int = 64,
        output_dim: int = 2,
        stiffness_feature_index: int = 8,
    ):
        super().__init__()
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.stiffness_feature_index = int(stiffness_feature_index)
        if not 0 <= self.stiffness_feature_index < self.edge_dim:
            raise ValueError("stiffness_feature_index is outside the edge features")

        hidden_size = int(hidden_size)
        output_dim = int(output_dim)
        global_dim = 2 * self.node_dim + 2
        self.global_encoder = nn.Sequential(
            nn.Linear(global_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.node_decoder = nn.Sequential(
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
        self.node_decoder[-1].weight.data.mul_(0.1)
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
        del edge_index, attention_bias
        node_mean = node_features.mean(dim=0)
        node_std = node_features.std(dim=0, unbiased=False)
        stiffness = edge_features[:, self.stiffness_feature_index]
        stiffness_stats = torch.stack(
            [stiffness.mean(), stiffness.std(unbiased=False)]
        )
        global_state = self.global_encoder(
            torch.cat([node_mean, node_std, stiffness_stats]).unsqueeze(0)
        )
        global_state = global_state.expand(node_features.size(0), -1)
        prediction = self.node_skip(node_features) + self.node_decoder(
            torch.cat([node_features, global_state], dim=-1)
        )
        if return_attention:
            return prediction, None
        return prediction


__all__ = ["GlobalStiffnessMLPSimulator"]
