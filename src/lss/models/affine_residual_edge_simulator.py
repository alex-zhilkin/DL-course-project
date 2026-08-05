"""One-shot edge simulator with separated affine and non-affine outputs."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch_geometric.utils import softmax


class AffineResidualEdgeSimulator(nn.Module):
    """Predict a global affine increment plus a non-affine local correction."""

    def __init__(
        self,
        *,
        node_dim: int,
        edge_dim: int,
        hidden_size: int = 96,
        output_dim: int = 2,
        stiffness_feature_index: int = 8,
    ):
        super().__init__()
        if int(output_dim) != 2:
            raise ValueError("AffineResidualEdgeSimulator requires output_dim=2")
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.hidden_size = int(hidden_size)
        self.stiffness_feature_index = int(stiffness_feature_index)

        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(self.edge_dim, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.edge_score = nn.Sequential(
            nn.Linear(3 * self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 1),
        )
        self.edge_value = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.node_fuse = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
        )

        # Pooled structure, mean/std stiffness, and the current 2D affine state.
        global_input_dim = self.hidden_size + 2 + 6
        self.global_encoder = nn.Sequential(
            nn.Linear(global_input_dim, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
        )
        self.affine_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 6),
        )
        self.local_decoder = nn.Sequential(
            nn.Linear(2 * self.hidden_size + self.node_dim, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, 2),
        )
        self.node_skip = nn.Linear(self.node_dim, 2)
        self.score_scale = math.sqrt(self.hidden_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Begin from the geometric distance prior and small displacement heads.
        nn.init.zeros_(self.edge_score[-1].weight)
        nn.init.zeros_(self.edge_score[-1].bias)
        self.affine_head[-1].weight.data.mul_(0.1)
        self.local_decoder[-1].weight.data.mul_(0.1)
        self.node_skip.weight.data.mul_(0.1)

    @staticmethod
    def _affine_coefficients(design: Tensor, values: Tensor) -> Tensor:
        identity = torch.eye(
            design.size(1), device=design.device, dtype=design.dtype
        )
        gram = design.transpose(0, 1) @ design + 1e-5 * identity
        return torch.linalg.solve(
            gram, design.transpose(0, 1) @ values
        )

    def forward(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        edge_index: Tensor,
        *,
        attention_bias: Tensor | None = None,
        return_attention: bool = False,
    ):
        node_h = self.node_encoder(node_features)
        edge_h = self.edge_encoder(edge_features)
        first, second = edge_index
        if torch.any(first >= second):
            raise ValueError(
                "Undirected edge_index must contain one canonical i < j entry per pair."
            )

        reverse_edge = edge_features.clone()
        reverse_edge[:, [0, 1, 2, 3, 9, 10]] *= -1
        reverse_edge_h = self.edge_encoder(reverse_edge)
        forward_logits = self.edge_score(
            torch.cat([node_h[first], node_h[second], edge_h], dim=-1)
        ).squeeze(-1) / self.score_scale
        reverse_logits = self.edge_score(
            torch.cat([node_h[second], node_h[first], reverse_edge_h], dim=-1)
        ).squeeze(-1) / self.score_scale
        if attention_bias is not None:
            bias = attention_bias.to(forward_logits)
            forward_logits = forward_logits + bias
            reverse_logits = reverse_logits + bias

        endpoints = torch.cat([second, first])
        logits = torch.cat([forward_logits, reverse_logits])
        attention = softmax(logits, endpoints, num_nodes=node_h.size(0))
        forward_values = self.edge_value(
            torch.cat([node_h[first], edge_h], dim=-1)
        )
        reverse_values = self.edge_value(
            torch.cat([node_h[second], reverse_edge_h], dim=-1)
        )
        values = torch.cat([forward_values, reverse_values])
        aggregate = torch.zeros_like(node_h)
        aggregate.index_add_(0, endpoints, values * attention.unsqueeze(-1))
        node_token = node_h + self.node_fuse(
            torch.cat([node_h, aggregate], dim=-1)
        )

        # The last two node channels are the normalized reference coordinates.
        reference = node_features[:, 4:6]
        design = torch.cat(
            [torch.ones_like(reference[:, :1]), reference], dim=-1
        )
        current_delta = node_features[:, 2:4]
        current_affine = self._affine_coefficients(
            design, current_delta
        ).reshape(-1)
        stiffness = edge_features[:, self.stiffness_feature_index]
        stiffness_stats = torch.stack(
            [stiffness.mean(), stiffness.std(unbiased=False)]
        )
        global_input = torch.cat(
            [node_token.mean(dim=0), stiffness_stats, current_affine]
        )
        global_state = self.global_encoder(global_input.unsqueeze(0))

        affine_coefficients = self.affine_head(global_state).reshape(3, 2)
        affine_increment = design @ affine_coefficients
        expanded_global = global_state.expand(node_features.size(0), -1)
        local_raw = self.node_skip(node_features) + self.local_decoder(
            torch.cat([node_token, expanded_global, node_features], dim=-1)
        )
        local_affine = design @ self._affine_coefficients(design, local_raw)
        non_affine_residual = local_raw - local_affine
        prediction = affine_increment + non_affine_residual

        if return_attention:
            return prediction, attention.reshape(2, -1).transpose(0, 1)
        return prediction


__all__ = ["AffineResidualEdgeSimulator"]
