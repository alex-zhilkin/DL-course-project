"""One-shot attention over a caller-supplied undirected edge set."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch_geometric.utils import softmax


class OneShotUndirectedEdgeAttentionSimulator(nn.Module):
    """Attend over each undirected edge once, then decode nodes independently.

    Unlike a message-passing GNN, updated node embeddings are never sent back
    across the graph. The caller supplies one canonical ``i < j`` edge per
    unordered pair. That edge contributes to both endpoints inside this single
    attention operation without duplicating the graph representation.
    """

    def __init__(
        self,
        *,
        node_dim: int,
        edge_dim: int,
        hidden_size: int = 64,
        edge_hidden_size: int | None = None,
        output_dim: int = 2,
        global_context: bool = False,
        edge_aggregation: str = "softmax",
    ):
        super().__init__()
        hidden_size = int(hidden_size)
        edge_hidden_size = int(edge_hidden_size or hidden_size)
        self.node_dim = int(node_dim)
        self.global_context = bool(global_context)
        self.edge_aggregation = str(edge_aggregation)
        if self.edge_aggregation not in {"softmax", "gated_sum"}:
            raise ValueError("edge_aggregation must be 'softmax' or 'gated_sum'")
        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(int(edge_dim), edge_hidden_size),
            nn.GELU(),
            nn.Linear(edge_hidden_size, hidden_size),
        )
        self.edge_score = nn.Sequential(
            nn.Linear(3 * hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.edge_value = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.node_fuse = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        if self.global_context:
            self.global_encoder = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.Linear(hidden_size, hidden_size),
                nn.LayerNorm(hidden_size),
            )
        self.decoder = nn.Sequential(
            nn.Linear(
                hidden_size
                + self.node_dim
                + (hidden_size if self.global_context else 0),
                hidden_size,
            ),
            nn.GELU(),
            nn.Linear(hidden_size, int(output_dim)),
        )
        self.node_skip = nn.Linear(self.node_dim, int(output_dim))
        self.score_scale = math.sqrt(hidden_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # At initialization logits equal the supplied geometric distance prior.
        nn.init.zeros_(self.edge_score[-1].weight)
        nn.init.zeros_(self.edge_score[-1].bias)
        self.decoder[-1].weight.data.mul_(0.1)
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
        node_h = self.node_encoder(node_features)
        edge_h = self.edge_encoder(edge_features)
        first, second = edge_index
        if torch.any(first >= second):
            raise ValueError(
                "Undirected edge_index must contain one canonical i < j entry per pair."
            )

        # The stored feature orientation is first -> second. Derive the
        # opposite endpoint view analytically rather than storing a second edge.
        reverse_edge_features = edge_features.clone()
        reverse_edge_features[:, [0, 1, 2, 3, 9, 10]] *= -1
        reverse_edge_h = self.edge_encoder(reverse_edge_features)
        forward_score_input = torch.cat(
            [node_h[first], node_h[second], edge_h], dim=-1
        )
        reverse_score_input = torch.cat(
            [node_h[second], node_h[first], reverse_edge_h], dim=-1
        )
        forward_logits = (
            self.edge_score(forward_score_input).squeeze(-1) / self.score_scale
        )
        reverse_logits = (
            self.edge_score(reverse_score_input).squeeze(-1) / self.score_scale
        )
        if attention_bias is not None:
            if attention_bias.shape != forward_logits.shape:
                raise ValueError(
                    "attention_bias must have one scalar per undirected edge; "
                    f"got {tuple(attention_bias.shape)} for "
                    f"{tuple(forward_logits.shape)} logits"
                )
            bias = attention_bias.to(forward_logits)
            forward_logits = forward_logits + bias
            reverse_logits = reverse_logits + bias

        # Normalize across all edges incident to each endpoint. `forward`
        # contributes first -> second; `reverse` contributes second -> first.
        endpoint = torch.cat([second, first])
        logits = torch.cat([forward_logits, reverse_logits])
        if self.edge_aggregation == "softmax":
            attention = softmax(logits, endpoint, num_nodes=node_h.size(0))
        else:
            # Pair interactions are additive: do not renormalize away the
            # coordination number or the magnitude of multiple LJ contacts.
            attention = torch.sigmoid(logits)
        forward_values = self.edge_value(torch.cat([node_h[first], edge_h], dim=-1))
        reverse_values = self.edge_value(
            torch.cat([node_h[second], reverse_edge_h], dim=-1)
        )
        values = torch.cat([forward_values, reverse_values])
        aggregate = torch.zeros_like(node_h)
        aggregate.index_add_(0, endpoint, values * attention.unsqueeze(-1))
        node_token = node_h + self.node_fuse(torch.cat([node_h, aggregate], dim=-1))
        decoder_parts = [node_token, node_features]
        if self.global_context:
            # Denoise the collective deformation across the whole graph, then
            # make that learned state available to every node update.
            graph_context = self.global_encoder(node_token.mean(dim=0, keepdim=True))
            decoder_parts.append(graph_context.expand(node_token.size(0), -1))
        prediction = self.node_skip(node_features) + self.decoder(
            torch.cat(decoder_parts, dim=-1)
        )
        if return_attention:
            return prediction, attention.reshape(2, -1).transpose(0, 1)
        return prediction


__all__ = ["OneShotUndirectedEdgeAttentionSimulator"]
