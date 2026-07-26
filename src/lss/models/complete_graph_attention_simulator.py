"""Node-level simulator with learned attention over directed complete-graph edges."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch_geometric.utils import softmax


class CompleteGraphAttentionLayer(nn.Module):
    """Attend over all incoming pair edges separately for every target node."""

    def __init__(self, hidden_size: int):
        super().__init__()
        hidden_size = int(hidden_size)
        pair_size = 3 * hidden_size
        self.score = nn.Sequential(
            nn.Linear(pair_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        self.value = nn.Sequential(
            nn.Linear(pair_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.scale = math.sqrt(hidden_size)

        # Start from the caller-provided geometric prior. The learned score is
        # initially exactly zero and can subsequently strengthen, weaken, or
        # reverse that prior from data.
        nn.init.zeros_(self.score[-1].weight)
        nn.init.zeros_(self.score[-1].bias)

    def forward(
        self,
        node_h: Tensor,
        edge_h: Tensor,
        edge_index: Tensor,
        attention_bias: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        source, target = edge_index
        pair_h = torch.cat([node_h[source], node_h[target], edge_h], dim=-1)
        logits = self.score(pair_h).squeeze(-1) / self.scale
        if attention_bias is not None:
            if attention_bias.shape != logits.shape:
                raise ValueError(
                    "attention_bias must have one scalar per directed edge; "
                    f"got {tuple(attention_bias.shape)} for {tuple(logits.shape)} logits"
                )
            logits = logits + attention_bias.to(device=logits.device, dtype=logits.dtype)
        attention = softmax(logits, target, num_nodes=node_h.size(0))
        messages = self.value(pair_h) * attention.unsqueeze(-1)
        aggregate = torch.zeros_like(node_h)
        aggregate.index_add_(0, target, messages)
        node_h = self.norm(node_h + self.update(torch.cat([node_h, aggregate], dim=-1)))
        return node_h, attention


class CompleteGraphAttentionSimulator(nn.Module):
    """Predict standardized per-node updates without a global bottleneck.

    The caller decides whether the update represents displacement, velocity,
    or acceleration and is responsible for integrating the predicted state.
    """

    def __init__(
        self,
        *,
        node_dim: int = 5,
        edge_dim: int = 13,
        hidden_size: int = 50,
        layers: int = 3,
        output_dim: int = 2,
    ):
        super().__init__()
        self.node_dim = int(node_dim)
        self.hidden_size = int(hidden_size)
        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_dim, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(int(edge_dim), self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.layers = nn.ModuleList(
            CompleteGraphAttentionLayer(self.hidden_size) for _ in range(int(layers))
        )
        self.node_skip = nn.Linear(self.node_dim, int(output_dim))
        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size + self.node_dim, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, int(output_dim)),
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
        attentions = []
        for layer in self.layers:
            node_h, attention = layer(
                node_h, edge_h, edge_index, attention_bias=attention_bias
            )
            if return_attention:
                attentions.append(attention)
        # Preserve the original node-specific structure explicitly at the
        # output. Attention learns the interaction correction; the linear
        # skip can represent simple local/affine motion directly.
        prediction = self.node_skip(node_features) + self.decoder(
            torch.cat([node_h, node_features], dim=-1)
        )
        return (prediction, attentions) if return_attention else prediction
