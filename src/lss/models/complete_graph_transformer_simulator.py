"""Simple node transformer preceded by attention over complete-graph edges."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch_geometric.utils import softmax


class CompleteGraphTransformerSimulator(nn.Module):
    """Pool complete-pair information into nodes, then globally mix all nodes.

    This is deliberately not an iterative spatial GNN. There is one incoming
    edge-attention pooling operation followed by ordinary Transformer encoder
    layers over the complete set of node tokens.
    """

    def __init__(
        self,
        *,
        node_dim: int,
        edge_dim: int,
        hidden_size: int = 64,
        edge_hidden_size: int | None = None,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_dropout: float = 0.0,
        edge_aggregation: str = "softmax",
        output_dim: int = 2,
    ):
        super().__init__()
        hidden_size = int(hidden_size)
        edge_hidden_size = int(edge_hidden_size or hidden_size)
        if hidden_size % int(transformer_heads) != 0:
            raise ValueError("hidden_size must be divisible by transformer_heads")
        if edge_aggregation not in {"softmax", "gated_sum"}:
            raise ValueError("edge_aggregation must be 'softmax' or 'gated_sum'")

        self.node_dim = int(node_dim)
        self.edge_aggregation = str(edge_aggregation)
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

        pair_dim = 3 * hidden_size
        self.edge_score = nn.Sequential(
            nn.Linear(pair_dim, hidden_size),
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
        )

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=int(transformer_heads),
            dim_feedforward=4 * hidden_size,
            dropout=float(transformer_dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=int(transformer_layers), norm=nn.LayerNorm(hidden_size)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size + self.node_dim, hidden_size),
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
        # Start edge selection from the supplied geometric prior.
        nn.init.zeros_(self.edge_score[-1].weight)
        nn.init.zeros_(self.edge_score[-1].bias)
        # Begin close to the easily learned local displacement baseline.
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
        source, target = edge_index

        score_input = torch.cat(
            [node_h[source], node_h[target], edge_h], dim=-1
        )
        logits = self.edge_score(score_input).squeeze(-1) / self.score_scale
        if attention_bias is not None:
            if attention_bias.shape != logits.shape:
                raise ValueError(
                    "attention_bias must have one scalar per directed edge; "
                    f"got {tuple(attention_bias.shape)} for {tuple(logits.shape)} logits"
                )
            logits = logits + attention_bias.to(logits)
        if self.edge_aggregation == "softmax":
            attention = softmax(logits, target, num_nodes=node_h.size(0))
        else:
            # Physical pair interactions add. A sigmoid gate keeps complete
            # edges selective without forcing all incoming weights to sum to 1.
            attention = torch.sigmoid(logits)

        values = self.edge_value(torch.cat([node_h[source], edge_h], dim=-1))
        aggregate = torch.zeros_like(node_h)
        aggregate.index_add_(0, target, values * attention.unsqueeze(-1))
        local_tokens = node_h + self.node_fuse(torch.cat([node_h, aggregate], dim=-1))

        # Notebook 09 processes one graph at a time, so a leading singleton is
        # the complete node sequence for a standard all-node Transformer.
        global_tokens = self.transformer(local_tokens.unsqueeze(0)).squeeze(0)
        prediction = self.node_skip(node_features) + self.decoder(
            torch.cat([global_tokens, node_features], dim=-1)
        )
        if return_attention:
            return prediction, attention
        return prediction


__all__ = ["CompleteGraphTransformerSimulator"]
