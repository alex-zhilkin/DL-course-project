"""Static-conditioned scalar dynamic autoencoder and latent propagator."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .models import PyramidAttentionPool


class ConditionalDynamicAutoEncoder(nn.Module):
    def __init__(self, *, static_node_dim: int, static_dim: int, hidden_size: int = 96, dynamic_dim: int = 1):
        super().__init__()
        self.dynamic_dim = int(dynamic_dim)
        self.node_encoder = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.dynamic_pool = PyramidAttentionPool(hidden_size, dynamic_dim)
        self.decoder = nn.Sequential(
            nn.Linear(static_node_dim + static_dim + dynamic_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 2),
        )

    def _raw_encode(self, displacement: Tensor) -> Tensor:
        node_h = self.node_encoder(displacement)
        return self.dynamic_pool(node_h).unsqueeze(0)

    def encode(self, displacement: Tensor) -> Tensor:
        """Encode displacement only and anchor every reference frame at d=0."""

        zero = torch.zeros_like(displacement)
        return self._raw_encode(displacement) - self._raw_encode(zero)

    def decode(self, dynamic: Tensor, static_nodes: Tensor, static: Tensor) -> Tensor:
        decoder_input = torch.cat([
            static_nodes,
            static.expand(static_nodes.size(0), -1),
            dynamic.expand(static_nodes.size(0), -1),
        ], dim=-1)
        zero_input = torch.cat([
            static_nodes,
            static.expand(static_nodes.size(0), -1),
            torch.zeros_like(dynamic).expand(static_nodes.size(0), -1),
        ], dim=-1)
        return self.decoder(decoder_input) - self.decoder(zero_input)

    def forward(self, displacement: Tensor, static_nodes: Tensor, static: Tensor):
        dynamic = self.encode(displacement)
        return self.decode(dynamic, static_nodes, static), dynamic


class StaticConditionedDeltaPropagator(nn.Module):
    def __init__(self, *, static_dim: int, dynamic_dim: int = 1, hidden_size: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(static_dim + dynamic_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, dynamic_dim),
        )

    def forward(self, dynamic: Tensor, static: Tensor) -> Tensor:
        return dynamic + self.network(torch.cat([dynamic, static], dim=-1))


class StaticConditionedNextStepSimulator(nn.Module):
    """Re-encode current displacement and predict its next residual increment."""

    def __init__(self, *, static_node_dim: int, static_dim: int, hidden_size: int = 96,
                 dynamic_dim: int = 1):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(2, hidden_size), nn.GELU(),
            nn.Linear(hidden_size, hidden_size), nn.GELU(),
        )
        self.to_dynamic = nn.Linear(hidden_size, dynamic_dim)
        self.increment_decoder = nn.Sequential(
            nn.Linear(static_node_dim + static_dim + dynamic_dim, hidden_size),
            nn.GELU(), nn.Linear(hidden_size, hidden_size), nn.GELU(),
            nn.Linear(hidden_size, 2),
        )

    def _raw_encode(self, displacement: Tensor) -> Tensor:
        return self.to_dynamic(self.node_encoder(displacement).mean(dim=0, keepdim=True))

    def encode(self, displacement: Tensor) -> Tensor:
        return self._raw_encode(displacement) - self._raw_encode(torch.zeros_like(displacement))

    def predict_increment(self, dynamic: Tensor, static_nodes: Tensor, static: Tensor) -> Tensor:
        return self.increment_decoder(torch.cat([
            static_nodes,
            static.expand(static_nodes.size(0), -1),
            dynamic.expand(static_nodes.size(0), -1),
        ], dim=-1))

    def step(self, displacement: Tensor, static_nodes: Tensor, static: Tensor):
        dynamic = self.encode(displacement)
        next_displacement = displacement + self.predict_increment(dynamic, static_nodes, static)
        return next_displacement, dynamic


class AttentionStaticConditionedPropagator(nn.Module):
    """Self-attention over latent-coordinate tokens followed by a residual update."""

    def __init__(self, *, static_dim: int, dynamic_dim: int, hidden_size: int = 64,
                 heads: int = 4, layers: int = 2):
        super().__init__()
        self.dynamic_dim = int(dynamic_dim)
        self.coordinate_embedding = nn.Parameter(torch.randn(dynamic_dim, hidden_size) * 0.01)
        self.token_in = nn.Linear(1 + static_dim, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=heads, dim_feedforward=2 * hidden_size,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.delta_out = nn.Linear(hidden_size, 1)

    def forward(self, dynamic: Tensor, static: Tensor) -> Tensor:
        static_tokens = static.unsqueeze(1).expand(-1, self.dynamic_dim, -1)
        tokens = self.token_in(torch.cat([dynamic.unsqueeze(-1), static_tokens], dim=-1))
        tokens = tokens + self.coordinate_embedding.unsqueeze(0)
        delta = self.delta_out(self.transformer(tokens)).squeeze(-1)
        return dynamic + delta


__all__ = [
    "ConditionalDynamicAutoEncoder",
    "StaticConditionedDeltaPropagator",
    "StaticConditionedNextStepSimulator",
    "AttentionStaticConditionedPropagator",
]
