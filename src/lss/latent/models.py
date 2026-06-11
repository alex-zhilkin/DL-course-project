"""Latent-space autoencoder and dynamics models used by notebook 04."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


class SimpleAttentionPool(nn.Module):
    def __init__(self, hidden_size: int, latent_tokens: int):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.latent_tokens = int(latent_tokens)

        self.pool_queries = nn.Parameter(torch.randn(latent_tokens, hidden_size) * 0.01)
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, h: Tensor) -> Tensor:
        q = self.query_proj(self.pool_queries)  # [T, H]
        k = self.key_proj(h)  # [N, H]
        v = self.value_proj(h)  # [N, H]

        scores = (q @ k.transpose(0, 1)) / math.sqrt(self.hidden_size)  # [T, N]
        attn = torch.softmax(scores, dim=-1)  # [T, N]
        tokens = attn @ v  # [T, H]
        return tokens


class NodeDeltaAttentionAutoEncoder(nn.Module):
    def __init__(
        self,
        *,
        pos_dim: int,
        edge_dim: int,
        hidden_size: int,
        latent_dim: int,
        latent_tokens: int,
    ):
        super().__init__()
        self.pos_dim = int(pos_dim)
        self.edge_dim = int(edge_dim)
        self.hidden_size = int(hidden_size)
        self.latent_dim = int(latent_dim)
        self.latent_tokens = int(latent_tokens)

        self.edge_in = nn.Linear(edge_dim, hidden_size)
        self.ref_node_in = nn.Linear(pos_dim + hidden_size, hidden_size)
        self.node_in = nn.Linear(pos_dim + hidden_size, hidden_size)
        self.pool = SimpleAttentionPool(hidden_size, latent_tokens)
        self.to_latent = nn.Linear(latent_tokens * hidden_size, latent_dim)

        self.decoder_token_proj = nn.Linear(latent_dim, latent_tokens * hidden_size)
        self.decoder_query_proj = nn.Linear(hidden_size, hidden_size)
        self.decoder_key_proj = nn.Linear(hidden_size, hidden_size)
        self.decoder_value_proj = nn.Linear(hidden_size, hidden_size)
        self.node_decoder = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, pos_dim),
        )

    def encode_reference_graph(
        self, ref_pos: Tensor, ref_edge_attr: Tensor, edge_index: Tensor
    ) -> Tensor:
        edge_node = self.aggregate_edges(ref_edge_attr, edge_index, ref_pos.size(0))
        return self.ref_node_in(torch.cat([ref_pos, edge_node], dim=-1))

    def aggregate_edges(self, edge_attr: Tensor, edge_index: Tensor, num_nodes: int) -> Tensor:
        if edge_attr.numel() == 0:
            return torch.zeros(
                num_nodes, self.hidden_size, device=edge_attr.device, dtype=edge_attr.dtype
            )

        edge_h = self.edge_in(edge_attr)
        _, col = edge_index

        node_sum = torch.zeros(num_nodes, edge_h.size(-1), device=edge_h.device, dtype=edge_h.dtype)
        node_count = torch.zeros(num_nodes, 1, device=edge_h.device, dtype=edge_h.dtype)

        node_sum.index_add_(0, col, edge_h)
        node_count.index_add_(
            0, col, torch.ones(edge_h.size(0), 1, device=edge_h.device, dtype=edge_h.dtype)
        )

        return node_sum / node_count.clamp_min(1.0)

    def encode_latent_graph(
        self,
        delta_pos_g: Tensor,
        h0_g: Tensor,
        edge_attr_g: Tensor,
        edge_index_g: Tensor,
    ) -> Tensor:
        edge_delta = self.aggregate_edges(edge_attr_g, edge_index_g, delta_pos_g.size(0))
        h = self.node_in(torch.cat([delta_pos_g, h0_g + edge_delta], dim=-1))  # [N, H]

        tokens = self.pool(h)  # [T, H]
        z = self.to_latent(tokens.reshape(1, -1)).squeeze(0)
        return z

    def encode_latent(
        self,
        delta_pos: Tensor,
        h0: Tensor,
        edge_attr: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ) -> Tensor:
        num_graphs = int(batch.max().item()) + 1
        z_list = []

        for graph_idx in range(num_graphs):
            node_idx = (batch == graph_idx).nonzero(as_tuple=False).flatten()
            delta_pos_g = delta_pos[node_idx]
            h0_g = h0[node_idx]

            global_to_local = torch.full(
                (delta_pos.size(0),),
                -1,
                dtype=torch.long,
                device=batch.device,
            )
            global_to_local[node_idx] = torch.arange(node_idx.numel(), device=batch.device)

            edge_mask = (batch[edge_index[0]] == graph_idx) & (batch[edge_index[1]] == graph_idx)
            edge_index_g = global_to_local[edge_index[:, edge_mask]]
            edge_attr_g = edge_attr[edge_mask]

            z_g = self.encode_latent_graph(delta_pos_g, h0_g, edge_attr_g, edge_index_g)
            z_list.append(z_g)

        return torch.stack(z_list, dim=0)

    def decode(self, z: Tensor, h0: Tensor, batch: Tensor) -> Tensor:
        z_tokens = self.decoder_token_proj(z).reshape(
            z.size(0), self.latent_tokens, self.hidden_size
        )
        node_tokens = z_tokens[batch]
        q = self.decoder_query_proj(h0).unsqueeze(1)
        k = self.decoder_key_proj(node_tokens)
        v = self.decoder_value_proj(node_tokens)
        scores = (q * k).sum(dim=-1) / math.sqrt(self.hidden_size)
        attn = torch.softmax(scores, dim=-1)
        z_context = (attn.unsqueeze(-1) * v).sum(dim=1)
        return self.node_decoder(torch.cat([z_context, h0], dim=-1))

    def encode(
        self,
        delta_pos: Tensor,
        ref_pos: Tensor,
        edge_attr: Tensor,
        ref_edge_attr: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ) -> tuple[Tensor, Tensor]:
        h0 = self.encode_reference_graph(ref_pos, ref_edge_attr, edge_index)
        z = self.encode_latent(delta_pos, h0, edge_attr, edge_index, batch)
        return z, h0

    def forward(
        self,
        delta_pos: Tensor,
        ref_pos: Tensor,
        edge_attr: Tensor,
        ref_edge_attr: Tensor,
        edge_index: Tensor,
        batch: Tensor,
    ):
        z, h0 = self.encode(delta_pos, ref_pos, edge_attr, ref_edge_attr, edge_index, batch)
        recon = self.decode(z, h0, batch)
        return recon, z


class StaticContextProjection(nn.Module):
    """Project a pooled graph embedding while preserving optional scalar temperature."""

    def __init__(
        self,
        raw_context_dim: int,
        graph_context_dim: int | None = None,
        *,
        include_temperature: bool = False,
    ):
        super().__init__()
        self.raw_context_dim = int(raw_context_dim)
        self.include_temperature = bool(include_temperature)
        self.raw_graph_dim = self.raw_context_dim - int(self.include_temperature)
        if self.raw_graph_dim < 0:
            raise ValueError("raw_context_dim is too small for temperature conditioning.")
        self.graph_context_dim = (
            self.raw_graph_dim
            if graph_context_dim is None
            else int(graph_context_dim)
        )
        self.output_dim = self.graph_context_dim + int(self.include_temperature)
        self.graph_projection = (
            nn.Identity()
            if self.graph_context_dim == self.raw_graph_dim
            else nn.Sequential(
                nn.Linear(self.raw_graph_dim, self.graph_context_dim),
                nn.GELU(),
            )
        )

    def forward(self, context: Tensor | None) -> Tensor | None:
        if self.raw_context_dim <= 0:
            return context
        if context is None:
            raise ValueError("This latent propagator requires static network context.")
        if context.size(-1) != self.raw_context_dim:
            raise ValueError(
                f"Expected raw_context_dim={self.raw_context_dim}, "
                f"received {context.size(-1)}."
            )
        graph_context = self.graph_projection(context[..., : self.raw_graph_dim])
        if not self.include_temperature:
            return graph_context
        return torch.cat([graph_context, context[..., -1:]], dim=-1)


class LatentDynamicsMLP(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_size: int,
        context_dim: int = 0,
        graph_context_dim: int | None = None,
        context_include_temperature: bool = False,
    ):
        super().__init__()
        self.context_projection = StaticContextProjection(
            context_dim,
            graph_context_dim,
            include_temperature=context_include_temperature,
        )
        self.context_dim = self.context_projection.output_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim + self.context_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, latent_dim),
        )

    def forward(self, z: Tensor, context: Tensor | None = None) -> Tensor:
        features = _append_context(z, self.context_projection(context), self.context_dim)
        return z + self.net(features)


class LinearLatentDynamics(nn.Module):
    """Linear residual latent dynamics: z_next = z + A z + b."""

    def __init__(
        self,
        latent_dim: int,
        context_dim: int = 0,
        graph_context_dim: int | None = None,
        context_include_temperature: bool = False,
    ):
        super().__init__()
        self.context_projection = StaticContextProjection(
            context_dim,
            graph_context_dim,
            include_temperature=context_include_temperature,
        )
        self.context_dim = self.context_projection.output_dim
        self.delta = nn.Linear(latent_dim + self.context_dim, latent_dim)

    def forward(self, z: Tensor, context: Tensor | None = None) -> Tensor:
        features = _append_context(z, self.context_projection(context), self.context_dim)
        return z + self.delta(features)


class DirectLatentDynamicsMLP(nn.Module):
    """MLP that predicts the next latent embedding directly."""

    def __init__(
        self,
        latent_dim: int,
        hidden_size: int,
        context_dim: int = 0,
        graph_context_dim: int | None = None,
        context_include_temperature: bool = False,
    ):
        super().__init__()
        self.context_projection = StaticContextProjection(
            context_dim,
            graph_context_dim,
            include_temperature=context_include_temperature,
        )
        self.context_dim = self.context_projection.output_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim + self.context_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, latent_dim),
        )

    def forward(self, z: Tensor, context: Tensor | None = None) -> Tensor:
        return self.net(
            _append_context(z, self.context_projection(context), self.context_dim)
        )


class VelocityLatentDynamicsMLP(nn.Module):
    """MLP that predicts normalized latent velocity from state and previous velocity."""

    def __init__(
        self,
        latent_dim: int,
        hidden_size: int,
        context_dim: int = 0,
        graph_context_dim: int | None = None,
        context_include_temperature: bool = False,
    ):
        super().__init__()
        self.context_projection = StaticContextProjection(
            context_dim,
            graph_context_dim,
            include_temperature=context_include_temperature,
        )
        self.context_dim = self.context_projection.output_dim
        self.net = nn.Sequential(
            nn.Linear(2 * latent_dim + self.context_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, latent_dim),
        )

    def forward(self, z_and_dz: Tensor, context: Tensor | None = None) -> Tensor:
        return self.net(
            _append_context(
                z_and_dz,
                self.context_projection(context),
                self.context_dim,
            )
        )


def _append_context(state: Tensor, context: Tensor | None, context_dim: int) -> Tensor:
    if context_dim <= 0:
        return state
    if context is None:
        raise ValueError("This latent propagator requires static network context.")
    if context.size(-1) != context_dim:
        raise ValueError(
            f"Expected context_dim={context_dim}, received {context.size(-1)}."
        )
    return torch.cat([state, context], dim=-1)


def make_latent_propagator(
    latent_dim: int,
    hidden_size: int,
    *,
    model_type: str = "residual_mlp",
    context_dim: int = 0,
    graph_context_dim: int | None = None,
    context_include_temperature: bool = False,
) -> nn.Module:
    """Create a latent propagator.

    ``residual_mlp`` preserves the historical behavior: model(z) returns
    z_next in the normalized latent coordinate via a residual update.
    ``direct_mlp`` predicts z_next directly and is useful for JEPA-style
    next-embedding objectives.
    """

    model_type = str(model_type).lower()
    context_kwargs = {
        "context_dim": context_dim,
        "graph_context_dim": graph_context_dim,
        "context_include_temperature": context_include_temperature,
    }
    if model_type in {"residual", "residual_mlp", "delta_mlp"}:
        return LatentDynamicsMLP(latent_dim, hidden_size, **context_kwargs)
    if model_type in {"linear", "linear_residual", "linear_delta"}:
        return LinearLatentDynamics(latent_dim, **context_kwargs)
    if model_type in {"direct", "direct_mlp", "jepa_mlp", "next_mlp"}:
        return DirectLatentDynamicsMLP(latent_dim, hidden_size, **context_kwargs)
    if model_type in {"velocity", "velocity_mlp", "second_order_mlp"}:
        return VelocityLatentDynamicsMLP(latent_dim, hidden_size, **context_kwargs)
    raise ValueError(f"Unknown latent propagator model_type: {model_type}")


__all__ = [
    "DirectLatentDynamicsMLP",
    "LatentDynamicsMLP",
    "LinearLatentDynamics",
    "NodeDeltaAttentionAutoEncoder",
    "SimpleAttentionPool",
    "StaticContextProjection",
    "VelocityLatentDynamicsMLP",
    "make_latent_propagator",
]
