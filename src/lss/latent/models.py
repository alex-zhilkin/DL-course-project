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


class DirectLatentAttentionPool(nn.Module):
    """Use one learned query to produce each scalar latent coordinate."""

    def __init__(self, hidden_size: int, latent_dim: int):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.latent_dim = int(latent_dim)
        self.queries = nn.Parameter(torch.randn(latent_dim, hidden_size) * 0.01)
        self.query_proj = nn.Linear(hidden_size, hidden_size)
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, latent_dim)

    def forward(self, tokens: Tensor) -> Tensor:
        q = self.query_proj(self.queries)  # [D, H]
        k = self.key_proj(tokens)  # [T, H]
        values = self.value_proj(tokens).transpose(0, 1)  # [D, T]
        attention = torch.softmax(
            (q @ k.transpose(0, 1)) / math.sqrt(self.hidden_size), dim=-1
        )  # [D, T]
        return (attention * values).sum(dim=-1)  # [D]


class PyramidAttentionPool(nn.Module):
    """Hierarchical attention reduction: N nodes -> 100 -> 20 -> latent_dim."""

    def __init__(self, hidden_size: int, latent_dim: int):
        super().__init__()
        self.nodes_to_100 = SimpleAttentionPool(hidden_size, 100)
        self.tokens_100_to_20 = SimpleAttentionPool(hidden_size, 20)
        self.tokens_20_to_latent = DirectLatentAttentionPool(
            hidden_size, latent_dim
        )

    def forward(self, node_features: Tensor) -> Tensor:
        tokens_100 = self.nodes_to_100(node_features)
        tokens_20 = self.tokens_100_to_20(tokens_100)
        return self.tokens_20_to_latent(tokens_20)


class SimpleMLPPool(nn.Module):
    """Pool a variable token set using MLP-produced weights, without Q/K attention."""

    def __init__(self, hidden_size: int, output_tokens: int):
        super().__init__()
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_tokens),
        )
        self.value_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )

    def forward(self, tokens: Tensor) -> Tensor:
        weights = torch.softmax(self.score_mlp(tokens).transpose(0, 1), dim=-1)
        return weights @ self.value_mlp(tokens)


class DirectLatentMLPPool(nn.Module):
    """Reduce tokens to scalar latent coordinates using MLP-produced weights."""

    def __init__(self, hidden_size: int, latent_dim: int):
        super().__init__()
        self.score_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, latent_dim),
        )
        self.value_proj = nn.Linear(hidden_size, latent_dim)

    def forward(self, tokens: Tensor) -> Tensor:
        weights = torch.softmax(self.score_mlp(tokens).transpose(0, 1), dim=-1)
        values = self.value_proj(tokens).transpose(0, 1)
        return (weights * values).sum(dim=-1)


class PyramidMLPPool(nn.Module):
    """Hierarchical MLP-weighted reduction: N nodes -> 100 -> 20 -> latent_dim."""

    def __init__(self, hidden_size: int, latent_dim: int):
        super().__init__()
        self.nodes_to_100 = SimpleMLPPool(hidden_size, 100)
        self.tokens_100_to_20 = SimpleMLPPool(hidden_size, 20)
        self.tokens_20_to_latent = DirectLatentMLPPool(hidden_size, latent_dim)

    def forward(self, node_features: Tensor) -> Tensor:
        tokens_100 = self.nodes_to_100(node_features)
        tokens_20 = self.tokens_100_to_20(tokens_100)
        return self.tokens_20_to_latent(tokens_20)


class NodeDeltaAttentionAutoEncoder(nn.Module):
    def __init__(
        self,
        *,
        pos_dim: int,
        edge_dim: int,
        hidden_size: int,
        latent_dim: int,
        latent_tokens: int,
        node_feature_dim: int | None = None,
        reconstruction_dim: int | None = None,
    ):
        super().__init__()
        self.pos_dim = int(pos_dim)
        self.edge_dim = int(edge_dim)
        self.hidden_size = int(hidden_size)
        self.latent_dim = int(latent_dim)
        self.latent_tokens = int(latent_tokens)
        self.node_feature_dim = int(node_feature_dim or pos_dim)
        self.reconstruction_dim = int(reconstruction_dim or pos_dim)

        self.edge_in = nn.Linear(edge_dim, hidden_size)
        self.ref_edge_in = nn.Linear(edge_dim, hidden_size)
        self.ref_node_in = nn.Linear(pos_dim + hidden_size, hidden_size)
        self.node_in = nn.Linear(self.node_feature_dim + hidden_size, hidden_size)
        self.pool = PyramidAttentionPool(hidden_size, latent_dim)
        self.to_latent = None

        self.decoder_token_proj = nn.Linear(latent_dim, latent_tokens * hidden_size)
        self.decoder_query_proj = nn.Linear(hidden_size, hidden_size)
        self.decoder_key_proj = nn.Linear(hidden_size, hidden_size)
        self.decoder_value_proj = nn.Linear(hidden_size, hidden_size)
        self.node_decoder = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.reconstruction_dim),
        )

    def encode_reference_graph(
        self, ref_pos: Tensor, ref_edge_attr: Tensor, edge_index: Tensor
    ) -> Tensor:
        edge_node = self.aggregate_edges(
            ref_edge_attr, edge_index, ref_pos.size(0), projection=self.ref_edge_in
        )
        return self.ref_node_in(torch.cat([ref_pos, edge_node], dim=-1))

    def aggregate_edges(
        self,
        edge_attr: Tensor,
        edge_index: Tensor,
        num_nodes: int,
        *,
        projection: nn.Linear | None = None,
    ) -> Tensor:
        if edge_attr.numel() == 0:
            return torch.zeros(
                num_nodes, self.hidden_size, device=edge_attr.device, dtype=edge_attr.dtype
            )

        row, col = edge_index
        # Serialized datasets now carry one canonical edge per unordered pair.
        # Treat that edge as incident to both endpoints inside the model.  This
        # also remains compatible with older reciprocal edge lists: each
        # endpoint view is merely repeated and the mean is unchanged.
        reverse_attr = edge_attr.clone()
        if edge_attr.size(-1) == 4:
            directional_columns = [0, 1]
        elif edge_attr.size(-1) >= 11:
            # Expanded latent edge layout:
            # ref_vec, cur_vec, lengths/stretch/stiffness, raw_cur_vec, ...
            directional_columns = [0, 1, 2, 3, 9, 10]
        else:
            directional_columns = list(range(min(self.pos_dim, edge_attr.size(-1))))
        reverse_attr[:, directional_columns] *= -1
        endpoint = torch.cat([col, row])
        endpoint_attr = torch.cat([edge_attr, reverse_attr], dim=0)
        # Linear projection commutes with mean aggregation:
        # mean(W e + b) == W mean(e) + b. Pooling the compact raw features
        # first avoids a potentially enormous [N*(N-1), hidden_size]
        # intermediate for complete graphs without changing the result.
        node_sum = torch.zeros(
            num_nodes, edge_attr.size(-1), device=edge_attr.device, dtype=edge_attr.dtype
        )
        node_count = torch.zeros(
            num_nodes, 1, device=edge_attr.device, dtype=edge_attr.dtype
        )
        node_sum.index_add_(0, endpoint, endpoint_attr)
        node_count.index_add_(
            0,
            endpoint,
            torch.ones(
                endpoint_attr.size(0), 1, device=edge_attr.device, dtype=edge_attr.dtype
            ),
        )
        projected = (projection or self.edge_in)(node_sum / node_count.clamp_min(1.0))
        return projected * (node_count > 0).to(projected.dtype)

    def encode_latent_graph(
        self,
        delta_pos_g: Tensor,
        h0_g: Tensor,
        edge_attr_g: Tensor,
        edge_index_g: Tensor,
    ) -> Tensor:
        edge_delta = self.aggregate_edges(edge_attr_g, edge_index_g, delta_pos_g.size(0))
        h = self.node_in(torch.cat([delta_pos_g, h0_g + edge_delta], dim=-1))  # [N, H]

        return self.pool(h)

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


class NodeDeltaDirectAttentionAutoEncoder(NodeDeltaAttentionAutoEncoder):
    """Attention autoencoder whose decoder directly returns node displacements.

    Each reference-node query attends over latent tokens whose values are
    two-dimensional displacement vectors. The attention-weighted value is the
    node prediction, with no additional node decoder MLP.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.decoder_value_proj = nn.Linear(self.hidden_size, self.pos_dim)
        self.node_decoder = None

    def decode(self, z: Tensor, h0: Tensor, batch: Tensor) -> Tensor:
        z_tokens = self.decoder_token_proj(z).reshape(
            z.size(0), self.latent_tokens, self.hidden_size
        )
        node_tokens = z_tokens[batch]
        q = self.decoder_query_proj(h0).unsqueeze(1)
        k = self.decoder_key_proj(node_tokens)
        displacement_values = self.decoder_value_proj(node_tokens)
        scores = (q * k).sum(dim=-1) / math.sqrt(self.hidden_size)
        attention = torch.softmax(scores, dim=-1)
        return (attention.unsqueeze(-1) * displacement_values).sum(dim=1)


class NodeDeltaSingleStageAttentionAutoEncoder(NodeDeltaAttentionAutoEncoder):
    """Reduce node states directly to one scalar per learned latent query."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pool = DirectLatentAttentionPool(self.hidden_size, self.latent_dim)


class NodeDeltaMLPAutoEncoder(NodeDeltaAttentionAutoEncoder):
    """Minimal expressive autoencoder using mean pooling and shallow MLPs."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.ref_node_in = nn.Sequential(
            nn.Linear(self.pos_dim + self.hidden_size, self.hidden_size),
            nn.GELU(),
        )
        self.node_in = nn.Sequential(
            nn.Linear(self.node_feature_dim + self.hidden_size, self.hidden_size),
            nn.GELU(),
        )
        self.pool = None
        self.to_latent = nn.Linear(self.hidden_size, self.latent_dim)
        self.decoder_token_proj = None
        self.decoder_query_proj = None
        self.decoder_key_proj = None
        self.decoder_value_proj = None
        self.decoder_latent_proj = None
        self.node_decoder = nn.Sequential(
            nn.Linear(self.latent_dim + self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.pos_dim),
        )

    def encode_latent_graph(
        self,
        delta_pos_g: Tensor,
        h0_g: Tensor,
        edge_attr_g: Tensor,
        edge_index_g: Tensor,
    ) -> Tensor:
        edge_delta = self.aggregate_edges(edge_attr_g, edge_index_g, delta_pos_g.size(0))
        h = self.node_in(torch.cat([delta_pos_g, h0_g + edge_delta], dim=-1))
        return self.to_latent(h.mean(dim=0))

    def decode(self, z: Tensor, h0: Tensor, batch: Tensor) -> Tensor:
        return self.node_decoder(torch.cat([z[batch], h0], dim=-1))


class NodeDeltaPyramidMLPAutoEncoder(NodeDeltaMLPAutoEncoder):
    """MLP-weighted N -> 100 -> 20 -> latent_dim pyramid without Q/K attention."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pool = PyramidMLPPool(self.hidden_size, self.latent_dim)
        self.to_latent = None
        self.node_decoder = nn.Sequential(
            nn.Linear(self.latent_dim + self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.pos_dim),
        )

    def encode_latent_graph(
        self,
        delta_pos_g: Tensor,
        h0_g: Tensor,
        edge_attr_g: Tensor,
        edge_index_g: Tensor,
    ) -> Tensor:
        edge_delta = self.aggregate_edges(edge_attr_g, edge_index_g, delta_pos_g.size(0))
        h = self.node_in(torch.cat([delta_pos_g, h0_g + edge_delta], dim=-1))
        return self.pool(h)


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


class StaticAttentionContextProjection(nn.Module):
    """Learn several topology-sensitive summaries of static node tokens."""

    def __init__(
        self,
        raw_context_dim: int,
        graph_context_dim: int | None = None,
        *,
        context_tokens: int = 8,
    ):
        super().__init__()
        self.raw_context_dim = int(raw_context_dim)
        self.output_dim = int(graph_context_dim or raw_context_dim)
        self.context_tokens = int(context_tokens)
        self.input_norm = nn.LayerNorm(self.raw_context_dim)
        self.keys = nn.Linear(self.raw_context_dim, self.output_dim)
        self.values = nn.Linear(self.raw_context_dim, self.output_dim)
        self.queries = nn.Parameter(
            torch.randn(self.context_tokens, self.output_dim) * 0.02
        )
        self.fuse = nn.Sequential(
            nn.Linear(self.context_tokens * self.output_dim, self.output_dim),
            nn.GELU(),
            nn.LayerNorm(self.output_dim),
        )

    def forward(self, context: Tensor | None) -> Tensor:
        if context is None:
            raise ValueError("This latent propagator requires static node context.")
        if context.ndim != 2 or context.size(-1) != self.raw_context_dim:
            raise ValueError(
                "Learned static attention expects [nodes, features] context; "
                f"received {tuple(context.shape)}."
            )
        context = self.input_norm(context)
        keys = self.keys(context)
        values = self.values(context)
        scores = (self.queries @ keys.transpose(0, 1)) / math.sqrt(self.output_dim)
        tokens = torch.softmax(scores, dim=-1) @ values
        return self.fuse(tokens.reshape(1, -1))


def _make_context_projection(
    context_dim: int,
    graph_context_dim: int | None,
    context_pool_mode: str,
    *,
    include_temperature: bool = False,
) -> nn.Module:
    pool_mode = str(context_pool_mode).lower()
    if pool_mode in {"learned_attention", "attention", "set_attention"}:
        if include_temperature:
            raise ValueError("Learned attention context does not support temperature features.")
        return StaticAttentionContextProjection(context_dim, graph_context_dim)
    return StaticContextProjection(
        context_dim,
        graph_context_dim,
        include_temperature=include_temperature,
    )


class LatentDynamicsMLP(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        hidden_size: int,
        context_dim: int = 0,
        graph_context_dim: int | None = None,
        context_include_temperature: bool = False,
        context_pool_mode: str = "mean",
    ):
        super().__init__()
        self.context_projection = _make_context_projection(
            context_dim,
            graph_context_dim,
            context_pool_mode,
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
        # Begin from the mean latent increment (normalized output zero), then
        # learn network- and state-specific corrections. This is substantially
        # safer for closed-loop rollout than a random initial velocity field.
        nn.init.xavier_uniform_(self.net[-1].weight)
        self.net[-1].weight.data.mul_(0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        z: Tensor,
        context: Tensor | None = None,
        *,
        context_is_encoded: bool = False,
    ) -> Tensor:
        encoded_context = (
            context if context_is_encoded else self.context_projection(context)
        )
        features = _append_context(z, encoded_context, self.context_dim)
        return z + self.net(features)


class DeltaLatentDynamicsMLP(nn.Module):
    """MLP that predicts normalized latent delta dz directly."""

    predicts_delta = True

    def __init__(
        self,
        latent_dim: int,
        hidden_size: int,
        context_dim: int = 0,
        graph_context_dim: int | None = None,
        context_include_temperature: bool = False,
        context_pool_mode: str = "mean",
    ):
        super().__init__()
        self.context_projection = _make_context_projection(
            context_dim,
            graph_context_dim,
            context_pool_mode,
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
        nn.init.xavier_uniform_(self.net[-1].weight)
        self.net[-1].weight.data.mul_(0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self,
        z: Tensor,
        context: Tensor | None = None,
        *,
        context_is_encoded: bool = False,
    ) -> Tensor:
        encoded_context = (
            context if context_is_encoded else self.context_projection(context)
        )
        features = _append_context(z, encoded_context, self.context_dim)
        return self.net(features)


class SourceClassifyingDeltaLatentDynamicsMLP(DeltaLatentDynamicsMLP):
    """Shared delta propagator with an auxiliary source-classification head."""

    def __init__(self, *args, source_names: list[str], **kwargs):
        super().__init__(*args, **kwargs)
        self.source_names = tuple(str(name) for name in source_names)
        feature_dim = self.net[0].in_features
        hidden_size = self.net[0].out_features
        self.source_classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_size), nn.GELU(),
            nn.Linear(hidden_size, len(self.source_names)),
        )

    def source_logits(self, z: Tensor, context: Tensor | None = None, *, context_is_encoded: bool = False) -> Tensor:
        encoded_context = context if context_is_encoded else self.context_projection(context)
        features = _append_context(z, encoded_context, self.context_dim)
        return self.source_classifier(features)


class LinearLatentDynamics(nn.Module):
    """Linear residual latent dynamics: z_next = z + A z + b."""

    def __init__(
        self,
        latent_dim: int,
        context_dim: int = 0,
        graph_context_dim: int | None = None,
        context_include_temperature: bool = False,
        context_pool_mode: str = "mean",
    ):
        super().__init__()
        self.context_projection = _make_context_projection(
            context_dim,
            graph_context_dim,
            context_pool_mode,
            include_temperature=context_include_temperature,
        )
        self.context_dim = self.context_projection.output_dim
        self.delta = nn.Linear(latent_dim + self.context_dim, latent_dim)

    def forward(
        self,
        z: Tensor,
        context: Tensor | None = None,
        *,
        context_is_encoded: bool = False,
    ) -> Tensor:
        encoded_context = (
            context if context_is_encoded else self.context_projection(context)
        )
        features = _append_context(z, encoded_context, self.context_dim)
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
        context_pool_mode: str = "mean",
    ):
        super().__init__()
        self.context_projection = _make_context_projection(
            context_dim,
            graph_context_dim,
            context_pool_mode,
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

    def forward(
        self,
        z: Tensor,
        context: Tensor | None = None,
        *,
        context_is_encoded: bool = False,
    ) -> Tensor:
        encoded_context = (
            context if context_is_encoded else self.context_projection(context)
        )
        return self.net(_append_context(z, encoded_context, self.context_dim))


class KinematicLatentDynamicsMLP(nn.Module):
    """Predict a next latent state from position-like and velocity-like state.

    The input is ``[q_t, dz_t, z_reference, progress]`` where ``q_t`` is the
    latent displacement from the initial encoded graph.  The output is the
    next normalized latent displacement.  A near-identity initialization keeps
    an untrained closed-loop rollout bounded while the forcing term is learned.
    """

    uses_kinematic_state = True

    def __init__(
        self,
        latent_dim: int,
        hidden_size: int,
        context_dim: int = 0,
        graph_context_dim: int | None = None,
        context_include_temperature: bool = False,
        context_pool_mode: str = "mean",
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.context_pool_mode = str(context_pool_mode).lower()
        if self.context_pool_mode in {
            "learned_attention",
            "attention",
            "set_attention",
        }:
            self.context_projection = StaticAttentionContextProjection(
                context_dim,
                graph_context_dim,
            )
        else:
            self.context_projection = StaticContextProjection(
                context_dim,
                graph_context_dim,
                include_temperature=context_include_temperature,
            )
        self.context_dim = self.context_projection.output_dim
        state_dim = 3 * self.latent_dim + 1
        self.state_norm = nn.LayerNorm(state_dim)
        self.net = nn.Sequential(
            nn.Linear(state_dim + self.context_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.latent_dim),
        )
        nn.init.xavier_uniform_(self.net[-1].weight)
        self.net[-1].weight.data.mul_(0.01)
        nn.init.zeros_(self.net[-1].bias)

    def encode_context(self, context: Tensor | None) -> Tensor | None:
        if self.context_dim <= 0:
            return context
        projected = self.context_projection(context)
        return projected.unsqueeze(0) if projected.ndim == 1 else projected

    def forward(
        self,
        state: Tensor,
        context: Tensor | None = None,
        *,
        context_is_encoded: bool = False,
    ) -> Tensor:
        if state.size(-1) != 3 * self.latent_dim + 1:
            raise ValueError(
                f"Expected kinematic state width {3 * self.latent_dim + 1}, "
                f"received {state.size(-1)}."
            )
        q = state[..., : self.latent_dim]
        encoded_context = (
            context if context_is_encoded else self.encode_context(context)
        )
        features = _append_context(
            self.state_norm(state),
            encoded_context,
            self.context_dim,
        )
        return q + self.net(features)


class HistoryLatentDynamicsMLP(nn.Module):
    """Predict a latent acceleration from three consecutive observations.

    The state is ``[q_t, v_t, a_t, z_reference]``.  Here ``q_t`` is the
    displacement from the reference latent, ``v_t`` is the latest finite
    difference, and ``a_t`` is the difference between the two latest finite
    differences.  The output is an acceleration correction; rollout updates
    velocity first and then adds that velocity to the current latent. No time,
    temperature, or future trajectory metadata is included.
    """

    uses_history_state = True

    def __init__(
        self,
        latent_dim: int,
        hidden_size: int,
        context_dim: int = 0,
        graph_context_dim: int | None = None,
        context_include_temperature: bool = False,
        context_pool_mode: str = "mean",
        depth: int = 3,
        activation: str = "gelu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.context_pool_mode = str(context_pool_mode).lower()
        if self.context_pool_mode in {
            "learned_attention",
            "attention",
            "set_attention",
        }:
            self.context_projection = StaticAttentionContextProjection(
                context_dim,
                graph_context_dim,
            )
        else:
            self.context_projection = StaticContextProjection(
                context_dim,
                graph_context_dim,
                include_temperature=context_include_temperature,
            )
        self.context_dim = self.context_projection.output_dim
        state_dim = 4 * self.latent_dim
        self.state_norm = nn.LayerNorm(state_dim)
        if int(depth) < 1:
            raise ValueError("History propagator depth must be at least one.")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("History propagator dropout must lie in [0, 1).")
        activation_cls = {
            "gelu": nn.GELU,
            "silu": nn.SiLU,
            "relu": nn.ReLU,
            "leaky_relu": nn.LeakyReLU,
        }.get(str(activation).lower())
        if activation_cls is None:
            raise ValueError(f"Unknown history propagator activation: {activation}")
        layers: list[nn.Module] = []
        input_dim = state_dim + self.context_dim
        for _ in range(int(depth)):
            layers.extend([nn.Linear(input_dim, hidden_size), activation_cls()])
            if float(dropout) > 0:
                layers.append(nn.Dropout(float(dropout)))
            input_dim = hidden_size
        layers.append(nn.Linear(hidden_size, self.latent_dim))
        self.net = nn.Sequential(*layers)
        nn.init.xavier_uniform_(self.net[-1].weight)
        self.net[-1].weight.data.mul_(0.01)
        nn.init.zeros_(self.net[-1].bias)

    def encode_context(self, context: Tensor | None) -> Tensor | None:
        if self.context_dim <= 0:
            return context
        projected = self.context_projection(context)
        return projected.unsqueeze(0) if projected.ndim == 1 else projected

    def forward(
        self,
        state: Tensor,
        context: Tensor | None = None,
        *,
        context_is_encoded: bool = False,
    ) -> Tensor:
        expected = 4 * self.latent_dim
        if state.size(-1) != expected:
            raise ValueError(
                f"Expected three-frame history state width {expected}, "
                f"received {state.size(-1)}."
            )
        encoded_context = (
            context if context_is_encoded else self.encode_context(context)
        )
        features = _append_context(
            self.state_norm(state),
            encoded_context,
            self.context_dim,
        )
        return self.net(features)


class HistoryDeltaLatentDynamicsMLP(HistoryLatentDynamicsMLP):
    """Predict the full next latent displacement from three-frame history.

    The input state is identical to :class:`HistoryLatentDynamicsMLP`, but the
    output represents ``(z_next - z_t) / dz_std`` directly rather than a
    correction to the latest latent velocity.
    """

    predicts_direct_history_delta = True


class HistoryLatentAttentionDynamics(nn.Module):
    """Self-attention acceleration model over the latent-coordinate tokens.

    Each of the ``latent_dim`` tokens contains its current displacement,
    velocity, acceleration, and reference value.  Learned positional tokens
    keep latent-coordinate identity, while attention can represent coupled
    coordinate dynamics.  It predicts the same acceleration correction as
    :class:`HistoryLatentDynamicsMLP`.
    """

    uses_history_state = True

    def __init__(
        self,
        latent_dim: int,
        hidden_size: int,
        context_dim: int = 0,
        graph_context_dim: int | None = None,
        context_include_temperature: bool = False,
        context_pool_mode: str = "mean",
        attention_heads: int = 2,
        attention_layers: int = 1,
    ):
        super().__init__()
        if int(hidden_size) % int(attention_heads) != 0:
            raise ValueError("attention hidden_size must divide evenly by attention_heads.")
        self.latent_dim = int(latent_dim)
        self.context_pool_mode = str(context_pool_mode).lower()
        if self.context_pool_mode in {"learned_attention", "attention", "set_attention"}:
            self.context_projection = StaticAttentionContextProjection(
                context_dim, graph_context_dim
            )
        else:
            self.context_projection = StaticContextProjection(
                context_dim,
                graph_context_dim,
                include_temperature=context_include_temperature,
            )
        self.context_dim = self.context_projection.output_dim
        self.token_norm = nn.LayerNorm(4)
        self.token_input = nn.Linear(4, hidden_size)
        self.position = nn.Parameter(torch.randn(1, self.latent_dim, hidden_size) * 0.01)
        self.context_to_token = (
            nn.Linear(self.context_dim, hidden_size)
            if self.context_dim > 0
            else None
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=int(attention_heads),
            dim_feedforward=2 * hidden_size,
            activation="gelu",
            batch_first=True,
            norm_first=True,
            dropout=0.0,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=int(attention_layers), enable_nested_tensor=False
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output = nn.Linear(hidden_size, 1)
        nn.init.xavier_uniform_(self.output.weight)
        self.output.weight.data.mul_(0.01)
        nn.init.zeros_(self.output.bias)

    def encode_context(self, context: Tensor | None) -> Tensor | None:
        if self.context_dim <= 0:
            return context
        projected = self.context_projection(context)
        return projected.unsqueeze(0) if projected.ndim == 1 else projected

    def forward(self, state: Tensor, context: Tensor | None = None, *, context_is_encoded: bool = False) -> Tensor:
        expected = 4 * self.latent_dim
        if state.size(-1) != expected:
            raise ValueError(f"Expected three-frame history state width {expected}, received {state.size(-1)}.")
        tokens = state.reshape(state.size(0), 4, self.latent_dim).transpose(1, 2)
        tokens = self.token_input(self.token_norm(tokens)) + self.position
        encoded_context = context if context_is_encoded else self.encode_context(context)
        if self.context_to_token is not None:
            if encoded_context is None:
                raise ValueError("This latent propagator requires static network context.")
            tokens = tokens + self.context_to_token(encoded_context).unsqueeze(1)
        tokens = self.encoder(tokens)
        return self.output(self.output_norm(tokens)).squeeze(-1)


class LaggedHistoryLatentDynamicsMLP(nn.Module):
    """Predict evolving motion from a noise-robust rolling latent window."""

    uses_lagged_history = True

    def __init__(
        self,
        latent_dim: int,
        hidden_size: int,
        context_dim: int = 0,
        graph_context_dim: int | None = None,
        context_include_temperature: bool = False,
        context_pool_mode: str = "mean",
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.context_pool_mode = str(context_pool_mode).lower()
        self.context_projection = StaticContextProjection(
            context_dim,
            graph_context_dim,
            include_temperature=context_include_temperature,
        )
        self.context_dim = self.context_projection.output_dim
        self.net = nn.Sequential(
            nn.Linear(3 * self.latent_dim + self.context_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.latent_dim),
        )
        nn.init.xavier_uniform_(self.net[-1].weight)
        self.net[-1].weight.data.mul_(0.01)
        nn.init.zeros_(self.net[-1].bias)

    def encode_context(self, context: Tensor | None) -> Tensor | None:
        if self.context_dim <= 0:
            return context
        projected = self.context_projection(context)
        return projected.unsqueeze(0) if projected.ndim == 1 else projected

    def forward(self, state, context=None, *, context_is_encoded=False):
        encoded_context = context if context_is_encoded else self.encode_context(context)
        return self.net(_append_context(state, encoded_context, self.context_dim))


class FixedObservedLatentDynamicsMLP(nn.Module):
    """Predict delta-z from the current state and two fixed observed latents.

    This model is intentionally autonomous: it receives no frame index or
    normalized trajectory progress.
    """

    uses_fixed_observed_state = True

    def __init__(
        self,
        latent_dim: int,
        hidden_size: int,
        context_dim: int = 0,
        graph_context_dim: int | None = None,
        context_include_temperature: bool = False,
        context_pool_mode: str = "mean",
        include_progress: bool = False,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.context_pool_mode = str(context_pool_mode).lower()
        if self.context_pool_mode in {
            "learned_attention",
            "attention",
            "set_attention",
        }:
            self.context_projection = StaticAttentionContextProjection(
                context_dim,
                graph_context_dim,
            )
        else:
            self.context_projection = StaticContextProjection(
                context_dim,
                graph_context_dim,
                include_temperature=context_include_temperature,
            )
        self.context_dim = self.context_projection.output_dim
        self.include_progress = bool(include_progress)
        state_dim = 3 * self.latent_dim + int(self.include_progress)
        # Each latent block is already standardized by LatentNormalizer.
        # Preserve relative magnitude because z(k)-z(1) carries speed.
        self.state_norm = nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(state_dim + self.context_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.latent_dim),
        )
        nn.init.xavier_uniform_(self.net[-1].weight)
        self.net[-1].weight.data.mul_(0.01)
        nn.init.zeros_(self.net[-1].bias)

    def encode_context(self, context: Tensor | None) -> Tensor | None:
        if self.context_dim <= 0:
            return context
        projected = self.context_projection(context)
        return projected.unsqueeze(0) if projected.ndim == 1 else projected

    def forward(
        self,
        state: Tensor,
        context: Tensor | None = None,
        *,
        context_is_encoded: bool = False,
    ) -> Tensor:
        expected = 3 * self.latent_dim + int(self.include_progress)
        if state.size(-1) != expected:
            raise ValueError(
                f"Expected fixed-history state width {expected}, "
                f"received {state.size(-1)}."
            )
        encoded_context = (
            context if context_is_encoded else self.encode_context(context)
        )
        features = _append_context(
            self.state_norm(state),
            encoded_context,
            self.context_dim,
        )
        return self.net(features)


class FixedVelocityResidualLatentDynamicsMLP(FixedObservedLatentDynamicsMLP):
    """Correct the constant velocity inferred from two fixed observations."""

    uses_fixed_velocity_residual = True


class FixedWindowLatentDynamicsMLP(FixedObservedLatentDynamicsMLP):
    """Predict delta-z from the current latent and a fixed observed window.

    Unlike the two-endpoint fixed-history model, this keeps every observed
    latent in the input.  The MLP can learn velocity/acceleration-like
    summaries implicitly; no hand-crafted finite differences are required.
    """

    uses_fixed_window_history = True

    def __init__(self, latent_dim: int, hidden_size: int, *, fixed_history_size: int = 4, **kwargs):
        self.fixed_history_size = int(fixed_history_size)
        if self.fixed_history_size < 2:
            raise ValueError("fixed_history_size must be at least 2.")
        super().__init__(latent_dim, hidden_size, **kwargs)
        state_dim = (1 + self.fixed_history_size) * self.latent_dim + int(self.include_progress)
        self.net = nn.Sequential(
            nn.Linear(state_dim + self.context_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.latent_dim),
        )
        nn.init.xavier_uniform_(self.net[-1].weight)
        self.net[-1].weight.data.mul_(0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state, context=None, *, context_is_encoded=False):
        expected = (1 + self.fixed_history_size) * self.latent_dim + int(self.include_progress)
        if state.size(-1) != expected:
            raise ValueError(
                f"Expected fixed-window state width {expected}, received {state.size(-1)}."
            )
        encoded_context = context if context_is_encoded else self.encode_context(context)
        return self.net(_append_context(self.state_norm(state), encoded_context, self.context_dim))


class FixedWindowVelocityResidualLatentDynamicsMLP(FixedWindowLatentDynamicsMLP):
    """Four-frame window model with a local constant-velocity residual anchor."""

    uses_fixed_velocity_residual = True


class FixedWindowHistoryContextLatentDynamicsMLP(FixedWindowLatentDynamicsMLP):
    """Compress a fixed observed prefix into a learned motion context."""

    def __init__(
        self,
        latent_dim: int,
        hidden_size: int,
        *,
        motion_context_dim: int = 6,
        **kwargs,
    ):
        self.motion_context_dim = int(motion_context_dim)
        if self.motion_context_dim < 1:
            raise ValueError("motion_context_dim must be positive.")
        super().__init__(latent_dim, hidden_size, **kwargs)
        self.history_encoder = nn.GRU(
            input_size=self.latent_dim,
            hidden_size=self.motion_context_dim,
            batch_first=True,
        )
        state_dim = self.latent_dim + self.motion_context_dim + int(self.include_progress)
        self.net = nn.Sequential(
            nn.Linear(state_dim + self.context_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.latent_dim),
        )
        nn.init.xavier_uniform_(self.net[-1].weight)
        self.net[-1].weight.data.mul_(0.01)
        nn.init.zeros_(self.net[-1].bias)

    def _motion_features(self, state):
        expected = (1 + self.fixed_history_size) * self.latent_dim + int(self.include_progress)
        if state.size(-1) != expected:
            raise ValueError(
                f"Expected fixed-window state width {expected}, received {state.size(-1)}."
            )
        current = state[..., :self.latent_dim]
        prefix_end = (1 + self.fixed_history_size) * self.latent_dim
        observed = state[..., self.latent_dim:prefix_end].reshape(
            state.size(0), self.fixed_history_size, self.latent_dim
        )
        _, hidden = self.history_encoder(observed)
        return current, hidden[-1], state[..., prefix_end:]

    def forward(self, state, context=None, *, context_is_encoded=False):
        current, motion_context, progress = self._motion_features(state)
        features = torch.cat([current, motion_context], dim=-1)
        if self.include_progress:
            features = torch.cat([features, progress], dim=-1)
        encoded_context = context if context_is_encoded else self.encode_context(context)
        return self.net(_append_context(features, encoded_context, self.context_dim))


class FixedWindowHistoryGatedContextLatentDynamicsMLP(
    FixedWindowHistoryContextLatentDynamicsMLP
):
    """Use early motion to gate, rather than replace, graph context."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.context_dim <= 0:
            raise ValueError("History-gated dynamics requires static graph context.")
        self.graph_context_gate = nn.Linear(self.motion_context_dim, 1)
        nn.init.zeros_(self.graph_context_gate.weight)
        nn.init.zeros_(self.graph_context_gate.bias)

    def forward(self, state, context=None, *, context_is_encoded=False):
        current, motion_context, progress = self._motion_features(state)
        encoded_context = context if context_is_encoded else self.encode_context(context)
        if encoded_context is None:
            raise ValueError("History-gated dynamics requires static graph context.")
        gate = torch.sigmoid(self.graph_context_gate(motion_context))
        features = torch.cat([current, motion_context], dim=-1)
        if self.include_progress:
            features = torch.cat([features, progress], dim=-1)
        return self.net(
            _append_context(features, gate * encoded_context, self.context_dim)
        )


class SourceConditionedFixedVelocityResidualLatentDynamicsMLP(
    FixedVelocityResidualLatentDynamicsMLP
):
    """Shared fixed-history trunk with a small source-specific output head."""

    def __init__(self, latent_dim: int, hidden_size: int, **kwargs):
        super().__init__(latent_dim, hidden_size, **kwargs)
        self.source_heads = nn.ModuleList(
            [nn.Linear(hidden_size, latent_dim) for _ in range(4)]
        )
        trunk_layers = list(self.net.children())[:-1]
        self.net = nn.Sequential(*trunk_layers)
        for head in self.source_heads:
            nn.init.xavier_uniform_(head.weight)
            head.weight.data.mul_(0.01)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        state: Tensor,
        context: Tensor | None = None,
        *,
        context_is_encoded: bool = False,
    ) -> Tensor:
        if context is None:
            raise ValueError("Source-conditioned dynamics requires a source token.")
        encoded_context = (
            context if context_is_encoded else self.encode_context(context)
        )
        features = _append_context(state, encoded_context, self.context_dim)
        hidden = self.net(features)
        heads = torch.stack([head(hidden) for head in self.source_heads], dim=-2)
        source_index = encoded_context[..., :4].argmax(dim=-1).long()
        gather_index = source_index.reshape(*source_index.shape, 1, 1).expand(
            *source_index.shape, 1, self.latent_dim
        )
        return heads.gather(-2, gather_index).squeeze(-2)


class NoisyResidualFixedVelocityResidualLatentDynamicsMLP(
    FixedVelocityResidualLatentDynamicsMLP
):
    """Shared fixed-history dynamics with a zero-initialized noisy-LJ residual."""

    def __init__(self, latent_dim: int, hidden_size: int, **kwargs):
        super().__init__(latent_dim, hidden_size, **kwargs)
        self.noisy_residual = nn.Sequential(
            nn.Linear(3 * latent_dim + self.context_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, latent_dim),
        )
        nn.init.zeros_(self.noisy_residual[-1].weight)
        nn.init.zeros_(self.noisy_residual[-1].bias)

    def forward(
        self,
        state: Tensor,
        context: Tensor | None = None,
        *,
        context_is_encoded: bool = False,
    ) -> Tensor:
        encoded_context = (
            context if context_is_encoded else self.encode_context(context)
        )
        features = _append_context(state, encoded_context, self.context_dim)
        shared = self.net(features)
        residual = self.noisy_residual(features)
        if encoded_context is None or encoded_context.size(-1) < 4:
            raise ValueError(
                "Noisy residual dynamics requires a four-component source token."
            )
        source_id = encoded_context[..., -4:].argmax(dim=-1)
        noisy_mask = (source_id == 3).to(shared.dtype).unsqueeze(-1)
        return shared + noisy_mask * residual


class RecurrentLatentDynamicsGRU(nn.Module):
    """Propagate latent states with a persistent learned recurrent memory."""

    uses_recurrent_memory = True

    def __init__(
        self,
        latent_dim: int,
        hidden_size: int,
        context_dim: int = 0,
        graph_context_dim: int | None = None,
        context_include_temperature: bool = False,
        context_pool_mode: str = "mean",
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden_size = int(hidden_size)
        self.context_pool_mode = str(context_pool_mode).lower()
        self.context_projection = StaticContextProjection(
            context_dim,
            graph_context_dim,
            include_temperature=context_include_temperature,
        )
        self.context_dim = self.context_projection.output_dim
        self.memory = nn.GRUCell(
            2 * self.latent_dim + self.context_dim,
            self.hidden_size,
        )
        self.delta_head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.latent_dim),
        )
        nn.init.xavier_uniform_(self.delta_head[-1].weight)
        self.delta_head[-1].weight.data.mul_(0.01)
        nn.init.zeros_(self.delta_head[-1].bias)

    def initial_memory(self, batch_size: int, *, device, dtype) -> Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)

    def encode_context(self, context: Tensor | None) -> Tensor | None:
        if self.context_dim <= 0:
            return context
        return self.context_projection(context)

    def forward(
        self,
        z: Tensor,
        dz: Tensor,
        memory: Tensor,
        context: Tensor | None = None,
        *,
        context_is_encoded: bool = False,
    ) -> tuple[Tensor, Tensor]:
        encoded_context = (
            context if context_is_encoded else self.encode_context(context)
        )
        features = torch.cat(
            [z, dz]
            + ([] if encoded_context is None else [encoded_context]),
            dim=-1,
        )
        memory = self.memory(features, memory)
        return self.delta_head(memory), memory


class PolarRhoLatentDynamics(nn.Module):
    """Constrained 2D latent dynamics: advance radius and preserve angle."""

    requires_next_z_loss = True
    uses_rho_progress_scale = True

    def __init__(
        self,
        latent_dim: int,
        hidden_size: int,
        context_dim: int = 0,
        graph_context_dim: int | None = None,
        context_include_temperature: bool = False,
    ):
        super().__init__()
        if int(latent_dim) != 2:
            raise ValueError("polar_rho latent dynamics requires latent_dim=2.")
        self.context_projection = StaticContextProjection(
            context_dim,
            graph_context_dim,
            include_temperature=context_include_temperature,
        )
        self.context_dim = self.context_projection.output_dim
        self.delta_rho = nn.Sequential(
            nn.Linear(1 + self.context_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        z: Tensor,
        context: Tensor | None = None,
        *,
        rho_scale: Tensor | None = None,
    ) -> Tensor:
        context = self.context_projection(context)
        rho = torch.linalg.vector_norm(z, dim=-1, keepdim=True).clamp_min(1e-8)
        direction = z / rho
        if rho_scale is None:
            rho_scale = torch.ones_like(rho)
        else:
            rho_scale = rho_scale.to(device=z.device, dtype=z.dtype).reshape_as(rho).clamp_min(1e-8)
        progress = rho / rho_scale
        features = _append_context(progress, context, self.context_dim)
        next_progress = (progress + self.delta_rho(features)).clamp_min(0.0)
        next_rho = next_progress * rho_scale
        return next_rho * direction


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
    context_pool_mode: str = "mean",
    fixed_history_include_progress: bool = False,
    fixed_history_size: int = 4,
    fixed_history_motion_context_dim: int = 6,
    history_depth: int = 3,
    history_activation: str = "gelu",
    history_dropout: float = 0.0,
    history_attention_heads: int = 2,
    history_attention_layers: int = 1,
    source_names: list[str] | None = None,
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
    if model_type in {"residual", "residual_mlp"}:
        return LatentDynamicsMLP(
            latent_dim, hidden_size,
            context_pool_mode=context_pool_mode,
            **context_kwargs,
        )
    if model_type in {"delta", "delta_mlp", "dz_mlp"}:
        return DeltaLatentDynamicsMLP(
            latent_dim, hidden_size,
            context_pool_mode=context_pool_mode,
            **context_kwargs,
        )
    if model_type in {"delta_source_classifier", "delta_source_classifier_mlp"}:
        return SourceClassifyingDeltaLatentDynamicsMLP(
            latent_dim, hidden_size, source_names=list(source_names or ("unknown",)),
            context_pool_mode=context_pool_mode, **context_kwargs,
        )
    if model_type in {"linear", "linear_residual", "linear_delta"}:
        return LinearLatentDynamics(
            latent_dim,
            context_pool_mode=context_pool_mode,
            **context_kwargs,
        )
    if model_type in {"direct", "direct_mlp", "jepa_mlp", "next_mlp"}:
        return DirectLatentDynamicsMLP(
            latent_dim, hidden_size,
            context_pool_mode=context_pool_mode,
            **context_kwargs,
        )
    if model_type in {
        "kinematic",
        "kinematic_mlp",
        "anchored",
        "anchored_mlp",
        "second_order_direct",
    }:
        return KinematicLatentDynamicsMLP(
            latent_dim,
            hidden_size,
            context_pool_mode=context_pool_mode,
            **context_kwargs,
        )
    if model_type in {
        "history",
        "history_mlp",
        "three_frame",
        "three_frame_mlp",
    }:
        return HistoryLatentDynamicsMLP(
            latent_dim,
            hidden_size,
            context_pool_mode=context_pool_mode,
            depth=history_depth,
            activation=history_activation,
            dropout=history_dropout,
            **context_kwargs,
        )
    if model_type in {
        "history_delta",
        "history_delta_mlp",
        "three_frame_delta",
        "three_frame_delta_mlp",
    }:
        return HistoryDeltaLatentDynamicsMLP(
            latent_dim,
            hidden_size,
            context_pool_mode=context_pool_mode,
            depth=history_depth,
            activation=history_activation,
            dropout=history_dropout,
            **context_kwargs,
        )
    if model_type in {
        "history_attention",
        "history_attention_mlp",
        "three_frame_attention",
    }:
        return HistoryLatentAttentionDynamics(
            latent_dim,
            hidden_size,
            context_pool_mode=context_pool_mode,
            attention_heads=history_attention_heads,
            attention_layers=history_attention_layers,
            **context_kwargs,
        )
    if model_type in {
        "fixed_velocity_residual",
        "fixed_velocity_residual_mlp",
    }:
        return FixedVelocityResidualLatentDynamicsMLP(
            latent_dim,
            hidden_size,
            context_pool_mode=context_pool_mode,
            include_progress=fixed_history_include_progress,
            **context_kwargs,
        )
    if model_type in {
        "source_conditioned_fixed_velocity_residual",
        "source_conditioned_fixed_velocity_residual_mlp",
    }:
        return SourceConditionedFixedVelocityResidualLatentDynamicsMLP(
            latent_dim,
            hidden_size,
            context_pool_mode=context_pool_mode,
            include_progress=fixed_history_include_progress,
            **context_kwargs,
        )
    if model_type in {
        "noisy_residual_fixed_velocity_residual",
        "noisy_residual_fixed_velocity_residual_mlp",
    }:
        return NoisyResidualFixedVelocityResidualLatentDynamicsMLP(
            latent_dim, hidden_size,
            context_pool_mode=context_pool_mode,
            include_progress=fixed_history_include_progress,
            **context_kwargs,
        )
    if model_type in {
        "fixed_window_velocity_residual",
        "fixed_window_velocity_residual_mlp",
    }:
        return FixedWindowVelocityResidualLatentDynamicsMLP(
            latent_dim,
            hidden_size,
            context_pool_mode=context_pool_mode,
            include_progress=fixed_history_include_progress,
            fixed_history_size=fixed_history_size,
            **context_kwargs,
        )
    if model_type in {
        "fixed_window_history_gated_context",
        "fixed_window_history_gated_context_mlp",
        "history_gated_graph_context",
        "history_gated_graph_context_mlp",
    }:
        return FixedWindowHistoryGatedContextLatentDynamicsMLP(
            latent_dim,
            hidden_size,
            context_pool_mode=context_pool_mode,
            include_progress=fixed_history_include_progress,
            fixed_history_size=fixed_history_size,
            motion_context_dim=fixed_history_motion_context_dim,
            **context_kwargs,
        )
    if model_type in {
        "fixed_window_history_context",
        "fixed_window_history_context_mlp",
        "learned_motion_context",
        "learned_motion_context_mlp",
    }:
        return FixedWindowHistoryContextLatentDynamicsMLP(
            latent_dim,
            hidden_size,
            context_pool_mode=context_pool_mode,
            include_progress=fixed_history_include_progress,
            fixed_history_size=fixed_history_size,
            motion_context_dim=fixed_history_motion_context_dim,
            **context_kwargs,
        )
    if model_type in {
        "fixed_window",
        "fixed_window_mlp",
        "fixed_four_frame",
        "fixed_four_frame_mlp",
    }:
        return FixedWindowLatentDynamicsMLP(
            latent_dim,
            hidden_size,
            context_pool_mode=context_pool_mode,
            include_progress=fixed_history_include_progress,
            fixed_history_size=fixed_history_size,
            **context_kwargs,
        )
    if model_type in {
        "fixed_history",
        "fixed_history_mlp",
        "fixed_observed",
        "fixed_observed_mlp",
    }:
        return FixedObservedLatentDynamicsMLP(
            latent_dim,
            hidden_size,
            context_pool_mode=context_pool_mode,
            **context_kwargs,
        )
    if model_type in {"recurrent_memory", "recurrent_memory_gru", "gru_memory"}:
        return RecurrentLatentDynamicsGRU(
            latent_dim,
            hidden_size,
            context_pool_mode=context_pool_mode,
            **context_kwargs,
        )
    if model_type in {"polar", "polar_rho", "rho_theta", "radial"}:
        return PolarRhoLatentDynamics(latent_dim, hidden_size, **context_kwargs)
    if model_type in {"velocity", "velocity_mlp", "second_order_mlp"}:
        return VelocityLatentDynamicsMLP(latent_dim, hidden_size, **context_kwargs)
    raise ValueError(f"Unknown latent propagator model_type: {model_type}")


__all__ = [
    "DirectLatentDynamicsMLP",
    "DeltaLatentDynamicsMLP",
    "FixedObservedLatentDynamicsMLP",
    "FixedWindowHistoryContextLatentDynamicsMLP",
    "FixedWindowHistoryGatedContextLatentDynamicsMLP",
    "FixedWindowLatentDynamicsMLP",
    "FixedVelocityResidualLatentDynamicsMLP",
    "NoisyResidualFixedVelocityResidualLatentDynamicsMLP",
    "LatentDynamicsMLP",
    "KinematicLatentDynamicsMLP",
    "HistoryLatentDynamicsMLP",
    "LaggedHistoryLatentDynamicsMLP",
    "RecurrentLatentDynamicsGRU",
    "LinearLatentDynamics",
    "NodeDeltaAttentionAutoEncoder",
    "NodeDeltaDirectAttentionAutoEncoder",
    "NodeDeltaSingleStageAttentionAutoEncoder",
    "NodeDeltaMLPAutoEncoder",
    "NodeDeltaPyramidMLPAutoEncoder",
    "DirectLatentAttentionPool",
    "PyramidAttentionPool",
    "PyramidMLPPool",
    "PolarRhoLatentDynamics",
    "SimpleAttentionPool",
    "StaticContextProjection",
    "StaticAttentionContextProjection",
    "VelocityLatentDynamicsMLP",
    "make_latent_propagator",
]
