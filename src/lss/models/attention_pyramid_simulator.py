"""Undirected edge attention followed by an attention U-shaped token pyramid."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch_geometric.utils import softmax


class _CrossAttentionBlock(nn.Module):
    def __init__(self, hidden_size: int, heads: int):
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_size)
        self.context_norm = nn.LayerNorm(hidden_size)
        self.attention = nn.MultiheadAttention(
            hidden_size, heads, batch_first=True
        )
        self.output_norm = nn.LayerNorm(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
        )

    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        attended, _ = self.attention(
            self.query_norm(query),
            self.context_norm(context),
            self.context_norm(context),
            need_weights=False,
        )
        output = query + attended
        return output + self.feed_forward(self.output_norm(output))


class _SelfAttentionBlock(nn.Module):
    def __init__(self, hidden_size: int, heads: int):
        super().__init__()
        self.block = _CrossAttentionBlock(hidden_size, heads)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.block(tokens, tokens)


class AttentionPyramidSimulator(nn.Module):
    """Predict every node jointly through a compress-expand attention pyramid.

    The complete undirected edge set is consumed once to construct the original
    node tokens. Learned queries then compress ``N`` node tokens to a sequence
    of smaller token sets. The decoder expands those tokens again using
    cross-attention and U-Net-style skips before the original nodes query the
    expanded representation. There is no recurrent message passing.
    """

    def __init__(
        self,
        *,
        node_dim: int,
        edge_dim: int,
        hidden_size: int = 96,
        pyramid_tokens: tuple[int, ...] = (32, 16),
        heads: int = 4,
        bottleneck_layers: int = 2,
        latent_dim: int = 0,
        output_dim: int = 2,
    ):
        super().__init__()
        hidden_size = int(hidden_size)
        pyramid_tokens = tuple(int(value) for value in pyramid_tokens)
        if not pyramid_tokens or any(value < 1 for value in pyramid_tokens):
            raise ValueError("pyramid_tokens must contain positive token counts")
        if any(right >= left for left, right in zip(pyramid_tokens, pyramid_tokens[1:])):
            raise ValueError("pyramid_tokens must be strictly decreasing")
        if hidden_size % int(heads):
            raise ValueError("hidden_size must be divisible by heads")

        self.node_dim = int(node_dim)
        self.pyramid_tokens = pyramid_tokens
        self.latent_dim = int(latent_dim)
        self.node_encoder = nn.Sequential(
            nn.Linear(self.node_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(int(edge_dim), hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
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

        self.encoder_queries = nn.ParameterList(
            [
                nn.Parameter(torch.empty(1, token_count, hidden_size))
                for token_count in pyramid_tokens
            ]
        )
        self.encoder_blocks = nn.ModuleList(
            [_CrossAttentionBlock(hidden_size, heads) for _ in pyramid_tokens]
        )
        self.bottleneck = nn.ModuleList(
            [_SelfAttentionBlock(hidden_size, heads) for _ in range(bottleneck_layers)]
        )

        # At each level, the smaller representation queries the saved encoder
        # tokens, restoring resolution without inventing a node ordering.
        self.decoder_queries = nn.ModuleList(
            [
                nn.Linear(hidden_size, hidden_size)
                for _ in reversed(pyramid_tokens[:-1])
            ]
        )
        self.decoder_query_tokens = nn.ParameterList(
            [
                nn.Parameter(torch.empty(1, token_count, hidden_size))
                for token_count in reversed(pyramid_tokens[:-1])
            ]
        )
        self.decoder_blocks = nn.ModuleList(
            [
                _CrossAttentionBlock(hidden_size, heads)
                for _ in reversed(pyramid_tokens[:-1])
            ]
        )
        self.node_query = nn.Linear(hidden_size, hidden_size)
        self.static_node_encoder = nn.Sequential(
            nn.Linear(self.node_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )
        self.node_decode = _CrossAttentionBlock(hidden_size, heads)
        self.local_decoder = nn.Sequential(
            nn.Linear(hidden_size + self.node_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, int(output_dim)),
        )
        self.pyramid_decoder = nn.Sequential(
            nn.Linear(hidden_size + self.node_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, int(output_dim)),
        )
        self.global_gate_logit = nn.Parameter(torch.tensor(-4.0))
        if self.latent_dim > 0:
            self.latent_down = nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, self.latent_dim),
            )
            self.latent_up = nn.Linear(
                self.latent_dim, pyramid_tokens[-1] * hidden_size
            )
        else:
            self.latent_down = None
            self.latent_up = None
        self.node_skip = nn.Linear(self.node_dim, int(output_dim))
        self.score_scale = math.sqrt(hidden_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        for query in self.encoder_queries:
            nn.init.normal_(query, std=0.02)
        for query in self.decoder_query_tokens:
            nn.init.normal_(query, std=0.02)
        # Geometry alone determines the initial edge attention.
        nn.init.zeros_(self.edge_score[-1].weight)
        nn.init.zeros_(self.edge_score[-1].bias)
        self.local_decoder[-1].weight.data.mul_(0.1)
        self.pyramid_decoder[-1].weight.data.mul_(0.1)
        self.node_skip.weight.data.mul_(0.1)

    def _edge_conditioned_nodes(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        edge_index: Tensor,
        attention_bias: Tensor | None,
    ) -> Tensor:
        node_h = self.node_encoder(node_features)
        edge_h = self.edge_encoder(edge_features)
        first, second = edge_index
        if torch.any(first >= second):
            raise ValueError(
                "Undirected edge_index must contain one canonical i < j entry per pair."
            )
        reverse_features = edge_features.clone()
        reverse_features[:, [0, 1, 2, 3, 9, 10]] *= -1
        reverse_h = self.edge_encoder(reverse_features)
        forward_logits = self.edge_score(
            torch.cat([node_h[first], node_h[second], edge_h], dim=-1)
        ).squeeze(-1) / self.score_scale
        reverse_logits = self.edge_score(
            torch.cat([node_h[second], node_h[first], reverse_h], dim=-1)
        ).squeeze(-1) / self.score_scale
        if attention_bias is not None:
            bias = attention_bias.to(forward_logits)
            if bias.shape != forward_logits.shape:
                raise ValueError("attention_bias must have one value per edge")
            forward_logits = forward_logits + bias
            reverse_logits = reverse_logits + bias
        endpoint = torch.cat([second, first])
        attention = softmax(
            torch.cat([forward_logits, reverse_logits]),
            endpoint,
            num_nodes=node_h.size(0),
        )
        values = torch.cat(
            [
                self.edge_value(torch.cat([node_h[first], edge_h], dim=-1)),
                self.edge_value(torch.cat([node_h[second], reverse_h], dim=-1)),
            ]
        )
        aggregate = torch.zeros_like(node_h)
        aggregate.index_add_(0, endpoint, values * attention.unsqueeze(-1))
        return node_h + self.node_fuse(torch.cat([node_h, aggregate], dim=-1))

    def forward(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        edge_index: Tensor,
        *,
        attention_bias: Tensor | None = None,
    ) -> Tensor:
        local, original, encoded, context = self._encode_pyramid(
            node_features, edge_features, edge_index, attention_bias
        )
        if self.latent_dim > 0:
            latent = self.latent_down(context.mean(dim=1))
            context = self.latent_up(latent).reshape(
                latent.size(0), self.pyramid_tokens[-1], -1
            )
            for query, block in zip(
                self.decoder_query_tokens, self.decoder_blocks
            ):
                context = block(query.expand(context.size(0), -1, -1), context)
            # The decoder sees reference geometry and loading progress, but not
            # current positions/displacements. All dynamic information must
            # therefore cross the exact scalar bottleneck.
            static_features = node_features.clone()
            static_features[:, :4] = 0
            static_nodes = self.static_node_encoder(static_features).unsqueeze(0)
            node_queries = self.node_query(static_nodes)
        else:
            for projection, block, skip in zip(
                self.decoder_queries,
                self.decoder_blocks,
                reversed(encoded[:-1]),
            ):
                context = block(projection(skip), context)
                context = context + skip
            node_queries = self.node_query(original)
        decoded_nodes = self.node_decode(node_queries, context).squeeze(0)
        if self.latent_dim > 0:
            return self.pyramid_decoder(
                torch.cat([decoded_nodes, static_features], dim=-1)
            )
        local_prediction = self.node_skip(node_features) + self.local_decoder(
            torch.cat([local, node_features], dim=-1)
        )
        global_correction = self.pyramid_decoder(
            torch.cat([decoded_nodes, node_features], dim=-1)
        )
        return local_prediction + torch.sigmoid(self.global_gate_logit) * global_correction

    def _encode_pyramid(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        edge_index: Tensor,
        attention_bias: Tensor | None,
    ):
        local = self._edge_conditioned_nodes(
            node_features, edge_features, edge_index, attention_bias
        )
        original = local.unsqueeze(0)
        encoded = []
        context = original
        for query, block in zip(self.encoder_queries, self.encoder_blocks):
            context = block(query.expand(context.size(0), -1, -1), context)
            encoded.append(context)
        for block in self.bottleneck:
            context = block(context)
        return local, original, encoded, context

    def encode_latent(
        self,
        node_features: Tensor,
        edge_features: Tensor,
        edge_index: Tensor,
        *,
        attention_bias: Tensor | None = None,
    ) -> Tensor:
        """Return the exact scalar bottleneck for one graph."""
        if self.latent_dim <= 0:
            raise ValueError("encode_latent requires latent_dim > 0")
        _, _, _, context = self._encode_pyramid(
            node_features, edge_features, edge_index, attention_bias
        )
        return self.latent_down(context.mean(dim=1)).squeeze(0)


__all__ = ["AttentionPyramidSimulator"]
