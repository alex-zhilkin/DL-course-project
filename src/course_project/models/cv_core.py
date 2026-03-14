from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .common import init_token_query_params, init_transformer_style_weights

@dataclass(frozen=True)
class CVCoreConfig:
    # Config for the shared CV bottleneck
    hidden_size: int
    output_dim: int
    transformer_layers: int
    transformer_heads: int
    transformer_dropout: float
    token_sizes: tuple[int, ...]
    use_local_skip: bool = False


# Nice interfaces for shared CVCore
@dataclass
class CVCoreInput:
    tokens: Tensor
    mask: Tensor | None = None
    local_skip: Tensor | None = None

@dataclass
class CVCoreOutput:
    prediction: Tensor
    cv: Tensor
    global_context: Tensor | None = None


class TransformerEncoderLayer(nn.Module):
    # Small self-attention for after pyramid mixing
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def _sa_block(self, x: Tensor) -> Tensor:
        out, _ = self.self_attn(x, x, x, need_weights=False, average_attn_weights=False)
        return self.dropout1(out)

    def _ff_block(self, x: Tensor) -> Tensor:
        return self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self._sa_block(self.norm1(x))
        x = x + self._ff_block(self.norm2(x))
        return x


class SharedCVCore(nn.Module):
    # Shared CV model: encoder pyramid -> transformer bottleneck -> CV heads -> decoder pyramid.
    
    def __init__(self, cfg: CVCoreConfig):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = int(cfg.hidden_size)
        self.cv_count = int(cfg.token_sizes[-1])
        self.latent_dim = int(self.cv_count)
        self.use_local_skip = bool(cfg.use_local_skip)
        self.token_sizes = [int(v) for v in cfg.token_sizes]
        
        if len(self.token_sizes) < 2:
            raise ValueError("token_sizes must have at least two entries")
            
        self.encoder_pyramid_queries = nn.ParameterList(
            [nn.Parameter(torch.randn(k, self.hidden_size)) for k in self.token_sizes[:-1]]
        )
        
        self.encoder_pyramid_pools = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=self.hidden_size,
                    num_heads=cfg.transformer_heads,
                    dropout=cfg.transformer_dropout,
                    batch_first=True,
                )
                for _ in self.token_sizes[:-1]
            ]
        )
        self.cv_query_tokens = nn.ParameterList(
            [nn.Parameter(torch.randn(1, self.hidden_size)) for _ in range(self.cv_count)]
        )
        self.cv_pools = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=self.hidden_size,
                    num_heads=cfg.transformer_heads,
                    dropout=cfg.transformer_dropout,
                    batch_first=True,
                )
                for _ in range(self.cv_count)
            ]
        )
        self.cv_heads = nn.ModuleList([nn.Linear(self.hidden_size, 1) for _ in range(self.cv_count)])
        self.token_transformer_layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    d_model=self.hidden_size,
                    nhead=cfg.transformer_heads,
                    dim_feedforward=4 * self.hidden_size,
                    dropout=cfg.transformer_dropout,
                    activation="gelu",
                )
                for _ in range(cfg.transformer_layers)
            ]
        )
        self.decoder_seed = nn.Linear(self.latent_dim, self.cv_count * self.hidden_size)
        reverse_sizes = list(reversed(self.token_sizes[:-1]))
        self.decoder_pyramid_queries = nn.ParameterList(
            [nn.Parameter(torch.randn(k, self.hidden_size)) for k in reverse_sizes]
        )
        self.decoder_pyramid_pools = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=self.hidden_size,
                    num_heads=cfg.transformer_heads,
                    dropout=cfg.transformer_dropout,
                    batch_first=True,
                )
                for _ in reverse_sizes
            ]
        )
        self.output_token_pool = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=cfg.transformer_heads,
            dropout=cfg.transformer_dropout,
            batch_first=True,
        )
        in_dim = self.hidden_size * 2 if self.use_local_skip else self.hidden_size
        self.out = nn.Linear(in_dim, int(cfg.output_dim))
        self.out_tau = nn.Linear(in_dim, int(cfg.output_dim))

        self.last_cv: Tensor | None = None
        self.init_weights()

    def init_weights(self) -> None:
        init_transformer_style_weights(self)
        init_token_query_params(
            list(self.encoder_pyramid_queries) + list(self.cv_query_tokens) + list(self.decoder_pyramid_queries)
        )
        self.out.weight.data.mul_(0.1)
        self.out_tau.weight.data.mul_(0.1)
        for head in self.cv_heads:
            nn.init.xavier_uniform_(head.weight)
            if head.bias is not None:
                nn.init.zeros_(head.bias)

    def encode(self, core_input: CVCoreInput):
        # Encoder pyramid first shrinks the token set, then the transformer mixes it globally.
        z = core_input.tokens
        key_padding_mask = None if core_input.mask is None else ~core_input.mask

        for i, (pool, query_tokens) in enumerate(zip(self.encoder_pyramid_pools, self.encoder_pyramid_queries)):
            q = query_tokens.unsqueeze(0).expand(z.size(0), -1, -1)
            kwargs = {
                "query": q,
                "key": z,
                "value": z,
                "need_weights": False,
                "average_attn_weights": False,
            }
            if i == 0 and key_padding_mask is not None:
                kwargs["key_padding_mask"] = key_padding_mask
            z, _ = pool(**kwargs)

        for layer in self.token_transformer_layers:
            z = layer(z)
        cv_rows: list[Tensor] = []
        for query_token, pool, head in zip(self.cv_query_tokens, self.cv_pools, self.cv_heads):
            qcv = query_token.unsqueeze(0).expand(z.size(0), -1, -1)
            zcv_i, _ = pool(
                query=qcv,
                key=z,
                value=z,
                need_weights=False,
                average_attn_weights=False,
            )
            cv_rows.append(head(zcv_i.squeeze(1)))
        cv = torch.cat(cv_rows, dim=1)
        self.last_cv = cv

    def _decode_tokens(self, core_input: CVCoreInput) -> Tensor:
        # Decoder pyramid grows the latent back up before reading out per-token outputs.
        cv = self.last_cv
        z = self.decoder_seed(cv).view(cv.size(0), self.cv_count, self.hidden_size)
        for pool, query_tokens in zip(self.decoder_pyramid_pools, self.decoder_pyramid_queries):
            q = query_tokens.unsqueeze(0).expand(z.size(0), -1, -1)
            z, _ = pool(
                query=q,
                key=z,
                value=z,
                need_weights=False,
                average_attn_weights=False,
            )
        out_tokens, _ = self.output_token_pool(
            query=core_input.tokens,
            key=z,
            value=z,
            need_weights=False,
            average_attn_weights=False,
        )
        return out_tokens

    def _per_token_output(self, core_input: CVCoreInput) -> tuple[Tensor, Tensor]:
        # Decode back to per-token hidden states before the final output layer.
        decoded_tokens = self._decode_tokens(core_input)
        fused = decoded_tokens
        if self.use_local_skip:
            fused = torch.cat([core_input.local_skip, decoded_tokens], dim=-1)
        return self.out(fused), decoded_tokens

    def forward(self, core_input: CVCoreInput) -> CVCoreOutput:
        self.encode(core_input)
        cv = self.last_cv
        pred, hg = self._per_token_output(core_input)
        return CVCoreOutput(
            prediction=pred,
            cv=cv,
            global_context=hg,
        )

    def predict_tau(self, core_input: CVCoreInput) -> Tensor:
        self.encode(core_input)
        decoded_tokens = self._decode_tokens(core_input)
        fused = decoded_tokens
        if self.use_local_skip:
            fused = torch.cat([core_input.local_skip, decoded_tokens], dim=-1)
        return self.out_tau(fused)
