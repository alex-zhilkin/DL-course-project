from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from .common import init_token_query_params, init_transformer_style_weights

@dataclass(frozen=True)
class CVCoreConfig:
    # Config for the shared CV encoder bottleneck.
    hidden_size: int
    transformer_layers: int
    transformer_heads: int
    transformer_dropout: float
    token_sizes: tuple[int, ...]


# Nice interfaces for shared CVCore
@dataclass
class CVCoreInput:
    tokens: Tensor
    mask: Tensor | None = None

class SharedCVCore(nn.Module):
    # Shared CV encoder: node tokens -> transformer bottleneck -> scalar CV heads.
    
    def __init__(self, cfg: CVCoreConfig):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = int(cfg.hidden_size)
        self.cv_count = int(cfg.token_sizes[-1])
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
        self.token_self_attn_layers = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=self.hidden_size,
                    num_heads=cfg.transformer_heads,
                    dropout=cfg.transformer_dropout,
                    batch_first=True,
                )
                for _ in range(cfg.transformer_layers)
            ]
        )
        self.last_cv: Tensor | None = None
        self.last_pyramid_attn: list[Tensor] | None = None
        self.last_cv_attn: Tensor | None = None
        self.init_weights()

    def init_weights(self) -> None:
        init_transformer_style_weights(self)
        init_token_query_params(
            list(self.encoder_pyramid_queries) + list(self.cv_query_tokens)
        )
        for head in self.cv_heads:
            nn.init.xavier_uniform_(head.weight)
            if head.bias is not None:
                nn.init.zeros_(head.bias)

    def encode(self, core_input: CVCoreInput, *, capture_attention: bool = False):
        # Encoder pyramid first shrinks the token set, then the transformer mixes it globally.
        z = core_input.tokens
        key_padding_mask = None if core_input.mask is None else ~core_input.mask
        pyramid_attn: list[Tensor] = []

        for i, (pool, query_tokens) in enumerate(zip(self.encoder_pyramid_pools, self.encoder_pyramid_queries)):
            q = query_tokens.unsqueeze(0).expand(z.size(0), -1, -1)
            kwargs = {
                "query": q,
                "key": z,
                "value": z,
                "need_weights": capture_attention,
                "average_attn_weights": False,
            }
            if i == 0 and key_padding_mask is not None:
                kwargs["key_padding_mask"] = key_padding_mask
            attn_out, attn = pool(**kwargs)
            z = q + attn_out
            if capture_attention and attn is not None:
                pyramid_attn.append(attn.detach())

        for self_attn in self.token_self_attn_layers:
            attn_out, _ = self_attn(
                query=z,
                key=z,
                value=z,
                need_weights=False,
                average_attn_weights=False,
            )
            z = z + attn_out
        cv_rows: list[Tensor] = []
        cv_attn_rows: list[Tensor] = []
        for query_token, pool, head in zip(self.cv_query_tokens, self.cv_pools, self.cv_heads):
            qcv = query_token.unsqueeze(0).expand(z.size(0), -1, -1)
            cv_attn_out, cv_attn = pool(
                query=qcv,
                key=z,
                value=z,
                need_weights=capture_attention,
                average_attn_weights=False,
            )
            zcv_i = qcv + cv_attn_out
            cv_rows.append(head(zcv_i.squeeze(1)))
            if capture_attention and cv_attn is not None:
                cv_attn_rows.append(cv_attn.detach())
        cv = torch.cat(cv_rows, dim=1)
        self.last_cv = cv
        self.last_pyramid_attn = pyramid_attn if capture_attention else None
        if capture_attention and cv_attn_rows:
            self.last_cv_attn = torch.cat(cv_attn_rows, dim=2)
        else:
            self.last_cv_attn = None

    def forward(self, core_input: CVCoreInput) -> Tensor:
        self.encode(core_input)
        return self.last_cv
