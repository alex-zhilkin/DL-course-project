from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from .common import init_token_query_params, init_transformer_style_weights
from .components import _TokenTransformerStack


@dataclass(frozen=True)
class CVCoreConfig:
    hidden_size: int
    output_dim: int
    transformer_layers: int
    transformer_heads: int
    transformer_dropout: float
    token_sizes: tuple[int, ...]
    use_local_skip: bool = False


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


class SharedCVCore(nn.Module):
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
        for i in range(1, len(self.token_sizes)):
            if self.token_sizes[i] > self.token_sizes[i - 1] or self.token_sizes[i] < 1:
                raise ValueError("token_sizes must be positive and non-increasing")
        self.query_tokens = nn.ParameterList(
            [nn.Parameter(torch.randn(k, self.hidden_size)) for k in self.token_sizes[:-1]]
        )
        self.down_pools = nn.ModuleList(
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
        self.token_transformer = _TokenTransformerStack(
            d_model=self.hidden_size,
            nhead=cfg.transformer_heads,
            dim_feedforward=4 * self.hidden_size,
            dropout=cfg.transformer_dropout,
            num_layers=cfg.transformer_layers,
        )
        in_dim = self.hidden_size * 2 if self.use_local_skip else self.hidden_size
        self.global_from_cv = nn.Linear(self.latent_dim, self.hidden_size)
        self.out = nn.Linear(in_dim, int(cfg.output_dim))
        self.out_tau = nn.Linear(in_dim, int(cfg.output_dim))

        self.last_cv: Tensor | None = None
        self.init_weights()

    def init_weights(self) -> None:
        init_transformer_style_weights(self)
        init_token_query_params(list(self.query_tokens) + list(self.cv_query_tokens))
        self.out.weight.data.mul_(0.1)
        self.out_tau.weight.data.mul_(0.1)
        for head in self.cv_heads:
            nn.init.xavier_uniform_(head.weight)
            if head.bias is not None:
                nn.init.zeros_(head.bias)

    def encode(self, core_input: CVCoreInput) -> None:
        z = core_input.tokens
        key_padding_mask = None if core_input.mask is None else ~core_input.mask

        for i, (pool, query_tokens) in enumerate(zip(self.down_pools, self.query_tokens)):
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

        z = self.token_transformer(z)
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

    def _per_token_output(self, core_input: CVCoreInput) -> tuple[Tensor, Tensor]:
        cv = self.last_cv
        hg = self.global_from_cv(cv).unsqueeze(1).expand(-1, core_input.tokens.size(1), -1)
        fused = hg
        if self.use_local_skip:
            fused = torch.cat([core_input.local_skip, hg], dim=-1)
        return self.out(fused), hg

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
        cv = self.last_cv
        hg = self.global_from_cv(cv).unsqueeze(1).expand(-1, core_input.tokens.size(1), -1)
        fused = hg
        if self.use_local_skip:
            fused = torch.cat([core_input.local_skip, hg], dim=-1)
        return self.out_tau(fused)
