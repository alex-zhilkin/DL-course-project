from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import torch
import torch.nn as nn
from torch import Tensor

from .common import init_token_query_params, init_transformer_style_weights
from .components import _TokenTransformerStack


@dataclass(frozen=True)
class CVCoreConfig:
    hidden_size: int
    K1: int
    cv_count: int
    output_dim: int
    transformer_layers: int
    transformer_heads: int
    transformer_dropout: float
    cv_hidden_size: int = 1
    decode_mode: Literal["per_token", "global"] = "per_token"
    use_local_skip: bool = False


@dataclass
class CVCoreInput:
    # Shared core expects hidden tokens [B, T, H].
    tokens: Tensor
    # Optional validity mask [B, T] where True means a real token.
    mask: Tensor | None = None
    # Optional local identity-style signal with the same shape as tokens.
    local_skip: Tensor | None = None


@dataclass
class CVCoreOutput:
    prediction: Tensor
    cv: Tensor
    global_context: Tensor | None = None
    token_state: Tensor | None = None
    attn: dict | None = None


class CVTokenAdapter(Protocol):
    def build_core_input(self, data) -> CVCoreInput:
        ...


def _build_reverse_pyramid(token_sizes: list[int], out_dim: int) -> nn.Sequential:
    dims = list(reversed([int(v) for v in token_sizes])) + [int(out_dim)]
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class SharedCVCore(nn.Module):
    def __init__(self, cfg: CVCoreConfig):
        super().__init__()
        self.cfg = cfg
        self.hidden_size = int(cfg.hidden_size)
        self.cv_count = max(1, int(cfg.cv_count))
        self.cv_hidden_size = max(1, int(cfg.cv_hidden_size))
        self.latent_dim = int(self.cv_count * self.cv_hidden_size)
        self.decode_mode = str(cfg.decode_mode)
        self.use_local_skip = bool(cfg.use_local_skip)
        self.token_sizes = self._resolve_token_sizes(K1=int(cfg.K1), cv_count=self.cv_count)

        self.query_tokens = nn.ParameterList(
            [nn.Parameter(torch.randn(k, self.hidden_size)) for k in self.token_sizes[:-1]]
        )
        self.cv_queries = nn.Parameter(torch.randn(self.token_sizes[-1], self.cv_hidden_size))
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
        self.pool_cv = nn.MultiheadAttention(
            embed_dim=self.cv_hidden_size,
            num_heads=cfg.transformer_heads,
            dropout=cfg.transformer_dropout,
            batch_first=True,
            kdim=self.hidden_size,
            vdim=self.hidden_size,
        )
        self.token_transformer = _TokenTransformerStack(
            d_model=self.hidden_size,
            nhead=cfg.transformer_heads,
            dim_feedforward=4 * self.hidden_size,
            dropout=cfg.transformer_dropout,
            num_layers=cfg.transformer_layers,
        )

        if self.decode_mode == "per_token":
            in_dim = self.hidden_size * 2 if self.use_local_skip else self.hidden_size
            self.global_from_cv = nn.Linear(self.latent_dim, self.hidden_size)
            self.out = nn.Linear(in_dim, int(cfg.output_dim))
            self.out_tau = nn.Linear(in_dim, int(cfg.output_dim))
        else:
            self.decoder = _build_reverse_pyramid(
                [self.token_sizes[0], self.token_sizes[1], self.latent_dim],
                int(cfg.output_dim),
            )
            self.out_tau = None

        self.last_cv: Tensor | None = None
        self.init_weights()

    @staticmethod
    def _resolve_token_sizes(K1: int, cv_count: int) -> list[int]:
        k1 = max(1, int(K1))
        k2 = max(1, int(cv_count))
        if k1 == k2:
            mids = [k1, k1]
        else:
            ratio = (k2 / k1) ** (1.0 / 3.0)
            mids = [
                max(1, int(round(k1 * ratio))),
                max(1, int(round(k1 * (ratio**2)))),
            ]
        sizes = [k1, mids[0], mids[1], k2]
        for i in range(1, len(sizes)):
            sizes[i] = min(sizes[i], sizes[i - 1])
        for i in range(len(sizes) - 2, -1, -1):
            sizes[i] = max(sizes[i], sizes[i + 1])
        return sizes

    def init_weights(self) -> None:
        init_transformer_style_weights(self)
        init_token_query_params(list(self.query_tokens) + [self.cv_queries])
        if hasattr(self, "out"):
            self.out.weight.data.mul_(0.1)
            self.out_tau.weight.data.mul_(0.1)
        if hasattr(self, "decoder"):
            out_linears = [m for m in self.decoder.modules() if isinstance(m, nn.Linear)]
            if out_linears:
                out_linears[-1].weight.data.mul_(0.1)

    def encode(self, core_input: CVCoreInput, *, return_attn: bool = False) -> tuple[Tensor, Tensor | None]:
        z = core_input.tokens
        key_padding_mask = None if core_input.mask is None else ~core_input.mask
        attn_rows: list[Tensor] = []

        for i, (pool, query_tokens) in enumerate(zip(self.down_pools, self.query_tokens)):
            q = query_tokens.unsqueeze(0).expand(z.size(0), -1, -1)
            kwargs = {
                "query": q,
                "key": z,
                "value": z,
                "need_weights": return_attn,
                "average_attn_weights": False,
            }
            if i == 0 and key_padding_mask is not None:
                kwargs["key_padding_mask"] = key_padding_mask
            z, attn = pool(**kwargs)
            if return_attn and attn is not None:
                attn_rows.append(attn)

        z, attn_global = self.token_transformer(z, return_attn=return_attn)
        qcv = self.cv_queries.unsqueeze(0).expand(z.size(0), -1, -1)
        zcv, attn_cv = self.pool_cv(
            query=qcv,
            key=z,
            value=z,
            need_weights=return_attn,
            average_attn_weights=False,
        )
        cv = zcv.reshape(zcv.size(0), -1)
        self.last_cv = cv

        if not return_attn:
            return zcv, None
        return zcv, {
            "pools": attn_rows + ([attn_cv] if attn_cv is not None else []),
            "global": attn_global,
            "token_sizes": list(self.token_sizes),
        }

    def _per_token_output(self, core_input: CVCoreInput) -> tuple[Tensor, Tensor]:
        if core_input.mask is None:
            raise ValueError("per_token CV core expects a mask")
        if self.use_local_skip and core_input.local_skip is None:
            raise ValueError("per_token CV core expects local_skip when use_local_skip=True")
        cv = self.last_cv
        hg = self.global_from_cv(cv).unsqueeze(1).expand(-1, core_input.tokens.size(1), -1)
        fused = torch.cat([core_input.local_skip, hg], dim=-1) if self.use_local_skip else hg
        return self.out(fused), hg

    def forward(
        self,
        core_input: CVCoreInput,
        *,
        return_attn: bool = False,
        return_z2: bool = False,
    ) -> CVCoreOutput:
        zcv, attn = self.encode(core_input, return_attn=return_attn)
        cv = self.last_cv
        if self.decode_mode == "per_token":
            pred, hg = self._per_token_output(core_input)
            return CVCoreOutput(
                prediction=pred,
                cv=cv,
                global_context=hg,
                token_state=zcv if return_z2 else None,
                attn=attn,
            )
        pred = self.decoder(cv)
        return CVCoreOutput(
            prediction=pred,
            cv=cv,
            global_context=None,
            token_state=zcv if return_z2 else None,
            attn=attn,
        )

    def predict_tau(self, core_input: CVCoreInput) -> Tensor:
        if self.decode_mode != "per_token":
            raise ValueError("time-lag head is only defined for per_token decode mode")
        self.encode(core_input, return_attn=False)
        cv = self.last_cv
        hg = self.global_from_cv(cv).unsqueeze(1).expand(-1, core_input.tokens.size(1), -1)
        fused = torch.cat([core_input.local_skip, hg], dim=-1) if self.use_local_skip else hg
        return self.out_tau(fused)
