from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch

from .common import init_token_query_params, init_transformer_style_weights
from .components import NodeEdgeFusionEncoder, _TokenTransformerStack
from .spatial_transformer_simulator import Model as SpatialTransformerModel


class CVBottleneckGlobalBlock(nn.Module):
    """Node tokens -> pyramid bottleneck CV tokens -> broadcast global node context."""

    def __init__(
        self,
        *,
        hidden_size: int,
        K1: int,
        cv_count: int,
        transformer_layers: int,
        transformer_heads: int,
        transformer_dropout: float,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.Kcv = max(1, int(cv_count))
        # Keep the intermediate K2 token width scalar for CV-focused training.
        self.k2_hidden_size = 1

        self.token_sizes = self._resolve_token_sizes(K1=int(K1), cv_count=int(cv_count))

        self.tokens_hidden = nn.ParameterList(
            [nn.Parameter(torch.randn(k, self.hidden_size)) for k in self.token_sizes[:-1]]
        )
        self.tokens2 = nn.Parameter(torch.randn(self.token_sizes[-1], self.k2_hidden_size))

        self.down_pools_hidden = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=self.hidden_size,
                    num_heads=transformer_heads,
                    dropout=transformer_dropout,
                    batch_first=True,
                )
                for _ in self.token_sizes[:-1]
            ]
        )
        self.pool2 = nn.MultiheadAttention(
            embed_dim=self.k2_hidden_size,
            num_heads=transformer_heads,
            dropout=transformer_dropout,
            batch_first=True,
            kdim=self.hidden_size,
            vdim=self.hidden_size,
        )
        self.token_transformer = _TokenTransformerStack(
            d_model=self.hidden_size,
            nhead=transformer_heads,
            dim_feedforward=4 * self.hidden_size,
            dropout=transformer_dropout,
            num_layers=transformer_layers,
        )
        self.global_from_cv = nn.Linear(self.Kcv, self.hidden_size)

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
        init_token_query_params(list(self.tokens_hidden) + [self.tokens2])

    def forward(
        self,
        node_embeddings: Tensor,
        batch: Tensor | None,
        *,
        return_attn: bool = False,
        return_z2: bool = False,
    ):
        if batch is None:
            batch = torch.zeros(node_embeddings.size(0), dtype=torch.long, device=node_embeddings.device)

        h_dense, mask = to_dense_batch(node_embeddings, batch)
        B, Nmax, _ = h_dense.shape
        key_padding_nodes = ~mask

        t_hidden = [tok.unsqueeze(0).expand(B, -1, -1) for tok in self.tokens_hidden]
        t2 = self.tokens2.unsqueeze(0).expand(B, -1, -1)
        z = h_dense
        down_attn: list[Tensor] = []
        for i, (pool, tq) in enumerate(zip(self.down_pools_hidden, t_hidden)):
            pool_kwargs = dict(
                query=tq,
                key=z,
                value=z,
                need_weights=return_attn,
                average_attn_weights=False,
            )
            if i == 0:
                pool_kwargs["key_padding_mask"] = key_padding_nodes
            z, a = pool(**pool_kwargs)
            if return_attn and a is not None:
                down_attn.append(a)

        z_mixed, a_global = self.token_transformer(z, return_attn=return_attn)
        z2, a_pool2 = self.pool2(
            query=t2,
            key=z_mixed,
            value=z_mixed,
            need_weights=return_attn,
            average_attn_weights=False,
        )

        # Match the modular-network-simulator CV setup: expose the deepest scalar
        # bottleneck tokens directly as the learned CVs.
        self.last_cv = z2.squeeze(-1)  # [B, Kcv]
        hg = self.global_from_cv(self.last_cv).unsqueeze(1).expand(-1, Nmax, -1)  # [B, Nmax, H]

        if not return_attn and not return_z2:
            return h_dense, hg, mask

        attn = None
        if return_attn:
            a_pool1 = down_attn[0] if down_attn else None
            attn = {
                "pool1": a_pool1,
                "pool2": a_pool2,
                "global": a_global,
                "pools": down_attn + ([a_pool2] if a_pool2 is not None else []),
                "unpools": [],
                "token_sizes": list(self.token_sizes),
                "cv_token_size": self.Kcv,
                "node_mask": mask,
            }

        if return_z2:
            return h_dense, hg, mask, attn, z2
        return h_dense, hg, mask, attn


class CVPredictorCore(nn.Module):
    """Local node/edge fusion -> CV bottleneck global context -> linear dV head."""

    def __init__(
        self,
        in_node_dim: int,
        in_edge_dim: int,
        hidden_size: int,
        pos_dim: int,
        *,
        K1: int,
        cv_count: int,
        transformer_layers: int,
        transformer_heads: int,
        transformer_dropout: float,
        num_mlp: int,
        edge_aggr: str,
        use_local_skip: bool,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.last_cv: Tensor | None = None
        # Intentionally bottleneck-only to force prediction signal through learned CVs.
        self.use_local_skip = False
        self._requested_use_local_skip = bool(use_local_skip)

        self.frontend = NodeEdgeFusionEncoder(
            in_node_dim=in_node_dim,
            in_edge_dim=in_edge_dim,
            hidden_size=self.hidden_size,
            num_mlp=num_mlp,
            edge_aggr=edge_aggr,
        )
        self.global_block = CVBottleneckGlobalBlock(
            hidden_size=self.hidden_size,
            K1=K1,
            cv_count=cv_count,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            transformer_dropout=transformer_dropout,
        )
        self.out = nn.Linear(self.hidden_size, pos_dim)
        self.init_weights()

    def init_weights(self) -> None:
        init_transformer_style_weights(self)
        init_token_query_params(list(self.global_block.tokens_hidden) + [self.global_block.tokens2])
        self.out.weight.data.mul_(0.1)

    def forward(
        self,
        data: Data,
        *,
        return_context: bool = False,
        return_attn: bool = False,
        return_z2: bool = False,
    ):
        local_nodes = self.frontend(data)
        if return_attn or return_z2:
            decoded = self.global_block(
                local_nodes,
                data.batch if hasattr(data, "batch") else None,
                return_attn=return_attn,
                return_z2=return_z2,
            )
            if return_z2:
                h_dense, hg, mask, attn, zcv = decoded
            else:
                h_dense, hg, mask, attn = decoded
                zcv = None
        else:
            h_dense, hg, mask = self.global_block(
                local_nodes,
                data.batch if hasattr(data, "batch") else None,
                return_attn=False,
                return_z2=False,
            )
            attn = None
            zcv = None

        self.last_cv = self.global_block.last_cv
        dv_dense = self.out(hg)
        dv = dv_dense[mask]

        if return_context and return_attn:
            if return_z2:
                return dv, hg[mask], attn, zcv
            return dv, hg[mask], attn
        if return_context:
            if return_z2:
                return dv, hg[mask], zcv
            return dv, hg[mask]
        if return_z2:
            return dv, zcv
        return dv


class Model(SpatialTransformerModel):
    """Dedicated CV simulator with explicit scalar-CV bottleneck pressure.

    `CV` is the number of scalar CVs.
    """

    def __init__(
        self,
        data: Data,
        hidden_size: int,
        n_layers: int,
        pos_dim: int,
        *,
        num_mlp: int,
        K1: int,
        CV: int,
        transformer_layers: int,
        transformer_heads: int,
        transformer_dropout: float,
        edge_aggr: str,
        use_local_skip: bool,
    ):
        super().__init__(
            data=data,
            hidden_size=hidden_size,
            n_layers=n_layers,
            pos_dim=pos_dim,
            num_mlp=num_mlp,
            K1=K1,
            K2=CV,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            transformer_dropout=transformer_dropout,
            edge_aggr=edge_aggr,
            k2_hidden_size=1,
            use_local_skip=use_local_skip,
        )
        self.core = CVPredictorCore(
            in_node_dim=self._expected_node_dim,
            in_edge_dim=data.num_edge_features,
            hidden_size=hidden_size,
            pos_dim=pos_dim,
            K1=K1,
            cv_count=CV,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            transformer_dropout=transformer_dropout,
            num_mlp=num_mlp,
            edge_aggr=edge_aggr,
            use_local_skip=use_local_skip,
        )
        self.k_cv = int(CV)
