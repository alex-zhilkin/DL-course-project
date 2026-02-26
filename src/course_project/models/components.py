from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_batch

from .common import build_mlp, init_token_query_params, init_transformer_style_weights


class Normalizer(nn.Module):
    def __init__(
        self,
        size: int,
        max_accumulations: int = 1_000_000,
        std_epsilon: float = 1e-8,
        name: str = "Normalizer",
        device: str = "cuda",
    ):
        super().__init__()
        self.name = name
        self._max_accumulations = max_accumulations
        self.register_buffer("_std_epsilon", torch.tensor(std_epsilon, dtype=torch.float, requires_grad=False))
        self.register_buffer("_acc_count", torch.tensor(0, dtype=torch.float, requires_grad=False))
        self.register_buffer("_num_accumulations", torch.tensor(0, dtype=torch.float, requires_grad=False))
        self.register_buffer("_acc_sum", torch.zeros((1, size), dtype=torch.float, requires_grad=False))
        self.register_buffer("_acc_sum_squared", torch.zeros((1, size), dtype=torch.float, requires_grad=False))

    def forward(self, data: Tensor, accumulate: bool = True, is_training: bool = True):
        if accumulate and is_training and self._num_accumulations < self._max_accumulations:
            self._accumulate(data.detach())
        return (data - self._mean()) / self._std_with_epsilon()

    def inverse(self, normalized_batch_data: Tensor):
        return normalized_batch_data * self._std_with_epsilon() + self._mean()

    def _accumulate(self, data: Tensor):
        count = data.shape[0]
        data_sum = torch.sum(data, axis=0, keepdims=True)
        squared_data_sum = torch.sum(data**2, axis=0, keepdims=True)
        self._acc_sum += data_sum
        self._acc_sum_squared += squared_data_sum
        self._acc_count += count
        self._num_accumulations += 1

    def _mean(self):
        safe_count = torch.maximum(
            self._acc_count, torch.tensor(1.0, dtype=torch.float, device=self._acc_count.device)
        )
        return self._acc_sum / safe_count

    def _std_with_epsilon(self):
        safe_count = torch.maximum(
            self._acc_count, torch.tensor(1.0, dtype=torch.float, device=self._acc_count.device)
        )
        std = torch.sqrt(self._acc_sum_squared / safe_count - self._mean() ** 2)
        return torch.maximum(std, self._std_epsilon)

    def get_variable(self):
        return {
            "_max_accumulations": self._max_accumulations,
            "_std_epsilon": self._std_epsilon,
            "_acc_count": self._acc_count,
            "_num_accumulations": self._num_accumulations,
            "_acc_sum": self._acc_sum,
            "_acc_sum_squared": self._acc_sum_squared,
            "name": self.name,
        }


class LocalMessagePassingBlock(MessagePassing):
    def __init__(self, hidden_size: int, num_mlp: int):
        super().__init__(aggr="add")
        self.node_layer = build_mlp(hidden_size * 4, hidden_size, hidden_size, num_mlp=num_mlp, lay_norm=False)
        self.edge_layer = build_mlp(hidden_size * 3, hidden_size, hidden_size * 3, num_mlp=num_mlp, lay_norm=False)

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        message_block = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.edge_layer(message_block)

    def update(self, agg: Tensor, x: Tensor) -> Tensor:
        new_nodes = torch.cat([agg, x], dim=1)
        return self.node_layer(new_nodes)

    def forward(self, data: Data) -> Data:
        x = self.propagate(edge_index=data.edge_index, x=data.x, edge_attr=data.edge_attr)
        return Data(
            x=x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            box=data.box if hasattr(data, "box") else None,
            batch=data.batch if hasattr(data, "batch") else None,
            dtype=torch.float,
        )


class LocalGNNBackbone(nn.Module):
    """Local graph encoder + message-passing stack reused by spatial and hybrid models."""

    def __init__(self, in_node_dim: int, in_edge_dim: int, hidden_size: int, n_layers: int, num_mlp: int):
        super().__init__()
        self.node_encoder = build_mlp(in_node_dim, hidden_size, hidden_size, num_mlp=num_mlp, lay_norm=False)
        self.edge_encoder = build_mlp(in_edge_dim, hidden_size, hidden_size, num_mlp=num_mlp, lay_norm=False)
        self.layers = nn.ModuleList([LocalMessagePassingBlock(hidden_size, num_mlp=num_mlp) for _ in range(n_layers)])

    def forward(self, data: Data) -> Data:
        out = Data(
            x=self.node_encoder(data.x),
            edge_index=data.edge_index,
            edge_attr=self.edge_encoder(data.edge_attr),
            box=data.box if hasattr(data, "box") else None,
            batch=data.batch if hasattr(data, "batch") else None,
            dtype=torch.float,
        )
        for layer in self.layers:
            out = layer(out)
        return out


class NodeEdgeFusionEncoder(nn.Module):
    """Transformer-local front-end: node/edge encoders + edge->node aggregation + fusion."""

    def __init__(
        self,
        in_node_dim: int,
        in_edge_dim: int,
        hidden_size: int,
        num_mlp: int,
        edge_aggr: str,
    ):
        super().__init__()
        self.edge_aggr = edge_aggr
        if self.edge_aggr not in {"mean", "sum"}:
            raise ValueError(f"edge_aggr must be 'mean' or 'sum', got {self.edge_aggr!r}")
        self.node_in = build_mlp(in_node_dim, hidden_size, hidden_size, num_mlp=num_mlp, lay_norm=False)
        self.edge_in = build_mlp(in_edge_dim, hidden_size, hidden_size, num_mlp=num_mlp, lay_norm=False)
        self.fuse = build_mlp(hidden_size * 2, hidden_size, hidden_size, num_mlp=num_mlp, lay_norm=False)

    def _edge_to_node(self, data: Data, e_emb: Tensor) -> Tensor:
        _, col = data.edge_index
        n_nodes = data.x.size(0)
        hidden = e_emb.size(1)
        node_sum = torch.zeros(n_nodes, hidden, device=e_emb.device, dtype=e_emb.dtype)
        node_cnt = torch.zeros(n_nodes, 1, device=e_emb.device, dtype=e_emb.dtype)
        node_sum.index_add_(0, col, e_emb)
        node_cnt.index_add_(0, col, torch.ones((e_emb.size(0), 1), device=e_emb.device, dtype=e_emb.dtype))
        if self.edge_aggr == "sum":
            return node_sum
        return node_sum / node_cnt.clamp(min=1.0)

    def forward(self, data: Data) -> Tensor:
        h_node = self.node_in(data.x)
        e_emb = self.edge_in(data.edge_attr)
        e_node = self._edge_to_node(data, e_emb)
        return self.fuse(torch.cat([h_node, e_node], dim=-1))


class _CapturableTransformerEncoderLayer(nn.Module):
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

    def _sa_block(self, x: Tensor, need_weights: bool = False) -> tuple[Tensor, Tensor | None]:
        out, attn = self.self_attn(x, x, x, need_weights=need_weights, average_attn_weights=False)
        out = self.dropout1(out)
        return out, attn

    def _ff_block(self, x: Tensor) -> Tensor:
        return self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))

    def forward(self, x: Tensor, *, return_attn: bool = False) -> tuple[Tensor, Tensor | None]:
        sa_out, attn = self._sa_block(self.norm1(x), need_weights=return_attn)
        x = x + sa_out
        x = x + self._ff_block(self.norm2(x))
        return x, attn


class _TokenTransformerStack(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _CapturableTransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    activation="gelu",
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: Tensor, *, return_attn: bool = False) -> tuple[Tensor, Tensor | None]:
        attn_layers = []
        for layer in self.layers:
            x, attn = layer(x, return_attn=return_attn)
            if return_attn and attn is not None:
                attn_layers.append(attn)
        if return_attn:
            if attn_layers:
                return x, torch.stack(attn_layers, dim=0)
            return x, None
        return x, None


class TokenGlobalDecoder(nn.Module):
    """Two-stage token bottleneck + transformer + unpool-to-node global decoder."""

    def __init__(
        self,
        hidden_size: int,
        K1: int,
        K2: int,
        k2_hidden_size: int,
        transformer_layers: int,
        transformer_heads: int,
        transformer_dropout: float,
    ):
        super().__init__()
        self.K1 = int(K1)
        self.K2 = int(K2)
        if self.K2 > self.K1:
            raise ValueError(f"K2 must be <= K1 (got K1={self.K1}, K2={self.K2})")
        self.hidden_size = int(hidden_size)
        self.k2_hidden_size = int(k2_hidden_size)
        if self.k2_hidden_size < 1:
            raise ValueError(f"k2_hidden_size must be >= 1, got {self.k2_hidden_size}")

        self.tokens1 = nn.Parameter(torch.randn(self.K1, self.hidden_size))
        self.tokens2 = nn.Parameter(torch.randn(self.K2, self.k2_hidden_size))
        self.pool1 = nn.MultiheadAttention(
            self.hidden_size, transformer_heads, dropout=transformer_dropout, batch_first=True
        )
        self.pool2 = nn.MultiheadAttention(
            self.k2_hidden_size,
            transformer_heads,
            dropout=transformer_dropout,
            batch_first=True,
            kdim=self.hidden_size,
            vdim=self.hidden_size,
        )
        self.token_transformer = _TokenTransformerStack(
            d_model=self.k2_hidden_size,
            nhead=transformer_heads,
            dim_feedforward=4 * self.k2_hidden_size,
            dropout=transformer_dropout,
            num_layers=transformer_layers,
        )
        self.unpool_from_z2 = nn.MultiheadAttention(
            self.hidden_size,
            transformer_heads,
            dropout=transformer_dropout,
            batch_first=True,
            kdim=self.k2_hidden_size,
            vdim=self.k2_hidden_size,
        )
        self.last_cv: Tensor | None = None
        self.init_weights()

    def init_weights(self) -> None:
        init_transformer_style_weights(self)
        init_token_query_params([self.tokens1, self.tokens2])

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
        B, _, _ = h_dense.shape
        key_padding_nodes = ~mask

        t1 = self.tokens1.unsqueeze(0).expand(B, -1, -1)
        t2 = self.tokens2.unsqueeze(0).expand(B, -1, -1)

        z1, a_pool1 = self.pool1(
            query=t1,
            key=h_dense,
            value=h_dense,
            key_padding_mask=key_padding_nodes,
            need_weights=return_attn,
            average_attn_weights=False,
        )
        z2, a_pool2 = self.pool2(
            query=t2,
            key=z1,
            value=z1,
            need_weights=return_attn,
            average_attn_weights=False,
        )
        z2_mixed, a_global = self.token_transformer(z2, return_attn=return_attn)
        h_from_bottleneck, a_unpool2 = self.unpool_from_z2(
            query=h_dense,
            key=z2_mixed,
            value=z2_mixed,
            need_weights=return_attn,
            average_attn_weights=False,
        )

        if self.k2_hidden_size == 1:
            self.last_cv = z2_mixed.squeeze(-1)
        else:
            self.last_cv = z2_mixed.mean(dim=-1)

        if not return_attn and not return_z2:
            return h_dense, h_from_bottleneck, mask

        attn = None
        if return_attn:
            attn = {
                "pool1": a_pool1,
                "pool2": a_pool2,
                "global": a_global,
                "unpool2": a_unpool2,
                "pools": [a_pool1, a_pool2],
                "unpools": [a_unpool2],
                "token_sizes": [self.K1, self.K2],
                "node_mask": mask,
            }

        if return_z2:
            return h_dense, h_from_bottleneck, mask, attn, z2_mixed
        return h_dense, h_from_bottleneck, mask, attn
