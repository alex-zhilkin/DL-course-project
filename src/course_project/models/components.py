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


class BasicMeshGNNLayer(MessagePassing):
    """Basic message-passing layer inspired by mesh graph networks."""

    def __init__(self, hidden_size: int, num_mlp: int):
        super().__init__(aggr="add")
        self.message_mlp = build_mlp(
            hidden_size * 3,
            hidden_size,
            hidden_size * 3,
            num_mlp=num_mlp,
            lay_norm=False,
        )
        self.node_mlp = build_mlp(
            hidden_size * 4,
            hidden_size,
            hidden_size,
            num_mlp=num_mlp,
            lay_norm=False,
        )

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        return self.message_mlp(torch.cat([x_i, x_j, edge_attr], dim=1))

    def update(self, aggr_out: Tensor, x: Tensor) -> Tensor:
        return self.node_mlp(torch.cat([aggr_out, x], dim=1))

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
    """Simple local GNN: encode nodes/edges, then run N basic message-passing layers."""

    def __init__(self, in_node_dim: int, in_edge_dim: int, hidden_size: int, n_layers: int, num_mlp: int):
        super().__init__()
        self.node_encoder = build_mlp(in_node_dim, hidden_size, hidden_size, num_mlp=num_mlp, lay_norm=False)
        self.edge_encoder = build_mlp(in_edge_dim, hidden_size, hidden_size, num_mlp=num_mlp, lay_norm=False)
        self.layers = nn.ModuleList([BasicMeshGNNLayer(hidden_size, num_mlp=num_mlp) for _ in range(n_layers)])

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
    ):
        super().__init__()
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

    def _sa_block(self, x: Tensor) -> Tensor:
        out, _ = self.self_attn(x, x, x, need_weights=False, average_attn_weights=False)
        out = self.dropout1(out)
        return out

    def _ff_block(self, x: Tensor) -> Tensor:
        return self.dropout2(self.linear2(self.dropout(self.activation(self.linear1(x)))))

    def forward(self, x: Tensor) -> Tensor:
        sa_out = self._sa_block(self.norm1(x))
        x = x + sa_out
        x = x + self._ff_block(self.norm2(x))
        return x


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

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
