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
            self._acc_count, torch.tensor(1.0, device=self._acc_count.device)
        )
        return self._acc_sum / safe_count

    def _std_with_epsilon(self):
        safe_count = torch.maximum(
            self._acc_count, torch.tensor(1.0, device=self._acc_count.device)
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
    """Basic message-passing layer"""

    def __init__(self, hidden_size: int, num_mlp: int, use_skip: bool = False):
        super().__init__(aggr="add")
        self.use_skip = bool(use_skip)
        
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
        out = self.node_mlp(torch.cat([aggr_out, x], dim=1))
        return x + out if self.use_skip else out

    def forward(self, data: Data) -> Data:
        x = self.propagate(edge_index=data.edge_index, x=data.x, edge_attr=data.edge_attr)
        return Data(
            x=x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            box=data.box,
            batch=data.batch,
        )

class LocalGNNBackbone(nn.Module):
    """Simple local GNN: encode nodes/edges, then run N basic message-passing layers."""

    def __init__(self, in_node_dim: int, in_edge_dim: int, hidden_size: int, n_layers: int, num_mlp: int, use_skip: bool = False):
        super().__init__()
        self.node_encoder = build_mlp(in_node_dim, hidden_size, hidden_size, num_mlp=num_mlp, lay_norm=False)
        self.edge_encoder = build_mlp(in_edge_dim, hidden_size, hidden_size, num_mlp=num_mlp, lay_norm=False)
        self.layers = nn.ModuleList([BasicMeshGNNLayer(hidden_size, num_mlp=num_mlp, use_skip=use_skip) for _ in range(n_layers)])

    def forward(self, data: Data, *, return_input_embedding: bool = False):
        x0 = self.node_encoder(data.x)
        out = Data(
            x=x0,
            edge_index=data.edge_index,
            edge_attr=self.edge_encoder(data.edge_attr),
            box=data.box,
            batch=data.batch ,
        )
        for layer in self.layers:
            out = layer(out)
        if return_input_embedding:
            return out, x0
        return out
