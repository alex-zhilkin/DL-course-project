from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch

from .base import BaseModelInputs, BaseSimulator
from .common import build_mlp, get_correct_edge_vec
from .components import Normalizer
from .cv_core import CVCoreConfig, CVCoreInput, SharedCVCore

ModelInputs = BaseModelInputs

class NodeEdgeFusionEncoder(torch.nn.Module):
    """Turns the graph into node tokens with edge info mixed in."""

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
        node_sum = torch.zeros(n_nodes, hidden, device=e_emb.device)
        node_cnt = torch.zeros(n_nodes, 1, device=e_emb.device)
        node_sum.index_add_(0, col, e_emb)
        node_cnt.index_add_(0, col, torch.ones((e_emb.size(0), 1), device=e_emb.device))
        return node_sum / node_cnt.clamp(min=1.0)

    def forward(self, data: Data) -> Tensor:
        h_node = self.node_in(data.x)
        e_emb = self.edge_in(data.edge_attr)
        e_node = self._edge_to_node(data, e_emb)
        return self.fuse(torch.cat([h_node, e_node], dim=-1))



class GraphTokenAdapter(torch.nn.Module):
    """Builds the token input that the CV core expects."""

    def __init__(
        self,
        *,
        in_node_dim: int,
        in_edge_dim: int,
        hidden_size: int,
        num_mlp: int,
        use_local_skip: bool,
    ):
        super().__init__()
        self.use_local_skip = bool(use_local_skip)
        self.frontend = NodeEdgeFusionEncoder(
            in_node_dim=in_node_dim,
            in_edge_dim=in_edge_dim,
            hidden_size=hidden_size,
            num_mlp=num_mlp,
        )

    def build_core_input(self, data: Data) -> CVCoreInput:
        local_nodes = self.frontend(data)
        batch = data.batch
        dense_tokens, mask = to_dense_batch(local_nodes, batch)
        local_skip = dense_tokens if self.use_local_skip else None
        return CVCoreInput(tokens=dense_tokens, mask=mask, local_skip=local_skip)


class Model(BaseSimulator):
    """Graph CV model: local graph encoder + transformer bottleneck."""

    def __init__(
        self,
        data: Data,
        hidden_size: int,
        n_layers: int,
        pos_dim: int,
        *,
        num_mlp: int,
        transformer_layers: int,
        transformer_heads: int,
        transformer_dropout: float,
        use_local_skip: bool,
        token_sizes: tuple[int, ...],
        time_lag_steps: int = 0,
        time_lag_weight: float = 0.0,
    ):
        super().__init__(pos_dim=pos_dim)
        self._validate_input_dims(data, min_node_features=2, min_edge_features=1)

        self.device = data.x.device
        self._expected_node_dim = int(data.num_features)
        self.node_normalizer = Normalizer(size=self._expected_node_dim, name="NodeNormalizer", device=self.device)
        self.edge_normalizer = Normalizer(size=data.num_edge_features, name="EdgeNormalizer", device=self.device)
        self.output_normalizer = Normalizer(size=pos_dim, name="OutputNormalizer", device=self.device)

        self.adapter = GraphTokenAdapter(
            in_node_dim=self._expected_node_dim,
            in_edge_dim=data.num_edge_features,
            hidden_size=hidden_size,
            num_mlp=num_mlp,
            use_local_skip=use_local_skip,
        )
        self.core = SharedCVCore(
            CVCoreConfig(
                hidden_size=hidden_size,
                output_dim=pos_dim,
                transformer_layers=transformer_layers,
                transformer_heads=transformer_heads,
                transformer_dropout=transformer_dropout,
                token_sizes=token_sizes,
                use_local_skip=use_local_skip,
            )
        )
        self.k_cv = int(token_sizes[-1])
        self.time_lag_steps = int(time_lag_steps)
        self.time_lag_weight = float(time_lag_weight)
        self.freeze_normalizers = False

    @staticmethod
    def _current_velocity(inputs: ModelInputs) -> Tensor:
        return inputs.cur_graph.vel_state

    def _norm_training(self, is_training: bool) -> bool:
        return bool(self.training) and bool(is_training) and (not self.freeze_normalizers)

    def normalize_graph(self, graph: Data, is_training: bool = True) -> Data:
        norm_training = self._norm_training(is_training)
        norm_nodes = self.node_normalizer(graph.x, is_training=norm_training)
        norm_edges = self.edge_normalizer(graph.edge_attr, is_training=norm_training)
        
        return Data(
            x=norm_nodes,
            edge_index=graph.edge_index,
            edge_attr=norm_edges,
            box=graph.box,
            batch=graph.batch,
        )

    def _forward_core(
        self,
        data: Data,
        use_tau_head: bool = False,
    ):
        core_input = self.adapter.build_core_input(data)
        if use_tau_head:
            pred_dense = self.core.predict_tau(core_input)
            return pred_dense[core_input.mask]

        output = self.core(core_input)
        pred = output.prediction[core_input.mask]
        return pred

    # In the end we don't use it, but it exists :) 
    def predict_time_lag_acc(self, data: Data, *, is_training: bool = True) -> Tensor:
        data = self.normalize_graph(data, is_training=is_training)
        return self._forward_core(data, use_tau_head=True)

    def forward(
        self,
        data: Data,
        is_training: bool = True,
    ):
        data = self.normalize_graph(data, is_training=is_training)
        return self._forward_core(data)

    def extract_cv(self, data: Data, *, is_training: bool = False) -> Tensor:
        data = self.normalize_graph(data, is_training=is_training)
        self._forward_core(data)
        return self.core.last_cv.detach()

    @classmethod
    def _recalc_edges(cls, data: Data, pos_dim: int | None = None) -> Tensor:
        edge_vectors = get_correct_edge_vec(data, pos_dim=pos_dim)
        distances = torch.norm(edge_vectors, dim=1)
        bond_coeffs = data.edge_attr[:, -1]
        return torch.column_stack([edge_vectors, distances, bond_coeffs])

    def update(self, inputs: ModelInputs, model_output: Tensor) -> Data:
        predicted = self.output_normalizer.inverse(model_output)
        cur_velocity = self._current_velocity(inputs)
        updated_velocity = cur_velocity + predicted
        predicted_position = inputs.cur_position + updated_velocity

        tmp = Data(
            x=predicted_position.clone().float(),
            edge_index=inputs.cur_graph.edge_index,
            edge_attr=inputs.cur_graph.edge_attr.float(),
            box=inputs.cur_graph.box,
        )
        new_edge_attr = self._recalc_edges(tmp, self.pos_dim)

        predicted_graph = Data(
            x=predicted_position.float(),
            edge_index=inputs.cur_graph.edge_index,
            edge_attr=new_edge_attr.float(),
            box=inputs.cur_graph.box,
        )
        predicted_graph.vel_state = updated_velocity.detach()
        return predicted_graph

    def loss(
        self,
        model_output: Tensor,
        inputs: ModelInputs,
        *,
        accumulate_norm_stats: bool | None = None,
    ) -> Tensor:
        cur_velocity = self._current_velocity(inputs)
        target_velocity = inputs.target_position - inputs.cur_position
        norm_training = self._norm_training(self.training if accumulate_norm_stats is None else accumulate_norm_stats)
        target_velocity_change = target_velocity - cur_velocity
        target_velocity_change_normalized = self.output_normalizer(target_velocity_change, is_training=norm_training)
        
        return F.mse_loss(model_output, target_velocity_change_normalized)
