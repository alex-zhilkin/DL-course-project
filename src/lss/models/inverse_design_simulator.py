from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import ModuleList
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing

from .base import BaseModelInputs, BaseSimulator
from .common import build_mlp, get_correct_edge_vec


class AxisSharedNodeEncoder(torch.nn.Module):
    """GNNInverseDesign node encoder: share weights across x/y axes."""

    def __init__(self, num_history_steps: int, hidden_dim: int, num_mlp: int):
        super().__init__()
        self.axis_mlp = build_mlp(
            num_history_steps,
            hidden_dim,
            hidden_dim,
            num_mlp=num_mlp,
            lay_norm=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        x_reshaped = x.view(x.size(0), 2, -1)
        encoded = self.axis_mlp(x_reshaped)
        return encoded.view(x.size(0), -1)


class Encoder(torch.nn.Module):
    def __init__(self, data: Data, hidden_size: int, num_mlp: int):
        super().__init__()
        self.num_history_steps = data.num_features // 2
        self.shared_node_encoder = AxisSharedNodeEncoder(
            num_history_steps=self.num_history_steps,
            hidden_dim=hidden_size,
            num_mlp=num_mlp,
        )
        self.node_projection = torch.nn.Linear(hidden_size * 2, hidden_size)
        self.edge_encoder = build_mlp(
            data.num_edge_features,
            hidden_size,
            hidden_size,
            num_mlp=num_mlp,
            lay_norm=False,
        )

    def forward(self, data: Data) -> Data:
        shared_features = self.shared_node_encoder(data.x)
        x_encoded = self.node_projection(shared_features)
        return Data(
            x=x_encoded,
            edge_index=data.edge_index,
            edge_attr=self.edge_encoder(data.edge_attr),
            box=data.box if hasattr(data, "box") else None,
            box_tensor=data.box_tensor if hasattr(data, "box_tensor") else None,
            batch=data.batch if hasattr(data, "batch") else None,
        )


class CustomMessagePassing(MessagePassing):
    def __init__(self, hidden_size: int, num_mlp: int):
        super().__init__(aggr="add")
        self.node_layer = build_mlp(
            hidden_size * 4,
            hidden_size,
            hidden_size,
            num_mlp=num_mlp,
            lay_norm=True,
        )
        self.edge_layer = build_mlp(
            hidden_size * 3,
            hidden_size,
            hidden_size * 3,
            num_mlp=num_mlp,
            lay_norm=True,
        )

    def message(self, x_i: Tensor, x_j: Tensor, edge_attr: Tensor) -> Tensor:
        message_block = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.edge_layer(message_block)

    def update(self, aggr: Tensor, x: Tensor) -> Tensor:
        new_nodes = torch.cat([aggr, x], dim=1)
        return self.node_layer(new_nodes)

    def forward(self, data: Data):
        x = self.propagate(edge_index=data.edge_index, x=data.x, edge_attr=data.edge_attr)
        return Data(
            x=x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            box=data.box if hasattr(data, "box") else None,
            box_tensor=data.box_tensor if hasattr(data, "box_tensor") else None,
            batch=data.batch if hasattr(data, "batch") else None,
        )


class Decoder(torch.nn.Module):
    def __init__(self, hidden_size: int, num_mlp: int):
        super().__init__()
        self.node_decoder = build_mlp(
            hidden_size,
            hidden_size,
            2,
            num_mlp=num_mlp,
            lay_norm=False,
        )

    def forward(self, data: Data) -> Tensor:
        return self.node_decoder(data.x)


class Model(BaseSimulator):
    """Port of GNNInverseDesign/simulator_model.py Model.

    The architecture is intentionally separate from the local `spatial` model:
    it uses the axis-shared velocity-history encoder and residual message
    passing stack from GNNInverseDesign.
    """

    def __init__(
        self,
        data: Data,
        hidden_size: int,
        n_layers: int,
        pos_dim: int,
        *,
        num_mlp: int,
    ):
        super().__init__(pos_dim=pos_dim)
        if data.num_features % 2 != 0:
            raise ValueError(
                "inverse_design_simulator expects an even node feature dimension "
                "representing x/y history channels."
            )
        if pos_dim != 2:
            raise ValueError("inverse_design_simulator is defined for 2D graphs.")

        from .components import Normalizer

        self.node_normalizer = Normalizer(
            size=data.num_features,
            name="NodeNormalizer",
            device=str(data.x.device),
        )
        self.edge_normalizer = Normalizer(
            size=data.num_edge_features,
            name="EdgeNormalizer",
            device=str(data.x.device),
        )
        self.output_normalizer = Normalizer(
            size=2,
            name="OutputNormalizer",
            device=str(data.x.device),
        )
        self.encoder = Encoder(data, hidden_size, num_mlp=num_mlp)
        self.gnn_layers = ModuleList(
            [CustomMessagePassing(hidden_size, num_mlp=num_mlp) for _ in range(n_layers)]
        )
        self.decoder = Decoder(hidden_size, num_mlp=num_mlp)
        self.device = data.x.device
        self.freeze_normalizers = False

    def normalize_graph(self, graph: Data, is_training: bool = True) -> Data:
        accumulate = bool(is_training) and not bool(self.freeze_normalizers)
        norm_nodes = self.node_normalizer(
            graph.x,
            accumulate=accumulate,
            is_training=is_training,
        )
        norm_edges = self.edge_normalizer(
            graph.edge_attr,
            accumulate=accumulate,
            is_training=is_training,
        )
        return Data(
            x=norm_nodes,
            edge_index=graph.edge_index,
            edge_attr=norm_edges,
            box=graph.box if hasattr(graph, "box") else None,
            box_tensor=graph.box_tensor if hasattr(graph, "box_tensor") else None,
            batch=graph.batch if hasattr(graph, "batch") else None,
        )

    def forward(self, data: Data, is_training: bool = True) -> torch.Tensor:
        data = self.normalize_graph(data, is_training=is_training)
        data = self.encoder(data)
        for gnn_layer in self.gnn_layers:
            residual = data.x
            data = gnn_layer(data)
            data.x = data.x + residual
        return self.decoder(data)

    @classmethod
    def _recalc_edges(cls, data: Data) -> Tensor:
        edge_vectors = get_correct_edge_vec(data, pos_dim=2)
        distances = torch.norm(edge_vectors, dim=1)
        bond_coeffs = data.edge_attr[:, -1]
        return torch.column_stack([edge_vectors, distances, bond_coeffs])

    def update(
        self,
        inputs: BaseModelInputs,
        model_output: Tensor,
        recalc_edges: bool = False,
    ) -> Data:
        predicted_acceleration = self.output_normalizer.inverse(model_output)
        cur_velocity = inputs.cur_position - inputs.prev_position
        updated_velocity = cur_velocity + predicted_acceleration
        predicted_position = inputs.cur_position + updated_velocity
        edge_attr = inputs.cur_graph.edge_attr
        out = Data(
            x=predicted_position,
            pos=predicted_position,
            edge_index=inputs.cur_graph.edge_index,
            edge_attr=edge_attr,
            box=inputs.cur_graph.box if hasattr(inputs.cur_graph, "box") else None,
            box_tensor=(
                inputs.cur_graph.box_tensor
                if hasattr(inputs.cur_graph, "box_tensor")
                else None
            ),
        )
        if recalc_edges:
            out.edge_attr = self._recalc_edges(out)
        out.vel_state = updated_velocity.detach()
        return out

    def loss(
        self,
        model_output: Tensor,
        inputs: BaseModelInputs,
        *,
        accumulate_norm_stats: bool | None = None,
    ) -> Tensor:
        cur_velocity = inputs.cur_position - inputs.prev_position
        target_velocity = inputs.target_position - inputs.cur_position
        target_velocity_change = target_velocity - cur_velocity
        accumulate = (
            self.training
            if accumulate_norm_stats is None
            else bool(accumulate_norm_stats)
        ) and not bool(self.freeze_normalizers)
        target_velocity_change_normalized = self.output_normalizer(
            target_velocity_change,
            is_training=self.training,
            accumulate=accumulate,
        )
        return torch.nn.functional.mse_loss(
            model_output,
            target_velocity_change_normalized,
        )
