from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data

from .base import BaseModelInputs, BaseSimulator
from ..utils import resolve_device
from .common import build_mlp, get_correct_edge_vec
from .components import LocalGNNBackbone, Normalizer

ModelInputs = BaseModelInputs
torch.set_default_dtype(torch.float32)


class Model(BaseSimulator):
    def __init__(
        self,
        data: Data,
        hidden_size: int,
        n_layers: int,
        pos_dim: int,
        num_mlp: int,
    ):
        super().__init__(pos_dim=pos_dim)
        self._validate_input_dims(data, min_node_features=2, min_edge_features=1)

        self.device = data.x.device if hasattr(data, "x") and data.x is not None else resolve_device("auto")

        self.node_normalizer = Normalizer(size=data.num_features, name="NodeNormalizer", device=self.device)
        self.edge_normalizer = Normalizer(size=data.num_edge_features, name="EdgeNormalizer", device=self.device)
        self.output_normalizer = Normalizer(size=pos_dim, name="OutputNormalizer", device=self.device)

        self.backbone = LocalGNNBackbone(
            in_node_dim=data.num_features,
            in_edge_dim=data.num_edge_features,
            hidden_size=hidden_size,
            n_layers=n_layers,
            num_mlp=num_mlp,
        )
        self.decoder = build_mlp(hidden_size, hidden_size, pos_dim, num_mlp=num_mlp, lay_norm=False)

        self.freeze_normalizers = False

    @staticmethod
    def _current_velocity(inputs: ModelInputs) -> Tensor:
        if hasattr(inputs.cur_graph, "vel_state"):
            return inputs.cur_graph.vel_state
        return inputs.cur_position - inputs.prev_position

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
            box=graph.box if hasattr(graph, "box") else None,
            batch=graph.batch if hasattr(graph, "batch") else None,
            dtype=torch.float,
        )

    def _encode(self, data: Data) -> Data:
        return self.backbone(data)

    def forward(self, data: Data, is_training: bool = True) -> Tensor:
        data = self.normalize_graph(data, is_training=is_training)
        latent = self._encode(data)
        return self.decoder(latent.x)

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
            box=inputs.cur_graph.box if hasattr(inputs.cur_graph, "box") else None,
            dtype=torch.float32,
        )
        new_edge_attr = self._recalc_edges(tmp, self.pos_dim)

        predicted_graph = Data(
            x=predicted_position.float(),
            edge_index=inputs.cur_graph.edge_index,
            edge_attr=new_edge_attr.float(),
            box=inputs.cur_graph.box if hasattr(inputs.cur_graph, "box") else None,
            dtype=torch.float32,
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
        norm_training = self._norm_training(
            self.training if accumulate_norm_stats is None else accumulate_norm_stats
        )
        target_velocity_change = target_velocity - cur_velocity
        target_velocity_change_normalized = self.output_normalizer(
            target_velocity_change,
            is_training=norm_training,
        )
        dv_loss = F.mse_loss(model_output, target_velocity_change_normalized)
        return dv_loss

    def save_checkpoint(self, savedir: str, *, training_state: dict):
        torch.save(self._checkpoint_payload(training_state), savedir)

    def load_checkpoint(self, ckpdir: str):
        self._load_checkpoint_payload(torch.load(ckpdir, weights_only=False, map_location="cpu"))
