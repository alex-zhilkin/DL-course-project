from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data

from .base import BaseModelInputs, BaseSimulator
from .common import build_mlp, get_correct_edge_vec
from .components import LocalGNNBackbone, Normalizer
from .cv_transformer_simulator import Model as CVTransformerModel

ModelInputs = BaseModelInputs
torch.set_default_dtype(torch.float32)


class Model(BaseSimulator):
    """Local GNN with a frozen CV model (global context) injected in the middle"""

    def __init__(
        self,
        data: Data,
        hidden_size: int,
        n_layers: int,
        pos_dim: int,
        *,
        num_mlp: int,
        cv_checkpoint_path: str,
        cv_inject_scale_init: float = 1.0,
        cv_consistency_weight: float = 0.0,
        time_lag_steps: int = 0,
        time_lag_weight: float = 0.0,
    ):
        super().__init__(pos_dim=pos_dim)
        if pos_dim not in (2, 3):
            raise ValueError(f"pos_dim must be 2 or 3, got {pos_dim}")
        if data.num_node_features < 2 or data.num_edge_features < 1:
            raise ValueError("hybrid expects at least 2 node features and 1 edge feature")

        self.device = data.x.device

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

        self.cv_checkpoint_path = str(cv_checkpoint_path)
        self.cv_encoder = self._load_frozen_cv_encoder(
            data=data,
            pos_dim=pos_dim,
        )
        self.cv_dim = int(self.cv_encoder.k_cv)
        self.cv_film = build_mlp(
            self.cv_dim,
            hidden_size,
            hidden_size * 2,
            num_mlp=2,
            lay_norm=False,
        )
        self.decoder = build_mlp(hidden_size, hidden_size, pos_dim, num_mlp=num_mlp, lay_norm=False)
        self.decoder_tau = build_mlp(
            hidden_size,
            hidden_size,
            pos_dim,
            num_mlp=num_mlp,
            lay_norm=False,
        )
        self.cv_inject_scale = torch.nn.Parameter(
            torch.tensor(float(cv_inject_scale_init))
        )
        self.cv_consistency_weight = float(cv_consistency_weight)
        self.time_lag_steps = int(time_lag_steps)
        self.time_lag_weight = float(time_lag_weight)

        self.last_cv = None
        self._film_mean_sum = 0.0
        self._film_mean_count = 0
        self._last_film_mean = 0.0
        self.freeze_normalizers = False

        self.cv_encoder.eval()
        for p in self.cv_encoder.parameters():
            p.requires_grad = False

    def _load_frozen_cv_encoder(
        self,
        *,
        data: Data,
        pos_dim: int,
    ) -> CVTransformerModel:
        payload = torch.load(str(Path(self.cv_checkpoint_path)), map_location="cpu", weights_only=False)
        cfg = payload["model_config"]
        extras = cfg["model_extras"]

        cv_hidden = int(cfg["hidden_size"])
        cv_layers = int(cfg["n_layers"])
        cv_num_mlp = int(extras["num_mlp"])
        cv_t_layers = int(extras["transformer_layers"])
        cv_t_heads = int(extras["transformer_heads"])
        cv_t_drop = float(extras["transformer_dropout"])
        token_sizes = tuple(int(v) for v in extras["token_sizes"])
        encoder = CVTransformerModel(
            data=data,
            hidden_size=cv_hidden,
            n_layers=cv_layers,
            pos_dim=pos_dim,
            num_mlp=cv_num_mlp,
            transformer_layers=cv_t_layers,
            transformer_heads=cv_t_heads,
            transformer_dropout=cv_t_drop,
            token_sizes=token_sizes,
        ).to(self.device)
        encoder.load_state_dict(payload["model"])
        return encoder

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

    @staticmethod
    def _node_batch(data: Data) -> Tensor:
        return data.batch

    def _apply_film(self, x: Tensor, cv: Tensor, batch: Tensor) -> tuple[Tensor, Tensor]:
        gamma_beta = self.cv_film(cv)[batch]
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gamma = torch.tanh(gamma)
        beta = torch.tanh(beta)
        out = x * (1.0 + self.cv_inject_scale * gamma) + self.cv_inject_scale * beta
        strength = 0.5 * (gamma.abs().mean() + beta.abs().mean())
        return out, strength

    def _encode_with_cv(self, norm_graph: Data, raw_graph: Data) -> Data:
        with torch.no_grad():
            self.cv_encoder.eval()
            cv = self.cv_encoder.extract_cv(raw_graph, is_training=False)
        self.last_cv = cv.detach()
        batch = self._node_batch(norm_graph)

        latent = Data(
            x=self.backbone.node_encoder(norm_graph.x),
            edge_index=norm_graph.edge_index,
            edge_attr=self.backbone.edge_encoder(norm_graph.edge_attr),
            box=norm_graph.box,
            batch=batch,
        )
        split_idx = max(1, len(self.backbone.layers) // 2)
        for layer in self.backbone.layers[:split_idx]:
            latent = layer(latent)
        latent.x, film_strength = self._apply_film(latent.x, cv, batch)
        for layer in self.backbone.layers[split_idx:]:
            latent = layer(latent)
        if self.training:
            self._last_film_mean = float(film_strength.detach().cpu().item())
            self._film_mean_sum += self._last_film_mean
            self._film_mean_count += 1
        return latent

    def get_global_film_stats(self) -> dict:
        return {
            "mean": self._film_mean_sum / self._film_mean_count,
            "last": self._last_film_mean,
        }

    def forward(self, data: Data, is_training: bool = True) -> Tensor:
        norm_graph = self.normalize_graph(data, is_training=is_training)
        latent = self._encode_with_cv(norm_graph, data)
        return self.decoder(latent.x)

    def predict_time_lag_acc(self, data: Data, *, is_training: bool = True) -> Tensor:
        norm_graph = self.normalize_graph(data, is_training=is_training)
        latent = self._encode_with_cv(norm_graph, data)
        return self.decoder_tau(latent.x)

    def extract_cv(self, data: Data, *, is_training: bool = False) -> Tensor:
        with torch.no_grad():
            self.cv_encoder.eval()
            return self.cv_encoder.extract_cv(data, is_training=False).detach()

    def cv_consistency_loss(self, pred_graph: Data, target_graph: Data) -> Tensor:
        with torch.no_grad():
            self.cv_encoder.eval()
            target_cv = self.cv_encoder.extract_cv(target_graph, is_training=False).detach()
        pred_cv = self.cv_encoder.extract_cv(pred_graph, is_training=False)
        return F.mse_loss(pred_cv, target_cv)


    def _recalc_edges(
        cls,
        data: Data,
        pos_dim: int | None = None,
    ) -> Tensor:
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
        norm_training = self._norm_training(
            self.training if accumulate_norm_stats is None else accumulate_norm_stats
        )
        target_velocity_change = target_velocity - cur_velocity
        target_velocity_change_normalized = self.output_normalizer(
            target_velocity_change,
            is_training=norm_training,
        )
        return F.mse_loss(model_output, target_velocity_change_normalized)
