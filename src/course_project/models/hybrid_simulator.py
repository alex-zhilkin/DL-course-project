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
    """Spatial simulator with frozen CV encoder injection."""

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
    ):
        super().__init__(pos_dim=pos_dim)
        self._validate_input_dims(data, min_node_features=2, min_edge_features=1)

        if hasattr(data, "x") and data.x is not None:
            dev = data.x.device
        elif torch.cuda.is_available():
            dev = "cuda"
        elif torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"
        self.device = dev

        self.node_normalizer = Normalizer(size=data.num_features, name="NodeNormalizer", device=dev)
        self.edge_normalizer = Normalizer(size=data.num_edge_features, name="EdgeNormalizer", device=dev)
        self.output_normalizer = Normalizer(size=pos_dim, name="OutputNormalizer", device=dev)

        self.backbone = LocalGNNBackbone(
            in_node_dim=data.num_features,
            in_edge_dim=data.num_edge_features,
            hidden_size=hidden_size,
            n_layers=n_layers,
            num_mlp=num_mlp,
        )
        self.decoder = build_mlp(hidden_size, hidden_size, pos_dim, num_mlp=num_mlp, lay_norm=False)

        self.cv_checkpoint_path = str(cv_checkpoint_path)
        self.cv_encoder = self._load_frozen_cv_encoder(
            data=data,
            pos_dim=pos_dim,
        )
        self.cv_dim = int(self.cv_encoder.k_cv)
        self.cv_injector = build_mlp(
            self.cv_dim,
            hidden_size,
            hidden_size,
            num_mlp=max(2, num_mlp - 1),
            lay_norm=False,
        )
        self.cv_node_gate = build_mlp(
            hidden_size + hidden_size,
            hidden_size,
            1,
            num_mlp=2,
            lay_norm=False,
        )
        self.cv_inject_scale = torch.nn.Parameter(
            torch.tensor(float(cv_inject_scale_init), dtype=torch.float32)
        )

        self.last_cv = None
        self._gate_mean_sum = 0.0
        self._gate_mean_count = 0
        self._last_gate_mean = 0.0
        self.freeze_normalizers = False

        self.cv_encoder.eval()
        for p in self.cv_encoder.parameters():
            p.requires_grad = False

    def set_epoch(self, epoch: int):
        super().set_epoch(epoch)
        self._gate_mean_sum = 0.0
        self._gate_mean_count = 0
        self._last_gate_mean = 0.0

    def _load_frozen_cv_encoder(
        self,
        *,
        data: Data,
        pos_dim: int,
    ) -> CVTransformerModel:
        payload = torch.load(str(Path(self.cv_checkpoint_path)), map_location="cpu", weights_only=False)
        cfg = payload["model_config"] if "model_config" in payload else payload["cfg"]
        extras = cfg["model_extras"]

        cv_hidden = int(cfg["hidden_size"])
        cv_layers = int(cfg["n_layers"])
        cv_num_mlp = int(extras["num_mlp"])
        cv_k1 = int(extras["K1"])
        cv_cv = int(extras["CV"])
        cv_t_layers = int(extras["transformer_layers"])
        cv_t_heads = int(extras["transformer_heads"])
        cv_t_drop = float(extras["transformer_dropout"])
        edge_aggr = extras["edge_aggr"]
        use_local_skip = bool(extras["use_local_skip"])

        encoder = CVTransformerModel(
            data=data,
            hidden_size=cv_hidden,
            n_layers=cv_layers,
            pos_dim=pos_dim,
            num_mlp=cv_num_mlp,
            K1=cv_k1,
            CV=cv_cv,
            transformer_layers=cv_t_layers,
            transformer_heads=cv_t_heads,
            transformer_dropout=cv_t_drop,
            edge_aggr=edge_aggr,
            use_local_skip=use_local_skip,
        ).to(self.device)
        state = payload["model"] if "model" in payload else payload["model_state_dict"]
        encoder.load_state_dict(state, strict=False)
        return encoder

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

    def _encode_with_cv(self, norm_graph: Data, raw_graph: Data) -> Data:
        latent = self.backbone(norm_graph)

        with torch.no_grad():
            self.cv_encoder.eval()
            cv = self.cv_encoder.extract_cv(raw_graph, is_training=False)
        self.last_cv = cv.detach()

        if hasattr(latent, "batch") and latent.batch is not None:
            batch = latent.batch
        else:
            batch = torch.zeros(latent.x.size(0), dtype=torch.long, device=latent.x.device)

        cv_hidden = self.cv_injector(cv)
        cv_nodes = cv_hidden[batch]
        node_gate = torch.sigmoid(self.cv_node_gate(torch.cat([latent.x, cv_nodes], dim=1)))
        latent.x = latent.x + self.cv_inject_scale * node_gate * cv_nodes
        if self.training:
            self._last_gate_mean = float(node_gate.detach().mean().cpu().item())
            self._gate_mean_sum += self._last_gate_mean
            self._gate_mean_count += 1
        return latent

    def get_global_gate_stats(self) -> dict:
        return {
            "mean": self._gate_mean_sum / self._gate_mean_count,
            "last": self._last_gate_mean,
        }

    def forward(self, data: Data, is_training: bool = True) -> Tensor:
        norm_graph = self.normalize_graph(data, is_training=is_training)
        latent = self._encode_with_cv(norm_graph, data)
        return self.decoder(latent.x)

    def extract_cv(self, data: Data, *, is_training: bool = False) -> Tensor:
        with torch.no_grad():
            self.cv_encoder.eval()
            return self.cv_encoder.extract_cv(data, is_training=False).detach()

    @classmethod
    def _recalc_edges(
        cls,
        data: Data,
        dummy_sep: bool = False,
        pos_dim: int | None = None,
    ) -> Tensor:
        edge_vectors = get_correct_edge_vec(data, pos_dim=pos_dim)
        distances = torch.norm(edge_vectors, dim=1)
        bond_coeffs = data.edge_attr[:, -1] if not dummy_sep else data.edge_attr[:, -2]
        return (
            torch.column_stack([edge_vectors, distances, bond_coeffs])
            if not dummy_sep
            else torch.column_stack([edge_vectors, distances, bond_coeffs, data.edge_attr[:, -1]])
        )

    def update(self, inputs: ModelInputs, model_output: Tensor, dummy_sep: bool = False) -> Data:
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
        new_edge_attr = self._recalc_edges(tmp, dummy_sep, self.pos_dim)

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
        return F.mse_loss(model_output, target_velocity_change_normalized)

    def save_checkpoint(self, savedir: str, *, training_state: dict):
        model = self.state_dict()
        _output_normalizer = self.output_normalizer.get_variable()
        _node_normalizer = self.node_normalizer.get_variable()
        _edge_normalizer = self.edge_normalizer.get_variable()

        to_save = {
            "model": model,
            "output_normalizer": _output_normalizer,
            "node_normalizer": _node_normalizer,
            "edge_normalizer": _edge_normalizer,
            "model_config": self.cfg,
        }
        to_save.update(training_state)
        torch.save(to_save, savedir)

    def load_checkpoint(self, ckpdir: str):
        dicts = torch.load(ckpdir, weights_only=False, map_location="cpu")
        self.load_state_dict(dicts["model"], strict=False)
        self.cfg = dicts["model_config"]
        for para, value in dicts["output_normalizer"].items():
            cur = getattr(self.output_normalizer, para)
            if isinstance(cur, torch.Tensor) and isinstance(value, torch.Tensor):
                cur.copy_(value)
            else:
                setattr(self.output_normalizer, para, value)
        for para, value in dicts["node_normalizer"].items():
            cur = getattr(self.node_normalizer, para)
            if isinstance(cur, torch.Tensor) and isinstance(value, torch.Tensor):
                cur.copy_(value)
            else:
                setattr(self.node_normalizer, para, value)
        for para, value in dicts["edge_normalizer"].items():
            cur = getattr(self.edge_normalizer, para)
            if isinstance(cur, torch.Tensor) and isinstance(value, torch.Tensor):
                cur.copy_(value)
            else:
                setattr(self.edge_normalizer, para, value)
