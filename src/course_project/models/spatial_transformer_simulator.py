from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data

from .base import BaseModelInputs, BaseSimulator
from .common import get_correct_edge_vec, init_scaled_linear_head
from .components import NodeEdgeFusionEncoder, Normalizer, TokenGlobalDecoder

ModelInputs = BaseModelInputs
torch.set_default_dtype(torch.float32)


class TwoStageDownUpTransformer(nn.Module):
    def __init__(
        self,
        in_node_dim: int,
        in_edge_dim: int,
        hidden_size: int,
        pos_dim: int,
        *,
        K1: int,
        K2: int,
        transformer_layers: int,
        transformer_heads: int,
        transformer_dropout: float,
        num_mlp: int,
        edge_aggr: str,
        k2_hidden_size: int,
        use_local_skip: bool,
    ):
        super().__init__()
        self.K1 = int(K1)
        self.K2 = int(K2)
        self.hidden_size = int(hidden_size)
        self.k2_hidden_size = int(k2_hidden_size)
        self.use_local_skip = bool(use_local_skip)
        # If true, we also pass local node features directly (identity signal).
        self.token_sizes = [self.K1, self.K2]
        self.last_cv = None

        self.frontend = NodeEdgeFusionEncoder(
            in_node_dim=in_node_dim,
            in_edge_dim=in_edge_dim,
            hidden_size=self.hidden_size,
            num_mlp=num_mlp,
            edge_aggr=edge_aggr,
        )
        self.global_decoder = TokenGlobalDecoder(
            hidden_size=self.hidden_size,
            K1=self.K1,
            K2=self.K2,
            k2_hidden_size=self.k2_hidden_size,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            transformer_dropout=transformer_dropout,
        )
        out_in_dim = self.hidden_size * 2 if self.use_local_skip else self.hidden_size
        self.out = nn.Linear(out_in_dim, pos_dim)
        self.init_weights()

    @property
    def tokens1(self) -> nn.Parameter:
        return self.global_decoder.tokens1

    @property
    def pool1(self) -> nn.MultiheadAttention:
        return self.global_decoder.pool1

    def init_weights(self) -> None:
        init_scaled_linear_head(self.out, scale=0.1)

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
            decoded = self.global_decoder(
                local_nodes,
                data.batch if hasattr(data, "batch") else None,
                return_attn=return_attn,
                return_z2=return_z2,
            )
            if return_z2:
                h_dense, h_from_bottleneck, mask, attn, z2_mixed = decoded
            else:
                h_dense, h_from_bottleneck, mask, attn = decoded
                z2_mixed = None
        else:
            h_dense, h_from_bottleneck, mask = self.global_decoder(
                local_nodes,
                data.batch if hasattr(data, "batch") else None,
                return_attn=False,
                return_z2=False,
            )
            attn = None
            z2_mixed = None

        self.last_cv = self.global_decoder.last_cv
        if self.use_local_skip:
            # Keep local node identity hint together with global context.
            h_fused = torch.cat([h_dense, h_from_bottleneck], dim=-1)
        else:
            h_fused = h_from_bottleneck
        dv_dense = self.out(h_fused)
        dv = dv_dense[mask]

        if return_context and return_attn:
            if return_z2:
                return dv, h_from_bottleneck[mask], attn, z2_mixed
            return dv, h_from_bottleneck[mask], attn
        if return_context:
            if return_z2:
                return dv, h_from_bottleneck[mask], z2_mixed
            return dv, h_from_bottleneck[mask]
        if return_z2:
            return dv, z2_mixed
        return dv


class Model(BaseSimulator):
    def __init__(
        self,
        data: Data,
        hidden_size: int,
        n_layers: int,
        pos_dim: int,
        *,
        num_mlp: int,
        K1: int,
        K2: int,
        transformer_layers: int,
        transformer_heads: int,
        transformer_dropout: float,
        edge_aggr: str,
        k2_hidden_size: int,
        use_local_skip: bool,
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

        self._expected_node_dim = int(data.num_features)
        self.node_normalizer = Normalizer(size=self._expected_node_dim, name="NodeNormalizer", device=dev)
        self.edge_normalizer = Normalizer(size=data.num_edge_features, name="EdgeNormalizer", device=dev)
        self.output_normalizer = Normalizer(size=pos_dim, name="OutputNormalizer", device=dev)

        self.core = TwoStageDownUpTransformer(
            in_node_dim=self._expected_node_dim,
            in_edge_dim=data.num_edge_features,
            hidden_size=hidden_size,
            pos_dim=pos_dim,
            K1=K1,
            K2=K2,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            transformer_dropout=transformer_dropout,
            num_mlp=num_mlp,
            edge_aggr=edge_aggr,
            k2_hidden_size=k2_hidden_size,
            use_local_skip=use_local_skip,
        )
        self.last_cv = None
        self.freeze_normalizers = False

    def _norm_training(self, is_training: bool) -> bool:
        return bool(self.training) and bool(is_training) and (not self.freeze_normalizers)

    @staticmethod
    def _current_velocity(inputs: ModelInputs) -> Tensor:
        if hasattr(inputs.cur_graph, "vel_state"):
            return inputs.cur_graph.vel_state
        return inputs.cur_position - inputs.prev_position

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

    def _predict_with_z2(self, data: Data) -> tuple[Tensor, Tensor]:
        pred, z2 = self.core(data, return_z2=True)
        return pred, z2

    def forward(self, data: Data, is_training: bool = True) -> Tensor:
        data = self.normalize_graph(data, is_training=is_training)
        pred, _z2 = self._predict_with_z2(data)
        self.last_cv = self.core.last_cv.detach()
        return pred

    def extract_cv(self, data: Data, *, is_training: bool = False) -> Tensor:
        data = self.normalize_graph(data, is_training=is_training)
        _pred, _z2 = self._predict_with_z2(data)
        return self.core.last_cv.detach()

    @classmethod
    def _recalc_edges(cls, data: Data, dummy_sep: bool = False, pos_dim: int | None = None) -> Tensor:
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
        if hasattr(inputs.cur_graph, "vel_state"):
            updated_velocity = cur_velocity + predicted
        else:
            updated_velocity = predicted
        predicted_position = inputs.cur_position + updated_velocity

        tmp = Data(
            x=predicted_position.clone().float(),
            edge_index=inputs.cur_graph.edge_index,
            edge_attr=inputs.cur_graph.edge_attr.float(),
            box=inputs.cur_graph.box if hasattr(inputs.cur_graph, "box") else None,
            dtype=torch.float32,
        )
        new_edge_attr = self._recalc_edges(tmp, dummy_sep, self.pos_dim)

        return Data(
            x=predicted_position.float(),
            edge_index=inputs.cur_graph.edge_index,
            edge_attr=new_edge_attr.float(),
            box=inputs.cur_graph.box if hasattr(inputs.cur_graph, "box") else None,
            dtype=torch.float32,
        )

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
            target_velocity_change, is_training=norm_training
        )
        dv_loss = F.mse_loss(model_output, target_velocity_change_normalized)
        return dv_loss

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
