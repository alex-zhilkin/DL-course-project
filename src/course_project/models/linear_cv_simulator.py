from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch

from .base import BaseModelInputs, BaseSimulator
from .common import get_correct_edge_vec
from .components import Normalizer

ModelInputs = BaseModelInputs


class LinearEdgeCVEncoder(torch.nn.Module):
    """Mean-pooled edge feature pyramid using only linear layers."""

    def __init__(
        self,
        *,
        in_edge_dim: int,
        edge_token_dim: int,
        token_sizes: tuple[int, ...],
    ):
        super().__init__()
        if len(token_sizes) < 1:
            raise ValueError("token_sizes must contain at least the CV count")
        self.in_edge_dim = int(in_edge_dim)
        self.edge_token_dim = int(edge_token_dim)
        self.edge_in = torch.nn.Linear(self.in_edge_dim, self.edge_token_dim)
        sizes = [int(v) for v in token_sizes]
        in_dim = self.edge_token_dim
        layers: list[torch.nn.Module] = []
        for out_dim in sizes[:-1]:
            layers.append(torch.nn.Linear(in_dim, out_dim))
            in_dim = out_dim
        layers.append(torch.nn.Linear(in_dim, sizes[-1]))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, data: Data) -> Tensor:
        graph_count = int(data.batch.max().item()) + 1 if data.batch.numel() else 1
        edge_tokens = self.edge_in(data.edge_attr)
        edge_batch = data.batch[data.edge_index[0]]
        edge_sum = torch.zeros(graph_count, self.edge_token_dim, dtype=edge_tokens.dtype, device=edge_tokens.device)
        edge_count = torch.zeros(graph_count, 1, dtype=edge_tokens.dtype, device=edge_tokens.device)
        edge_sum.index_add_(0, edge_batch, edge_tokens)
        edge_count.index_add_(0, edge_batch, torch.ones(edge_batch.size(0), 1, dtype=edge_tokens.dtype, device=edge_tokens.device))
        edge_summary = edge_sum / edge_count.clamp(min=1.0)

        return self.net(edge_summary)


class Model(BaseSimulator):
    """Linear edge-CV simulator: edge attributes -> linear CV pyramid -> motion decoder."""

    def __init__(
        self,
        data: Data,
        hidden_size: int,
        n_layers: int,
        pos_dim: int,
        *,
        num_mlp: int,
        token_sizes: tuple[int, ...],
        use_normalization: bool = True,
        prediction_target: str = "acceleration",
        global_decoder_max_nodes: int | None = None,
        global_decoder_layers: int = 1,
        linear_encoder_max_edges: int | None = None,
        edge_token_dim: int | None = None,
        node_token_dim: int | None = None,
    ):
        super().__init__(pos_dim=pos_dim)
        if pos_dim not in (2, 3):
            raise ValueError(f"pos_dim must be 2 or 3, got {pos_dim}")
        if data.num_node_features < 2 or data.num_edge_features < 1:
            raise ValueError("linear_cv_simulator expects at least 2 node features and 1 edge feature")
        if prediction_target not in {"acceleration", "velocity"}:
            raise ValueError("prediction_target must be 'acceleration' or 'velocity'")
        if int(global_decoder_layers) < 1:
            raise ValueError("global_decoder_layers must be >= 1")

        self.device = data.x.device
        self._expected_node_dim = int(data.num_features)
        self.node_normalizer = Normalizer(size=self._expected_node_dim, name="NodeNormalizer", device=self.device)
        self.edge_normalizer = Normalizer(size=data.num_edge_features, name="EdgeNormalizer", device=self.device)
        self.output_normalizer = Normalizer(size=pos_dim, name="OutputNormalizer", device=self.device)
        self.use_normalization = bool(use_normalization)
        self.prediction_target = str(prediction_target)
        self.global_decoder_layers = int(global_decoder_layers)
        self.global_decoder_max_nodes = int(global_decoder_max_nodes) if global_decoder_max_nodes is not None else int(data.num_nodes)
        self.linear_encoder_max_edges = int(linear_encoder_max_edges) if linear_encoder_max_edges is not None else int(data.edge_index.size(1))
        self.edge_token_dim = int(edge_token_dim) if edge_token_dim is not None else int(data.num_edge_features)
        self.k_cv = int(token_sizes[-1])

        self.encoder = LinearEdgeCVEncoder(
            in_edge_dim=data.num_edge_features,
            edge_token_dim=self.edge_token_dim,
            token_sizes=token_sizes,
        )

        global_decoder_out = int(self.global_decoder_max_nodes) * pos_dim
        if self.global_decoder_layers == 1:
            self.global_cv_head = torch.nn.Linear(self.k_cv, global_decoder_out, bias=False)
        else:
            layers: list[torch.nn.Module] = []
            in_dim = self.k_cv
            for _ in range(self.global_decoder_layers - 1):
                layers.append(torch.nn.Linear(in_dim, hidden_size))
                layers.append(torch.nn.GELU())
                in_dim = hidden_size
            layers.append(torch.nn.Linear(in_dim, global_decoder_out, bias=False))
            self.global_cv_head = torch.nn.Sequential(*layers)

        self.last_cv: Tensor | None = None
        self.freeze_normalizers = False

    @staticmethod
    def _current_velocity(inputs: ModelInputs) -> Tensor:
        return inputs.cur_graph.vel_state

    def _norm_training(self, is_training: bool) -> bool:
        return bool(self.training) and bool(is_training) and (not self.freeze_normalizers)

    def normalize_graph(self, graph: Data, is_training: bool = True) -> Data:
        batch = graph.batch if getattr(graph, "batch", None) is not None else torch.zeros(graph.x.size(0), dtype=torch.long, device=graph.x.device)
        if not self.use_normalization:
            return Data(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr, box=graph.box, batch=batch)
        norm_training = self._norm_training(is_training)
        return Data(
            x=self.node_normalizer(graph.x, is_training=norm_training),
            edge_index=graph.edge_index,
            edge_attr=self.edge_normalizer(graph.edge_attr, is_training=norm_training),
            box=graph.box,
            batch=batch,
        )

    def _decode_cv(self, data: Data, cv: Tensor) -> Tensor:
        return self.decode_cv_from_normalized(data, cv)

    def encode_cv_from_normalized(
        self,
        data: Data,
        *,
        capture_attention: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        del capture_attention
        cv = self.encoder(data)
        self.last_cv = cv
        return cv, {}

    def decode_cv_from_normalized(
        self,
        data: Data,
        cv: Tensor,
        *,
        aux: dict[str, Tensor] | None = None,
        zero_local_skip: bool = False,
    ) -> Tensor:
        del aux, zero_local_skip
        counts = torch.bincount(data.batch)
        if bool((counts > self.global_decoder_max_nodes).any()):
            max_nodes = int(counts.max().item())
            raise ValueError(
                f"linear_cv_simulator received graph with {max_nodes} nodes, "
                f"but global_decoder_max_nodes={self.global_decoder_max_nodes}"
            )
        dense_pred = self.global_cv_head(cv).view(cv.size(0), self.global_decoder_max_nodes, self.pos_dim)
        _, mask = to_dense_batch(data.x[:, :1], data.batch, max_num_nodes=self.global_decoder_max_nodes)
        return dense_pred[mask]

    def predict_motion_from_cv(
        self,
        data: Data,
        *,
        cv_mask: Tensor | None = None,
        include_output_offset: bool = True,
        is_training: bool = False,
        zero_local_skip: bool = False,
        subtract_zero_cv: bool = False,
    ) -> tuple[Tensor, Tensor]:
        norm_graph = self.normalize_graph(data, is_training=is_training)
        cv, aux = self.encode_cv_from_normalized(norm_graph)
        cv_for_head = cv
        if cv_mask is not None:
            mask = torch.as_tensor(cv_mask, dtype=cv.dtype, device=cv.device).reshape(1, -1)
            cv_for_head = cv * mask
        pred_norm = self.decode_cv_from_normalized(
            norm_graph,
            cv_for_head,
            aux=aux,
            zero_local_skip=zero_local_skip,
        )
        if subtract_zero_cv:
            zero_norm = self.decode_cv_from_normalized(
                norm_graph,
                torch.zeros_like(cv_for_head),
                aux=aux,
                zero_local_skip=zero_local_skip,
            )
            pred_norm = pred_norm - zero_norm
        if not self.use_normalization:
            pred = pred_norm
        elif include_output_offset:
            pred = self.output_normalizer.inverse(pred_norm)
        else:
            pred = pred_norm * self.output_normalizer._std_with_epsilon()
        return cv.detach(), pred

    def forward(self, data: Data, is_training: bool = True):
        data = self.normalize_graph(data, is_training=is_training)
        cv, _ = self.encode_cv_from_normalized(data)
        return self._decode_cv(data, cv)

    def extract_cv(self, data: Data, *, is_training: bool = False) -> Tensor:
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                data = self.normalize_graph(data, is_training=False)
                cv = self.encoder(data)
                self.last_cv = cv
                return cv.detach()
        finally:
            self.train(was_training)

    @classmethod
    def _recalc_edges(cls, data: Data, pos_dim: int | None = None) -> Tensor:
        edge_vectors = get_correct_edge_vec(data, pos_dim=pos_dim)
        distances = torch.norm(edge_vectors, dim=1)
        bond_coeffs = data.edge_attr[:, -1]
        return torch.column_stack([edge_vectors, distances, bond_coeffs])

    def update(self, inputs: ModelInputs, model_output: Tensor) -> Data:
        predicted = model_output if not self.use_normalization else self.output_normalizer.inverse(model_output)
        cur_velocity = self._current_velocity(inputs)
        updated_velocity = predicted if self.prediction_target == "velocity" else cur_velocity + predicted
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
        target = target_velocity if self.prediction_target == "velocity" else target_velocity - cur_velocity
        if self.use_normalization:
            norm_training = self._norm_training(self.training if accumulate_norm_stats is None else accumulate_norm_stats)
            target = self.output_normalizer(target, is_training=norm_training)
        return F.mse_loss(model_output, target)
