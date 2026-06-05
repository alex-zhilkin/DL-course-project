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

class EdgeTransformerEncoder(torch.nn.Module):
    """Turns edge attributes into edge tokens and mixes them with self-attention."""

    def __init__(
        self,
        in_edge_dim: int,
        hidden_size: int,
        num_mlp: int,
        transformer_layers: int,
        transformer_heads: int,
        transformer_dropout: float,
    ):
        super().__init__()
        self.edge_in = build_mlp(in_edge_dim, hidden_size, hidden_size, num_mlp=num_mlp, lay_norm=False)
        if int(transformer_layers) > 0:
            layer = torch.nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=transformer_heads,
                dim_feedforward=4 * hidden_size,
                dropout=transformer_dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.token_transformer = torch.nn.TransformerEncoder(layer, num_layers=int(transformer_layers))
        else:
            self.token_transformer = None

    def forward(self, data: Data) -> Tensor:
        h_edge = self.edge_in(data.edge_attr)
        edge_batch = data.batch[data.edge_index[0]]
        dense_tokens, mask = to_dense_batch(h_edge, edge_batch)
        if self.token_transformer is not None:
            dense_tokens = self.token_transformer(dense_tokens, src_key_padding_mask=~mask)
        return dense_tokens, mask



class GraphTokenAdapter(torch.nn.Module):
    """Builds the token input that the CV core expects."""

    def __init__(
        self,
        *,
        in_edge_dim: int,
        hidden_size: int,
        num_mlp: int,
        transformer_layers: int,
        transformer_heads: int,
        transformer_dropout: float,
    ):
        super().__init__()
        self.frontend = EdgeTransformerEncoder(
            in_edge_dim=in_edge_dim,
            hidden_size=hidden_size,
            num_mlp=num_mlp,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            transformer_dropout=transformer_dropout,
        )

    def build_core_input(self, data: Data) -> CVCoreInput:
        dense_tokens, mask = self.frontend(data)
        return CVCoreInput(tokens=dense_tokens, mask=mask)


class Model(BaseSimulator):
    """Graph CV simulator: graph encoder -> CVs -> node motion decoder."""

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
        token_sizes: tuple[int, ...],
        use_normalization: bool = True,
        prediction_target: str = "acceleration",
        global_decoder_max_nodes: int | None = None,
        global_decoder_layers: int = 1,
        global_decoder_local_skip: bool = False,
    ):
        super().__init__(pos_dim=pos_dim)
        if pos_dim not in (2, 3):
            raise ValueError(f"pos_dim must be 2 or 3, got {pos_dim}")
        if data.num_node_features < 2 or data.num_edge_features < 1:
            raise ValueError("cv_transformer expects at least 2 node features and 1 edge feature")
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
        self.global_decoder_local_skip = bool(global_decoder_local_skip)
        self.global_decoder_max_nodes = int(global_decoder_max_nodes) if global_decoder_max_nodes is not None else int(data.num_nodes)

        self.adapter = GraphTokenAdapter(
            in_edge_dim=data.num_edge_features,
            hidden_size=hidden_size,
            num_mlp=num_mlp,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            transformer_dropout=transformer_dropout,
        )
        self.core = SharedCVCore(
            CVCoreConfig(
                hidden_size=hidden_size,
                transformer_layers=0,
                transformer_heads=transformer_heads,
                transformer_dropout=transformer_dropout,
                token_sizes=token_sizes,
            )
        )
        self.k_cv = int(token_sizes[-1])
        if self.global_decoder_local_skip:
            self.global_cv_head = None
            self.global_decoder_output_head = torch.nn.Linear(hidden_size + self.k_cv, self.pos_dim)
        else:
            global_decoder_out = int(self.global_decoder_max_nodes) * self.pos_dim
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
            self.global_decoder_output_head = None
        self.freeze_normalizers = False

    @staticmethod
    def _current_velocity(inputs: ModelInputs) -> Tensor:
        return inputs.cur_graph.vel_state

    def _norm_training(self, is_training: bool) -> bool:
        return bool(self.training) and bool(is_training) and (not self.freeze_normalizers)

    def normalize_graph(self, graph: Data, is_training: bool = True) -> Data:
        batch = graph.batch if getattr(graph, "batch", None) is not None else torch.zeros(graph.x.size(0), dtype=torch.long, device=graph.x.device)
        if not self.use_normalization:
            return Data(
                x=graph.x,
                edge_index=graph.edge_index,
                edge_attr=graph.edge_attr,
                box=graph.box,
                batch=batch,
            )
        norm_training = self._norm_training(is_training)
        norm_nodes = self.node_normalizer(graph.x, is_training=norm_training)
        norm_edges = self.edge_normalizer(graph.edge_attr, is_training=norm_training)
        
        return Data(
            x=norm_nodes,
            edge_index=graph.edge_index,
            edge_attr=norm_edges,
            box=graph.box,
            batch=batch,
        )

    def _forward_core(self, data: Data) -> Tensor:
        core_input = self.adapter.build_core_input(data)
        self.core.encode(core_input)
        return self._forward_global_cv_decoder(data, core_input)

    def _node_mask(self, data: Data) -> Tensor:
        counts = torch.bincount(data.batch)
        if bool((counts > self.global_decoder_max_nodes).any()):
            max_nodes = int(counts.max().item())
            raise ValueError(
                f"cv_transformer received graph with {max_nodes} nodes, "
                f"but global_decoder_max_nodes={self.global_decoder_max_nodes}"
            )
        _, mask = to_dense_batch(
            data.x[:, :1],
            data.batch,
            max_num_nodes=self.global_decoder_max_nodes,
        )
        return mask

    def encode_cv_from_normalized(
        self,
        data: Data,
        *,
        capture_attention: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor | CVCoreInput]]:
        core_input = self.adapter.build_core_input(data)
        self.core.encode(core_input, capture_attention=capture_attention)
        return self.core.last_cv, {"core_input": core_input}

    def decode_cv_from_normalized(
        self,
        data: Data,
        cv: Tensor,
        *,
        aux: dict[str, Tensor | CVCoreInput] | None = None,
        zero_local_skip: bool = False,
    ) -> Tensor:
        mask = self._node_mask(data)
        if self.global_decoder_local_skip:
            if self.global_decoder_output_head is None:
                raise RuntimeError("global decoder local skip head was not initialized")
            core_input = None if aux is None else aux.get("core_input")
            if not isinstance(core_input, CVCoreInput):
                raise RuntimeError("core_input is required for the local-skip global decoder")
            
            local_tokens = torch.zeros_like(core_input.tokens) if zero_local_skip else core_input.tokens
            dense_local = torch.zeros(
                cv.size(0),
                self.global_decoder_max_nodes,
                self.core.cfg.hidden_size,
                dtype=local_tokens.dtype,
                device=local_tokens.device,
            )
            dense_local[:, : local_tokens.size(1), :] = local_tokens
            dense_cv = cv.unsqueeze(1).expand(-1, self.global_decoder_max_nodes, -1)
            dense_pred = self.global_decoder_output_head(torch.cat([dense_local, dense_cv], dim=-1))
        else:
            if self.global_cv_head is None:
                raise RuntimeError("global CV head was not initialized")
            
            dense_decoded = self.global_cv_head(cv).view(
                cv.size(0),
                self.global_decoder_max_nodes,
                self.pos_dim,
            )
            dense_pred = dense_decoded
        return dense_pred[mask]

    def _forward_global_cv_decoder(self, data: Data, core_input: CVCoreInput) -> Tensor:
        return self.decode_cv_from_normalized(data, self.core.last_cv, aux={"core_input": core_input})

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

    def forward(
        self,
        data: Data,
        is_training: bool = True,
    ):
        data = self.normalize_graph(data, is_training=is_training)
        return self._forward_core(data)

    def extract_cv(self, data: Data, *, is_training: bool = False) -> Tensor:
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                data = self.normalize_graph(data, is_training=False)
                self._forward_core(data)
                return self.core.last_cv.detach()
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
