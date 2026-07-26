from __future__ import annotations

import hashlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.transforms import AddLaplacianEigenvectorPE
from torch_geometric.utils import to_dense_batch

from .base import BaseModelInputs, BaseSimulator
from .common import build_mlp, get_correct_edge_vec
from .components import Normalizer

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
        K1: int = 32,
        K2: int = 8,
        K3: int | None = None,
        K4: int | None = None,
        transformer_layers: int = 2,
        transformer_heads: int = 1,
        transformer_dropout: float = 0.0,
        num_mlp: int = 3,
        edge_aggr: str = "mean",
        cv_dim: int = 2,
        k2_hidden_size: int | None = 1,
        use_local_skip: bool = False,
    ):
        super().__init__()
        if edge_aggr not in {"mean", "sum"}:
            raise ValueError("edge_aggr must be 'mean' or 'sum'")

        self.K1 = int(K1)
        self.K2 = int(K2)
        self.hidden_size = int(hidden_size)
        self.k2_hidden_size = int(k2_hidden_size) if k2_hidden_size is not None else int(hidden_size)
        self.edge_aggr = str(edge_aggr)
        self.cv_dim = int(cv_dim)
        # Kept as an accepted config key for old notebooks/checkpoints, but the
        # transformer decoder no longer receives a direct input/local skip.
        self.use_local_skip = False
        self.last_cv = None

        self.node_in = build_mlp(in_node_dim, hidden_size, hidden_size, num_mlp=num_mlp, lay_norm=False)
        self.edge_in = build_mlp(in_edge_dim, hidden_size, hidden_size, num_mlp=num_mlp, lay_norm=False)
        self.fuse = build_mlp(hidden_size * 2, hidden_size, hidden_size, num_mlp=num_mlp, lay_norm=False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=transformer_heads,
            dim_feedforward=4 * hidden_size,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.node_transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        self.out = nn.Linear(hidden_size, pos_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=1.0)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                if getattr(module, "in_proj_weight", None) is not None:
                    nn.init.xavier_uniform_(module.in_proj_weight, gain=1.0)
                if getattr(module, "in_proj_bias", None) is not None:
                    nn.init.zeros_(module.in_proj_bias)
                if getattr(module, "q_proj_weight", None) is not None:
                    nn.init.xavier_uniform_(module.q_proj_weight, gain=1.0)
                if getattr(module, "k_proj_weight", None) is not None:
                    nn.init.xavier_uniform_(module.k_proj_weight, gain=1.0)
                if getattr(module, "v_proj_weight", None) is not None:
                    nn.init.xavier_uniform_(module.v_proj_weight, gain=1.0)
                nn.init.xavier_uniform_(module.out_proj.weight, gain=1.0)
                if module.out_proj.bias is not None:
                    nn.init.zeros_(module.out_proj.bias)

        self.out.weight.data.mul_(0.1)

    def _edge_to_node(self, data: Data, e_emb: Tensor) -> Tensor:
        _, col = data.edge_index
        n_nodes = data.x.size(0)
        hidden = e_emb.size(1)
        node_sum = torch.zeros(n_nodes, hidden, device=e_emb.device, dtype=e_emb.dtype)
        node_cnt = torch.zeros(n_nodes, 1, device=e_emb.device, dtype=e_emb.dtype)
        node_sum.index_add_(0, col, e_emb)
        node_cnt.index_add_(0, col, torch.ones((e_emb.size(0), 1), device=e_emb.device, dtype=e_emb.dtype))
        if self.edge_aggr == "sum":
            return node_sum
        return node_sum / node_cnt.clamp(min=1.0)

    def forward(
        self,
        data: Data,
        *,
        return_context: bool = False,
        return_attn: bool = False,
        return_z2: bool = False,
    ):
        h_node = self.node_in(data.x)
        e_emb = self.edge_in(data.edge_attr)
        e_node = self._edge_to_node(data, e_emb)
        h = self.fuse(torch.cat([h_node, e_node], dim=-1))

        batch = data.batch if getattr(data, "batch", None) is not None else torch.zeros(h.size(0), dtype=torch.long, device=h.device)
        h_dense, mask = to_dense_batch(h, batch)
        batch_size, nmax, _ = h_dense.shape
        key_padding_nodes = ~mask

        encoded_nodes = self.node_transformer(h_dense, src_key_padding_mask=key_padding_nodes)
        valid_counts = mask.sum(dim=1).clamp(min=1).to(encoded_nodes.dtype).unsqueeze(-1)
        self.last_cv = (encoded_nodes * mask.unsqueeze(-1).to(encoded_nodes.dtype)).sum(dim=1) / valid_counts
        dv_dense = self.out(encoded_nodes)
        dv = dv_dense[mask]

        if return_context and return_attn:
            attn = {
                "pool1": None,
                "pool2": None,
                "global": None,
                "unpool2": None,
                "pools": [],
                "unpools": [],
                "token_sizes": [int(nmax)],
                "node_mask": mask,
            }
            if return_z2:
                return dv, encoded_nodes[mask], attn, self.last_cv
            return dv, encoded_nodes[mask], attn
        if return_context:
            if return_z2:
                return dv, encoded_nodes[mask], self.last_cv
            return dv, encoded_nodes[mask]
        if return_z2:
            return dv, self.last_cv
        return dv


class Model(BaseSimulator):
    def __init__(
        self,
        data: Data,
        hidden_size: int,
        n_layers: int,
        pos_dim: int,
        *,
        num_mlp: int = 3,
        K1: int = 32,
        K2: int = 8,
        K3: int | None = None,
        K4: int | None = None,
        transformer_layers: int = 2,
        transformer_heads: int = 1,
        transformer_dropout: float = 0.0,
        cv_dim: int = 2,
        k2_hidden_size: int | None = 1,
        use_local_skip: bool = False,
        use_lap_pe: bool = False,
        lap_pe_k: int = 0,
        lap_pe_is_undirected: bool = True,
        edge_aggr: str = "mean",
        use_normalization: bool = True,
        prediction_target: str = "acceleration",
        **_,
    ):
        super().__init__(pos_dim=pos_dim)
        self._validate_input_dims(data, min_node_features=2, min_edge_features=1)
        if prediction_target not in {"acceleration", "velocity"}:
            raise ValueError("prediction_target must be 'acceleration' or 'velocity'")

        self.device = data.x.device if getattr(data, "x", None) is not None else torch.device("cpu")
        self.use_normalization = bool(use_normalization)
        self.prediction_target = str(prediction_target)
        self.use_lap_pe = bool(use_lap_pe)
        self.lap_pe_k = int(lap_pe_k)
        self.lap_pe_is_undirected = bool(lap_pe_is_undirected)
        self._lap_pe_attr_name = "laplacian_eigenvector_pe"
        self._lap_pe_transform = (
            AddLaplacianEigenvectorPE(
                k=self.lap_pe_k,
                attr_name=self._lap_pe_attr_name,
                is_undirected=self.lap_pe_is_undirected,
            )
            if self.use_lap_pe and self.lap_pe_k > 0
            else None
        )
        self._lap_pe_cache: dict[str, Tensor] = {}
        self._lap_validated_keys: set[str] = set()
        self._base_node_dim = int(data.num_features)
        self._expected_node_dim = self._base_node_dim + (self.lap_pe_k if self._lap_pe_transform is not None else 0)

        self.node_normalizer = Normalizer(size=self._expected_node_dim, name="NodeNormalizer", device=self.device)
        self.edge_normalizer = Normalizer(size=data.num_edge_features, name="EdgeNormalizer", device=self.device)
        self.output_normalizer = Normalizer(size=pos_dim, name="OutputNormalizer", device=self.device)
        self.core = TwoStageDownUpTransformer(
            in_node_dim=self._expected_node_dim,
            in_edge_dim=data.num_edge_features,
            hidden_size=hidden_size,
            pos_dim=pos_dim,
            K1=K1,
            K2=K2,
            K3=K3,
            K4=K4,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            transformer_dropout=transformer_dropout,
            num_mlp=num_mlp,
            edge_aggr=edge_aggr,
            cv_dim=cv_dim,
            k2_hidden_size=k2_hidden_size,
            use_local_skip=use_local_skip,
        )
        self.last_dv_loss = None
        self.cv_dim = int(cv_dim)
        self.last_cv = None
        self.freeze_normalizers = False

    @staticmethod
    def _validate_input_dims(data: Data, *, min_node_features: int, min_edge_features: int) -> None:
        if data.num_node_features < min_node_features or data.num_edge_features < min_edge_features:
            raise ValueError(
                f"transformer_simulator expects at least {min_node_features} node features "
                f"and {min_edge_features} edge features"
            )

    def _canonicalize_lap_pe(self, pe: Tensor) -> Tensor:
        pe = pe.clone()
        if pe.numel() == 0:
            return pe
        for j in range(pe.shape[1]):
            col = pe[:, j]
            idx = torch.argmax(torch.abs(col))
            sign = torch.sign(col[idx])
            if sign == 0:
                sign = torch.tensor(1.0, dtype=col.dtype, device=col.device)
            pe[:, j] = col * sign
        return pe

    def _lap_cache_key(self, graph: Data) -> str | None:
        if self._lap_pe_transform is None or not hasattr(graph, "edge_index") or graph.edge_index is None:
            return None
        try:
            edge_index_cpu = graph.edge_index.detach().cpu().contiguous()
            if hasattr(graph, "edge_attr") and graph.edge_attr is not None:
                edge_weight_cpu = graph.edge_attr[:, -1].detach().cpu().contiguous()
            elif hasattr(graph, "edge_weight") and graph.edge_weight is not None:
                edge_weight_cpu = graph.edge_weight.detach().cpu().contiguous()
            else:
                edge_weight_cpu = torch.ones(edge_index_cpu.size(1), dtype=torch.float32)
            h = hashlib.sha1()
            h.update(str(int(graph.num_nodes)).encode())
            h.update(edge_index_cpu.numpy().tobytes())
            h.update(edge_weight_cpu.numpy().astype("float32", copy=False).tobytes())
            h.update(str(self.lap_pe_k).encode())
            return h.hexdigest()
        except Exception:
            return None

    @staticmethod
    def _validate_undirected_edge_pairs(edge_index_cpu: Tensor, edge_weight_cpu: Tensor, *, atol: float = 1e-8) -> tuple[bool, int]:
        edges = edge_index_cpu.detach().cpu().numpy().T.tolist()
        weights = edge_weight_cpu.detach().cpu().numpy()
        idx_by_edge = {tuple(edge): i for i, edge in enumerate(edges)}
        mismatches = 0
        for (src, dst), i in idx_by_edge.items():
            j = idx_by_edge.get((dst, src))
            if j is None or abs(float(weights[i]) - float(weights[j])) > atol:
                mismatches += 1
        return mismatches == 0, mismatches

    def _compute_lap_pe(self, graph: Data) -> Tensor:
        if self._lap_pe_transform is None:
            return torch.zeros((graph.x.size(0), 0), dtype=graph.x.dtype, device=graph.x.device)
        if hasattr(graph, self._lap_pe_attr_name):
            existing = getattr(graph, self._lap_pe_attr_name)
            if isinstance(existing, torch.Tensor) and existing.dim() == 2:
                if existing.size(0) == graph.x.size(0) and existing.size(1) == self.lap_pe_k:
                    return existing.to(dtype=graph.x.dtype, device=graph.x.device)

        cache_key = self._lap_cache_key(graph)
        if cache_key is not None and cache_key in self._lap_pe_cache:
            return self._lap_pe_cache[cache_key].to(dtype=graph.x.dtype, device=graph.x.device)

        edge_index_cpu = graph.edge_index.detach().cpu()
        if hasattr(graph, "edge_attr") and graph.edge_attr is not None:
            edge_weight_cpu = graph.edge_attr[:, -1].detach().cpu()
        elif hasattr(graph, "edge_weight") and graph.edge_weight is not None:
            edge_weight_cpu = graph.edge_weight.detach().cpu()
        else:
            edge_weight_cpu = torch.ones(edge_index_cpu.size(1), dtype=torch.float32)

        if self.lap_pe_is_undirected:
            validation_key = cache_key if cache_key is not None else f"uncached_{id(graph)}"
            if validation_key not in self._lap_validated_keys:
                if torch.any(edge_weight_cpu <= 0):
                    raise RuntimeError("LapPE (undirected) expects strictly positive edge weights.")
                ok, mismatches = self._validate_undirected_edge_pairs(edge_index_cpu, edge_weight_cpu)
                if not ok:
                    raise RuntimeError(
                        "lap_pe_is_undirected=True requires reciprocal edges with matching weights. "
                        f"Found {mismatches} mismatched/missing pairs."
                    )
                self._lap_validated_keys.add(validation_key)

        tmp = Data(edge_index=edge_index_cpu, edge_weight=edge_weight_cpu, num_nodes=int(graph.x.size(0)))
        tmp = self._lap_pe_transform(tmp)
        pe = self._canonicalize_lap_pe(getattr(tmp, self._lap_pe_attr_name)).to(dtype=torch.float32)
        if cache_key is not None:
            if len(self._lap_pe_cache) >= 2048:
                self._lap_pe_cache.clear()
            self._lap_pe_cache[cache_key] = pe.detach().cpu()
        return pe.to(dtype=graph.x.dtype, device=graph.x.device)

    def _norm_training(self, is_training: bool) -> bool:
        return bool(self.training) and bool(is_training) and (not self.freeze_normalizers)

    @staticmethod
    def _current_velocity(inputs: ModelInputs) -> Tensor:
        if hasattr(inputs.cur_graph, "vel_state"):
            return inputs.cur_graph.vel_state
        return inputs.cur_position - inputs.prev_position

    def normalize_graph(self, graph: Data, is_training: bool = True) -> Data:
        x = graph.x
        if self._lap_pe_transform is not None:
            if x.size(1) == self._base_node_dim:
                lap_pe = self._compute_lap_pe(graph)
                x = torch.cat([x, lap_pe], dim=1)
            elif x.size(1) != self._expected_node_dim:
                raise RuntimeError(
                    f"Unexpected node feature dim {x.size(1)} for LapPE-enabled model "
                    f"(expected {self._base_node_dim} or {self._expected_node_dim})."
                )
        if self.use_normalization:
            norm_training = self._norm_training(is_training)
            x = self.node_normalizer(x, is_training=norm_training)
            edge_attr = self.edge_normalizer(graph.edge_attr, is_training=norm_training)
        else:
            edge_attr = graph.edge_attr
        return Data(
            x=x,
            edge_index=graph.edge_index,
            edge_attr=edge_attr,
            box=graph.box if hasattr(graph, "box") else None,
            batch=graph.batch if hasattr(graph, "batch") else None,
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
        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                data = self.normalize_graph(data, is_training=False)
                _pred, _z2 = self._predict_with_z2(data)
                return self.core.last_cv.detach()
        finally:
            self.train(was_training)

    @classmethod
    def _recalc_edges(cls, data: Data, dummy_sep: bool = False, pos_dim: int | None = None) -> Tensor:
        edge_vectors = get_correct_edge_vec(data, pos_dim=pos_dim)
        distances = torch.norm(edge_vectors, dim=1)
        bond_coeffs = data.edge_attr[:, -1] if not dummy_sep else data.edge_attr[:, -2]
        if dummy_sep:
            return torch.column_stack([edge_vectors, distances, bond_coeffs, data.edge_attr[:, -1]])
        return torch.column_stack([edge_vectors, distances, bond_coeffs])

    def update(self, inputs: ModelInputs, model_output: Tensor, dummy_sep: bool = False) -> Data:
        predicted = model_output if not self.use_normalization else self.output_normalizer.inverse(model_output)
        cur_velocity = self._current_velocity(inputs)
        updated_velocity = predicted if self.prediction_target == "velocity" else cur_velocity + predicted
        predicted_position = inputs.cur_position + updated_velocity
        if hasattr(inputs.cur_graph, "x") and inputs.cur_graph.x is not None and inputs.cur_graph.x.size(1) > self.pos_dim:
            static_tail = inputs.cur_graph.x[:, self.pos_dim :]
            predicted_x = torch.cat([predicted_position, static_tail], dim=1)
        else:
            predicted_x = predicted_position

        tmp = Data(
            x=predicted_x.clone().float(),
            edge_index=inputs.cur_graph.edge_index,
            edge_attr=inputs.cur_graph.edge_attr.float(),
            box=inputs.cur_graph.box if hasattr(inputs.cur_graph, "box") else None,
        )
        new_edge_attr = self._recalc_edges(tmp, dummy_sep, self.pos_dim)
        predicted_graph = Data(
            x=predicted_x.float(),
            edge_index=inputs.cur_graph.edge_index,
            edge_attr=new_edge_attr.float(),
            box=inputs.cur_graph.box if hasattr(inputs.cur_graph, "box") else None,
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
        dv_loss = F.mse_loss(model_output, target)
        self.last_dv_loss = dv_loss.detach()
        return dv_loss
