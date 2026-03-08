from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch

from .components import NodeEdgeFusionEncoder
from .cv_core import CVCoreConfig, CVCoreInput, SharedCVCore
from .spatial_transformer_simulator import Model as SpatialTransformerModel


class GraphTokenAdapter(nn.Module):
    def __init__(
        self,
        *,
        in_node_dim: int,
        in_edge_dim: int,
        hidden_size: int,
        num_mlp: int,
        edge_aggr: str,
        use_local_skip: bool,
    ):
        super().__init__()
        self.use_local_skip = bool(use_local_skip)
        self.frontend = NodeEdgeFusionEncoder(
            in_node_dim=in_node_dim,
            in_edge_dim=in_edge_dim,
            hidden_size=hidden_size,
            num_mlp=num_mlp,
            edge_aggr=edge_aggr,
        )

    def build_core_input(self, data: Data) -> CVCoreInput:
        local_nodes = self.frontend(data)
        batch = data.batch if hasattr(data, "batch") else None
        dense_tokens, mask = to_dense_batch(local_nodes, batch)
        local_skip = dense_tokens if self.use_local_skip else None
        return CVCoreInput(tokens=dense_tokens, mask=mask, local_skip=local_skip)


class Model(SpatialTransformerModel):
    """Graph CV model with a shared transformer bottleneck core."""

    def __init__(
        self,
        data: Data,
        hidden_size: int,
        n_layers: int,
        pos_dim: int,
        *,
        num_mlp: int,
        K1: int,
        CV: int,
        transformer_layers: int,
        transformer_heads: int,
        transformer_dropout: float,
        edge_aggr: str,
        use_local_skip: bool,
        time_lag_steps: int = 0,
        time_lag_weight: float = 0.0,
    ):
        super().__init__(
            data=data,
            hidden_size=hidden_size,
            n_layers=n_layers,
            pos_dim=pos_dim,
            num_mlp=num_mlp,
            K1=K1,
            K2=CV,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            transformer_dropout=transformer_dropout,
            edge_aggr=edge_aggr,
            k2_hidden_size=1,
            use_local_skip=use_local_skip,
        )
        self.adapter = GraphTokenAdapter(
            in_node_dim=self._expected_node_dim,
            in_edge_dim=data.num_edge_features,
            hidden_size=hidden_size,
            num_mlp=num_mlp,
            edge_aggr=edge_aggr,
            use_local_skip=use_local_skip,
        )
        self.core = SharedCVCore(
            CVCoreConfig(
                hidden_size=hidden_size,
                K1=K1,
                cv_count=CV,
                output_dim=pos_dim,
                transformer_layers=transformer_layers,
                transformer_heads=transformer_heads,
                transformer_dropout=transformer_dropout,
                decode_mode="per_token",
                use_local_skip=use_local_skip,
            )
        )
        self.k_cv = int(CV)
        self.time_lag_steps = int(time_lag_steps)
        self.time_lag_weight = float(time_lag_weight)

    def _forward_core(
        self,
        data: Data,
        *,
        return_context: bool = False,
        return_attn: bool = False,
        return_z2: bool = False,
        use_tau_head: bool = False,
    ):
        core_input = self.adapter.build_core_input(data)
        if use_tau_head:
            pred_dense = self.core.predict_tau(core_input)
            self.last_cv = self.core.last_cv.detach()
            return pred_dense[core_input.mask]

        output = self.core(core_input, return_attn=return_attn, return_z2=return_z2)
        self.last_cv = output.cv.detach()
        pred = output.prediction[core_input.mask]

        if return_context and return_attn:
            if return_z2:
                return pred, output.global_context[core_input.mask], output.attn, output.token_state
            return pred, output.global_context[core_input.mask], output.attn
        if return_context:
            if return_z2:
                return pred, output.global_context[core_input.mask], output.token_state
            return pred, output.global_context[core_input.mask]
        if return_z2:
            return pred, output.token_state
        return pred

    def predict_time_lag_acc(self, data: Data, *, is_training: bool = True) -> Tensor:
        data = self.normalize_graph(data, is_training=is_training)
        return self._forward_core(data, use_tau_head=True)

    def forward(
        self,
        data: Data,
        *,
        return_context: bool = False,
        return_attn: bool = False,
        return_z2: bool = False,
        is_training: bool = True,
    ):
        data = self.normalize_graph(data, is_training=is_training)
        return self._forward_core(
            data,
            return_context=return_context,
            return_attn=return_attn,
            return_z2=return_z2,
        )

    def extract_cv(self, data: Data, *, is_training: bool = False) -> Tensor:
        data = self.normalize_graph(data, is_training=is_training)
        _pred, _z2 = self._forward_core(data, return_z2=True)
        return self.last_cv
