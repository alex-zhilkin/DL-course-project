from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data

from .common import build_mlp, init_token_query_params, init_transformer_style_weights
from .components import NodeEdgeFusionEncoder
from .cv_transformer_simulator import CVBottleneckGlobalBlock
from .spatial_transformer_simulator import Model as SpatialTransformerModel


class LatentSpaceDynamics(nn.Module):
    """Simple residual dynamics in latent space: z_next = z + alpha * f(z)."""

    def __init__(
        self,
        latent_dim: int,
        hidden_size: int,
        *,
        num_mlp: int,
        delta_scale: float = 0.1,
    ):
        super().__init__()
        self.delta_scale = float(delta_scale)
        self.delta_net = build_mlp(
            in_size=int(latent_dim),
            hidden_size=int(hidden_size),
            out_size=int(latent_dim),
            num_mlp=max(2, int(num_mlp)),
            lay_norm=False,
        )

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor]:
        delta = self.delta_net(z) * self.delta_scale
        return z + delta, delta


class LatentSpacePredictorCore(nn.Module):
    """Encode to CV bottleneck, propagate in latent space, decode back to dV."""

    def __init__(
        self,
        in_node_dim: int,
        in_edge_dim: int,
        hidden_size: int,
        pos_dim: int,
        *,
        K1: int,
        cv_count: int,
        cv_hidden_size: int,
        transformer_layers: int,
        transformer_heads: int,
        transformer_dropout: float,
        num_mlp: int,
        edge_aggr: str,
        use_local_skip: bool,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.cv_count = max(1, int(cv_count))
        self.cv_hidden_size = max(1, int(cv_hidden_size))
        self.latent_dim = int(self.cv_count * self.cv_hidden_size)
        self.use_local_skip = bool(use_local_skip)
        # If true, we also pass local node features directly (identity signal).
        self._requested_use_local_skip = bool(use_local_skip)

        self.frontend = NodeEdgeFusionEncoder(
            in_node_dim=in_node_dim,
            in_edge_dim=in_edge_dim,
            hidden_size=self.hidden_size,
            num_mlp=num_mlp,
            edge_aggr=edge_aggr,
        )
        self.global_block = CVBottleneckGlobalBlock(
            hidden_size=self.hidden_size,
            K1=K1,
            cv_count=self.cv_count,
            cv_hidden_size=self.cv_hidden_size,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            transformer_dropout=transformer_dropout,
        )
        self.latent_dynamics = LatentSpaceDynamics(
            latent_dim=self.latent_dim,
            hidden_size=max(8, self.hidden_size // 2),
            num_mlp=max(2, num_mlp - 1),
            delta_scale=0.1,
        )
        self.global_from_latent = nn.Linear(self.latent_dim, self.hidden_size)

        out_in_dim = self.hidden_size * 2 if self.use_local_skip else self.hidden_size
        self.out = nn.Linear(out_in_dim, pos_dim)

        self.last_cv: Tensor | None = None
        self.last_cv_next: Tensor | None = None
        self.last_cv_delta: Tensor | None = None
        self.init_weights()

    def init_weights(self) -> None:
        init_transformer_style_weights(self)
        init_token_query_params(list(self.global_block.tokens_hidden) + [self.global_block.tokens2])
        self.out.weight.data.mul_(0.1)

    def _encode_context(self, data: Data, *, return_attn: bool = False):
        local_nodes = self.frontend(data)
        if return_attn:
            h_dense, _hg, mask, attn = self.global_block(
                local_nodes,
                data.batch if hasattr(data, "batch") else None,
                return_attn=True,
                return_z2=False,
            )
        else:
            h_dense, _hg, mask = self.global_block(
                local_nodes,
                data.batch if hasattr(data, "batch") else None,
                return_attn=False,
                return_z2=False,
            )
            attn = None
        z_current = self.global_block.last_cv
        return h_dense, mask, z_current, attn

    def decode_from_latent(self, h_dense: Tensor, mask: Tensor, latent: Tensor) -> Tensor:
        if latent.dim() == 1:
            latent = latent.unsqueeze(0)
        h_from_latent = self.global_from_latent(latent).unsqueeze(1).expand(-1, h_dense.size(1), -1)
        if self.use_local_skip:
            # Keep local node identity hint together with latent global context.
            h_fused = torch.cat([h_dense, h_from_latent], dim=-1)
        else:
            h_fused = h_from_latent
        pos_dense = self.out(h_fused)
        return pos_dense[mask]

    def encode_latent(self, data: Data) -> Tensor:
        _h_dense, _mask, z_current, _attn = self._encode_context(data, return_attn=False)
        self.last_cv = z_current
        return z_current

    def autoencode(self, data: Data) -> tuple[Tensor, Tensor]:
        h_dense, mask, z_current, _attn = self._encode_context(data, return_attn=False)
        self.last_cv = z_current
        pos = self.decode_from_latent(h_dense, mask, z_current)
        return pos, z_current

    def decode_with_graph_context(self, data: Data, latent: Tensor) -> Tensor:
        h_dense, mask, _z_current, _attn = self._encode_context(data, return_attn=False)
        return self.decode_from_latent(h_dense, mask, latent)

    def propagate_latent(self, z_current: Tensor) -> tuple[Tensor, Tensor]:
        z_next, z_delta = self.latent_dynamics(z_current)
        self.last_cv = z_current
        self.last_cv_next = z_next
        self.last_cv_delta = z_delta
        return z_next, z_delta

    def forward(
        self,
        data: Data,
        *,
        return_context: bool = False,
        return_attn: bool = False,
        return_z2: bool = False,
    ):
        h_dense, mask, z_current, attn = self._encode_context(data, return_attn=return_attn)
        z_next, _z_delta = self.propagate_latent(z_current)
        dv = self.decode_from_latent(h_dense, mask, z_next)

        if return_context and return_attn:
            if return_z2:
                return dv, z_next, attn, z_next
            return dv, z_next, attn
        if return_context:
            if return_z2:
                return dv, z_next, z_next
            return dv, z_next
        if return_z2:
            return dv, z_next
        return dv


class Model(SpatialTransformerModel):
    """Spatial transformer variant with explicit latent-space propagation."""

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
        cv_hidden_size: int = 1,
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
            k2_hidden_size=cv_hidden_size,
            use_local_skip=use_local_skip,
        )
        self.core = LatentSpacePredictorCore(
            in_node_dim=self._expected_node_dim,
            in_edge_dim=data.num_edge_features,
            hidden_size=hidden_size,
            pos_dim=pos_dim,
            K1=K1,
            cv_count=CV,
            cv_hidden_size=cv_hidden_size,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            transformer_dropout=transformer_dropout,
            num_mlp=num_mlp,
            edge_aggr=edge_aggr,
            use_local_skip=use_local_skip,
        )
        self.k_cv = int(CV)
        self.cv_hidden_size = int(cv_hidden_size)
        self.latent_dim = int(self.k_cv * self.cv_hidden_size)

    def normalize_graph(self, graph: Data, is_training: bool = True) -> Data:
        return graph

    def autoencode_positions(self, data: Data, *, is_training: bool = True) -> tuple[Tensor, Tensor]:
        data = self.normalize_graph(data, is_training=is_training)
        pos, z = self.core.autoencode(data)
        self.last_cv = z.detach()
        return pos, z

    def encode_latent(self, data: Data, *, is_training: bool = False) -> Tensor:
        data = self.normalize_graph(data, is_training=is_training)
        z = self.core.encode_latent(data)
        self.last_cv = z.detach()
        return z

    def decode_positions_from_latent(self, data: Data, latent: Tensor, *, is_training: bool = False) -> Tensor:
        data = self.normalize_graph(data, is_training=is_training)
        return self.core.decode_with_graph_context(data, latent)

    def propagate_latent(self, latent: Tensor) -> Tensor:
        z_next, _z_delta = self.core.propagate_latent(latent)
        return z_next

    def predict_tplus_positions(self, data: Data, *, is_training: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        data = self.normalize_graph(data, is_training=is_training)
        z0 = self.core.encode_latent(data)
        zt = self.propagate_latent(z0)
        pred_pos = self.core.decode_with_graph_context(data, zt)
        self.last_cv = z0.detach()
        return pred_pos, z0, zt

    def extract_cv(self, data: Data, *, is_training: bool = False) -> Tensor:
        z = self.encode_latent(data, is_training=is_training)
        return z.detach()

    def update(self, inputs, model_output: Tensor, dummy_sep: bool = False) -> Data:
        predicted = model_output
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
        inputs,
        *,
        accumulate_norm_stats: bool | None = None,
    ) -> Tensor:
        cur_velocity = self._current_velocity(inputs)
        target_velocity = inputs.target_position - inputs.cur_position
        target_velocity_change = target_velocity - cur_velocity
        return torch.nn.functional.mse_loss(model_output, target_velocity_change)
