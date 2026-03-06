from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_batch

from .common import build_mlp, init_token_query_params, init_transformer_style_weights
from .components import NodeEdgeFusionEncoder, _TokenTransformerStack


class GraphTokenEncoder(nn.Module):
    def __init__(
        self,
        *,
        in_node_dim: int,
        in_edge_dim: int,
        hidden_size: int,
        num_mlp: int,
        edge_aggr: str,
        K1: int,
        CV: int,
        cv_hidden_size: int,
        transformer_layers: int,
        transformer_heads: int,
        transformer_dropout: float,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.K1 = int(K1)
        self.Kcv = int(CV)
        self.cv_hidden_size = int(cv_hidden_size)
        self.latent_dim = int(self.Kcv * self.cv_hidden_size)

        self.frontend = NodeEdgeFusionEncoder(
            in_node_dim=in_node_dim,
            in_edge_dim=in_edge_dim,
            hidden_size=self.hidden_size,
            num_mlp=num_mlp,
            edge_aggr=edge_aggr,
        )
        self.edge_encoder = build_mlp(
            in_size=in_edge_dim,
            hidden_size=self.hidden_size,
            out_size=self.hidden_size,
            num_mlp=num_mlp,
            lay_norm=False,
        )

        self.tokens1 = nn.Parameter(torch.randn(self.K1, self.hidden_size))
        self.tokens2 = nn.Parameter(torch.randn(self.Kcv, self.cv_hidden_size))
        self.token_bias1 = nn.Parameter(torch.randn(self.K1, self.hidden_size))
        self.token_bias2 = nn.Parameter(torch.randn(self.Kcv, self.cv_hidden_size))

        self.pool1 = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=transformer_heads,
            dropout=transformer_dropout,
            batch_first=True,
        )
        self.pool2 = nn.MultiheadAttention(
            embed_dim=self.cv_hidden_size,
            num_heads=transformer_heads,
            dropout=transformer_dropout,
            batch_first=True,
            kdim=self.hidden_size,
            vdim=self.hidden_size,
        )
        self.token_transformer1 = _TokenTransformerStack(
            d_model=self.hidden_size,
            nhead=transformer_heads,
            dim_feedforward=4 * self.hidden_size,
            dropout=transformer_dropout,
            num_layers=transformer_layers,
        )
        self.edge_to_z1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.edge_to_z2 = nn.Linear(self.hidden_size, self.cv_hidden_size)
        self.init_weights()

    def init_weights(self) -> None:
        init_transformer_style_weights(self)
        init_token_query_params([self.tokens1, self.tokens2], std=0.5)
        init_token_query_params([self.token_bias1, self.token_bias2], std=0.5)

    def _edge_graph_embedding(self, data: Data, edge_emb: Tensor, B: int) -> Tensor:
        if hasattr(data, "batch") and data.batch is not None:
            edge_batch = data.batch[data.edge_index[0]]
        else:
            edge_batch = torch.zeros(edge_emb.size(0), dtype=torch.long, device=edge_emb.device)
        graph_sum = torch.zeros(B, self.hidden_size, dtype=edge_emb.dtype, device=edge_emb.device)
        graph_cnt = torch.zeros(B, 1, dtype=edge_emb.dtype, device=edge_emb.device)
        graph_sum.index_add_(0, edge_batch, edge_emb)
        graph_cnt.index_add_(0, edge_batch, torch.ones(edge_emb.size(0), 1, dtype=edge_emb.dtype, device=edge_emb.device))
        return graph_sum / graph_cnt.clamp(min=1.0)

    def forward(self, data: Data) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        node_local = self.frontend(data)
        edge_emb = self.edge_encoder(data.edge_attr)

        if hasattr(data, "batch") and data.batch is not None:
            batch = data.batch
        else:
            batch = torch.zeros(node_local.size(0), dtype=torch.long, device=node_local.device)
        node_dense, mask = to_dense_batch(node_local, batch)
        B, _, _ = node_dense.shape

        edge_graph = self._edge_graph_embedding(data, edge_emb, B=B)

        t1 = self.tokens1.unsqueeze(0).expand(B, -1, -1)
        t2 = self.tokens2.unsqueeze(0).expand(B, -1, -1)

        z1, _ = self.pool1(query=t1, key=node_dense, value=node_dense, need_weights=False)
        z1 = z1 + self.token_bias1.unsqueeze(0)
        z1 = z1 + self.edge_to_z1(edge_graph).unsqueeze(1)
        z1, _ = self.token_transformer1(z1, return_attn=False)
        z2, _ = self.pool2(query=t2, key=z1, value=z1, need_weights=False)
        z2 = z2 + self.token_bias2.unsqueeze(0)

        z2 = z2 + self.edge_to_z2(edge_graph).unsqueeze(1)
        latent = z2.reshape(B, -1)
        return latent, z2, node_dense, mask


class GraphTokenDecoder(nn.Module):
    def __init__(
        self,
        *,
        hidden_size: int,
        K1: int,
        CV: int,
        cv_hidden_size: int,
        pos_dim: int,
        edge_dim: int,
        num_mlp: int,
        transformer_layers: int,
        transformer_heads: int,
        transformer_dropout: float,
        use_local_skip: bool,
        max_nodes: int = 4096,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.K1 = int(K1)
        self.Kcv = int(CV)
        self.cv_hidden_size = int(cv_hidden_size)
        self.latent_dim = int(self.Kcv * self.cv_hidden_size)
        self.use_local_skip = bool(use_local_skip)
        # If true, decoder also gets local node features as an identity signal.
        self.max_nodes = int(max_nodes)

        self.tokens1 = nn.Parameter(torch.randn(self.K1, self.hidden_size))
        self.up_from_z2 = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=transformer_heads,
            dropout=transformer_dropout,
            batch_first=True,
            kdim=self.cv_hidden_size,
            vdim=self.cv_hidden_size,
        )
        self.token_transformer = _TokenTransformerStack(
            d_model=self.hidden_size,
            nhead=transformer_heads,
            dim_feedforward=4 * self.hidden_size,
            dropout=transformer_dropout,
            num_layers=transformer_layers,
        )
        self.unpool_to_nodes = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=transformer_heads,
            dropout=transformer_dropout,
            batch_first=True,
        )
        self.node_id_embed = nn.Embedding(self.max_nodes, self.hidden_size)
        if self.use_local_skip:
            self.node_fuse = nn.Linear(self.hidden_size * 2, self.hidden_size)
        else:
            self.node_fuse = nn.Identity()

        self.pos_head = build_mlp(
            in_size=self.hidden_size,
            hidden_size=self.hidden_size,
            out_size=pos_dim,
            num_mlp=max(2, num_mlp),
            lay_norm=False,
        )
        self.edge_head = build_mlp(
            in_size=self.hidden_size * 2,
            hidden_size=self.hidden_size,
            out_size=edge_dim,
            num_mlp=max(2, num_mlp),
            lay_norm=False,
        )
        self.init_weights()

    def init_weights(self) -> None:
        init_transformer_style_weights(self)
        init_token_query_params([self.tokens1], std=0.5)
        nn.init.normal_(self.node_id_embed.weight, mean=0.0, std=0.2)

    def build_node_queries(self, data: Data) -> tuple[Tensor, Tensor]:
        if hasattr(data, "x") and data.x is not None:
            n = data.x.size(0)
            device = data.x.device
            dtype = data.x.dtype
        else:
            n = int(data.edge_index.max().item()) + 1
            device = data.edge_index.device
            dtype = torch.float32

        idx = torch.arange(n, device=device, dtype=dtype)
        idx = idx / max(1, n - 1)

        src = data.edge_index[0]
        deg = torch.bincount(src, minlength=n).to(device=device, dtype=dtype)
        deg = deg / deg.clamp(min=1.0).max()

        feats = [idx, deg]
        for f in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
            w = 2.0 * torch.pi * f
            feats.append(torch.sin(w * idx))
            feats.append(torch.cos(w * idx))
            feats.append(torch.sin(w * deg))
            feats.append(torch.cos(w * deg))
        base = torch.stack(feats, dim=-1)
        rep = (self.hidden_size + base.size(1) - 1) // base.size(1)
        q = base.repeat(1, rep)[:, : self.hidden_size]
        idx_long = torch.arange(n, device=device, dtype=torch.long).clamp(max=self.max_nodes - 1)
        q = q + self.node_id_embed(idx_long).to(dtype=dtype)

        if hasattr(data, "batch") and data.batch is not None:
            batch = data.batch
        else:
            batch = torch.zeros(n, dtype=torch.long, device=device)
        q_dense, mask = to_dense_batch(q, batch)
        return q_dense, mask

    def forward(
        self,
        *,
        z2: Tensor,
        node_query_dense: Tensor,
        mask: Tensor,
        edge_index: Tensor,
    ) -> tuple[Tensor, Tensor]:
        B, _, _ = z2.shape
        t1 = self.tokens1.unsqueeze(0).expand(B, -1, -1)

        z1, _ = self.up_from_z2(query=t1, key=z2, value=z2, need_weights=False)
        z1, _ = self.token_transformer(z1, return_attn=False)
        node_global, _ = self.unpool_to_nodes(query=node_query_dense, key=z1, value=z1, need_weights=False)

        if self.use_local_skip:
            # Mix local node identity hint with global decoded context.
            node_h = self.node_fuse(torch.cat([node_query_dense, node_global], dim=-1))
        else:
            node_h = self.node_fuse(node_global)

        pos_dense = self.pos_head(node_h)
        pos = pos_dense[mask]

        src = edge_index[0]
        dst = edge_index[1]
        edge_h = torch.cat([node_h[mask][src], node_h[mask][dst]], dim=-1)
        edge_attr = self.edge_head(edge_h)
        return pos, edge_attr


class Model(nn.Module):
    """Graph autoencoder: encode node+edge attributes, decode positions+edge_attr."""

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
        cv_hidden_size: int = 32,
    ):
        super().__init__()
        self.pos_dim = int(pos_dim)
        self.k_cv = int(CV)
        self.cv_hidden_size = int(cv_hidden_size)
        self.latent_dim = int(self.k_cv * self.cv_hidden_size)

        in_node_dim = int(data.num_features)
        in_edge_dim = int(data.num_edge_features)
        self.encoder = GraphTokenEncoder(
            in_node_dim=in_node_dim,
            in_edge_dim=in_edge_dim,
            hidden_size=hidden_size,
            num_mlp=num_mlp,
            edge_aggr=edge_aggr,
            K1=K1,
            CV=CV,
            cv_hidden_size=cv_hidden_size,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            transformer_dropout=transformer_dropout,
        )
        self.decoder = GraphTokenDecoder(
            hidden_size=hidden_size,
            K1=K1,
            CV=CV,
            cv_hidden_size=cv_hidden_size,
            pos_dim=pos_dim,
            edge_dim=in_edge_dim,
            num_mlp=num_mlp,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            transformer_dropout=transformer_dropout,
            use_local_skip=use_local_skip,
        )
        # Final pass: keep initialization consistent across all linear/attention layers.
        init_transformer_style_weights(self)
        self.last_cv: Tensor | None = None

    def _encode_full(self, data: Data) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return self.encoder(data)

    def encode_latent(self, data: Data, *, is_training: bool = False) -> Tensor:
        latent, _z2, _node_dense, _mask = self._encode_full(data)
        self.last_cv = latent.detach()
        return latent

    def decode_graph_from_latent(self, data: Data, latent: Tensor, *, is_training: bool = False) -> Data:
        if self.decoder.use_local_skip:
            # use_local_skip: pull local node identity hint from encoder features.
            _latent_enc, _z2_enc, node_dense, mask = self._encode_full(data)
        else:
            node_dense, mask = self.decoder.build_node_queries(data)
        B = node_dense.size(0)
        z2 = latent.reshape(B, self.k_cv, self.cv_hidden_size)
        pos, edge_attr = self.decoder(
            z2=z2,
            node_query_dense=node_dense,
            mask=mask,
            edge_index=data.edge_index,
        )
        return Data(
            x=pos,
            edge_index=data.edge_index,
            edge_attr=edge_attr,
            box=data.box if hasattr(data, "box") else None,
            t=data.t if hasattr(data, "t") else None,
            dtype=data.x.dtype,
        )

    def autoencode_graph(self, data: Data, *, is_training: bool = True) -> tuple[Data, Tensor]:
        latent, z2, node_dense_enc, mask_enc = self._encode_full(data)
        if self.decoder.use_local_skip:
            # use_local_skip: keep local node identity hint from encoder side.
            node_dense = node_dense_enc
            mask = mask_enc
        else:
            node_dense, mask = self.decoder.build_node_queries(data)
        self.last_cv = latent.detach()
        pos, edge_attr = self.decoder(
            z2=z2,
            node_query_dense=node_dense,
            mask=mask,
            edge_index=data.edge_index,
        )
        graph = Data(
            x=pos,
            edge_index=data.edge_index,
            edge_attr=edge_attr,
            box=data.box if hasattr(data, "box") else None,
            t=data.t if hasattr(data, "t") else None,
            dtype=data.x.dtype,
        )
        return graph, latent

    def autoencode_positions(self, data: Data, *, is_training: bool = True) -> tuple[Tensor, Tensor]:
        graph, latent = self.autoencode_graph(data, is_training=is_training)
        return graph.x[:, : self.pos_dim], latent

    def decode_positions_from_latent(self, data: Data, latent: Tensor, *, is_training: bool = False) -> Tensor:
        graph = self.decode_graph_from_latent(data, latent, is_training=is_training)
        return graph.x[:, : self.pos_dim]

    def extract_cv(self, data: Data, *, is_training: bool = False) -> Tensor:
        return self.encode_latent(data, is_training=is_training).detach()

    def forward(self, data: Data, is_training: bool = True) -> Data:
        graph, _latent = self.autoencode_graph(data, is_training=is_training)
        return graph
