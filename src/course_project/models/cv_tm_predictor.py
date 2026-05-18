from __future__ import annotations

import math

import torch
import torch.nn as nn

from .cv_core import CVCoreConfig, CVCoreInput, SharedCVCore


class SequenceHistoryAdapter(nn.Module):
    def __init__(self, feat_dim: int, history: int, hidden_size: int):
        super().__init__()
        self.in_proj = nn.Linear(history, hidden_size)
        self.desc_emb = nn.Parameter(torch.randn(feat_dim, hidden_size) * 0.01)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.in_proj.weight)
        if self.in_proj.bias is not None:
            nn.init.zeros_(self.in_proj.bias)
        nn.init.normal_(self.desc_emb, mean=0.0, std=0.01)

    def build_core_input(self, x_hist: torch.Tensor) -> CVCoreInput:
        x_desc = x_hist.transpose(1, 2)
        tokens = self.in_proj(x_desc) + self.desc_emb.unsqueeze(0)
        mask = torch.ones(tokens.size(0), tokens.size(1), dtype=torch.bool, device=tokens.device)
        return CVCoreInput(tokens=tokens, mask=mask)


class TriangularAttention(nn.Module):
    def __init__(self, hidden_size: int, heads: int, dropout: float):
        super().__init__()
        if hidden_size % heads != 0:
            raise ValueError(f"hidden_size={hidden_size} must be divisible by heads={heads}")
        self.heads = heads
        self.head_dim = hidden_size // heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v1_proj = nn.Linear(hidden_size, hidden_size)
        self.v2_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in (self.q_proj, self.k_proj, self.v1_proj, self.v2_proj, self.out_proj):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _, d = x.shape
        q = self.q_proj(x).view(b, n, n, self.heads, self.head_dim)
        k = self.k_proj(x).view(b, n, n, self.heads, self.head_dim)
        v1 = self.v1_proj(x).view(b, n, n, self.heads, self.head_dim)
        v2 = self.v2_proj(x).view(b, n, n, self.heads, self.head_dim)
        scores = torch.einsum("bilhd,bljhd->biljh", q, k) / math.sqrt(self.head_dim)
        attn = self.dropout(torch.softmax(scores, dim=2))
        values = v1.unsqueeze(3) * v2.unsqueeze(1)
        out = torch.einsum("biljh,biljhd->bijhd", attn, values).reshape(b, n, n, d)
        return self.out_proj(out)


class EdgeTransformerLayer(nn.Module):
    def __init__(self, hidden_size: int, heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = TriangularAttention(hidden_size, heads, dropout)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_size, hidden_size),
        )
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in self.ffn:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class TripleAttentionTokenLayer(nn.Module):
    def __init__(self, hidden_size: int, heads: int, dropout: float):
        super().__init__()
        self.left_proj = nn.Linear(hidden_size, hidden_size)
        self.right_proj = nn.Linear(hidden_size, hidden_size)
        self.edge_layer = EdgeTransformerLayer(hidden_size, heads, dropout)
        self.token_mlp = nn.Sequential(
            nn.Linear(3 * hidden_size, 2 * hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_size, hidden_size),
        )
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for layer in (self.left_proj, self.right_proj):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        for module in self.token_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        left = self.left_proj(tokens).unsqueeze(2)
        right = self.right_proj(tokens).unsqueeze(1)
        edge_states = self.edge_layer(left + right)
        row_context = edge_states.mean(dim=2)
        col_context = edge_states.mean(dim=1)
        update = self.token_mlp(torch.cat([tokens, row_context, col_context], dim=-1))
        return tokens + self.dropout(update)


class BackboneModel(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        history: int,
        hidden: int,
        token_sizes: tuple[int, ...],
        heads: int,
        token_layers: int,
        dropout: float,
        pre_pyramid_layers: int = 0,
        linear_cv_decoder: bool = True,
    ):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.cv_count = int(token_sizes[-1])
        self.linear_cv_decoder = bool(linear_cv_decoder)
        self.adapter = SequenceHistoryAdapter(feat_dim, history, hidden)
        self.pre_token_layers = nn.ModuleList(
            [TripleAttentionTokenLayer(hidden, heads, dropout) for _ in range(pre_pyramid_layers)]
        )
        self.core = SharedCVCore(
            CVCoreConfig(
                hidden_size=hidden,
                transformer_layers=token_layers,
                transformer_heads=heads,
                transformer_dropout=dropout,
                token_sizes=token_sizes,
            )
        )
        self.dv_head = nn.Linear(self.cv_count, self.feat_dim, bias=False)
        self.cls_head = nn.Sequential(
            nn.Linear(self.cv_count, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.dv_head.weight)
        for module in self.cls_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def encode_cv(self, x_hist: torch.Tensor) -> torch.Tensor:
        core_input = self.adapter.build_core_input(x_hist)
        tokens = core_input.tokens
        for layer in self.pre_token_layers:
            tokens = layer(tokens)
        core_input = CVCoreInput(tokens=tokens, mask=core_input.mask)
        self.core.encode(core_input)
        return self.core.last_cv

    def forward(self, x_hist: torch.Tensor):
        cv = self.encode_cv(x_hist)
        dv_pred = self.dv_head(cv)
        cls_logit = self.cls_head(cv).squeeze(-1)
        return dv_pred, cls_logit, cv


class LinearBackboneModel(nn.Module):
    """Linear CV backbone over selected descriptor-history tokens."""

    def __init__(
        self,
        feat_dim: int,
        history: int,
        hidden: int,
        token_sizes: tuple[int, ...],
        dropout: float,
        descriptor_token_dim: int | None = None,
        descriptor_indices: list[int] | torch.Tensor | None = None,
        descriptor_metadata: torch.Tensor | None = None,
    ):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.history = int(history)
        self.cv_count = int(token_sizes[-1])
        self.descriptor_token_dim = int(descriptor_token_dim) if descriptor_token_dim is not None else int(hidden)
        if descriptor_indices is None:
            descriptor_indices_t = torch.arange(self.feat_dim, dtype=torch.long)
        else:
            descriptor_indices_t = torch.as_tensor(descriptor_indices, dtype=torch.long)
        self.register_buffer("descriptor_indices", descriptor_indices_t, persistent=False)
        if descriptor_metadata is None:
            descriptor_metadata_t = torch.zeros((descriptor_indices_t.numel(), 0), dtype=torch.float32)
        else:
            descriptor_metadata_t = torch.as_tensor(descriptor_metadata, dtype=torch.float32)
        self.register_buffer("descriptor_metadata", descriptor_metadata_t, persistent=False)
        self.desc_in = nn.Linear(self.history + descriptor_metadata_t.size(1), self.descriptor_token_dim)
        self.desc_identity = nn.Parameter(torch.randn(descriptor_indices_t.numel(), self.descriptor_token_dim) * 0.01)
        self.token_mlp = nn.Sequential(
            nn.Linear(self.descriptor_token_dim, self.descriptor_token_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.descriptor_token_dim, self.descriptor_token_dim),
            nn.GELU(),
        )
        sizes = [int(v) for v in token_sizes]
        in_dim = 3 * self.descriptor_token_dim
        layers: list[nn.Module] = []
        for out_dim in sizes[:-1]:
            layers.append(nn.Linear(in_dim, out_dim))
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, sizes[-1]))
        self.cv_net = nn.Sequential(*layers)
        self.dv_head = nn.Linear(self.cv_count, self.feat_dim, bias=False)
        self.cls_head = nn.Sequential(
            nn.Linear(self.cv_count, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.desc_in.weight)
        nn.init.zeros_(self.desc_in.bias)
        nn.init.normal_(self.desc_identity, mean=0.0, std=0.01)
        for module in self.token_mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for module in self.cv_net.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.dv_head.weight)
        for module in self.cls_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def encode_cv(self, x_hist: torch.Tensor) -> torch.Tensor:
        x_selected = x_hist.index_select(dim=2, index=self.descriptor_indices.to(x_hist.device))
        descriptor_histories = x_selected.transpose(1, 2)
        if self.descriptor_metadata.size(1) > 0:
            metadata = self.descriptor_metadata.to(x_hist.device).unsqueeze(0).expand(x_hist.size(0), -1, -1)
            descriptor_histories = torch.cat([descriptor_histories, metadata], dim=-1)
        descriptor_tokens = self.desc_in(descriptor_histories)
        descriptor_tokens = descriptor_tokens + self.desc_identity.to(x_hist.device).unsqueeze(0)
        descriptor_tokens = self.token_mlp(descriptor_tokens)
        graph_summary = torch.cat(
            [
                descriptor_tokens.mean(dim=1),
                descriptor_tokens.std(dim=1, unbiased=False),
                descriptor_tokens.max(dim=1).values,
            ],
            dim=-1,
        )
        return self.cv_net(graph_summary)

    def forward(self, x_hist: torch.Tensor):
        cv = self.encode_cv(x_hist)
        dv_pred = self.dv_head(cv)
        cls_logit = self.cls_head(cv).squeeze(-1)
        return dv_pred, cls_logit, cv


class MetricHead(nn.Module):
    def __init__(self, cv_dim: int, hidden_size: int, dropout: float, inner_size: int | None = None):
        super().__init__()
        inner = int(inner_size) if inner_size is not None else max(16, hidden_size // 2)
        self.input_proj = nn.Linear(cv_dim, inner)
        self.gru = nn.GRU(
            input_size=inner,
            hidden_size=inner,
            num_layers=1,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(inner, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        for name, param in self.gru.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(self, cv_seq: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(cv_seq)
        _, h_n = self.gru(x.unsqueeze(0))
        return self.out(self.dropout(h_n[-1])).squeeze(-1)
