"""Static-conditioned latent trajectory propagators for prescribed loading."""

from __future__ import annotations

import torch
from torch import Tensor, nn


def progress_features(progress: Tensor) -> Tensor:
    progress = progress.reshape(-1, 1)
    return torch.cat(
        [
            progress,
            progress.square(),
            torch.sin(torch.pi * progress),
            torch.cos(torch.pi * progress) - 1.0,
        ],
        dim=-1,
    )


class StaticLatentPathMLP(nn.Module):
    """Predict the latent state at a requested time without recurrent drift."""

    def __init__(
        self,
        *,
        context_dim: int,
        latent_dim: int,
        hidden_size: int = 128,
    ):
        super().__init__()
        self.context_encoder = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
        )
        self.path = nn.Sequential(
            nn.Linear(hidden_size + 4, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, latent_dim),
        )

    def forward(self, context: Tensor, progress: Tensor) -> Tensor:
        encoded = self.context_encoder(context)
        residual = self.path(torch.cat([encoded, progress_features(progress)], dim=-1))
        # Exactly anchor every trajectory at the encoded reference state.
        return progress.reshape(-1, 1) * residual


class StaticLatentGRU(nn.Module):
    """Autoregress a latent displacement while retaining static network memory."""

    def __init__(
        self,
        *,
        context_dim: int,
        latent_dim: int,
        hidden_size: int = 128,
    ):
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.context_encoder = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, hidden_size),
            nn.Tanh(),
        )
        self.cell = nn.GRUCell(2 * self.latent_dim + 4, hidden_size)
        self.delta = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.latent_dim),
        )
        self.delta[-1].weight.data.mul_(0.01)
        nn.init.zeros_(self.delta[-1].bias)

    def rollout(self, context: Tensor, steps: int) -> Tensor:
        batch = context.size(0)
        hidden = self.context_encoder(context)
        q = context.new_zeros((batch, self.latent_dim))
        velocity = torch.zeros_like(q)
        path = [q]
        for step in range(1, int(steps) + 1):
            progress = context.new_full((batch,), step / max(1, int(steps)))
            hidden = self.cell(
                torch.cat([q, velocity, progress_features(progress)], dim=-1),
                hidden,
            )
            next_q = q + self.delta(hidden)
            velocity, q = next_q - q, next_q
            path.append(q)
        return torch.stack(path, dim=1)


__all__ = ["StaticLatentGRU", "StaticLatentPathMLP", "progress_features"]
