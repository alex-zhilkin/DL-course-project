"""Shared latent autoencoder and propagator training helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from graph_utils import (
    calc_p_ratio_box,
    calc_p_ratio_rollout_all,
    calc_p_ratio_rollout_outer,
    calc_p_ratio_rollout_sides,
)

from ..graph import clone_graph
from .simulation import (
    ae_target_tensor,
    batch_delta_graphs,
    edge_features,
    filtered_frame_ids,
    frame_node_feature,
    frame_for_filtered_step,
    iter_batches,
    pearson_r,
    r2_score,
)
from .models import make_latent_propagator


@dataclass(frozen=True)
class TrainingConfig:
    """Shared optimization and early-stopping settings."""

    max_epochs: int = 250
    patience: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5
    min_delta: float = 1e-5
    log_every: int = 5


@dataclass
class TrainingResult:
    """Best restored model plus its training history."""

    model: torch.nn.Module
    history: pd.DataFrame
    best_val_loss: float
    best_epoch: int


def _train_with_early_stopping(
    model: torch.nn.Module,
    *,
    train_epoch: Callable,
    val_epoch: Callable,
    config: TrainingConfig,
    label: str,
    metric_key: str | None = None,
    verbose: bool = True,
) -> TrainingResult:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    stale = 0
    rows = []

    for epoch in range(1, int(config.max_epochs) + 1):
        train_info = train_epoch(optimizer)
        with torch.no_grad():
            val_info = val_epoch()

        train_loss = float(train_info[metric_key] if metric_key else train_info)
        val_loss = float(val_info[metric_key] if metric_key else val_info)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        if isinstance(train_info, dict):
            row.update({f"train_{key}": float(value) for key, value in train_info.items()})
        if isinstance(val_info, dict):
            row.update({f"val_{key}": float(value) for key, value in val_info.items()})
        rows.append(row)

        improved = np.isfinite(val_loss) and val_loss < best_val - float(config.min_delta)
        if improved:
            best_val = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1

        if verbose and (epoch == 1 or epoch % int(config.log_every) == 0 or improved):
            print(
                f"{label} {epoch:04d} train={train_loss:.6g} "
                f"val={val_loss:.6g} stale={stale}"
            )
        if stale >= int(config.patience):
            break

    if best_state is None:
        raise RuntimeError(f"{label} training produced no finite validation loss.")
    model.load_state_dict(best_state)
    model.eval()
    return TrainingResult(
        model=model,
        history=pd.DataFrame(rows),
        best_val_loss=float(best_val),
        best_epoch=int(best_epoch),
    )


@dataclass
class LatentNormalizer:
    """Normalization state for one-step latent propagation."""

    z_mean: torch.Tensor
    z_std: torch.Tensor
    dz_mean: torch.Tensor
    dz_std: torch.Tensor
    z_next_mean: torch.Tensor | None = None
    z_next_std: torch.Tensor | None = None
    context_mean: torch.Tensor | None = None
    context_std: torch.Tensor | None = None

    def to(self, device) -> "LatentNormalizer":
        return LatentNormalizer(
            z_mean=self.z_mean.to(device),
            z_std=self.z_std.to(device),
            dz_mean=self.dz_mean.to(device),
            dz_std=self.dz_std.to(device),
            z_next_mean=None if self.z_next_mean is None else self.z_next_mean.to(device),
            z_next_std=None if self.z_next_std is None else self.z_next_std.to(device),
            context_mean=None if self.context_mean is None else self.context_mean.to(device),
            context_std=None if self.context_std is None else self.context_std.to(device),
        )

    def as_dict(self) -> dict[str, torch.Tensor]:
        out = {
            "z_mean": self.z_mean,
            "z_std": self.z_std,
            "dz_mean": self.dz_mean,
            "dz_std": self.dz_std,
        }
        if self.z_next_mean is not None:
            out["z_next_mean"] = self.z_next_mean
        if self.z_next_std is not None:
            out["z_next_std"] = self.z_next_std
        if self.context_mean is not None:
            out["context_mean"] = self.context_mean
        if self.context_std is not None:
            out["context_std"] = self.context_std
        return out

    @classmethod
    def from_dict(cls, values: dict[str, torch.Tensor]) -> "LatentNormalizer":
        return cls(
            z_mean=values["z_mean"],
            z_std=values["z_std"],
            dz_mean=values["dz_mean"],
            dz_std=values["dz_std"],
            z_next_mean=values.get("z_next_mean"),
            z_next_std=values.get("z_next_std"),
            context_mean=values.get("context_mean"),
            context_std=values.get("context_std"),
        )

    def normalize_z(self, z: torch.Tensor) -> torch.Tensor:
        return (z - self.z_mean) / self.z_std

    def normalize_dz(self, dz: torch.Tensor) -> torch.Tensor:
        return (dz - self.dz_mean) / self.dz_std

    def unnormalize_dz(self, dz_norm: torch.Tensor) -> torch.Tensor:
        return dz_norm * self.dz_std + self.dz_mean

    def normalize_z_next(self, z: torch.Tensor) -> torch.Tensor:
        mean = self.z_next_mean if self.z_next_mean is not None else self.z_mean
        std = self.z_next_std if self.z_next_std is not None else self.z_std
        return (z - mean) / std

    def unnormalize_z_next(self, z_norm: torch.Tensor) -> torch.Tensor:
        mean = self.z_next_mean if self.z_next_mean is not None else self.z_mean
        std = self.z_next_std if self.z_next_std is not None else self.z_std
        return z_norm * std + mean

    def normalize_context(self, context: torch.Tensor | None) -> torch.Tensor | None:
        if context is None:
            return None
        if self.context_mean is None or self.context_std is None:
            return context
        return (context - self.context_mean) / self.context_std


def encode_reference_context(
    ae_model,
    sim,
    *,
    pos_dim: int,
    normalizers: dict[str, torch.Tensor],
    device,
    include_temperature: bool = False,
) -> torch.Tensor:
    """Pool the learned reference-node representation into static network context."""
    ref_graph = sim[0]
    ref_pos = ref_graph.x[:, :pos_dim].to(device).float()
    ref_edge_attr_norm = (
        edge_features(ref_graph, ref_graph, pos_dim=pos_dim, device=device)
        - normalizers["edge_mean"].to(device)
    ) / normalizers["edge_std"].to(device)
    h0 = ae_model.encode_reference_graph(
        ref_pos,
        ref_edge_attr_norm,
        ref_graph.edge_index.to(device).long(),
    )
    context = h0.mean(dim=0)
    if include_temperature:
        temperature = float(getattr(ref_graph, "temperature", 0.0))
        temperature_feature = torch.tensor(
            [np.log1p(max(temperature, 0.0))],
            dtype=context.dtype,
            device=context.device,
        )
        context = torch.cat([context, temperature_feature], dim=0)
    return context


def encode_frame_latent(
    ae_model,
    sim,
    t: int,
    *,
    pos_dim: int,
    node_feature_mode: str,
    normalizers: dict[str, torch.Tensor],
    device,
) -> torch.Tensor:
    ref_graph = sim[0]
    cur_graph = sim[int(t)]
    ref_pos = ref_graph.x[:, :pos_dim].to(device).float()
    node_feature = frame_node_feature(
        sim,
        t,
        pos_dim=pos_dim,
        mode=node_feature_mode,
        device=device,
    )
    node_feature_norm = (node_feature - normalizers["node_feature_mean"].to(device)) / normalizers[
        "node_feature_std"
    ].to(device)
    edge_attr_norm = (
        edge_features(ref_graph, cur_graph, pos_dim=pos_dim, device=device)
        - normalizers["edge_mean"].to(device)
    ) / normalizers["edge_std"].to(device)
    ref_edge_attr_norm = (
        edge_features(ref_graph, ref_graph, pos_dim=pos_dim, device=device)
        - normalizers["edge_mean"].to(device)
    ) / normalizers["edge_std"].to(device)
    edge_index = ref_graph.edge_index.to(device).long()
    batch = torch.zeros(ref_pos.size(0), dtype=torch.long, device=device)
    z, _ = ae_model.encode(
        node_feature_norm,
        ref_pos,
        edge_attr_norm,
        ref_edge_attr_norm,
        edge_index,
        batch,
    )
    return z.squeeze(0)


def encode_transition_batch(
    ae_model,
    sims,
    rows,
    *,
    pos_dim: int,
    node_feature_mode: str,
    normalizers: dict[str, torch.Tensor],
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    z0 = []
    z1 = []
    with torch.no_grad():
        for row in rows:
            if len(row) == 3:
                sim_idx, t0, t1 = row
            else:
                sim_idx, t0 = row
                t1 = int(t0) + 1
            sim = sims[int(sim_idx)]
            z0.append(
                encode_frame_latent(
                    ae_model,
                    sim,
                    int(t0),
                    pos_dim=pos_dim,
                    node_feature_mode=node_feature_mode,
                    normalizers=normalizers,
                    device=device,
                )
            )
            z1.append(
                encode_frame_latent(
                    ae_model,
                    sim,
                    int(t1),
                    pos_dim=pos_dim,
                    node_feature_mode=node_feature_mode,
                    normalizers=normalizers,
                    device=device,
                )
            )
    return torch.stack(z0, dim=0), torch.stack(z1, dim=0)


def fit_latent_step_stats(
    ae_model,
    sims,
    rows,
    *,
    batch_graphs: int,
    pos_dim: int,
    node_feature_mode: str,
    normalizers: dict[str, torch.Tensor],
    device,
    use_static_context: bool = False,
    context_include_temperature: bool = False,
) -> LatentNormalizer:
    z_chunks = []
    z_next_chunks = []
    dz_chunks = []
    context_chunks = []
    for rows_batch in iter_batches(rows, batch_graphs, shuffle=False):
        z0, z1 = encode_transition_batch(
            ae_model,
            sims,
            rows_batch,
            pos_dim=pos_dim,
            node_feature_mode=node_feature_mode,
            normalizers=normalizers,
            device=device,
        )
        z_chunks.append(z0.detach())
        z_next_chunks.append(z1.detach())
        dz_chunks.append((z1 - z0).detach())
        if use_static_context:
            context_chunks.extend(
                encode_reference_context(
                    ae_model,
                    sims[int(row[0])],
                    pos_dim=pos_dim,
                    normalizers=normalizers,
                    device=device,
                    include_temperature=context_include_temperature,
                ).detach()
                for row in rows_batch
            )
    z_all = torch.cat(z_chunks, dim=0)
    z_next_all = torch.cat(z_next_chunks, dim=0)
    dz_all = torch.cat(dz_chunks, dim=0)
    context_all = torch.stack(context_chunks, dim=0) if context_chunks else None
    return LatentNormalizer(
        z_mean=z_all.mean(dim=0, keepdim=True),
        z_std=z_all.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6),
        dz_mean=dz_all.mean(dim=0, keepdim=True),
        dz_std=dz_all.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6),
        z_next_mean=z_next_all.mean(dim=0, keepdim=True),
        z_next_std=z_next_all.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6),
        context_mean=None if context_all is None else context_all.mean(dim=0, keepdim=True),
        context_std=(
            None
            if context_all is None
            else context_all.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
        ),
    )


def make_propagator(latent_dim: int, hidden_size: int, *, loss_mode: str, model_type: str | None = None):
    loss_mode = str(loss_mode).lower()
    if model_type is None:
        if loss_mode in {"velocity_delta", "second_order"}:
            model_type = "velocity_mlp"
        elif loss_mode in {"next_z", "jepa", "next_embedding"}:
            model_type = "direct_mlp"
        else:
            model_type = "residual_mlp"
    return make_latent_propagator(latent_dim, hidden_size, model_type=model_type)


def latent_step(
    model,
    z: torch.Tensor,
    stats: LatentNormalizer,
    *,
    loss_mode: str,
    context: torch.Tensor | None = None,
) -> torch.Tensor:
    loss_mode = str(loss_mode).lower()
    z_norm = stats.normalize_z(z.unsqueeze(0))
    context_norm = stats.normalize_context(
        None if context is None else context.unsqueeze(0)
    )
    pred = model(z_norm, context_norm)
    if loss_mode in {"delta", "dz", "residual_delta", "hybrid_delta_next"}:
        pred_dz_norm = pred - z_norm
        return z + stats.unnormalize_dz(pred_dz_norm).squeeze(0)
    if loss_mode in {"next_z", "jepa", "next_embedding"}:
        return stats.unnormalize_z_next(pred).squeeze(0)
    raise ValueError(f"Unknown propagator loss_mode: {loss_mode}")


def epoch_autoencoder(
    model,
    sims,
    frame_rows,
    *,
    batch_graphs: int,
    pos_dim: int,
    node_feature_mode: str,
    ae_target_mode: str,
    normalizers: dict[str, torch.Tensor],
    device,
    coordinate_weights=None,
    optimizer=None,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    losses = []
    for rows in iter_batches(frame_rows, batch_graphs, shuffle=is_train):
        batch_data = batch_delta_graphs(
            sims,
            rows,
            pos_dim=pos_dim,
            device=device,
            node_feature_mode=node_feature_mode,
        )
        target_norm = (ae_target_tensor(batch_data, ae_target_mode) - normalizers["target_mean"].to(device)) / normalizers[
            "target_std"
        ].to(device)
        node_feature_norm = (
            batch_data["node_feature"] - normalizers["node_feature_mean"].to(device)
        ) / normalizers["node_feature_std"].to(device)
        edge_attr_norm = (batch_data["edge_attr"] - normalizers["edge_mean"].to(device)) / normalizers[
            "edge_std"
        ].to(device)
        ref_edge_attr_norm = (
            batch_data["ref_edge_attr"] - normalizers["edge_mean"].to(device)
        ) / normalizers["edge_std"].to(device)
        recon_norm, _ = model(
            node_feature_norm,
            batch_data["ref_pos"],
            edge_attr_norm,
            ref_edge_attr_norm,
            batch_data["edge_index"],
            batch_data["batch"],
        )
        squared_error = (recon_norm - target_norm).square()
        if coordinate_weights is not None:
            weights = torch.as_tensor(
                coordinate_weights,
                dtype=squared_error.dtype,
                device=squared_error.device,
            ).reshape(1, -1)
            if weights.size(-1) != squared_error.size(-1):
                raise ValueError(
                    "coordinate_weights must have one value per reconstructed coordinate."
                )
            squared_error = squared_error * weights / weights.mean().clamp_min(1e-8)
        loss = squared_error.mean()
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else float("nan")


def train_autoencoder(
    model,
    train_sims,
    val_sims,
    train_rows,
    val_rows,
    *,
    batch_graphs: int,
    pos_dim: int,
    node_feature_mode: str,
    ae_target_mode: str,
    normalizers: dict[str, torch.Tensor],
    device,
    config: TrainingConfig,
    coordinate_weights=None,
    verbose: bool = True,
) -> TrainingResult:
    """Train and restore the best latent autoencoder."""

    common = {
        "batch_graphs": batch_graphs,
        "pos_dim": pos_dim,
        "node_feature_mode": node_feature_mode,
        "ae_target_mode": ae_target_mode,
        "normalizers": normalizers,
        "device": device,
        "coordinate_weights": coordinate_weights,
    }
    return _train_with_early_stopping(
        model,
        train_epoch=lambda optimizer: epoch_autoencoder(
            model, train_sims, train_rows, optimizer=optimizer, **common
        ),
        val_epoch=lambda: epoch_autoencoder(model, val_sims, val_rows, **common),
        config=config,
        label="ae",
        verbose=verbose,
    )


def epoch_propagator(
    model,
    ae_model,
    sims,
    transition_rows,
    stats: LatentNormalizer,
    *,
    batch_graphs: int,
    pos_dim: int,
    node_feature_mode: str,
    normalizers: dict[str, torch.Tensor],
    device,
    loss_mode: str,
    use_static_context: bool = False,
    context_include_temperature: bool = False,
    optimizer=None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    losses = []
    raw_losses = []
    loss_mode = str(loss_mode).lower()
    for rows in iter_batches(transition_rows, batch_graphs, shuffle=is_train):
        z0, z1 = encode_transition_batch(
            ae_model,
            sims,
            rows,
            pos_dim=pos_dim,
            node_feature_mode=node_feature_mode,
            normalizers=normalizers,
            device=device,
        )
        z0_norm = stats.normalize_z(z0)
        context = None
        if use_static_context:
            context = torch.stack(
                [
                    encode_reference_context(
                        ae_model,
                        sims[int(row[0])],
                        pos_dim=pos_dim,
                        normalizers=normalizers,
                        device=device,
                        include_temperature=context_include_temperature,
                    )
                    for row in rows
                ],
                dim=0,
            )
            context = stats.normalize_context(context)
        pred = model(z0_norm, context)
        if loss_mode in {"delta", "dz", "residual_delta", "hybrid_delta_next"}:
            target_norm = stats.normalize_dz(z1 - z0)
            pred_norm = pred - z0_norm
            pred_raw = z0 + stats.unnormalize_dz(pred_norm)
            if loss_mode == "hybrid_delta_next":
                next_weight = float(getattr(model, "next_loss_weight", 0.1))
                next_pred_norm = stats.normalize_z_next(pred_raw)
                next_target_norm = stats.normalize_z_next(z1)
                loss = F.mse_loss(pred_norm, target_norm) + next_weight * F.mse_loss(
                    next_pred_norm, next_target_norm
                )
            else:
                loss = F.mse_loss(pred_norm, target_norm)
        elif loss_mode in {"next_z", "jepa", "next_embedding"}:
            target_norm = stats.normalize_z_next(z1)
            pred_norm = pred
            pred_raw = stats.unnormalize_z_next(pred_norm)
            loss = F.mse_loss(pred_norm, target_norm)
        else:
            raise ValueError(f"Unknown propagator loss_mode: {loss_mode}")
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        losses.append(float(loss.item()))
        raw_losses.append(float(F.mse_loss(pred_raw, z1).item()))
    return {
        "loss_norm": float(np.mean(losses)) if losses else float("nan"),
        "loss_raw": float(np.mean(raw_losses)) if raw_losses else float("nan"),
    }


def make_velocity_transition_index(
    sims,
    *,
    frame_skip: int = 1,
    max_frames_per_sim: int | None = None,
) -> list[tuple[int, int, int, int]]:
    rows = []
    for sim_idx, sim in enumerate(sims):
        frame_ids = filtered_frame_ids(sim, frame_skip=frame_skip, include_last=True)
        triples = list(zip(frame_ids[:-2], frame_ids[1:-1], frame_ids[2:]))
        if max_frames_per_sim is not None:
            triples = triples[: int(max_frames_per_sim)]
        rows.extend((int(sim_idx), int(t_prev), int(t0), int(t1)) for t_prev, t0, t1 in triples)
    return rows


def epoch_velocity_propagator(
    model,
    ae_model,
    sims,
    rows,
    stats: LatentNormalizer,
    *,
    batch_graphs: int,
    pos_dim: int,
    node_feature_mode: str,
    normalizers: dict[str, torch.Tensor],
    device,
    use_static_context: bool = False,
    context_include_temperature: bool = False,
    optimizer=None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    losses = []
    raw_losses = []
    for batch_rows in iter_batches(rows, batch_graphs, shuffle=is_train):
        z_prev = []
        z0 = []
        z1 = []
        with torch.no_grad():
            for sim_idx, t_prev, t0, t1 in batch_rows:
                sim = sims[int(sim_idx)]
                z_prev.append(
                    encode_frame_latent(
                        ae_model,
                        sim,
                        int(t_prev),
                        pos_dim=pos_dim,
                        node_feature_mode=node_feature_mode,
                        normalizers=normalizers,
                        device=device,
                    )
                )
                z0.append(
                    encode_frame_latent(
                        ae_model,
                        sim,
                        int(t0),
                        pos_dim=pos_dim,
                        node_feature_mode=node_feature_mode,
                        normalizers=normalizers,
                        device=device,
                    )
                )
                z1.append(
                    encode_frame_latent(
                        ae_model,
                        sim,
                        int(t1),
                        pos_dim=pos_dim,
                        node_feature_mode=node_feature_mode,
                        normalizers=normalizers,
                        device=device,
                    )
                )
        z_prev = torch.stack(z_prev, dim=0)
        z0 = torch.stack(z0, dim=0)
        z1 = torch.stack(z1, dim=0)
        prev_dz = z0 - z_prev
        target_dz = z1 - z0
        inp = torch.cat([stats.normalize_z(z0), stats.normalize_dz(prev_dz)], dim=-1)
        context = None
        if use_static_context:
            context = torch.stack(
                [
                    encode_reference_context(
                        ae_model,
                        sims[int(row[0])],
                        pos_dim=pos_dim,
                        normalizers=normalizers,
                        device=device,
                        include_temperature=context_include_temperature,
                    )
                    for row in batch_rows
                ],
                dim=0,
            )
            context = stats.normalize_context(context)
        pred_dz_norm = model(inp, context)
        target_dz_norm = stats.normalize_dz(target_dz)
        pred_z1 = z0 + stats.unnormalize_dz(pred_dz_norm)
        loss = F.mse_loss(pred_dz_norm, target_dz_norm)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        losses.append(float(loss.item()))
        raw_losses.append(float(F.mse_loss(pred_z1, z1).item()))
    return {
        "loss_norm": float(np.mean(losses)) if losses else float("nan"),
        "loss_raw": float(np.mean(raw_losses)) if raw_losses else float("nan"),
    }


def latent_step_velocity(
    model,
    z: torch.Tensor,
    prev_dz: torch.Tensor,
    stats: LatentNormalizer,
    context: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    inp = torch.cat([stats.normalize_z(z.unsqueeze(0)), stats.normalize_dz(prev_dz.unsqueeze(0))], dim=-1)
    context_norm = stats.normalize_context(
        None if context is None else context.unsqueeze(0)
    )
    pred_dz = stats.unnormalize_dz(model(inp, context_norm)).squeeze(0)
    return z + pred_dz, pred_dz


def make_multistep_transition_index(
    sims,
    *,
    horizons,
    frame_skip: int = 1,
    max_starts_per_sim: int | None = None,
) -> list[tuple[int, int, list[int]]]:
    """Build rows of ``(sim_idx, start_frame, target_frames)`` for latent unroll training."""

    horizons = sorted({int(h) for h in horizons if int(h) > 0})
    rows = []
    for sim_idx, sim in enumerate(sims):
        frame_ids = filtered_frame_ids(sim, frame_skip=frame_skip, include_last=True)
        if len(frame_ids) <= max(horizons, default=0):
            continue
        starts = list(range(0, len(frame_ids) - max(horizons)))
        if max_starts_per_sim is not None:
            starts = starts[: int(max_starts_per_sim)]
        for start_order in starts:
            target_orders = [start_order + horizon for horizon in horizons]
            rows.append(
                (
                    int(sim_idx),
                    int(frame_ids[start_order]),
                    [int(frame_ids[target_order]) for target_order in target_orders],
                )
            )
    return rows


def epoch_multistep_propagator(
    model,
    ae_model,
    sims,
    rows,
    stats: LatentNormalizer,
    *,
    batch_graphs: int,
    pos_dim: int,
    node_feature_mode: str,
    normalizers: dict[str, torch.Tensor],
    device,
    loss_mode: str,
    horizons,
    use_static_context: bool = False,
    context_include_temperature: bool = False,
    optimizer=None,
) -> dict[str, float]:
    """Train a propagator through autoregressive latent unrolls.

    The loss stays fully in representation space: for each start frame, the
    model is repeatedly applied and the predicted latent is matched at selected
    horizons. This directly penalizes rollout drift without using decoder loss.
    """

    is_train = optimizer is not None
    model.train(is_train)
    losses = []
    raw_losses = []
    horizons = sorted({int(h) for h in horizons if int(h) > 0})
    horizon_to_idx = {horizon: idx for idx, horizon in enumerate(horizons)}
    max_horizon = max(horizons, default=0)
    loss_mode = str(loss_mode).lower()

    for batch_rows in iter_batches(rows, batch_graphs, shuffle=is_train):
        row_losses = []
        row_raw_losses = []
        for sim_idx, start_frame, target_frames in batch_rows:
            sim = sims[int(sim_idx)]
            z = encode_frame_latent(
                ae_model,
                sim,
                int(start_frame),
                pos_dim=pos_dim,
                node_feature_mode=node_feature_mode,
                normalizers=normalizers,
                device=device,
            )
            target_z = [
                encode_frame_latent(
                    ae_model,
                    sim,
                    int(target_frame),
                    pos_dim=pos_dim,
                    node_feature_mode=node_feature_mode,
                    normalizers=normalizers,
                    device=device,
                )
                for target_frame in target_frames
            ]
            context = (
                encode_reference_context(
                    ae_model,
                    sim,
                    pos_dim=pos_dim,
                    normalizers=normalizers,
                    device=device,
                    include_temperature=context_include_temperature,
                )
                if use_static_context
                else None
            )
            for step_idx in range(1, max_horizon + 1):
                z = latent_step(
                    model,
                    z,
                    stats,
                    loss_mode=loss_mode,
                    context=context,
                )
                if step_idx not in horizon_to_idx:
                    continue
                true_z = target_z[horizon_to_idx[step_idx]]
                if loss_mode in {"next_z", "jepa", "next_embedding"}:
                    pred_norm = stats.normalize_z_next(z.unsqueeze(0))
                    target_norm = stats.normalize_z_next(true_z.unsqueeze(0))
                else:
                    # For multi-step residual training, comparing normalized z is
                    # more stable than comparing cumulative dz at long horizons.
                    pred_norm = stats.normalize_z(z.unsqueeze(0))
                    target_norm = stats.normalize_z(true_z.unsqueeze(0))
                row_losses.append(F.mse_loss(pred_norm, target_norm))
                row_raw_losses.append(F.mse_loss(z, true_z))
        if not row_losses:
            continue
        loss = torch.stack(row_losses).mean()
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
        raw_losses.append(float(torch.stack(row_raw_losses).mean().detach().cpu()))

    return {
        "loss_norm": float(np.mean(losses)) if losses else float("nan"),
        "loss_raw": float(np.mean(raw_losses)) if raw_losses else float("nan"),
    }


def train_propagator(
    model,
    autoencoder,
    train_sims,
    val_sims,
    train_rows,
    val_rows,
    stats: LatentNormalizer,
    *,
    batch_graphs: int,
    pos_dim: int,
    node_feature_mode: str,
    normalizers: dict[str, torch.Tensor],
    device,
    loss_mode: str,
    config: TrainingConfig,
    objective: str = "one_step",
    horizons=None,
    use_static_context: bool = False,
    context_include_temperature: bool = False,
    verbose: bool = True,
) -> TrainingResult:
    """Train any latent propagator through the shared training lifecycle."""

    autoencoder.eval()
    for parameter in autoencoder.parameters():
        parameter.requires_grad_(False)

    common = {
        "batch_graphs": batch_graphs,
        "pos_dim": pos_dim,
        "node_feature_mode": node_feature_mode,
        "normalizers": normalizers,
        "device": device,
        "use_static_context": use_static_context,
        "context_include_temperature": context_include_temperature,
    }
    objective = str(objective).lower()

    if objective in {"one_step", "next_step"}:
        epoch_fn = epoch_propagator
        extra = {"loss_mode": loss_mode}
    elif objective in {"multistep", "multi_step"}:
        if not horizons:
            raise ValueError("horizons are required for multistep propagator training.")
        epoch_fn = epoch_multistep_propagator
        extra = {"loss_mode": loss_mode, "horizons": horizons}
    elif objective in {"velocity", "second_order"}:
        epoch_fn = epoch_velocity_propagator
        extra = {}
    else:
        raise ValueError(f"Unknown propagator objective: {objective}")

    def run_epoch(sims, rows, optimizer=None):
        return epoch_fn(
            model,
            autoencoder,
            sims,
            rows,
            stats,
            optimizer=optimizer,
            **common,
            **extra,
        )

    return _train_with_early_stopping(
        model,
        train_epoch=lambda optimizer: run_epoch(train_sims, train_rows, optimizer),
        val_epoch=lambda: run_epoch(val_sims, val_rows),
        config=config,
        label="propagator",
        metric_key="loss_norm",
        verbose=verbose,
    )


def rollout_position_metrics(df: pd.DataFrame, *, dataset, split_name, rollout_steps) -> dict[str, float]:
    if len(df) == 0:
        return {"dataset": dataset, "split": split_name, "rollout_steps": int(rollout_steps), "used": 0}
    final_pos_mse = float(df["final_pos_mse"].mean())
    initial_to_target_mse = float(df["initial_to_target_mse"].mean())
    movement_fraction_mse = (
        float(final_pos_mse / initial_to_target_mse) if initial_to_target_mse > 0 else float("nan")
    )
    out = {
        "dataset": dataset,
        "split": split_name,
        "rollout_steps": int(rollout_steps),
        "used": len(df),
        "p_ratio_r2": r2_score(df["true_p_ratio"], df["pred_p_ratio"]),
        "p_ratio_pearson": pearson_r(df["true_p_ratio"], df["pred_p_ratio"]),
        "p_ratio_mse": float(np.mean((df["true_p_ratio"] - df["pred_p_ratio"]) ** 2)),
        "final_pos_mse": final_pos_mse,
        "initial_to_target_mse": initial_to_target_mse,
        "pred_to_initial_mse": float(df["pred_to_initial_mse"].mean()),
        "movement_fraction_mse": movement_fraction_mse,
    }
    out["rollout_position_r2"] = (
        float(np.clip(1.0 - movement_fraction_mse, 0.0, 1.0))
        if np.isfinite(movement_fraction_mse)
        else float("nan")
    )
    return out


def calc_rollout_p_ratio_by_method(rollout: list, idx: int = -1, *, method: str = "rollout_sides") -> float:
    method = str(method).lower()
    if method in {"rollout_sides", "sides", "side"}:
        return float(calc_p_ratio_rollout_sides(rollout, idx))
    if method in {"box", "global_box", "normal"}:
        return float(calc_p_ratio_box(rollout, idx))
    if method in {"rollout_all", "all"}:
        return float(calc_p_ratio_rollout_all(rollout, idx))
    if method in {"rollout_outer", "outer"}:
        return float(calc_p_ratio_rollout_outer(rollout, idx))
    raise ValueError(f"Unknown p-ratio method: {method}")


def decode_latent_to_graph(
    ae_model,
    sim,
    z: torch.Tensor,
    target_index: int,
    *,
    pos_dim: int,
    ae_target_mode: str,
    normalizers: dict[str, torch.Tensor],
    device,
):
    ref = clone_graph(sim[0]).to(device)
    ref_pos = ref.x[:, :pos_dim].float()
    ref_edge_attr_norm = (
        edge_features(sim[0], sim[0], pos_dim=pos_dim, device=device) - normalizers["edge_mean"].to(device)
    ) / normalizers["edge_std"].to(device)
    batch = torch.zeros(ref_pos.size(0), dtype=torch.long, device=device)
    h0 = ae_model.encode_reference_graph(ref_pos, ref_edge_attr_norm, ref.edge_index.to(device).long())
    target_norm = ae_model.decode(z.unsqueeze(0), h0, batch)
    target = target_norm * normalizers["target_std"].to(device) + normalizers["target_mean"].to(device)
    pred = clone_graph(sim[target_index]).to(device)
    pred.x = pred.x.clone().float()
    if ae_target_mode in ("position", "positions"):
        pred.x[:, :pos_dim] = target
    else:
        pred.x[:, :pos_dim] = ref_pos + target
    return pred.cpu()


def latent_rollout_eval(
    ae_model,
    dyn_model,
    sims,
    stats: LatentNormalizer,
    *,
    dataset,
    split_name,
    rollout_steps: int,
    pos_dim: int,
    ae_target_mode: str,
    node_feature_mode: str,
    normalizers: dict[str, torch.Tensor],
    device,
    loss_mode: str,
    p_ratio_method: str = "rollout_sides",
):
    rows = []
    ae_model.eval()
    dyn_model.eval()
    with torch.no_grad():
        for sim_idx, sim in enumerate(sims):
            filtered_steps = min(int(rollout_steps), max(len(filtered_frame_ids(sim, include_last=True)) - 1, 0))
            target_index = frame_for_filtered_step(sim, filtered_steps)
            if filtered_steps <= 0 or target_index <= 0:
                continue
            z = encode_frame_latent(
                ae_model,
                sim,
                0,
                pos_dim=pos_dim,
                node_feature_mode=node_feature_mode,
                normalizers=normalizers,
                device=device,
            )
            for _ in range(filtered_steps):
                z = latent_step(dyn_model, z, stats, loss_mode=loss_mode)
            pred_graph = decode_latent_to_graph(
                ae_model,
                sim,
                z,
                target_index,
                pos_dim=pos_dim,
                ae_target_mode=ae_target_mode,
                normalizers=normalizers,
                device=device,
            )
            pred_pr = calc_rollout_p_ratio_by_method(
                [clone_graph(sim[0]).cpu(), pred_graph], -1, method=p_ratio_method
            )
            true_pr = calc_rollout_p_ratio_by_method(sim, target_index, method=p_ratio_method)
            initial_pos = sim[0].x[:, :pos_dim].cpu().float()
            target_pos = sim[target_index].x[:, :pos_dim].cpu().float()
            pred_pos = pred_graph.x[:, :pos_dim].cpu().float()
            initial_to_target_mse = float(F.mse_loss(initial_pos, target_pos).item())
            pred_to_initial_mse = float(F.mse_loss(pred_pos, initial_pos).item())
            final_pos_mse = float(F.mse_loss(pred_pos, target_pos).item())
            movement_fraction_mse = final_pos_mse / initial_to_target_mse if initial_to_target_mse > 0 else float("nan")
            rows.append(
                {
                    "dataset": dataset,
                    "split": split_name,
                    "sim_idx": sim_idx,
                    "target_index": target_index,
                    "rollout_steps": int(rollout_steps),
                    "filtered_steps": int(filtered_steps),
                    "pred_p_ratio": pred_pr,
                    "true_p_ratio": true_pr,
                    "p_ratio_method": str(p_ratio_method),
                    "final_pos_mse": final_pos_mse,
                    "initial_to_target_mse": initial_to_target_mse,
                    "pred_to_initial_mse": pred_to_initial_mse,
                    "movement_fraction_mse": movement_fraction_mse,
                    "rollout_position_r2": (
                        float(np.clip(1.0 - movement_fraction_mse, 0.0, 1.0))
                        if np.isfinite(movement_fraction_mse)
                        else float("nan")
                    ),
                }
            )
    df = pd.DataFrame(rows)
    stats = rollout_position_metrics(df, dataset=dataset, split_name=split_name, rollout_steps=rollout_steps)
    stats["p_ratio_method"] = str(p_ratio_method)
    return df, stats


def latent_velocity_rollout_eval(
    ae_model,
    dyn_model,
    sims,
    stats: LatentNormalizer,
    *,
    dataset,
    split_name,
    rollout_steps: int,
    pos_dim: int,
    ae_target_mode: str,
    node_feature_mode: str,
    normalizers: dict[str, torch.Tensor],
    device,
    initial_velocity: str = "mean",
    p_ratio_method: str = "rollout_sides",
):
    rows = []
    ae_model.eval()
    dyn_model.eval()
    with torch.no_grad():
        for sim_idx, sim in enumerate(sims):
            filtered_steps = min(int(rollout_steps), max(len(filtered_frame_ids(sim, include_last=True)) - 1, 0))
            target_index = frame_for_filtered_step(sim, filtered_steps)
            if filtered_steps <= 0 or target_index <= 0:
                continue
            z = encode_frame_latent(
                ae_model,
                sim,
                0,
                pos_dim=pos_dim,
                node_feature_mode=node_feature_mode,
                normalizers=normalizers,
                device=device,
            )
            if initial_velocity == "zero":
                prev_dz = torch.zeros_like(z)
            elif initial_velocity == "mean":
                prev_dz = stats.dz_mean.squeeze(0).to(device)
            else:
                raise ValueError(f"Unknown initial_velocity: {initial_velocity}")
            for _ in range(filtered_steps):
                z, prev_dz = latent_step_velocity(dyn_model, z, prev_dz, stats)
            pred_graph = decode_latent_to_graph(
                ae_model,
                sim,
                z,
                target_index,
                pos_dim=pos_dim,
                ae_target_mode=ae_target_mode,
                normalizers=normalizers,
                device=device,
            )
            pred_pr = calc_rollout_p_ratio_by_method(
                [clone_graph(sim[0]).cpu(), pred_graph], -1, method=p_ratio_method
            )
            true_pr = calc_rollout_p_ratio_by_method(sim, target_index, method=p_ratio_method)
            initial_pos = sim[0].x[:, :pos_dim].cpu().float()
            target_pos = sim[target_index].x[:, :pos_dim].cpu().float()
            pred_pos = pred_graph.x[:, :pos_dim].cpu().float()
            initial_to_target_mse = float(F.mse_loss(initial_pos, target_pos).item())
            pred_to_initial_mse = float(F.mse_loss(pred_pos, initial_pos).item())
            final_pos_mse = float(F.mse_loss(pred_pos, target_pos).item())
            movement_fraction_mse = final_pos_mse / initial_to_target_mse if initial_to_target_mse > 0 else float("nan")
            rows.append(
                {
                    "dataset": dataset,
                    "split": split_name,
                    "sim_idx": sim_idx,
                    "target_index": target_index,
                    "rollout_steps": int(rollout_steps),
                    "filtered_steps": int(filtered_steps),
                    "pred_p_ratio": pred_pr,
                    "true_p_ratio": true_pr,
                    "p_ratio_method": str(p_ratio_method),
                    "final_pos_mse": final_pos_mse,
                    "initial_to_target_mse": initial_to_target_mse,
                    "pred_to_initial_mse": pred_to_initial_mse,
                    "movement_fraction_mse": movement_fraction_mse,
                    "rollout_position_r2": (
                        float(np.clip(1.0 - movement_fraction_mse, 0.0, 1.0))
                        if np.isfinite(movement_fraction_mse)
                        else float("nan")
                    ),
                }
            )
    df = pd.DataFrame(rows)
    stats = rollout_position_metrics(df, dataset=dataset, split_name=split_name, rollout_steps=rollout_steps)
    stats["p_ratio_method"] = str(p_ratio_method)
    return df, stats
