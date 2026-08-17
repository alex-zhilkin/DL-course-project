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

from ..graph import box_tensor, clone_graph
from .simulation import (
    ae_target_tensor,
    batch_delta_graphs,
    complete_graph_edge_data,
    edge_features,
    filtered_frame_ids,
    frame_node_feature,
    frame_for_filtered_step,
    iter_batches,
    pearson_r,
    reference_edge_features,
    reference_positions_for_model,
    r2_score,
    undirected_complete_graph_edge_data,
)
from .models import make_latent_propagator
from .physics import PhysicsLossConfig, elastic_implicit_euler_energy


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
    epoch_callback: Callable | None = None,
    selection_metric_key: str | None = None,
    selection_mode: str = "min",
    verbose: bool = True,
) -> TrainingResult:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    best_state = None
    selection_mode = str(selection_mode).lower()
    if selection_mode not in {"min", "max"}:
        raise ValueError("selection_mode must be 'min' or 'max'.")
    best_val = float("inf") if selection_mode == "min" else float("-inf")
    # A rollout metric such as R² can be undefined when the validation targets
    # are (nearly) constant.  Keep a validation-loss checkpoint until the first
    # finite requested metric appears instead of finishing with no checkpoint.
    using_selection_metric = selection_metric_key is None
    best_fallback_val_loss = float("inf")
    best_value_label = selection_metric_key or "val"
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
        if epoch_callback is not None:
            with torch.no_grad():
                callback_info = epoch_callback(epoch, model)
            if callback_info:
                row.update(
                    {
                        str(key): float(value)
                        for key, value in callback_info.items()
                    }
                )
        rows.append(row)

        selection_value = float(
            row.get(selection_metric_key, float("nan"))
            if selection_metric_key is not None
            else val_loss
        )
        finite_selection = np.isfinite(selection_value)
        if selection_metric_key is not None and not using_selection_metric:
            if finite_selection:
                # The requested metric takes precedence over any provisional
                # validation-loss checkpoint as soon as it becomes available.
                improved = True
                using_selection_metric = True
            else:
                improved = np.isfinite(val_loss) and (
                    val_loss < best_fallback_val_loss - float(config.min_delta)
                )
        else:
            improved = finite_selection and (
                selection_value < best_val - float(config.min_delta)
                if selection_mode == "min"
                else selection_value > best_val + float(config.min_delta)
            )
        if improved:
            if selection_metric_key is not None and not finite_selection:
                best_fallback_val_loss = val_loss
                best_val = val_loss
                best_value_label = "val_loss_fallback"
            else:
                best_val = selection_value
                best_value_label = selection_metric_key or "val"
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1

        if verbose and (
            epoch_callback is not None
            or epoch == 1
            or epoch % int(config.log_every) == 0
            or improved
        ):
            callback_text = " ".join(
                f"{key}={value:.4g}"
                for key, value in row.items()
                if key.startswith("val_rollout_")
            )
            print(
                f"{label} {epoch:04d} train={train_loss:.6g} "
                f"val={val_loss:.6g} stale={stale}"
                + (f" {callback_text}" if callback_text else "")
            )
        if stale >= int(config.patience):
            if verbose:
                print(
                    f"{label} early stop at epoch {epoch:04d}; "
                    f"best_epoch={best_epoch:04d} best_{best_value_label}={best_val:.6g}"
                )
            break

    if best_state is None:
        raise RuntimeError(f"{label} training produced no finite validation loss.")
    model.load_state_dict(best_state)
    model.eval()
    if verbose and stale < int(config.patience):
        print(
            f"{label} finished at max epoch {int(config.max_epochs):04d}; "
            f"best_epoch={best_epoch:04d} best_{best_value_label}={best_val:.6g}"
        )
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
    rho_scale_mean: torch.Tensor | None = None

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
            rho_scale_mean=(
                None if self.rho_scale_mean is None else self.rho_scale_mean.to(device)
            ),
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
        if self.rho_scale_mean is not None:
            out["rho_scale_mean"] = self.rho_scale_mean
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
            rho_scale_mean=values.get("rho_scale_mean"),
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

    def normalize_rho_scale(self, rho_scale: torch.Tensor | None) -> torch.Tensor | None:
        if rho_scale is None:
            return None
        if self.rho_scale_mean is None:
            return torch.ones_like(rho_scale)
        return rho_scale / self.rho_scale_mean.to(rho_scale.device).clamp_min(1e-8)


def initial_structure_scale(
    sim,
    *,
    mode: str | None,
    pos_dim: int,
    device,
) -> torch.Tensor | None:
    """Return a static per-simulation scale for radial progress calibration."""

    if mode is None:
        return None
    mode = str(mode).lower()
    if mode in {"", "none", "off", "raw"}:
        return None

    graph = sim[0]
    box = box_tensor(graph, device=device, dtype=torch.float32)
    if box is None:
        pos = graph.x[:, :pos_dim].to(device).float()
        width = (pos[:, 0].max() - pos[:, 0].min()).clamp_min(1e-8)
        height = (pos[:, 1].max() - pos[:, 1].min()).clamp_min(1e-8)
    else:
        width = box[0].clamp_min(1e-8)
        height = box[1].clamp_min(1e-8)

    if mode in {"box_width", "width", "x", "lx"}:
        value = width
    elif mode in {"box_height", "height", "y", "ly"}:
        value = height
    elif mode in {"box_area_sqrt", "area_sqrt", "sqrt_area", "geometric_mean"}:
        value = torch.sqrt(width * height)
    else:
        raise ValueError(f"Unknown polar_rho_scale_mode: {mode}")
    return value.reshape(1, 1)


def encode_reference_context(
    ae_model,
    sim,
    *,
    pos_dim: int,
    normalizers: dict[str, torch.Tensor],
    device,
    include_temperature: bool = False,
    pool_mode: str = "mean",
) -> torch.Tensor:
    """Pool the learned reference-node representation into static network context."""
    ref_graph = sim[0]
    ref_pos = reference_positions_for_model(
        ref_graph, pos_dim=pos_dim, device=device
    )
    edge_mode = str(getattr(ae_model, "edge_mode", "stored"))
    if edge_mode == "complete":
        edge_index, _, ref_edge_attr = undirected_complete_graph_edge_data(
            ref_graph, ref_graph, pos_dim=pos_dim, device=device
        )
    elif edge_mode == "stored":
        edge_index = ref_graph.edge_index.to(device).long()
        ref_edge_attr = reference_edge_features(
            ref_graph, pos_dim=pos_dim, device=device
        )
    else:
        raise ValueError(f"Unknown edge_mode: {edge_mode}")
    ref_edge_attr_norm = (
        ref_edge_attr - normalizers.get("ref_edge_mean", normalizers["edge_mean"]).to(device)
    ) / normalizers.get("ref_edge_std", normalizers["edge_std"]).to(device)
    h0 = ae_model.encode_reference_graph(
        ref_pos,
        ref_edge_attr_norm,
        edge_index,
    )
    pool_mode = str(pool_mode).lower()
    if pool_mode in {"learned_attention", "attention", "set_attention"}:
        context = h0
    elif pool_mode in {"mean", "average"}:
        context = h0.mean(dim=0)
    elif pool_mode in {"moments", "distribution", "mean_std_min_max"}:
        context = torch.cat(
            [
                h0.mean(dim=0),
                h0.std(dim=0, unbiased=False),
                h0.amin(dim=0),
                h0.amax(dim=0),
            ],
            dim=0,
        )
    else:
        raise ValueError(f"Unknown propagator_context_pool: {pool_mode}")
    if include_temperature:
        temperature = float(getattr(ref_graph, "temperature", 0.0))
        temperature_feature = torch.tensor(
            [np.log1p(max(temperature, 0.0))],
            dtype=context.dtype,
            device=context.device,
        )
        if context.ndim == 2:
            context = torch.cat(
                [context, temperature_feature.reshape(1, 1).expand(context.size(0), 1)],
                dim=-1,
            )
        else:
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
    edge_mode: str | None = None,
) -> torch.Tensor:
    ref_graph = sim[0]
    cur_graph = sim[int(t)]
    ref_pos = reference_positions_for_model(
        ref_graph, pos_dim=pos_dim, device=device
    )
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
    edge_mode = str(edge_mode or getattr(ae_model, "edge_mode", "stored"))
    if edge_mode == "complete":
        edge_index, edge_attr, ref_edge_attr = undirected_complete_graph_edge_data(
            ref_graph, cur_graph, pos_dim=pos_dim, device=device
        )
    elif edge_mode == "stored":
        edge_index = ref_graph.edge_index.to(device).long()
        edge_attr = edge_features(ref_graph, cur_graph, pos_dim=pos_dim, device=device)
        ref_edge_attr = reference_edge_features(
            ref_graph, pos_dim=pos_dim, device=device
        )
    else:
        raise ValueError(f"Unknown edge_mode: {edge_mode}")
    edge_attr_norm = (edge_attr - normalizers["edge_mean"].to(device)) / normalizers[
        "edge_std"
    ].to(device)
    ref_edge_attr_norm = (
        ref_edge_attr - normalizers.get("ref_edge_mean", normalizers["edge_mean"]).to(device)
    ) / normalizers.get("ref_edge_std", normalizers["edge_std"]).to(device)
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
    context_pool_mode: str = "mean",
    rho_scale_mode: str | None = None,
) -> LatentNormalizer:
    z_chunks = []
    z_next_chunks = []
    dz_chunks = []
    context_chunks = []
    rho_scale_chunks = []
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
        if use_static_context and str(context_pool_mode).lower() not in {
            "learned_attention",
            "attention",
            "set_attention",
        }:
            context_chunks.extend(
                encode_reference_context(
                    ae_model,
                    sims[int(row[0])],
                    pos_dim=pos_dim,
                    normalizers=normalizers,
                    device=device,
                    include_temperature=context_include_temperature,
                    pool_mode=context_pool_mode,
                ).detach()
                for row in rows_batch
            )
        if rho_scale_mode not in {None, "", "none", "off", "raw"}:
            rho_scale_chunks.extend(
                initial_structure_scale(
                    sims[int(row[0])],
                    mode=rho_scale_mode,
                    pos_dim=pos_dim,
                    device=device,
                )
                for row in rows_batch
            )
    z_all = torch.cat(z_chunks, dim=0)
    z_next_all = torch.cat(z_next_chunks, dim=0)
    dz_all = torch.cat(dz_chunks, dim=0)
    context_all = torch.stack(context_chunks, dim=0) if context_chunks else None
    rho_scale_all = (
        torch.cat(rho_scale_chunks, dim=0) if rho_scale_chunks else None
    )
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
        rho_scale_mean=(
            None
            if rho_scale_all is None
            else rho_scale_all.mean(dim=0, keepdim=True).clamp_min(1e-8)
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
    rho_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    loss_mode = str(loss_mode).lower()
    if getattr(model, "requires_next_z_loss", False) and loss_mode not in {
        "next_z",
        "jepa",
        "next_embedding",
    }:
        raise ValueError(
            f"{model.__class__.__name__} predicts next z directly; use "
            "propagator_loss='next_z'."
        )
    z_norm = stats.normalize_z(z.unsqueeze(0))
    context_norm = stats.normalize_context(
        None if context is None else context.unsqueeze(0)
    )
    rho_scale_norm = stats.normalize_rho_scale(
        None if rho_scale is None else rho_scale.reshape(1, 1)
    )
    if getattr(model, "uses_rho_progress_scale", False):
        pred = model(z_norm, context_norm, rho_scale=rho_scale_norm)
    else:
        pred = model(z_norm, context_norm)
    if loss_mode in {"delta", "dz", "residual_delta", "hybrid_delta_next"}:
        pred_dz_norm = pred if getattr(model, "predicts_delta", False) else pred - z_norm
        return z + stats.unnormalize_dz(pred_dz_norm).squeeze(0)
    if loss_mode in {"next_z", "jepa", "next_embedding"}:
        return stats.unnormalize_z_next(pred).squeeze(0)
    raise ValueError(f"Unknown propagator loss_mode: {loss_mode}")


def latent_step_kinematic(
    model,
    z: torch.Tensor,
    z_previous: torch.Tensor,
    z_reference: torch.Tensor,
    stats: LatentNormalizer,
    *,
    progress: float | torch.Tensor,
    context: torch.Tensor | None = None,
    context_is_encoded: bool = False,
) -> torch.Tensor:
    """Advance an anchored, second-order latent state by one closed-loop step."""

    if not getattr(model, "uses_kinematic_state", False):
        raise ValueError(
            f"{model.__class__.__name__} is not a kinematic latent propagator."
        )
    z_std = stats.z_std.to(z).clamp_min(1e-6)
    q = (z.unsqueeze(0) - z_reference.unsqueeze(0)) / z_std
    # Zero physical velocity should be represented by exactly zero.  We use
    # the learned delta scale but deliberately do not subtract its mean.
    velocity = (z.unsqueeze(0) - z_previous.unsqueeze(0)) / stats.dz_std.to(z).clamp_min(
        1e-6
    )
    reference = stats.normalize_z(z_reference.unsqueeze(0))
    progress_tensor = torch.as_tensor(progress, dtype=z.dtype, device=z.device).reshape(1, 1)
    progress_tensor = 2.0 * progress_tensor.clamp(0.0, 1.0) - 1.0
    state = torch.cat([q, velocity, reference, progress_tensor], dim=-1)
    if context_is_encoded:
        context_norm = context
    else:
        context_value = context
        if context_value is not None and context_value.ndim == 1:
            context_value = context_value.unsqueeze(0)
        context_norm = stats.normalize_context(context_value)
    next_q = model(
        state,
        context_norm,
        context_is_encoded=context_is_encoded,
    )
    return (z_reference.unsqueeze(0) + next_q * z_std).squeeze(0)


def latent_step_history(
    model,
    z: torch.Tensor,
    z_previous: torch.Tensor,
    z_previous_previous: torch.Tensor,
    z_reference: torch.Tensor,
    stats: LatentNormalizer,
    *,
    context: torch.Tensor | None = None,
    context_is_encoded: bool = False,
) -> torch.Tensor:
    """Advance a latent state using position, velocity, and acceleration."""

    if not getattr(model, "uses_history_state", False):
        raise ValueError(
            f"{model.__class__.__name__} is not a three-frame latent propagator."
        )
    z_std = stats.z_std.to(z).clamp_min(1e-6)
    dz_std = stats.dz_std.to(z).clamp_min(1e-6)
    q = (z.unsqueeze(0) - z_reference.unsqueeze(0)) / z_std
    velocity = (z.unsqueeze(0) - z_previous.unsqueeze(0)) / dz_std
    previous_velocity = (
        z_previous.unsqueeze(0) - z_previous_previous.unsqueeze(0)
    ) / dz_std
    acceleration = velocity - previous_velocity
    reference = stats.normalize_z(z_reference.unsqueeze(0))
    state = torch.cat([q, velocity, acceleration, reference], dim=-1)
    if context_is_encoded:
        context_norm = context
    else:
        context_value = context
        if context_value is not None and context_value.ndim == 1:
            context_value = context_value.unsqueeze(0)
        context_norm = stats.normalize_context(context_value)
    predicted_acceleration = model(
        state,
        context_norm,
        context_is_encoded=context_is_encoded,
    )
    next_velocity = velocity + predicted_acceleration
    return (z.unsqueeze(0) + next_velocity * dz_std).squeeze(0)


def latent_step_fixed_history(
    model,
    z: torch.Tensor,
    observed_first: torch.Tensor,
    observed_second: torch.Tensor,
    stats: LatentNormalizer,
    *,
    observed_frame_gap: int = 1,
    context: torch.Tensor | None = None,
    context_is_encoded: bool = False,
) -> torch.Tensor:
    """Advance z from current and fixed observed states, without a time input."""

    if not getattr(model, "uses_fixed_observed_state", False):
        raise ValueError(
            f"{model.__class__.__name__} is not a fixed-history propagator."
        )
    velocity_residual = bool(
        getattr(model, "uses_fixed_velocity_residual", False)
    )
    if velocity_residual:
        frame_gap = max(int(observed_frame_gap), 1)
        observed_velocity = (observed_second - observed_first) / frame_gap
        state = torch.cat(
            [
                stats.normalize_z(z.unsqueeze(0)),
                stats.normalize_z(observed_second.unsqueeze(0)),
                observed_velocity.unsqueeze(0)
                / stats.dz_std.to(z).clamp_min(1e-6),
            ],
            dim=-1,
        )
    else:
        observed_velocity = torch.zeros_like(z)
        state = torch.cat(
            [
                stats.normalize_z(z.unsqueeze(0)),
                stats.normalize_z(observed_first.unsqueeze(0)),
                stats.normalize_z(observed_second.unsqueeze(0)),
            ],
            dim=-1,
        )
    if context_is_encoded:
        context_norm = context
    else:
        context_value = context
        if context_value is not None and context_value.ndim == 1:
            context_value = context_value.unsqueeze(0)
        context_norm = stats.normalize_context(context_value)
    predicted_delta_norm = model(
        state,
        context_norm,
        context_is_encoded=context_is_encoded,
    )
    if velocity_residual:
        # Zero network output must mean an exact constant-velocity rollout.
        predicted_residual = (
            predicted_delta_norm * stats.dz_std.to(z).clamp_min(1e-6)
        ).squeeze(0)
    else:
        predicted_residual = stats.unnormalize_dz(predicted_delta_norm).squeeze(0)
    return z + observed_velocity + predicted_residual


def _source_mixed_rows(sims, rows, *, shuffle: bool) -> list:
    """Interleave all source rows across an epoch without changing their counts."""

    grouped: dict[str, list] = {}
    for row in rows:
        sim_idx = int(row[0])
        source = str(getattr(sims[sim_idx][0], "source_name", "unknown"))
        grouped.setdefault(source, []).append(row)
    if len(grouped) <= 1:
        return list(rows)

    prepared: dict[str, list] = {}
    for source in sorted(grouped):
        source_rows = list(grouped[source])
        if shuffle and len(source_rows) > 1:
            order = torch.randperm(len(source_rows)).tolist()
            source_rows = [source_rows[index] for index in order]
        prepared[source] = source_rows

    # With equal budgets, use literal round-robin source mixing: exactly one
    # row from every source per cycle. Randomize each cycle's source order
    # during training so the ordering itself cannot become a learned cue.
    source_lengths = {len(source_rows) for source_rows in prepared.values()}
    if len(source_lengths) == 1:
        sources = sorted(prepared)
        mixed_rows = []
        for row_order in range(next(iter(source_lengths))):
            cycle = sources
            if shuffle:
                order = torch.randperm(len(sources)).tolist()
                cycle = [sources[index] for index in order]
            mixed_rows.extend(prepared[source][row_order] for source in cycle)
        return mixed_rows

    keyed_rows = []
    for source_order, source in enumerate(sorted(prepared)):
        source_rows = prepared[source]
        jitters = torch.rand(len(source_rows)).tolist() if shuffle else [0.5] * len(source_rows)
        denominator = max(len(source_rows), 1)
        for row_order, (row, jitter) in enumerate(zip(source_rows, jitters)):
            # Normalized progress spreads every source over the full epoch;
            # source_order only resolves the vanishingly rare exact tie.
            key = ((row_order + float(jitter)) / denominator, source_order)
            keyed_rows.append((key, row))
    keyed_rows.sort(key=lambda item: item[0])
    return [row for _, row in keyed_rows]


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
    edge_mode: str = "stored",
    coordinate_weights=None,
    mix_sources: bool = False,
    optimizer=None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    losses = []
    epoch_rows = (
        _source_mixed_rows(sims, frame_rows, shuffle=True)
        if is_train and mix_sources
        else frame_rows
    )
    for rows in iter_batches(
        epoch_rows,
        batch_graphs,
        shuffle=is_train and not mix_sources,
    ):
        batch_data = batch_delta_graphs(
            sims,
            rows,
            pos_dim=pos_dim,
            device=device,
            node_feature_mode=node_feature_mode,
            edge_mode=edge_mode,
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
            batch_data["ref_edge_attr"]
            - normalizers.get("ref_edge_mean", normalizers["edge_mean"]).to(device)
        ) / normalizers.get("ref_edge_std", normalizers["edge_std"]).to(device)
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
        reconstruction_loss = squared_error.mean()
        loss = reconstruction_loss
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        losses.append(
            {
                "loss": float(loss.item()),
                "reconstruction": float(reconstruction_loss.item()),
            }
        )
    if not losses:
        return {"loss": float("nan"), "reconstruction": float("nan")}
    return {key: float(np.mean([row[key] for row in losses])) for key in losses[0]}


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
    edge_mode: str = "stored",
    coordinate_weights=None,
    mix_sources: bool = False,
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
        "edge_mode": edge_mode,
        "coordinate_weights": coordinate_weights,
        "mix_sources": mix_sources,
    }
    return _train_with_early_stopping(
        model,
        train_epoch=lambda optimizer: epoch_autoencoder(
            model, train_sims, train_rows, optimizer=optimizer, **common
        ),
        val_epoch=lambda: epoch_autoencoder(model, val_sims, val_rows, **common),
        config=config,
        label="ae",
        metric_key="loss",
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
    ae_target_mode: str | None = None,
    use_static_context: bool = False,
    context_include_temperature: bool = False,
    context_pool_mode: str = "mean",
    rho_scale_mode: str | None = None,
    physics_config: PhysicsLossConfig | None = None,
    optimizer=None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    losses = []
    raw_losses = []
    physics_logs = []
    ae_target_mode = ae_target_mode or node_feature_mode
    loss_mode = str(loss_mode).lower()
    if getattr(model, "requires_next_z_loss", False) and loss_mode not in {
        "next_z",
        "jepa",
        "next_embedding",
    }:
        raise ValueError(
            f"{model.__class__.__name__} predicts next z directly; use "
            "propagator_loss='next_z'."
        )
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
        z0_used = z0
        if is_train and physics_config is not None and physics_config.latent_noise_std > 0:
            noise_norm = torch.randn_like(z0_norm) * float(physics_config.latent_noise_std)
            z0_norm = z0_norm + noise_norm
            z0_used = z0 + noise_norm * stats.z_std
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
                        pool_mode=context_pool_mode,
                    )
                    for row in rows
                ],
                dim=0,
            )
            context = stats.normalize_context(context)
        rho_scale = None
        if (
            getattr(model, "uses_rho_progress_scale", False)
            and rho_scale_mode not in {None, "", "none", "off", "raw"}
        ):
            raw_scale = torch.cat(
                [
                    initial_structure_scale(
                        sims[int(row[0])],
                        mode=rho_scale_mode,
                        pos_dim=pos_dim,
                        device=device,
                    )
                    for row in rows
                ],
                dim=0,
            )
            rho_scale = stats.normalize_rho_scale(raw_scale)
        if getattr(model, "uses_rho_progress_scale", False):
            pred = model(z0_norm, context, rho_scale=rho_scale)
        else:
            pred = model(z0_norm, context)
        if loss_mode in {"delta", "dz", "residual_delta", "hybrid_delta_next"}:
            target_norm = stats.normalize_dz(z1 - z0)
            pred_norm = pred if getattr(model, "predicts_delta", False) else pred - z0_norm
            pred_raw = z0_used + stats.unnormalize_dz(pred_norm)
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
        latent_supervised_loss = loss
        physics_values = None
        position_mse = torch.zeros((), device=z0.device, dtype=z0.dtype)
        if physics_config is not None:
            physical_parts = []
            position_parts = []
            for local_idx, (sim_idx, t0, t1) in enumerate(rows):
                sim = sims[int(sim_idx)]
                x_pred = decode_latent_positions(
                    ae_model, sim, pred_raw[local_idx], int(t1), pos_dim=pos_dim,
                    ae_target_mode=ae_target_mode, normalizers=normalizers, device=device,
                )
                if is_train and physics_config.latent_noise_std > 0:
                    x_prev = decode_latent_positions(
                        ae_model, sim, z0_used[local_idx], int(t0), pos_dim=pos_dim,
                        ae_target_mode=ae_target_mode, normalizers=normalizers, device=device,
                    )
                else:
                    x_prev = sim[int(t0)].x[:, :pos_dim].to(device).float()
                previous_index = max(0, int(t0) - 1)
                x_prev_prev = sim[previous_index].x[:, :pos_dim].to(device).float()
                energy = elastic_implicit_euler_energy(
                    x_pred, x_prev, x_prev_prev, reference_graph=sim[0],
                    target_graph=sim[int(t1)], config=physics_config,
                )
                physical_parts.append(energy)
                scale = (
                    sim[0].x[:, :pos_dim].to(device).float().amax(dim=0)
                    - sim[0].x[:, :pos_dim].to(device).float().amin(dim=0)
                ).clamp_min(1e-6)
                target_position = sim[int(t1)].x[:, :pos_dim].to(device).float()
                position_parts.append(F.mse_loss(x_pred / scale, target_position / scale))
            physics_values = {
                key: torch.stack([part[key] for part in physical_parts]).mean()
                for key in physical_parts[0]
            }
            position_mse = torch.stack(position_parts).mean()
            loss = (
                float(physics_config.lambda_phys) * physics_values["physical"]
                + float(physics_config.lambda_mse) * position_mse
            )
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        losses.append(float(loss.item()))
        raw_losses.append(float(F.mse_loss(pred_raw, z1).item()))
        if physics_values is not None:
            physics_logs.append({
                **{key: float(value.detach()) for key, value in physics_values.items()},
                "position_mse": float(position_mse.detach()),
                "latent_supervised": float(latent_supervised_loss.detach()),
            })
    result = {
        "loss_norm": float(np.mean(losses)) if losses else float("nan"),
        "loss_raw": float(np.mean(raw_losses)) if raw_losses else float("nan"),
    }
    if physics_logs:
        for key in physics_logs[0]:
            result[key] = float(np.mean([row[key] for row in physics_logs]))
    return result


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
    ae_target_mode: str | None = None,
    use_static_context: bool = False,
    context_include_temperature: bool = False,
    rho_scale_mode: str | None = None,
    physics_config: PhysicsLossConfig | None = None,
    optimizer=None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    losses = []
    raw_losses = []
    physics_logs = []
    ae_target_mode = ae_target_mode or node_feature_mode
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
        z_prev_used = z_prev
        z0_used = z0
        if is_train and physics_config is not None and physics_config.latent_noise_std > 0:
            latent_batch_std = torch.cat([z_prev, z0], dim=0).std(
                dim=0, keepdim=True, unbiased=False
            ).clamp_min(1e-6)
            noise_scale = float(physics_config.latent_noise_std) * latent_batch_std
            z_prev_used = z_prev + torch.randn_like(z_prev) * noise_scale
            z0_used = z0 + torch.randn_like(z0) * noise_scale
        prev_dz = z0_used - z_prev_used
        target_dz = z1 - z0_used
        inp = torch.cat([stats.normalize_z(z0_used), stats.normalize_dz(prev_dz)], dim=-1)
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
        pred_z1 = z0_used + stats.unnormalize_dz(pred_dz_norm)
        latent_supervised_loss = F.mse_loss(pred_dz_norm, target_dz_norm)
        loss = latent_supervised_loss
        physics_values = None
        position_mse = torch.zeros((), device=z0.device, dtype=z0.dtype)
        if physics_config is not None:
            physical_parts = []
            position_parts = []
            for local_idx, (sim_idx, t_prev, t0, t1) in enumerate(batch_rows):
                sim = sims[int(sim_idx)]
                x_pred = decode_latent_positions(
                    ae_model,
                    sim,
                    pred_z1[local_idx],
                    int(t1),
                    pos_dim=pos_dim,
                    ae_target_mode=ae_target_mode,
                    normalizers=normalizers,
                    device=device,
                )
                if is_train and physics_config.latent_noise_std > 0:
                    x_prev = decode_latent_positions(
                        ae_model,
                        sim,
                        z0_used[local_idx],
                        int(t0),
                        pos_dim=pos_dim,
                        ae_target_mode=ae_target_mode,
                        normalizers=normalizers,
                        device=device,
                    )
                    x_prev_prev = decode_latent_positions(
                        ae_model,
                        sim,
                        z_prev_used[local_idx],
                        int(t_prev),
                        pos_dim=pos_dim,
                        ae_target_mode=ae_target_mode,
                        normalizers=normalizers,
                        device=device,
                    )
                else:
                    x_prev = sim[int(t0)].x[:, :pos_dim].to(device).float()
                    x_prev_prev = sim[int(t_prev)].x[:, :pos_dim].to(device).float()
                energy = elastic_implicit_euler_energy(
                    x_pred,
                    x_prev,
                    x_prev_prev,
                    reference_graph=sim[0],
                    target_graph=sim[int(t1)],
                    config=physics_config,
                )
                physical_parts.append(energy)
                ref_pos = sim[0].x[:, :pos_dim].to(device).float()
                scale = (ref_pos.amax(dim=0) - ref_pos.amin(dim=0)).clamp_min(1e-6)
                target_position = sim[int(t1)].x[:, :pos_dim].to(device).float()
                position_parts.append(F.mse_loss(x_pred / scale, target_position / scale))
            physics_values = {
                key: torch.stack([part[key] for part in physical_parts]).mean()
                for key in physical_parts[0]
            }
            position_mse = torch.stack(position_parts).mean()
            loss = (
                float(physics_config.lambda_phys) * physics_values["physical"]
                + float(physics_config.lambda_mse) * position_mse
            )
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        losses.append(float(loss.item()))
        raw_losses.append(float(F.mse_loss(pred_z1, z1).item()))
        if physics_values is not None:
            physics_logs.append({
                **{key: float(value.detach()) for key, value in physics_values.items()},
                "position_mse": float(position_mse.detach()),
                "latent_supervised": float(latent_supervised_loss.detach()),
            })
    result = {
        "loss_norm": float(np.mean(losses)) if losses else float("nan"),
        "loss_raw": float(np.mean(raw_losses)) if raw_losses else float("nan"),
    }
    if physics_logs:
        for key in physics_logs[0]:
            result[key] = float(np.mean([row[key] for row in physics_logs]))
    return result


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
    rho_scale_mode: str | None = None,
    optimizer=None,
) -> dict[str, float]:
    """Train a propagator through autoregressive latent unrolls.

    The loss stays fully in representation space: for each start frame, the
    model is repeatedly applied through the requested horizon and the predicted
    latent is matched only at the final kth step.
    """

    is_train = optimizer is not None
    model.train(is_train)
    losses = []
    raw_losses = []
    horizons = sorted({int(h) for h in horizons if int(h) > 0})
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
            true_z = encode_frame_latent(
                ae_model,
                sim,
                int(target_frames[-1]),
                pos_dim=pos_dim,
                node_feature_mode=node_feature_mode,
                normalizers=normalizers,
                device=device,
            )
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
            rho_scale = initial_structure_scale(
                sim,
                mode=rho_scale_mode,
                pos_dim=pos_dim,
                device=device,
            )
            for step_idx in range(1, max_horizon + 1):
                z = latent_step(
                    model,
                    z,
                    stats,
                    loss_mode=loss_mode,
                    context=context,
                    rho_scale=rho_scale,
                )
            if loss_mode in {"next_z", "jepa", "next_embedding"}:
                pred_norm = stats.normalize_z_next(z.unsqueeze(0))
                target_norm = stats.normalize_z_next(true_z.unsqueeze(0))
            else:
                pred_norm = stats.normalize_z(z.unsqueeze(0))
                target_norm = stats.normalize_z(true_z.unsqueeze(0))
            row_losses.append(F.mse_loss(pred_norm, target_norm))
            row_raw_losses.append(F.mse_loss(z, true_z))
        if not row_losses:
            continue
        loss = torch.stack(row_losses).sum()
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
        raw_losses.append(float(torch.stack(row_raw_losses).sum().detach().cpu()))

    return {
        "loss_norm": float(np.mean(losses)) if losses else float("nan"),
        "loss_raw": float(np.mean(raw_losses)) if raw_losses else float("nan"),
    }


def _previous_filtered_frame(sim, frame: int, *, frame_skip: int) -> int:
    frame_ids = filtered_frame_ids(sim, frame_skip=frame_skip, include_last=True)
    try:
        order = frame_ids.index(int(frame))
    except ValueError:
        return max(0, int(frame) - max(1, int(frame_skip)))
    return int(frame_ids[max(0, order - 1)])


def _nth_previous_filtered_frame(
    sim,
    frame: int,
    *,
    frame_skip: int,
    steps: int,
) -> int:
    previous = int(frame)
    for _ in range(max(0, int(steps))):
        previous = _previous_filtered_frame(
            sim,
            previous,
            frame_skip=frame_skip,
        )
    return previous


def _precompute_kinematic_latents(
    ae_model,
    sims,
    rows,
    *,
    pos_dim: int,
    node_feature_mode: str,
    normalizers: dict[str, torch.Tensor],
    device,
    frame_skip: int,
    use_static_context: bool,
    context_include_temperature: bool,
    context_pool_mode: str,
    fixed_observed_frames: tuple[int, int] | None = None,
) -> tuple[dict[tuple[int, int], torch.Tensor], dict[int, torch.Tensor]]:
    """Encode every latent needed by history training once."""

    required: dict[int, set[int]] = {}
    for sim_idx, start_frame, target_frames in rows:
        sim_idx = int(sim_idx)
        start_frame = int(start_frame)
        required.setdefault(sim_idx, {0}).update(
            {
                start_frame,
                _previous_filtered_frame(
                    sims[sim_idx], start_frame, frame_skip=frame_skip
                ),
                _nth_previous_filtered_frame(
                    sims[sim_idx],
                    start_frame,
                    frame_skip=frame_skip,
                    steps=2,
                ),
                *(int(frame) for frame in target_frames),
            }
        )
        if fixed_observed_frames is not None:
            required[sim_idx].update(int(frame) for frame in fixed_observed_frames)
    latent_cache: dict[tuple[int, int], torch.Tensor] = {}
    context_cache: dict[int, torch.Tensor] = {}
    ae_model.eval()
    with torch.no_grad():
        for sim_idx, frames in required.items():
            sim = sims[sim_idx]
            for frame in sorted(frames):
                latent_cache[(sim_idx, frame)] = encode_frame_latent(
                    ae_model,
                    sim,
                    frame,
                    pos_dim=pos_dim,
                    node_feature_mode=node_feature_mode,
                    normalizers=normalizers,
                    device=device,
                ).detach()
            if use_static_context:
                context_cache[sim_idx] = encode_reference_context(
                    ae_model,
                    sim,
                    pos_dim=pos_dim,
                    normalizers=normalizers,
                    device=device,
                    include_temperature=context_include_temperature,
                    pool_mode=context_pool_mode,
                ).detach()
    return latent_cache, context_cache


def _per_frame_latent_variation(
    latent_cache: dict[tuple[int, int], torch.Tensor],
    *,
    global_scale: torch.Tensor,
    floor_fraction: float,
) -> dict[int, torch.Tensor]:
    """Estimate coordinate-wise between-network scale at each physical frame."""

    grouped: dict[int, list[torch.Tensor]] = {}
    for (_sim_idx, frame), latent in latent_cache.items():
        grouped.setdefault(int(frame), []).append(latent)
    floor = (
        global_scale.detach().reshape(-1) * float(floor_fraction)
    ).clamp_min(1e-6)
    return {
        frame: torch.stack(values, dim=0)
        .std(dim=0, unbiased=False)
        .clamp_min(floor)
        for frame, values in grouped.items()
        if len(values) >= 2
    }


def epoch_kinematic_multistep_propagator(
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
    unroll_steps: int,
    frame_skip: int = 1,
    use_static_context: bool = False,
    context_include_temperature: bool = False,
    latent_cache=None,
    context_cache=None,
    context_pool_mode: str = "mean",
    ae_target_mode: str | None = None,
    position_loss_weight: float = 0.0,
    position_boundary_weight: float = 1.0,
    position_boundary_fraction: float = 0.10,
    position_coordinate_weights=None,
    network_variation_weight: float = 0.0,
    frame_variation=None,
    fixed_observed_frames: tuple[int, int] | None = None,
    mix_sources: bool = False,
    optimizer=None,
    **_unused,
) -> dict[str, float]:
    """Supervise every state of a fully autoregressive latent unroll."""

    del node_feature_mode
    is_train = optimizer is not None
    model.train(is_train)
    unroll_steps = int(unroll_steps)
    losses, raw_losses, position_losses = [], [], []

    epoch_rows = (
        _source_mixed_rows(sims, rows, shuffle=True)
        if is_train and mix_sources
        else rows
    )
    for batch_rows in iter_batches(
        epoch_rows,
        batch_graphs,
        shuffle=is_train and not mix_sources,
    ):
        row_losses, row_raw_losses = [], []
        for sim_idx, start_frame, target_frames in batch_rows:
            sim_idx, start_frame = int(sim_idx), int(start_frame)
            if len(target_frames) < unroll_steps:
                continue
            sim = sims[sim_idx]
            z_reference = latent_cache[(sim_idx, 0)]
            z = latent_cache[(sim_idx, start_frame)]
            previous_frame = _previous_filtered_frame(
                sim, start_frame, frame_skip=frame_skip
            )
            z_previous = latent_cache[(sim_idx, previous_frame)]
            previous_previous_frame = _previous_filtered_frame(
                sim, previous_frame, frame_skip=frame_skip
            )
            z_previous_previous = latent_cache[
                (sim_idx, previous_previous_frame)
            ]
            context = (
                context_cache[sim_idx] if use_static_context else None
            )
            encoded_context = None
            if context is not None:
                context_value = (
                    context.unsqueeze(0) if context.ndim == 1 else context
                )
                encoded_context = model.encode_context(
                    stats.normalize_context(context_value)
                )
            step_losses, step_raw_losses, weights = [], [], []
            for offset in range(unroll_steps):
                target_frame = int(target_frames[offset])
                if getattr(model, "uses_fixed_observed_state", False):
                    if fixed_observed_frames is None:
                        raise ValueError(
                            "fixed_observed_frames are required by the fixed-history model."
                        )
                    observed_first = latent_cache[
                        (sim_idx, int(fixed_observed_frames[0]))
                    ]
                    observed_second = latent_cache[
                        (sim_idx, int(fixed_observed_frames[1]))
                    ]
                    z_next = latent_step_fixed_history(
                        model,
                        z,
                        observed_first,
                        observed_second,
                        stats,
                        observed_frame_gap=(
                            int(fixed_observed_frames[1])
                            - int(fixed_observed_frames[0])
                        ),
                        context=encoded_context,
                        context_is_encoded=encoded_context is not None,
                    )
                elif getattr(model, "uses_history_state", False):
                    z_next = latent_step_history(
                        model,
                        z,
                        z_previous,
                        z_previous_previous,
                        z_reference,
                        stats,
                        context=encoded_context,
                        context_is_encoded=encoded_context is not None,
                    )
                else:
                    progress = target_frame / max(1, len(sim) - 1)
                    z_next = latent_step_kinematic(
                        model,
                        z,
                        z_previous,
                        z_reference,
                        stats,
                        progress=progress,
                        context=encoded_context,
                        context_is_encoded=encoded_context is not None,
                    )
                true_z = latent_cache[(sim_idx, target_frame)]
                weight = 1.0 + offset / max(1, unroll_steps)
                if (
                    getattr(model, "uses_history_state", False)
                    or getattr(model, "uses_fixed_observed_state", False)
                ):
                    # The history model predicts motion, not an absolute
                    # coordinate. Train it on the increment at the natural
                    # delta-Z scale so small velocity errors are not hidden by
                    # the much larger overall latent range.
                    delta_scale = stats.dz_std.squeeze(0).to(z).clamp_min(1e-6)
                    predicted_delta = (z_next - z) / delta_scale
                    target_delta = (true_z - z) / delta_scale
                    step_loss = F.mse_loss(predicted_delta, target_delta)
                else:
                    pred_q = (
                        z_next - z_reference
                    ) / stats.z_std.squeeze(0).to(z).clamp_min(1e-6)
                    true_q = (
                        true_z - z_reference
                    ) / stats.z_std.squeeze(0).to(z).clamp_min(1e-6)
                    step_loss = F.mse_loss(pred_q, true_q)
                step_losses.append(step_loss * weight)
                if float(network_variation_weight) > 0 and target_frame in frame_variation:
                    variation_error = (
                        z_next - true_z
                    ) / frame_variation[target_frame].to(z).clamp_min(1e-6)
                    step_losses[-1] = step_losses[-1] + (
                        float(network_variation_weight)
                        * variation_error.square().mean()
                        * weight
                    )
                step_raw_losses.append(F.mse_loss(z_next, true_z) * weight)
                weights.append(weight)
                if getattr(model, "uses_fixed_observed_state", False):
                    z = z_next
                else:
                    z_previous_previous, z_previous, z = z_previous, z, z_next
            row_losses.append(torch.stack(step_losses).sum() / sum(weights))
            row_raw_losses.append(torch.stack(step_raw_losses).sum() / sum(weights))
            if float(position_loss_weight) > 0:
                target_frame = int(target_frames[unroll_steps - 1])
                if (
                    str(context_pool_mode).lower()
                    in {"learned_attention", "attention", "set_attention"}
                    and context is not None
                ):
                    h0 = context[:, : ae_model.hidden_size]
                    batch = torch.zeros(
                        h0.size(0), dtype=torch.long, device=device
                    )
                    target_norm = ae_model.decode(z.unsqueeze(0), h0, batch)
                    target_value = (
                        target_norm * normalizers["target_std"].to(device)
                        + normalizers["target_mean"].to(device)
                    )
                    ref_pos = sim[0].x[:, :pos_dim].to(device).float()
                    if (ae_target_mode or "normalized_delta") in {
                        "normalized_delta",
                        "self_normalized_delta",
                        "relative_delta",
                    }:
                        decode_scale = (
                            ref_pos.amax(dim=0) - ref_pos.amin(dim=0)
                        ).clamp_min(1e-6)
                        pred_pos = (
                            ref_pos + target_value * decode_scale.reshape(1, -1)
                        )
                    elif (ae_target_mode or "normalized_delta") in {
                        "position",
                        "positions",
                    }:
                        pred_pos = target_value
                    else:
                        pred_pos = ref_pos + target_value
                else:
                    pred_pos = decode_latent_positions(
                        ae_model,
                        sim,
                        z,
                        target_frame,
                        pos_dim=pos_dim,
                        ae_target_mode=ae_target_mode or "normalized_delta",
                        normalizers=normalizers,
                        device=device,
                    )
                ref_pos = sim[0].x[:, :pos_dim].to(device).float()
                target_pos = sim[target_frame].x[:, :pos_dim].to(device).float()
                if float(position_loss_weight) > 0:
                    scale = (
                        ref_pos.amax(dim=0) - ref_pos.amin(dim=0)
                    ).clamp_min(1e-6)
                    squared_position_error = (
                        (pred_pos - target_pos) / scale.reshape(1, -1)
                    ).square()
                    if position_coordinate_weights is not None:
                        coordinate_weights = torch.as_tensor(
                            position_coordinate_weights,
                            dtype=pred_pos.dtype,
                            device=device,
                        ).reshape(1, -1)
                        squared_position_error = (
                            squared_position_error
                            * coordinate_weights
                            / coordinate_weights.mean().clamp_min(1e-8)
                        )
                    node_error = squared_position_error.mean(dim=-1)
                    if float(position_boundary_weight) != 1.0:
                        side_count = max(
                            1,
                            int(
                                np.ceil(
                                    float(position_boundary_fraction)
                                    * ref_pos.size(0)
                                )
                            ),
                        )
                        boundary = torch.cat(
                            [
                                torch.topk(
                                    ref_pos[:, axis], side_count, largest=largest
                                ).indices
                                for axis in range(min(2, ref_pos.size(1)))
                                for largest in (False, True)
                            ]
                        ).unique()
                        node_weights = torch.ones_like(node_error)
                        node_weights[boundary] = float(position_boundary_weight)
                        position_loss = (
                            node_error * node_weights
                        ).sum() / node_weights.sum()
                    else:
                        position_loss = node_error.mean()
                    row_losses[-1] = (
                        row_losses[-1]
                        + float(position_loss_weight) * position_loss
                    )
                    position_losses.append(float(position_loss.detach().cpu()))
        if not row_losses:
            continue
        loss = torch.stack(row_losses).mean()
        raw_loss = torch.stack(row_raw_losses).mean()
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
        raw_losses.append(float(raw_loss.detach().cpu()))

    return {
        "loss_norm": float(np.mean(losses)) if losses else float("nan"),
        "loss_raw": float(np.mean(raw_losses)) if raw_losses else float("nan"),
        "position_loss": (
            float(np.mean(position_losses))
            if position_losses
            else 0.0
        ),
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
    ae_target_mode: str | None = None,
    objective: str = "one_step",
    horizons=None,
    frame_skip: int = 1,
    context_pool_mode: str = "mean",
    position_loss_weight: float = 0.0,
    position_boundary_weight: float = 1.0,
    position_boundary_fraction: float = 0.10,
    position_coordinate_weights=None,
    network_variation_weight: float = 0.0,
    network_variation_floor_fraction: float = 0.05,
    fixed_observed_frames: tuple[int, int] | None = None,
    unroll_curriculum=None,
    unroll_stage_epochs=None,
    mix_sources: bool = False,
    use_static_context: bool = False,
    context_include_temperature: bool = False,
    rho_scale_mode: str | None = None,
    physics_config: PhysicsLossConfig | None = None,
    epoch_callback: Callable | None = None,
    selection_metric_key: str | None = None,
    selection_mode: str = "min",
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
        "rho_scale_mode": rho_scale_mode,
    }
    objective = str(objective).lower()

    if objective in {"one_step", "next_step"}:
        epoch_fn = epoch_propagator
        extra = {
            "loss_mode": loss_mode,
            "ae_target_mode": ae_target_mode,
            "physics_config": physics_config,
            "context_pool_mode": context_pool_mode,
        }
    elif objective in {"multistep", "multi_step"}:
        if not horizons:
            raise ValueError("horizons are required for multistep propagator training.")
        epoch_fn = epoch_multistep_propagator
        extra = {
            "loss_mode": loss_mode,
            "horizons": horizons,
        }
    elif objective in {
        "kinematic_multistep",
        "kinematic",
        "anchored_multistep",
        "closed_loop",
        "history_one_step",
        "fixed_history_one_step",
    }:
        if not horizons:
            raise ValueError("horizons are required for kinematic multistep training.")
        max_horizon = max(int(horizon) for horizon in horizons)
        if sorted({int(horizon) for horizon in horizons}) != list(
            range(1, max_horizon + 1)
        ):
            raise ValueError(
                "Kinematic multistep horizons must contain every step from 1 "
                f"through {max_horizon}."
            )
        if getattr(model, "uses_history_state", False):
            def has_three_frames(sims, row):
                sim_idx, start_frame, _ = row
                previous = _previous_filtered_frame(
                    sims[int(sim_idx)],
                    int(start_frame),
                    frame_skip=frame_skip,
                )
                previous_previous = _previous_filtered_frame(
                    sims[int(sim_idx)],
                    previous,
                    frame_skip=frame_skip,
                )
                has_latent_history = (
                    previous_previous != previous
                    and previous != int(start_frame)
                )
                if node_feature_mode in {
                    "normalized_delta_velocity_history3",
                    "displacement_velocity_history3",
                    "modular_history3",
                }:
                    third_previous = _previous_filtered_frame(
                        sims[int(sim_idx)],
                        previous_previous,
                        frame_skip=frame_skip,
                    )
                    return has_latent_history and third_previous != previous_previous
                return has_latent_history

            train_rows = [
                row for row in train_rows if has_three_frames(train_sims, row)
            ]
            val_rows = [
                row for row in val_rows if has_three_frames(val_sims, row)
            ]
        if getattr(model, "uses_fixed_observed_state", False):
            if fixed_observed_frames is None or len(fixed_observed_frames) != 2:
                raise ValueError(
                    "The fixed-history model requires exactly two fixed_observed_frames."
                )
            fixed_observed_frames = tuple(int(frame) for frame in fixed_observed_frames)
            if not 0 <= fixed_observed_frames[0] < fixed_observed_frames[1]:
                raise ValueError(
                    "fixed_observed_frames must be increasing non-negative indices."
                )
            last_observed = fixed_observed_frames[1]
            train_rows = [row for row in train_rows if int(row[1]) >= last_observed]
            val_rows = [row for row in val_rows if int(row[1]) >= last_observed]
        training_label = (
            "one-step history training"
            if max_horizon == 1
            else "closed-loop training"
        )
        print(f"precomputing frozen AE latents for {training_label}")
        train_cache, train_context_cache = _precompute_kinematic_latents(
            autoencoder,
            train_sims,
            train_rows,
            pos_dim=pos_dim,
            node_feature_mode=node_feature_mode,
            normalizers=normalizers,
            device=device,
            frame_skip=frame_skip,
            use_static_context=use_static_context,
            context_include_temperature=context_include_temperature,
            context_pool_mode=context_pool_mode,
            fixed_observed_frames=fixed_observed_frames,
        )
        val_cache, val_context_cache = _precompute_kinematic_latents(
            autoencoder,
            val_sims,
            val_rows,
            pos_dim=pos_dim,
            node_feature_mode=node_feature_mode,
            normalizers=normalizers,
            device=device,
            frame_skip=frame_skip,
            use_static_context=use_static_context,
            context_include_temperature=context_include_temperature,
            context_pool_mode=context_pool_mode,
            fixed_observed_frames=fixed_observed_frames,
        )
        frame_variation = _per_frame_latent_variation(
            train_cache,
            global_scale=stats.z_std,
            floor_fraction=network_variation_floor_fraction,
        )
        if unroll_curriculum is None:
            stages = [max_horizon]
        else:
            stages = [int(value) for value in unroll_curriculum]
            if (
                not stages
                or stages != sorted(set(stages))
                or min(stages) < 1
                or max(stages) != max_horizon
            ):
                raise ValueError(
                    "unroll_curriculum must be increasing, unique, positive, "
                    f"and end at the maximum horizon ({max_horizon})."
                )
        if unroll_stage_epochs is None:
            base, remainder = divmod(int(config.max_epochs), len(stages))
            stage_epochs = [base + int(idx < remainder) for idx in range(len(stages))]
        else:
            stage_epochs = [int(value) for value in unroll_stage_epochs]
            if len(stage_epochs) != len(stages) or min(stage_epochs) < 1:
                raise ValueError(
                    "unroll_stage_epochs must provide one positive epoch count "
                    "per curriculum stage."
                )
            if sum(stage_epochs) != int(config.max_epochs):
                raise ValueError(
                    "unroll_stage_epochs must sum to the configured maximum "
                    f"epochs ({int(config.max_epochs)})."
                )

        histories = []
        epoch_offset = 0
        final_result = None
        for stage_index, (stage_horizon, max_epochs) in enumerate(
            zip(stages, stage_epochs), start=1
        ):
            print(
                f"latent history training: steps={stage_horizon}, "
                f"max_epochs={max_epochs}"
            )
            stage_config = TrainingConfig(
                max_epochs=max_epochs,
                patience=min(int(config.patience), max(2, max_epochs // 2)),
                learning_rate=float(config.learning_rate),
                weight_decay=float(config.weight_decay),
                min_delta=float(config.min_delta),
                log_every=int(config.log_every),
            )
            stage_common = {
                **common,
                "frame_skip": frame_skip,
                "unroll_steps": stage_horizon,
                "context_pool_mode": context_pool_mode,
                "ae_target_mode": ae_target_mode,
                "position_loss_weight": position_loss_weight,
                "position_boundary_weight": position_boundary_weight,
                "position_boundary_fraction": position_boundary_fraction,
                "position_coordinate_weights": position_coordinate_weights,
                "network_variation_weight": network_variation_weight,
                "frame_variation": frame_variation,
                "fixed_observed_frames": fixed_observed_frames,
                "mix_sources": mix_sources,
            }

            def stage_epoch(sims, rows, cache, context_cache, optimizer=None):
                return epoch_kinematic_multistep_propagator(
                    model,
                    autoencoder,
                    sims,
                    rows,
                    stats,
                    optimizer=optimizer,
                    latent_cache=cache,
                    context_cache=context_cache,
                    **stage_common,
                )

            final_result = _train_with_early_stopping(
                model,
                train_epoch=lambda optimizer: stage_epoch(
                    train_sims,
                    train_rows,
                    train_cache,
                    train_context_cache,
                    optimizer,
                ),
                val_epoch=lambda: stage_epoch(
                    val_sims,
                    val_rows,
                    val_cache,
                    val_context_cache,
                ),
                config=stage_config,
                label=f"propagator-h{stage_horizon}",
                metric_key="loss_norm",
                epoch_callback=epoch_callback,
                selection_metric_key=selection_metric_key,
                selection_mode=selection_mode,
                verbose=verbose,
            )
            history = final_result.history.copy()
            history["stage"] = stage_index
            history["unroll_steps"] = stage_horizon
            history["stage_epoch"] = history["epoch"]
            history["epoch"] = history["epoch"] + epoch_offset
            histories.append(history)
            epoch_offset += len(history)
        combined = pd.concat(histories, ignore_index=True)
        return TrainingResult(
            model=model,
            history=combined,
            best_val_loss=float(final_result.best_val_loss),
            best_epoch=int(
                histories[-1]["epoch"].iloc[0] - 1 + final_result.best_epoch
            ),
        )
    elif objective in {"velocity", "second_order"}:
        epoch_fn = epoch_velocity_propagator
        extra = {
            "ae_target_mode": ae_target_mode,
            "physics_config": physics_config,
        }
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
        epoch_callback=epoch_callback,
        selection_metric_key=selection_metric_key,
        selection_mode=selection_mode,
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


def decode_latent_positions(
    ae_model,
    sim,
    z: torch.Tensor,
    target_index: int,
    *,
    pos_dim: int,
    ae_target_mode: str,
    normalizers: dict[str, torch.Tensor],
    device,
) -> torch.Tensor:
    """Decode a latent state to differentiable full-space node positions."""

    ref = sim[0]
    # The encoder may receive the original physical reference geometry as
    # static context, but decoded displacements live in the stored/model
    # coordinate system.  Keep these two reference positions separate: using
    # the physical context position as the decoder origin would add normalized
    # displacements to dimensional coordinates.
    model_ref_pos = ref.x[:, :pos_dim].to(device).float()
    context_ref_pos = reference_positions_for_model(
        ref, pos_dim=pos_dim, device=device
    )
    edge_mode = str(getattr(ae_model, "edge_mode", "stored"))
    if edge_mode == "complete":
        edge_index, _, ref_edge_attr = undirected_complete_graph_edge_data(
            ref, ref, pos_dim=pos_dim, device=device
        )
    elif edge_mode == "stored":
        edge_index = ref.edge_index.to(device).long()
        ref_edge_attr = reference_edge_features(ref, pos_dim=pos_dim, device=device)
    else:
        raise ValueError(f"Unknown edge_mode: {edge_mode}")
    ref_edge_attr_norm = (
        ref_edge_attr - normalizers.get("ref_edge_mean", normalizers["edge_mean"]).to(device)
    ) / normalizers.get("ref_edge_std", normalizers["edge_std"]).to(device)
    batch = torch.zeros(model_ref_pos.size(0), dtype=torch.long, device=device)
    h0 = ae_model.encode_reference_graph(
        context_ref_pos, ref_edge_attr_norm, edge_index
    )
    target_norm = ae_model.decode(z.unsqueeze(0), h0, batch)
    target = target_norm * normalizers["target_std"].to(device) + normalizers["target_mean"].to(device)
    if ae_target_mode in ("position", "positions"):
        return target
    if ae_target_mode in {
        "normalized_delta",
        "self_normalized_delta",
        "relative_delta",
        "normalized_delta_velocity_history3",
        "displacement_velocity_history3",
        "modular_history3",
    }:
        scale = (
            model_ref_pos.max(dim=0).values - model_ref_pos.min(dim=0).values
        ).clamp_min(1e-6)
        return model_ref_pos + target[:, :pos_dim] * scale.reshape(1, -1)
    return model_ref_pos + target


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
    target = decode_latent_positions(
        ae_model, sim, z, target_index, pos_dim=pos_dim,
        ae_target_mode=ae_target_mode, normalizers=normalizers, device=device,
    )
    # A predicted frame may reuse static topology and node metadata, but must
    # not inherit target-frame box or dynamic attributes.
    pred = clone_graph(sim[0]).to(device)
    pred.x = pred.x.clone().float()
    pred.x[:, :pos_dim] = target
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
    use_static_context: bool = False,
    context_include_temperature: bool = False,
    rho_scale_mode: str | None = None,
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
            rho_scale = initial_structure_scale(
                sim,
                mode=rho_scale_mode,
                pos_dim=pos_dim,
                device=device,
            )
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
            for _ in range(filtered_steps):
                z = latent_step(
                    dyn_model,
                    z,
                    stats,
                    loss_mode=loss_mode,
                    context=context,
                    rho_scale=rho_scale,
                )
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
                    "rho_scale_mode": str(rho_scale_mode or "none"),
                    "rho_scale": (
                        float(rho_scale.detach().cpu().reshape(-1)[0])
                        if rho_scale is not None
                        else float("nan")
                    ),
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
