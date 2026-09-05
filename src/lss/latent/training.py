"""Shared latent autoencoder and propagator training helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
    canonical_edge_mode,
    compact_edge_features,
    compact_delta_edge_features,
    compact_delta_reference_edge_features,
    compact_reference_edge_features,
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


def _pcgrad_combined_gradients(
    source_losses: dict[str, torch.Tensor],
    parameters: list[torch.nn.Parameter],
) -> list[torch.Tensor]:
    """Return PCGrad-adjusted mean gradients for one multi-source batch."""

    source_gradients = []
    for loss in source_losses.values():
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        source_gradients.append(
            [
                gradient.detach().clone()
                if gradient is not None
                else torch.zeros_like(parameter)
                for gradient, parameter in zip(gradients, parameters)
            ]
        )
    projected = [[gradient.clone() for gradient in gradients] for gradients in source_gradients]
    for task_index, gradients in enumerate(projected):
        for other_index in torch.randperm(len(source_gradients)).tolist():
            if other_index == task_index:
                continue
            other_gradients = source_gradients[other_index]
            dot = sum(
                (gradient * other_gradient).sum()
                for gradient, other_gradient in zip(gradients, other_gradients)
            )
            if dot >= 0:
                continue
            squared_norm = sum(
                other_gradient.square().sum() for other_gradient in other_gradients
            ).clamp_min(1e-12)
            scale = dot / squared_norm
            for parameter_index in range(len(gradients)):
                gradients[parameter_index].sub_(scale * other_gradients[parameter_index])
    return [
        torch.stack([gradients[index] for gradients in projected], dim=0).mean(dim=0)
        for index in range(len(parameters))
    ]


def _solve_nash_mtl_coefficients(
    gram: torch.Tensor,
    *,
    max_iter: int = 50,
    tolerance: float = 1e-6,
    initial_alpha: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float, torch.Tensor]:
    """Solve ``C alpha = 1 / alpha`` by damped Newton in positive alpha-space."""

    stable_gram = gram.detach().to(device="cpu", dtype=torch.float64)
    if stable_gram.shape != (3, 3):
        raise ValueError("This Nash-MTL solver requires exactly three tasks.")
    if not torch.isfinite(stable_gram).all():
        raise RuntimeError("Nash-MTL received a non-finite gradient Gram matrix.")
    stable_gram = 0.5 * (stable_gram + stable_gram.T)
    gradient_norms = stable_gram.diag().clamp_min(0).sqrt()
    if torch.any(gradient_norms <= 1e-12):
        raise RuntimeError("Nash-MTL received a task with a zero gradient norm.")

    # A Gram matrix should be PSD. Permit only roundoff-scale negativity.
    eigenvalues = torch.linalg.eigvalsh(stable_gram)
    spectral_scale = float(eigenvalues.abs().max().clamp_min(1.0))
    if float(eigenvalues.min()) < -1e-10 * spectral_scale:
        raise RuntimeError("Nash-MTL gradient Gram matrix is not positive semidefinite.")

    if initial_alpha is None:
        alpha = gradient_norms.reciprocal()
    else:
        alpha = initial_alpha.detach().to(device="cpu", dtype=torch.float64).clone()
        if alpha.shape != (3,) or not torch.isfinite(alpha).all() or torch.any(alpha <= 0):
            alpha = gradient_norms.reciprocal()

    # Preserve warm-start ratios but put the start at the objective-optimal
    # scale along its ray: s^2 alpha^T C alpha = number of tasks.
    quadratic = float(alpha @ stable_gram @ alpha)
    if not np.isfinite(quadratic) or quadratic <= 1e-20:
        alpha = gradient_norms.reciprocal()
        quadratic = float(alpha @ stable_gram @ alpha)
    alpha *= np.sqrt(3.0 / max(quadratic, 1e-20))

    def objective(candidate: torch.Tensor) -> torch.Tensor:
        return 0.5 * candidate @ stable_gram @ candidate - candidate.log().sum()

    for _ in range(max(1, int(max_iter))):
        reciprocal = alpha.reciprocal()
        projection = stable_gram @ alpha
        gradient = projection - reciprocal
        residual = float(
            torch.linalg.vector_norm(gradient)
            / torch.linalg.vector_norm(reciprocal).clamp_min(1e-12)
        )
        elementwise_residual = float(
            torch.max(torch.abs(alpha * projection - 1.0))
        )
        if (
            np.isfinite(residual)
            and residual <= float(tolerance)
            and elementwise_residual <= float(tolerance)
            and float(projection.min()) > 0
        ):
            return alpha, residual, projection

        hessian = stable_gram + torch.diag(reciprocal.square())
        try:
            direction = torch.linalg.solve(hessian, -gradient)
        except RuntimeError as error:
            raise RuntimeError("Nash-MTL Newton system could not be solved.") from error
        directional_derivative = float(gradient @ direction)
        if not torch.isfinite(direction).all() or directional_derivative >= 0:
            raise RuntimeError("Nash-MTL Newton direction is not a finite descent direction.")

        negative = direction < 0
        step = 1.0
        if torch.any(negative):
            positive_limit = float(torch.min(-0.99 * alpha[negative] / direction[negative]))
            step = min(step, positive_limit)
        current_objective = float(objective(alpha))
        accepted = False
        for _ in range(50):
            candidate = alpha + step * direction
            if torch.all(candidate > 0) and torch.isfinite(candidate).all():
                candidate_objective = float(objective(candidate))
                if np.isfinite(candidate_objective) and candidate_objective <= (
                    current_objective + 1e-4 * step * directional_derivative
                ):
                    alpha = candidate
                    accepted = True
                    break
            step *= 0.5
        if not accepted:
            raise RuntimeError("Nash-MTL backtracking could not find a positive descent step.")

    reciprocal = alpha.reciprocal()
    projection = stable_gram @ alpha
    residual = float(
        torch.linalg.vector_norm(projection - reciprocal)
        / torch.linalg.vector_norm(reciprocal).clamp_min(1e-12)
    )
    elementwise_residual = float(
        torch.max(torch.abs(alpha * projection - 1.0))
    )
    if (
        not np.isfinite(residual)
        or residual > float(tolerance)
        or elementwise_residual > float(tolerance)
        or float(projection.min()) <= 0
    ):
        raise RuntimeError(
            "Nash-MTL Newton solver did not converge; "
            f"residual={residual:.3e}, "
            f"elementwise residual={elementwise_residual:.3e}."
        )
    return alpha, residual, projection


def _nash_mtl_combined_gradients(
    source_losses: dict[str, torch.Tensor],
    parameters: list[torch.nn.Parameter],
    *,
    max_iter: int = 50,
    initial_alpha: torch.Tensor | None = None,
) -> tuple[
    list[torch.Tensor],
    dict[str, float],
    float,
    float,
    dict[str, float],
    bool,
    torch.Tensor | None,
]:
    """Return a three-task Nash gradient, or safe normalized averaging."""

    source_names = sorted(source_losses)
    source_gradients: list[list[torch.Tensor]] = []
    for source_name in source_names:
        loss = source_losses[source_name]
        gradients = torch.autograd.grad(
            loss,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        source_gradients.append(
            [
                gradient.detach().clone()
                if gradient is not None
                else torch.zeros_like(parameter)
                for gradient, parameter in zip(gradients, parameters)
            ]
        )

    task_count = len(source_gradients)
    gram = torch.empty((task_count, task_count), dtype=torch.float64)
    for left in range(task_count):
        for right in range(left, task_count):
            value = sum(
                (left_gradient * right_gradient).sum().double().cpu()
                for left_gradient, right_gradient in zip(
                    source_gradients[left], source_gradients[right]
                )
            )
            gram[left, right] = value
            gram[right, left] = value
    cosine_by_pair = {}
    norms = gram.diag().clamp_min(0).sqrt()
    for left in range(task_count):
        for right in range(left + 1, task_count):
            denominator = float(norms[left] * norms[right])
            cosine_by_pair[f"{source_names[left]}__{source_names[right]}"] = (
                float(gram[left, right]) / denominator
                if denominator > 1e-12
                else float("nan")
            )

    used_fallback = False
    solved_alpha = None
    try:
        alpha, residual, projections = _solve_nash_mtl_coefficients(
            gram,
            max_iter=max_iter,
            initial_alpha=initial_alpha if task_count == 3 else None,
        )
        solved_alpha = alpha.clone()
    except (RuntimeError, ValueError):
        # A failed Newton solve must never produce an optimizer update. Average
        # unit task gradients instead, preserving equal directional influence.
        active = norms > 1e-12
        active_count = int(active.sum())
        if active_count == 0:
            alpha = torch.zeros_like(norms)
        else:
            alpha = torch.where(
                active,
                1.0 / (active_count * norms.clamp_min(1e-12)),
                torch.zeros_like(norms),
            )
        residual = float("nan")
        projections = torch.full_like(alpha, float("nan"))
        used_fallback = True
    combined = [
        sum(
            float(alpha[source_index]) * source_gradients[source_index][parameter_index]
            for source_index in range(task_count)
        )
        for parameter_index in range(len(parameters))
    ]
    alpha_by_source = {
        source: float(alpha[index]) for index, source in enumerate(source_names)
    }
    return (
        combined,
        alpha_by_source,
        residual,
        float(projections.min()),
        cosine_by_pair,
        used_fallback,
        solved_alpha,
    )


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
            source_r2_keys = [
                key
                for key in row
                if (
                    key.startswith("val_rollout_source_")
                    or key.startswith("val_ae_source_")
                )
                and "_p_ratio_r2" in key
                and "_endpoint_" not in key
            ]
            source_order = {
                "reid": 0,
                "depablo_low_temp": 1,
                "depablo_mixed_temp": 2,
                "lj_noisy": 3,
            }

            def source_r2_label(key: str) -> tuple[int, str]:
                prefix = (
                    "val_rollout_source_"
                    if key.startswith("val_rollout_source_")
                    else "val_ae_source_"
                )
                source = key.removeprefix(prefix).split("_p_ratio_r2", 1)[0]
                label = {
                    "reid": "Reid",
                    "depablo_low_temp": "low-T",
                    "depablo_mixed_temp": "mixed-T",
                    "lj_noisy": "noisy-LJ",
                }.get(source, source.replace("_", "-"))
                return source_order.get(source, len(source_order)), label

            source_r2_keys.sort(key=lambda key: source_r2_label(key)[0])
            if source_r2_keys:
                callback_text = " ".join(
                    f"{source_r2_label(key)[1]} r2={row[key]:.4g}"
                    for key in source_r2_keys
                )
            else:
                callback_keys = [
                    key
                    for key in row
                    if key.startswith(("val_rollout_", "val_ae_"))
                ]
                callback_text = " ".join(
                    f"{key}={row[key]:.4g}"
                    for key in callback_keys
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


_SOURCE_CONTEXT_ORDER = (
    "reid",
    "depablo_low_temp",
    "depablo_mixed_temp",
    "lj_noisy",
)


def encode_reference_context(
    ae_model,
    sim,
    *,
    pos_dim: int,
    normalizers: dict[str, torch.Tensor],
    device,
    include_temperature: bool = False,
    include_source_id: bool = False,
    pool_mode: str = "mean",
) -> torch.Tensor:
    """Pool the learned reference-node representation into static network context."""
    ref_graph = sim[0]
    ref_pos = reference_positions_for_model(
        ref_graph, pos_dim=pos_dim, device=device
    )
    edge_mode = canonical_edge_mode(getattr(ae_model, "edge_mode", "stored"))
    if edge_mode == "complete":
        edge_index, _, ref_edge_attr = undirected_complete_graph_edge_data(
            ref_graph, ref_graph, pos_dim=pos_dim, device=device
        )
    elif edge_mode in {"stored", "compact_stored", "compact_delta_stored"}:
        edge_index = ref_graph.edge_index.to(device).long()
        if edge_mode == "compact_stored":
            ref_edge_attr = compact_reference_edge_features(ref_graph, pos_dim=pos_dim, device=device)
        elif edge_mode == "compact_delta_stored":
            ref_edge_attr = compact_delta_reference_edge_features(ref_graph, pos_dim=pos_dim, device=device)
        else:
            ref_edge_attr = reference_edge_features(ref_graph, pos_dim=pos_dim, device=device)
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
    if pool_mode in {"source_id", "source", "dataset_id"}:
        context = torch.empty(0, dtype=h0.dtype, device=h0.device)
    elif pool_mode in {"learned_attention", "attention", "set_attention"}:
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
    if include_source_id:
        source_name = str(getattr(ref_graph, "source_name", ""))
        source_id = torch.zeros(
            len(_SOURCE_CONTEXT_ORDER), dtype=context.dtype, device=context.device
        )
        if source_name in _SOURCE_CONTEXT_ORDER:
            source_id[_SOURCE_CONTEXT_ORDER.index(source_name)] = 1.0
        if context.ndim == 2:
            context = torch.cat(
                [context, source_id.reshape(1, -1).expand(context.size(0), -1)],
                dim=-1,
            )
        else:
            context = torch.cat([context, source_id], dim=0)
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
    edge_mode = canonical_edge_mode(
        edge_mode or getattr(ae_model, "edge_mode", "stored")
    )
    if edge_mode == "complete":
        edge_index, edge_attr, ref_edge_attr = undirected_complete_graph_edge_data(
            ref_graph, cur_graph, pos_dim=pos_dim, device=device
        )
    elif edge_mode in {"stored", "compact_stored", "compact_delta_stored"}:
        edge_index = ref_graph.edge_index.to(device).long()
        if edge_mode == "compact_stored":
            edge_attr = compact_edge_features(
                ref_graph, cur_graph, pos_dim=pos_dim, device=device
            )
            ref_edge_attr = compact_reference_edge_features(
                ref_graph, pos_dim=pos_dim, device=device
            )
        elif edge_mode == "compact_delta_stored":
            edge_attr = compact_delta_edge_features(ref_graph, cur_graph, pos_dim=pos_dim, device=device)
            ref_edge_attr = compact_delta_reference_edge_features(ref_graph, pos_dim=pos_dim, device=device)
        else:
            edge_attr = edge_features(
                ref_graph, cur_graph, pos_dim=pos_dim, device=device
            )
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
    context_include_source_id: bool = False,
    context_pool_mode: str = "mean",
    rho_scale_mode: str | None = None,
) -> LatentNormalizer:
    z_chunks = []
    z_next_chunks = []
    dz_chunks = []
    context_chunks = []
    rho_scale_chunks = []
    static_context_by_sim: dict[int, torch.Tensor] = {}
    latent_by_frame: dict[tuple[int, int], torch.Tensor] = {}
    for rows_batch in iter_batches(rows, batch_graphs, shuffle=False):
        z0_values, z1_values = [], []
        with torch.no_grad():
            for row in rows_batch:
                sim_idx, t0, t1 = int(row[0]), int(row[1]), int(row[2])
                for frame in (t0, t1):
                    key = (sim_idx, frame)
                    if key not in latent_by_frame:
                        latent_by_frame[key] = encode_frame_latent(
                            ae_model,
                            sims[sim_idx],
                            frame,
                            pos_dim=pos_dim,
                            node_feature_mode=node_feature_mode,
                            normalizers=normalizers,
                            device=device,
                        ).detach()
                z0_values.append(latent_by_frame[(sim_idx, t0)])
                z1_values.append(latent_by_frame[(sim_idx, t1)])
        z0 = torch.stack(z0_values, dim=0)
        z1 = torch.stack(z1_values, dim=0)
        z_chunks.append(z0.detach())
        z_next_chunks.append(z1.detach())
        dz_chunks.append((z1 - z0).detach())
        if use_static_context and str(context_pool_mode).lower() not in {
            "learned_attention",
            "attention",
            "set_attention",
        }:
            for row in rows_batch:
                sim_idx = int(row[0])
                if sim_idx not in static_context_by_sim:
                    static_context_by_sim[sim_idx] = encode_reference_context(
                        ae_model,
                        sims[sim_idx],
                        pos_dim=pos_dim,
                        normalizers=normalizers,
                        device=device,
                        include_temperature=context_include_temperature,
                        include_source_id=context_include_source_id,
                        pool_mode=context_pool_mode,
                    ).detach()
                # Keep the original transition-weighted statistics while
                # avoiding an identical AE reference encoding for every row.
                context_chunks.append(static_context_by_sim[sim_idx])
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
    context_is_encoded: bool = False,
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
    context_norm = (
        context
        if context_is_encoded
        else stats.normalize_context(
            None if context is None else context.unsqueeze(0)
        )
    )
    rho_scale_norm = stats.normalize_rho_scale(
        None if rho_scale is None else rho_scale.reshape(1, 1)
    )
    if getattr(model, "uses_rho_progress_scale", False):
        if context_is_encoded:
            pred = model(
                z_norm,
                context_norm,
                rho_scale=rho_scale_norm,
                context_is_encoded=True,
            )
        else:
            pred = model(z_norm, context_norm, rho_scale=rho_scale_norm)
    else:
        if context_is_encoded:
            pred = model(z_norm, context_norm, context_is_encoded=True)
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
    prediction = model(
        state,
        context_norm,
        context_is_encoded=context_is_encoded,
    )
    if getattr(model, "predicts_direct_history_delta", False):
        return (z.unsqueeze(0) + prediction * dz_std).squeeze(0)
    next_velocity = velocity + prediction
    return (z.unsqueeze(0) + next_velocity * dz_std).squeeze(0)


def latent_step_lagged_history(
    model,
    z: torch.Tensor,
    z_lagged: torch.Tensor,
    z_reference: torch.Tensor,
    stats: LatentNormalizer,
    *,
    frame_gap: int,
    context: torch.Tensor | None = None,
    context_is_encoded: bool = False,
) -> torch.Tensor:
    """Predict the full next-step latent displacement from rolling history."""

    gap = max(1, int(frame_gap))
    z_std = stats.z_std.to(z).clamp_min(1e-6)
    dz_std = stats.dz_std.to(z).clamp_min(1e-6)
    q = (z.unsqueeze(0) - z_reference.unsqueeze(0)) / z_std
    velocity = ((z - z_lagged) / gap).unsqueeze(0) / dz_std
    reference = stats.normalize_z(z_reference.unsqueeze(0))
    state = torch.cat([q, velocity, reference], dim=-1)
    if context_is_encoded:
        context_norm = context
    else:
        context_value = context
        if context_value is not None and context_value.ndim == 1:
            context_value = context_value.unsqueeze(0)
        context_norm = stats.normalize_context(context_value)
    predicted_dz_norm = model(
        state, context_norm, context_is_encoded=context_is_encoded
    )
    predicted_dz = stats.unnormalize_dz(predicted_dz_norm)
    return (z.unsqueeze(0) + predicted_dz).squeeze(0)


def latent_step_recurrent_memory(
    model,
    z: torch.Tensor,
    z_previous: torch.Tensor,
    memory: torch.Tensor,
    stats: LatentNormalizer,
    *,
    context: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance one step while updating a persistent recurrent latent memory."""

    z_batch = z.unsqueeze(0) if z.ndim == 1 else z
    previous_batch = z_previous.unsqueeze(0) if z_previous.ndim == 1 else z_previous
    context_value = context
    if context_value is not None and context_value.ndim == 1:
        context_value = context_value.unsqueeze(0)
    predicted_delta_norm, memory = model(
        stats.normalize_z(z_batch),
        (z_batch - previous_batch) / stats.dz_std.to(z_batch).clamp_min(1e-6),
        memory,
        stats.normalize_context(context_value),
    )
    next_z = z_batch + stats.unnormalize_dz(predicted_delta_norm)
    return (next_z.squeeze(0) if z.ndim == 1 else next_z), memory


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
    progress: float | torch.Tensor | None = None,
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
    if bool(getattr(model, "include_progress", False)):
        if progress is None:
            raise ValueError("This fixed-history model requires rollout progress.")
        progress_value = torch.as_tensor(
            progress, dtype=state.dtype, device=state.device
        )
        if progress_value.ndim == 0:
            progress_value = progress_value.reshape(1, 1)
        elif progress_value.ndim == 1:
            progress_value = progress_value.reshape(-1, 1)
        state = torch.cat([state, progress_value.expand(state.size(0), 1)], dim=-1)
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


def latent_step_fixed_window(
    model,
    z: torch.Tensor,
    observed_latents: list[torch.Tensor] | tuple[torch.Tensor, ...],
    stats: LatentNormalizer,
    *,
    context: torch.Tensor | None = None,
    context_is_encoded: bool = False,
    progress: float | torch.Tensor | None = None,
    observed_frame_gap: int | None = None,
) -> torch.Tensor:
    """Advance from a fixed observed latent window using all frames directly."""

    if not getattr(model, "uses_fixed_window_history", False):
        raise ValueError(f"{model.__class__.__name__} is not a fixed-window propagator.")
    expected = int(getattr(model, "fixed_history_size", len(observed_latents)))
    if len(observed_latents) != expected:
        raise ValueError(f"Expected {expected} observed latents, received {len(observed_latents)}.")
    state = torch.cat(
        [stats.normalize_z(z.unsqueeze(0))]
        + [stats.normalize_z(observed.unsqueeze(0)) for observed in observed_latents],
        dim=-1,
    )
    if bool(getattr(model, "include_progress", False)):
        if progress is None:
            raise ValueError("This fixed-window model requires rollout progress.")
        progress_value = torch.as_tensor(progress, dtype=state.dtype, device=state.device)
        if progress_value.ndim == 0:
            progress_value = progress_value.reshape(1, 1)
        elif progress_value.ndim == 1:
            progress_value = progress_value.reshape(-1, 1)
        state = torch.cat([state, progress_value.expand(state.size(0), 1)], dim=-1)
    velocity_residual = bool(getattr(model, "uses_fixed_velocity_residual", False))
    observed_velocity = torch.zeros_like(z)
    if velocity_residual:
        frame_gap = max(int(observed_frame_gap or 1), 1)
        observed_velocity = (observed_latents[-1] - observed_latents[-2]) / frame_gap
    if context_is_encoded:
        context_norm = context
    else:
        context_value = context.unsqueeze(0) if context is not None and context.ndim == 1 else context
        context_norm = stats.normalize_context(context_value)
    predicted_delta_norm = model(state, context_norm, context_is_encoded=context_is_encoded)
    predicted_residual = (
        predicted_delta_norm * stats.dz_std.to(z).clamp_min(1e-6)
        if velocity_residual
        else stats.unnormalize_dz(predicted_delta_norm)
    ).squeeze(0)
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

    # With equal budgets, use a fixed round-robin source cycle. Source rows
    # themselves are shuffled above, while the fixed cycle guarantees that
    # every contiguous training batch differs by at most one row per source.
    source_lengths = {len(source_rows) for source_rows in prepared.values()}
    if len(source_lengths) == 1:
        sources = sorted(prepared)
        mixed_rows = []
        for row_order in range(next(iter(source_lengths))):
            mixed_rows.extend(prepared[source][row_order] for source in sources)
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
    gradient_method: str = "mean",
    nash_max_iter: int = 50,
    nash_state: dict | None = None,
    optimizer=None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    gradient_method = str(gradient_method).strip().lower()
    if gradient_method not in {"mean", "standard", "nash", "nash_mtl", "nash-mtl"}:
        raise ValueError("gradient_method must be 'mean' or 'nash_mtl'.")
    losses = []
    source_reconstruction: dict[str, list[float]] = {}
    nash_alphas: dict[str, list[float]] = {}
    nash_cosines: dict[str, list[float]] = {}
    nash_residuals: list[float] = []
    nash_min_projections: list[float] = []
    nash_fallbacks: list[float] = []
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
        per_node_loss = squared_error.mean(dim=-1)
        per_graph_loss = torch.stack(
            [
                per_node_loss[batch_data["batch"] == graph_index].mean()
                for graph_index in range(len(rows))
            ]
        )
        source_graph_indices: dict[str, list[int]] = {}
        for graph_index, row in enumerate(rows):
            source_name = str(
                getattr(sims[int(row[0])][0], "source_name", "unknown")
            )
            source_graph_indices.setdefault(source_name, []).append(graph_index)
        source_losses = {
            source_name: per_graph_loss[graph_indices].mean()
            for source_name, graph_indices in source_graph_indices.items()
        }
        for source_name, source_loss in source_losses.items():
            source_reconstruction.setdefault(source_name, []).append(
                float(
                    per_graph_loss[source_graph_indices[source_name]].mean().detach()
                )
            )
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            if gradient_method in {"nash", "nash_mtl", "nash-mtl"} and len(source_losses) > 1:
                parameters = [
                    parameter for parameter in model.parameters() if parameter.requires_grad
                ]
                source_order = tuple(sorted(source_losses))
                warm_alpha = None
                if (
                    nash_state is not None
                    and nash_state.get("sources") == source_order
                ):
                    warm_alpha = nash_state.get("alpha")
                (
                    combined,
                    alpha_by_source,
                    residual,
                    min_projection,
                    cosine_by_pair,
                    used_fallback,
                    solved_alpha,
                ) = (
                    _nash_mtl_combined_gradients(
                        source_losses,
                        parameters,
                        max_iter=nash_max_iter,
                        initial_alpha=warm_alpha,
                    )
                )
                if nash_state is not None and solved_alpha is not None:
                    nash_state["sources"] = source_order
                    nash_state["alpha"] = solved_alpha
                for parameter, gradient in zip(parameters, combined):
                    parameter.grad = gradient
                for source_name, alpha in alpha_by_source.items():
                    nash_alphas.setdefault(source_name, []).append(alpha)
                for pair, cosine in cosine_by_pair.items():
                    nash_cosines.setdefault(pair, []).append(cosine)
                if np.isfinite(residual):
                    nash_residuals.append(residual)
                if np.isfinite(min_projection):
                    nash_min_projections.append(min_projection)
                nash_fallbacks.append(float(used_fallback))
            else:
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
    summary = {
        key: float(np.mean([row[key] for row in losses])) for key in losses[0]
    }
    for source_name, values in source_reconstruction.items():
        source_key = "".join(
            character if character.isalnum() else "_"
            for character in source_name.lower()
        ).strip("_")
        summary[f"source_{source_key}_reconstruction"] = float(np.mean(values))
    for source_name, values in nash_alphas.items():
        source_key = "".join(
            character if character.isalnum() else "_"
            for character in source_name.lower()
        ).strip("_")
        summary[f"nash_alpha_{source_key}"] = float(np.mean(values))
    for pair, values in nash_cosines.items():
        pair_key = "__".join(
            "".join(
                character if character.isalnum() else "_"
                for character in source_name.lower()
            ).strip("_")
            for source_name in pair.split("__")
        )
        finite_values = [value for value in values if np.isfinite(value)]
        summary[f"nash_cosine_{pair_key}"] = (
            float(np.mean(finite_values)) if finite_values else float("nan")
        )
    if nash_residuals:
        summary["nash_residual"] = float(np.mean(nash_residuals))
    if nash_min_projections:
        summary["nash_min_task_projection"] = float(
            np.mean(nash_min_projections)
        )
    if nash_fallbacks:
        summary["nash_fallback_rate"] = float(np.mean(nash_fallbacks))
    return summary


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
    gradient_method: str = "mean",
    nash_max_iter: int = 50,
    epoch_callback: Callable | None = None,
    selection_metric_key: str | None = None,
    selection_mode: str = "min",
    verbose: bool = True,
) -> TrainingResult:
    """Train and restore the best latent autoencoder."""

    nash_state: dict = {}
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
        "gradient_method": gradient_method,
        "nash_max_iter": nash_max_iter,
        "nash_state": nash_state,
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
        epoch_callback=epoch_callback,
        selection_metric_key=selection_metric_key,
        selection_mode=selection_mode,
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
    context_include_source_id: bool = False,
    context_pool_mode: str = "mean",
    rho_scale_mode: str | None = None,
    max_progress_frame: int | None = None,
    physics_config: PhysicsLossConfig | None = None,
    source_loss_reduction: str = "pooled",
    use_pcgrad: bool = False,
    mix_sources: bool = False,
    source_classification_weight: float = 0.0,
    optimizer=None,
    **_unused,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    losses = []
    raw_losses = []
    physics_logs = []
    source_loss_logs: dict[str, list[float]] = {}
    source_accuracy_logs: dict[str, list[float]] = {}
    classification_losses = []
    classification_accuracies = []
    ae_target_mode = ae_target_mode or node_feature_mode
    loss_mode = str(loss_mode).lower()
    source_loss_reduction = str(source_loss_reduction).lower()
    if source_loss_reduction not in {"pooled", "equal", "per_source_mean"}:
        raise ValueError(
            "source_loss_reduction must be 'pooled' or 'equal'."
        )
    if use_pcgrad and physics_config is not None:
        raise ValueError("PCGrad is not implemented for physics losses.")
    if getattr(model, "requires_next_z_loss", False) and loss_mode not in {
        "next_z",
        "jepa",
        "next_embedding",
    }:
        raise ValueError(
            f"{model.__class__.__name__} predicts next z directly; use "
            "propagator_loss='next_z'."
        )
    epoch_rows = (
        _source_mixed_rows(sims, transition_rows, shuffle=True)
        if is_train and mix_sources
        else transition_rows
    )
    for rows in iter_batches(
        epoch_rows,
        batch_graphs,
        shuffle=is_train and not mix_sources,
    ):
        source_names = [
            str(getattr(sims[int(row[0])][0], "source_name", "unknown"))
            for row in rows
        ]
        batch_stats = stats
        z0, z1 = encode_transition_batch(
            ae_model,
            sims,
            rows,
            pos_dim=pos_dim,
            node_feature_mode=node_feature_mode,
            normalizers=normalizers,
            device=device,
        )
        z0_norm = batch_stats.normalize_z(z0)
        z0_used = z0
        if is_train and physics_config is not None and physics_config.latent_noise_std > 0:
            noise_norm = torch.randn_like(z0_norm) * float(physics_config.latent_noise_std)
            z0_norm = z0_norm + noise_norm
            z0_used = z0 + noise_norm * batch_stats.z_std
        context = None
        context_is_encoded = False
        if use_static_context:
            raw_contexts = [
                encode_reference_context(
                    ae_model,
                    sims[int(row[0])],
                    pos_dim=pos_dim,
                    normalizers=normalizers,
                    device=device,
                    include_temperature=context_include_temperature,
                    include_source_id=context_include_source_id,
                    pool_mode=context_pool_mode,
                )
                for row in rows
            ]
            if str(context_pool_mode).lower() in {
                "learned_attention",
                "attention",
                "set_attention",
            }:
                context = torch.cat(
                    [model.context_projection(raw_context) for raw_context in raw_contexts],
                    dim=0,
                )
                context_is_encoded = True
            else:
                context = batch_stats.normalize_context(
                    torch.stack(raw_contexts, dim=0)
                )
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
            rho_scale = batch_stats.normalize_rho_scale(raw_scale)
        if getattr(model, "uses_rho_progress_scale", False):
            pred = model(
                z0_norm,
                context,
                rho_scale=rho_scale,
                context_is_encoded=context_is_encoded,
            )
        else:
            pred = model(
                z0_norm,
                context,
                context_is_encoded=context_is_encoded,
            )
        if loss_mode in {"delta", "dz", "residual_delta", "hybrid_delta_next"}:
            target_norm = batch_stats.normalize_dz(z1 - z0)
            pred_norm = pred if getattr(model, "predicts_delta", False) else pred - z0_norm
            pred_raw = z0_used + batch_stats.unnormalize_dz(pred_norm)
            if loss_mode == "hybrid_delta_next":
                next_weight = float(getattr(model, "next_loss_weight", 0.1))
                next_pred_norm = batch_stats.normalize_z_next(pred_raw)
                next_target_norm = batch_stats.normalize_z_next(z1)
                per_row_loss = (pred_norm - target_norm).square().mean(dim=-1) + next_weight * (
                    next_pred_norm - next_target_norm
                ).square().mean(dim=-1)
            else:
                per_row_loss = (pred_norm - target_norm).square().mean(dim=-1)
        elif loss_mode in {"next_z", "jepa", "next_embedding"}:
            target_norm = batch_stats.normalize_z_next(z1)
            pred_norm = pred
            pred_raw = batch_stats.unnormalize_z_next(pred_norm)
            per_row_loss = (pred_norm - target_norm).square().mean(dim=-1)
        else:
            raise ValueError(f"Unknown propagator loss_mode: {loss_mode}")

        source_row_indices: dict[str, list[int]] = {}
        for row_index, source in enumerate(source_names):
            source_row_indices.setdefault(source, []).append(row_index)
        source_losses = {
            source: per_row_loss[indices].mean()
            for source, indices in source_row_indices.items()
        }
        if source_loss_reduction in {"equal", "per_source_mean"}:
            loss = torch.stack(list(source_losses.values())).mean()
        else:
            loss = per_row_loss.mean()

        if hasattr(model, "source_logits") and float(source_classification_weight) > 0:
            source_to_index = {name: index for index, name in enumerate(model.source_names)}
            class_targets = torch.tensor(
                [source_to_index[source] for source in source_names], device=z0.device, dtype=torch.long
            )
            class_logits = model.source_logits(
                z0_norm, context, context_is_encoded=context_is_encoded
            )
            classification_loss = F.cross_entropy(class_logits, class_targets)
            class_predictions = class_logits.argmax(dim=-1)
            class_accuracy = (class_predictions == class_targets).float()
            loss = loss + float(source_classification_weight) * classification_loss
            classification_losses.append(float(classification_loss.detach()))
            classification_accuracies.append(float(class_accuracy.mean().detach()))
            for source, indices in source_row_indices.items():
                source_accuracy_logs.setdefault(source, []).append(
                    float(class_accuracy[indices].mean().detach())
                )

        latent_supervised_loss = loss
        physics_values = None
        position_mse = torch.zeros((), device=z0.device, dtype=z0.dtype)
        if physics_config is not None:
            if source_loss_reduction in {"equal", "per_source_mean"}:
                raise ValueError(
                    "Equal per-source loss reduction is not implemented for physics losses."
                )
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
            if use_pcgrad and len(source_losses) > 1:
                parameters = [
                    parameter for parameter in model.parameters() if parameter.requires_grad
                ]
                combined_gradients = _pcgrad_combined_gradients(
                    source_losses, parameters
                )
                for parameter, gradient in zip(parameters, combined_gradients):
                    parameter.grad = gradient
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        losses.append(float(loss.item()))
        raw_losses.append(float(F.mse_loss(pred_raw, z1).item()))
        for source, source_loss in source_losses.items():
            source_loss_logs.setdefault(source, []).append(float(source_loss.detach()))
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
    for source, values in source_loss_logs.items():
        source_key = "".join(
            character if character.isalnum() else "_"
            for character in source.lower()
        ).strip("_")
        result[f"source_{source_key}_loss_norm"] = float(np.mean(values))
    if classification_losses:
        result["source_classification_loss"] = float(np.mean(classification_losses))
        result["source_classification_accuracy"] = float(np.mean(classification_accuracies))
    for source, values in source_accuracy_logs.items():
        source_key = "".join(
            character if character.isalnum() else "_"
            for character in source.lower()
        ).strip("_")
        result[f"source_{source_key}_classification_accuracy"] = float(np.mean(values))
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
    context_include_source_id: bool = False,
    rho_scale_mode: str | None = None,
    max_progress_frame: int | None = None,
    physics_config: PhysicsLossConfig | None = None,
    optimizer=None,
    **_unused,
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
                        include_source_id=context_include_source_id,
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
    context_include_source_id: bool = False,
    rho_scale_mode: str | None = None,
    max_progress_frame: int | None = None,
    optimizer=None,
    **_unused,
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
                    include_source_id=context_include_source_id,
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
    context_include_source_id: bool = False,
    context_pool_mode: str,
    fixed_observed_frames: tuple[int, ...] | None = None,
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
            observed_frames = tuple(int(frame) for frame in fixed_observed_frames)
            first, last = observed_frames[0], observed_frames[-1]
            required[sim_idx].update(range(first, last + 1))
            required[sim_idx].add(
                _nth_previous_filtered_frame(
                    sims[sim_idx],
                    start_frame,
                    frame_skip=frame_skip,
                    steps=max(1, last - first),
                )
            )
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
                    include_source_id=context_include_source_id,
                    pool_mode=context_pool_mode,
                ).detach()
    return latent_cache, context_cache


def _load_or_precompute_frozen_latents(
    cache_path: str | Path | None,
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
    context_include_source_id: bool,
    context_pool_mode: str,
    fixed_observed_frames: tuple[int, ...] | None,
) -> tuple[dict[tuple[int, int], torch.Tensor], dict[int, torch.Tensor]]:
    """Load fixed AE features, or encode all frames once and persist them."""
    expected = [(len(sim), str(getattr(sim[0], "source_name", "unknown"))) for sim in sims]
    feature_signature = {
        "pos_dim": int(pos_dim),
        "node_feature_mode": str(node_feature_mode),
        "frame_skip": int(frame_skip),
        "use_static_context": bool(use_static_context),
        "context_include_temperature": bool(context_include_temperature),
        "context_include_source_id": bool(context_include_source_id),
        "context_pool_mode": str(context_pool_mode),
        "fixed_observed_frames": tuple(fixed_observed_frames or ()),
    }
    path = Path(cache_path) if cache_path is not None else None
    if path is not None and path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if (
            payload.get("sim_signature") == expected
            and payload.get("feature_signature") == feature_signature
        ):
            print(f"loaded frozen AE features: {path}")
            latent_cache = {
                tuple(key): value.to(device)
                for key, value in payload["latents"].items()
            }
            context_cache = {
                int(key): value.to(device)
                for key, value in payload["contexts"].items()
            }
            return latent_cache, context_cache
        print(f"ignoring incompatible frozen AE feature cache: {path}")

    if path is None:
        return _precompute_kinematic_latents(
            ae_model, sims, rows, pos_dim=pos_dim,
            node_feature_mode=node_feature_mode, normalizers=normalizers,
            device=device, frame_skip=frame_skip,
            use_static_context=use_static_context,
            context_include_temperature=context_include_temperature,
            context_include_source_id=context_include_source_id,
            context_pool_mode=context_pool_mode,
            fixed_observed_frames=fixed_observed_frames,
        )

    print(f"encoding and caching all frozen AE frames: {path}")
    all_rows = [
        (sim_idx, 0, tuple(range(len(sim))))
        for sim_idx, sim in enumerate(sims)
    ]
    latent_cache, context_cache = _precompute_kinematic_latents(
        ae_model, sims, all_rows, pos_dim=pos_dim,
        node_feature_mode=node_feature_mode, normalizers=normalizers,
        device=device, frame_skip=frame_skip,
        use_static_context=use_static_context,
        context_include_temperature=context_include_temperature,
        context_include_source_id=context_include_source_id,
        context_pool_mode=context_pool_mode,
        fixed_observed_frames=fixed_observed_frames,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "sim_signature": expected,
            "feature_signature": feature_signature,
            "latents": {key: value.detach().cpu() for key, value in latent_cache.items()},
            "contexts": {key: value.detach().cpu() for key, value in context_cache.items()},
        },
        path,
    )
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
    max_progress_frame: int | None = None,
    mix_sources: bool = False,
    source_loss_reduction: str = "pooled",
    history_noise_std: float = 0.0,
    truncated_rollout_horizon: int | None = None,
    optimizer=None,
    **_unused,
) -> dict[str, float]:
    """Supervise a fully autoregressive latent unroll.

    When ``truncated_rollout_horizon`` is set, retain gradients only for the
    first and final calls. Intermediate autoregressive states are detached.
    This pairs a local next-step loss with an endpoint-recovery loss without
    backpropagating through the intervening rollout.
    """

    del node_feature_mode
    is_train = optimizer is not None
    model.train(is_train)
    source_loss_reduction = str(source_loss_reduction).lower()
    if source_loss_reduction not in {"pooled", "equal", "per_source_mean"}:
        raise ValueError("source_loss_reduction must be 'pooled' or 'equal'.")
    history_noise_std = float(history_noise_std)
    if history_noise_std < 0:
        raise ValueError("history_noise_std must be non-negative.")
    unroll_steps = int(unroll_steps)
    if truncated_rollout_horizon is not None:
        truncated_rollout_horizon = int(truncated_rollout_horizon)
        if truncated_rollout_horizon != unroll_steps or unroll_steps < 2:
            raise ValueError(
                "truncated_rollout_horizon must equal the multistep unroll length and be at least 2."
            )
    losses, raw_losses, position_losses = [], [], []
    first_step_losses, endpoint_losses = [], []
    source_loss_logs: dict[str, list[float]] = {}
    source_first_step_logs: dict[str, list[float]] = {}
    source_endpoint_logs: dict[str, list[float]] = {}

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
        row_losses, row_raw_losses, row_sources = [], [], []
        can_batch_fixed_history = (
            unroll_steps == 1
            and getattr(model, "uses_fixed_observed_state", False)
            and fixed_observed_frames is not None
            and float(position_loss_weight) == 0.0
            and str(context_pool_mode).lower()
            not in {"learned_attention", "attention", "set_attention"}
        )
        if can_batch_fixed_history:
            valid_rows = [row for row in batch_rows if len(row[2]) >= 1]
            if not valid_rows:
                continue
            sim_indices = [int(row[0]) for row in valid_rows]
            batch_stats = stats
            start_frames = [int(row[1]) for row in valid_rows]
            target_frames = [int(row[2][0]) for row in valid_rows]
            z = torch.stack(
                [
                    latent_cache[(sim_idx, start_frame)]
                    for sim_idx, start_frame in zip(sim_indices, start_frames)
                ],
                dim=0,
            )
            observed_window = [
                torch.stack(
                    [latent_cache[(sim_idx, int(frame))] for sim_idx in sim_indices],
                    dim=0,
                )
                for frame in fixed_observed_frames
            ]
            true_z = torch.stack(
                [
                    latent_cache[(sim_idx, target_frame)]
                    for sim_idx, target_frame in zip(sim_indices, target_frames)
                ],
                dim=0,
            )
            if is_train:
                noise_scale = 0.25 * batch_stats.dz_std.to(z).clamp_min(1e-6)
                z_input = z + torch.randn_like(z) * noise_scale
                observed_window_input = [
                    observed + torch.randn_like(observed) * noise_scale
                    for observed in observed_window
                ]
            else:
                z_input = z
                observed_window_input = observed_window
            window_history = bool(getattr(model, "uses_fixed_window_history", False))
            velocity_residual = bool(getattr(model, "uses_fixed_velocity_residual", False))
            if window_history:
                if velocity_residual:
                    observed_velocity = (
                        observed_window_input[-1] - observed_window_input[-2]
                    ) / max(
                        int(fixed_observed_frames[-1])
                        - int(fixed_observed_frames[-2]),
                        1,
                    )
                else:
                    observed_velocity = torch.zeros_like(z_input)
                state = torch.cat(
                    [batch_stats.normalize_z(z_input)]
                    + [batch_stats.normalize_z(observed) for observed in observed_window_input],
                    dim=-1,
                )
            elif velocity_residual:
                observed_first_input, observed_second_input = observed_window_input[:2]
                observed_velocity = (observed_second_input - observed_first_input) / max(
                    int(fixed_observed_frames[1])
                    - int(fixed_observed_frames[0]),
                    1,
                )
                state = torch.cat(
                    [
                        batch_stats.normalize_z(z_input),
                        batch_stats.normalize_z(observed_second_input),
                        observed_velocity / batch_stats.dz_std.to(z).clamp_min(1e-6),
                    ],
                    dim=-1,
                )
            else:
                observed_first_input, observed_second_input = observed_window_input[:2]
                observed_velocity = torch.zeros_like(z_input)
                state = torch.cat(
                    [
                        batch_stats.normalize_z(z_input),
                        batch_stats.normalize_z(observed_first_input),
                        batch_stats.normalize_z(observed_second_input),
                    ],
                    dim=-1,
                )
            if bool(getattr(model, "include_progress", False)):
                progress = torch.tensor(
                    [
                        float(target_frame)
                        / max(1, int(max_progress_frame or 1))
                        for target_frame in target_frames
                    ],
                    dtype=state.dtype,
                    device=state.device,
                ).reshape(-1, 1)
                state = torch.cat([state, progress], dim=-1)
            encoded_context = None
            if use_static_context:
                context_values = torch.stack(
                    [context_cache[sim_idx] for sim_idx in sim_indices], dim=0
                )
                encoded_context = model.encode_context(
                    batch_stats.normalize_context(context_values)
                )
            predicted_delta_norm = model(
                state,
                encoded_context,
                context_is_encoded=encoded_context is not None,
            )
            predicted_delta = (
                predicted_delta_norm * batch_stats.dz_std.to(z)
                if velocity_residual
                else batch_stats.unnormalize_dz(predicted_delta_norm)
            )
            z_next = z + observed_velocity + predicted_delta
            # Fixed-history models predict an increment.  Use the same
            # source-aware delta-Z scale as the plain one-step propagator:
            # raw latent coordinates from a frozen AE can be very small,
            # making their unnormalised MSE appear as zero and starving the
            # learned history encoder of a useful training signal.
            delta_scale = batch_stats.dz_std.to(z).clamp_min(1e-6)
            per_row_loss = (
                ((z_next - z) / delta_scale - (true_z - z) / delta_scale)
                .square()
                .mean(dim=-1)
            )
            if float(network_variation_weight) > 0:
                variation_scale = torch.stack(
                    [frame_variation[target].to(z) for target in target_frames],
                    dim=0,
                ).clamp_min(1e-6)
                per_row_loss = per_row_loss + float(network_variation_weight) * (
                    (z_next - true_z) / variation_scale
                ).square().mean(dim=-1)
            loss = per_row_loss.mean()
            raw_loss = (z_next - true_z).square().mean(dim=-1).mean()
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
            raw_losses.append(float(raw_loss.detach().cpu()))
            continue
        can_batch_lagged_history = (
            unroll_steps == 1
            and getattr(model, "uses_lagged_history", False)
            and float(position_loss_weight) == 0.0
            and str(context_pool_mode).lower()
            not in {"learned_attention", "attention", "set_attention"}
        )
        if can_batch_lagged_history:
            valid_rows = [row for row in batch_rows if len(row[2]) >= 1]
            if not valid_rows:
                continue
            sim_indices = [int(row[0]) for row in valid_rows]
            start_frames = [int(row[1]) for row in valid_rows]
            target_frames = [int(row[2][0]) for row in valid_rows]
            history_gap = max(
                1,
                int(fixed_observed_frames[1])
                - int(fixed_observed_frames[0]),
            )
            lagged_frames = [
                _nth_previous_filtered_frame(
                    sims[i], t, frame_skip=frame_skip, steps=history_gap
                )
                for i, t in zip(sim_indices, start_frames)
            ]
            z = torch.stack(
                [latent_cache[(i, t)] for i, t in zip(sim_indices, start_frames)]
            )
            z_lagged = torch.stack(
                [latent_cache[(i, t)] for i, t in zip(sim_indices, lagged_frames)]
            )
            z_reference = torch.stack(
                [latent_cache[(i, 0)] for i in sim_indices]
            )
            true_z = torch.stack(
                [latent_cache[(i, t)] for i, t in zip(sim_indices, target_frames)]
            )
            z_std = stats.z_std.to(z).clamp_min(1e-6)
            dz_std = stats.dz_std.to(z).clamp_min(1e-6)
            if is_train:
                noise_scale = 0.25 * dz_std
                z_input = z + torch.randn_like(z) * noise_scale
                z_lagged_input = (
                    z_lagged + torch.randn_like(z_lagged) * noise_scale
                )
            else:
                z_input = z
                z_lagged_input = z_lagged
            velocity = (
                (z_input - z_lagged_input) / history_gap
            ) / dz_std
            state = torch.cat(
                [
                    (z_input - z_reference) / z_std,
                    velocity,
                    stats.normalize_z(z_reference),
                ],
                dim=-1,
            )
            encoded_context = None
            if use_static_context:
                context_values = torch.stack(
                    [context_cache[i] for i in sim_indices], dim=0
                )
                encoded_context = model.encode_context(
                    stats.normalize_context(context_values)
                )
            predicted_dz_norm = model(
                state,
                encoded_context,
                context_is_encoded=encoded_context is not None,
            )
            predicted_dz = stats.unnormalize_dz(predicted_dz_norm)
            target_dz = true_z - z
            # Direct displacement prediction: noisy history inputs, clean
            # displacement target, and raw latent MSE only.
            per_row_loss = (predicted_dz - target_dz).square().mean(dim=-1)
            loss = per_row_loss.mean()
            raw_loss = (predicted_dz - target_dz).square().mean()
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
            raw_losses.append(float(raw_loss.detach().cpu()))
            continue
        can_batch_rolling_history = (
            unroll_steps == 1
            and getattr(model, "uses_history_state", False)
            and float(position_loss_weight) == 0.0
            and str(context_pool_mode).lower()
            not in {"learned_attention", "attention", "set_attention"}
        )
        if can_batch_rolling_history:
            valid_rows = [row for row in batch_rows if len(row[2]) >= 1]
            if not valid_rows:
                continue
            sim_indices = [int(row[0]) for row in valid_rows]
            start_frames = [int(row[1]) for row in valid_rows]
            target_frames = [int(row[2][0]) for row in valid_rows]
            previous_frames = [
                _previous_filtered_frame(
                    sims[sim_idx], start_frame, frame_skip=frame_skip
                )
                for sim_idx, start_frame in zip(sim_indices, start_frames)
            ]
            previous_previous_frames = [
                _previous_filtered_frame(
                    sims[sim_idx], previous_frame, frame_skip=frame_skip
                )
                for sim_idx, previous_frame in zip(sim_indices, previous_frames)
            ]
            z = torch.stack(
                [latent_cache[(i, t)] for i, t in zip(sim_indices, start_frames)]
            )
            z_previous = torch.stack(
                [latent_cache[(i, t)] for i, t in zip(sim_indices, previous_frames)]
            )
            z_previous_previous = torch.stack(
                [
                    latent_cache[(i, t)]
                    for i, t in zip(sim_indices, previous_previous_frames)
                ]
            )
            z_reference = torch.stack(
                [latent_cache[(i, 0)] for i in sim_indices]
            )
            true_z = torch.stack(
                [latent_cache[(i, t)] for i, t in zip(sim_indices, target_frames)]
            )
            z_std = stats.z_std.to(z).clamp_min(1e-6)
            dz_std = stats.dz_std.to(z).clamp_min(1e-6)
            velocity = (z - z_previous) / dz_std
            previous_velocity = (z_previous - z_previous_previous) / dz_std
            state = torch.cat(
                [
                    (z - z_reference) / z_std,
                    velocity,
                    velocity - previous_velocity,
                    stats.normalize_z(z_reference),
                ],
                dim=-1,
            )
            encoded_context = None
            if use_static_context:
                context_values = torch.stack(
                    [context_cache[i] for i in sim_indices], dim=0
                )
                encoded_context = model.encode_context(
                    stats.normalize_context(context_values)
                )
            prediction = model(
                state,
                encoded_context,
                context_is_encoded=encoded_context is not None,
            )
            if getattr(model, "predicts_direct_history_delta", False):
                z_next = z + prediction * dz_std
            else:
                z_next = z + (velocity + prediction) * dz_std
            per_row_loss = (
                ((z_next - z) / dz_std - (true_z - z) / dz_std)
                .square()
                .mean(dim=-1)
            )
            if float(network_variation_weight) > 0:
                variation_scale = torch.stack(
                    [frame_variation[target].to(z) for target in target_frames]
                ).clamp_min(1e-6)
                per_row_loss = per_row_loss + float(network_variation_weight) * (
                    (z_next - true_z) / variation_scale
                ).square().mean(dim=-1)
            source_indices: dict[str, list[int]] = {}
            for row_index, sim_idx in enumerate(sim_indices):
                source_name = str(
                    getattr(sims[sim_idx][0], "source_name", "unknown")
                )
                source_indices.setdefault(source_name, []).append(row_index)
            source_losses = {
                source_name: per_row_loss[indices].mean()
                for source_name, indices in source_indices.items()
            }
            if source_loss_reduction in {"equal", "per_source_mean"}:
                loss = torch.stack(list(source_losses.values())).mean()
            else:
                loss = per_row_loss.mean()
            raw_loss = (z_next - true_z).square().mean()
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            losses.append(float(loss.detach().cpu()))
            raw_losses.append(float(raw_loss.detach().cpu()))
            for source_name, indices in source_indices.items():
                source_loss_logs.setdefault(source_name, []).extend(
                    float(per_row_loss[index].detach().cpu()) for index in indices
                )
            continue
        for sim_idx, start_frame, target_frames in batch_rows:
            sim_idx, start_frame = int(sim_idx), int(start_frame)
            if len(target_frames) < unroll_steps:
                continue
            sim = sims[sim_idx]
            row_stats = stats
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
            if is_train and history_noise_std > 0:
                history_noise_scale = (
                    history_noise_std
                    * row_stats.dz_std.squeeze(0).to(z).clamp_min(1e-6)
                )
                z = z + torch.randn_like(z) * history_noise_scale
                z_previous = (
                    z_previous + torch.randn_like(z_previous) * history_noise_scale
                )
                z_previous_previous = (
                    z_previous_previous
                    + torch.randn_like(z_previous_previous) * history_noise_scale
                )
            lagged_history = None
            if getattr(model, "uses_lagged_history", False):
                if fixed_observed_frames is None:
                    raise ValueError(
                        "fixed_observed_frames are required by lagged history."
                    )
                history_gap = (
                    int(fixed_observed_frames[1])
                    - int(fixed_observed_frames[0])
                )
                history_frames = [
                    _nth_previous_filtered_frame(
                        sim,
                        start_frame,
                        frame_skip=frame_skip,
                        steps=steps,
                    )
                    for steps in range(history_gap, -1, -1)
                ]
                lagged_history = [
                    latent_cache[(sim_idx, frame)] for frame in history_frames
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
            z_start = z
            step_losses, step_raw_losses, weights = [], [], []
            first_step_loss = None
            endpoint_loss = None
            for offset in range(unroll_steps):
                target_frame = int(target_frames[offset])
                if getattr(model, "uses_lagged_history", False):
                    z_next = latent_step_lagged_history(
                        model,
                        z,
                        lagged_history[0],
                        z_reference,
                        row_stats,
                        frame_gap=len(lagged_history) - 1,
                        context=encoded_context,
                        context_is_encoded=encoded_context is not None,
                    )
                elif getattr(model, "uses_fixed_observed_state", False):
                    if fixed_observed_frames is None:
                        raise ValueError(
                            "fixed_observed_frames are required by the fixed-history model."
                        )
                    progress = (
                        float(target_frame) / max(1, int(max_progress_frame or 1))
                        if getattr(model, "include_progress", False)
                        else None
                    )
                    if getattr(model, "uses_fixed_window_history", False):
                        observed_window = [
                            latent_cache[(sim_idx, int(frame))]
                            for frame in fixed_observed_frames
                        ]
                        z_next = latent_step_fixed_window(
                            model, z, observed_window, row_stats,
                            context=encoded_context,
                            context_is_encoded=encoded_context is not None,
                            progress=progress,
                            observed_frame_gap=(
                                int(fixed_observed_frames[-1])
                                - int(fixed_observed_frames[-2])
                            ),
                        )
                    else:
                        observed_first = latent_cache[(sim_idx, int(fixed_observed_frames[0]))]
                        observed_second = latent_cache[(sim_idx, int(fixed_observed_frames[1]))]
                        z_next = latent_step_fixed_history(
                            model, z, observed_first, observed_second, row_stats,
                            observed_frame_gap=(int(fixed_observed_frames[1]) - int(fixed_observed_frames[0])),
                            context=encoded_context,
                            context_is_encoded=encoded_context is not None,
                            progress=progress,
                        )
                elif getattr(model, "uses_history_state", False):
                    z_next = latent_step_history(
                        model,
                        z,
                        z_previous,
                        z_previous_previous,
                        z_reference,
                        row_stats,
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
                        row_stats,
                        progress=progress,
                        context=encoded_context,
                        context_is_encoded=encoded_context is not None,
                    )
                true_z = latent_cache[(sim_idx, target_frame)]
                if truncated_rollout_horizon is not None:
                    if offset == 0:
                        delta_scale = row_stats.dz_std.squeeze(0).to(z).clamp_min(1e-6)
                        first_step_loss = F.mse_loss(
                            (z_next - z) / delta_scale,
                            (true_z - z) / delta_scale,
                        )
                    elif offset == truncated_rollout_horizon - 1:
                        delta_scale = row_stats.dz_std.squeeze(0).to(z).clamp_min(1e-6)
                        endpoint_loss = F.mse_loss(
                            (z_next - z_start) / delta_scale,
                            (true_z - z_start) / delta_scale,
                        )
                    if offset < truncated_rollout_horizon - 1:
                        z_next = z_next.detach()
                    if getattr(model, "uses_fixed_observed_state", False):
                        z = z_next
                    elif getattr(model, "uses_lagged_history", False):
                        lagged_history = [*lagged_history[1:], z_next]
                        z = z_next
                    else:
                        z_previous_previous, z_previous, z = z_previous, z, z_next
                    continue
                weight = 1.0 + offset / max(1, unroll_steps)
                if (
                    getattr(model, "uses_history_state", False)
                    or getattr(model, "uses_fixed_observed_state", False)
                    or getattr(model, "uses_lagged_history", False)
                ):
                    # The history model predicts motion, not an absolute
                    # coordinate. Train it on the increment at the natural
                    # delta-Z scale so small velocity errors are not hidden by
                    # the much larger overall latent range.
                    delta_scale = row_stats.dz_std.squeeze(0).to(z).clamp_min(1e-6)
                    predicted_delta = (z_next - z) / delta_scale
                    target_delta = (true_z - z) / delta_scale
                    step_loss = F.mse_loss(predicted_delta, target_delta)
                    # The mixed sources can have very different latent-motion
                    # scales.  Without this source/trajectory scale balancing,
                    # the largest-delta family dominates the shared dynamics
                    # objective and the smaller-motion families collapse to
                    # their mean increment.
                    motion_scale = target_delta.square().mean().detach().clamp_min(1e-3)
                    step_loss = step_loss / motion_scale
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
                elif getattr(model, "uses_lagged_history", False):
                    lagged_history = [*lagged_history[1:], z_next]
                    z = z_next
                else:
                    z_previous_previous, z_previous, z = z_previous, z, z_next
            if truncated_rollout_horizon is not None:
                if first_step_loss is None or endpoint_loss is None:
                    raise RuntimeError("Missing first-step or endpoint loss in truncated rollout.")
                row_losses.append(0.5 * (first_step_loss + endpoint_loss))
                row_raw_losses.append(F.mse_loss(z_next, true_z))
                first_step_losses.append(float(first_step_loss.detach().cpu()))
                endpoint_losses.append(float(endpoint_loss.detach().cpu()))
                source_name = str(
                    getattr(sim[0], "source_name", "unknown")
                )
                source_first_step_logs.setdefault(source_name, []).append(
                    float(first_step_loss.detach().cpu())
                )
                source_endpoint_logs.setdefault(source_name, []).append(
                    float(endpoint_loss.detach().cpu())
                )
            else:
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
                        "normalized_delta_velocity_acceleration",
                        "displacement_velocity_acceleration",
                        "kinematic_state",
                        "normalized_delta_velocity_history3",
                        "displacement_velocity_history3",
                        "modular_history3",
                    }:
                        decode_scale = (
                            ref_pos.amax(dim=0) - ref_pos.amin(dim=0)
                        ).clamp_min(1e-6)
                        pred_pos = (
                            ref_pos
                            + target_value[:, :pos_dim] * decode_scale.reshape(1, -1)
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
            row_sources.append(str(getattr(sim[0], "source_name", "unknown")))
        if not row_losses:
            continue
        source_indices: dict[str, list[int]] = {}
        for row_index, source_name in enumerate(row_sources):
            source_indices.setdefault(source_name, []).append(row_index)
        source_losses = {
            source_name: torch.stack([row_losses[index] for index in indices]).mean()
            for source_name, indices in source_indices.items()
        }
        if source_loss_reduction in {"equal", "per_source_mean"}:
            loss = torch.stack(list(source_losses.values())).mean()
        else:
            loss = torch.stack(row_losses).mean()
        raw_loss = torch.stack(row_raw_losses).mean()
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        losses.append(float(loss.detach().cpu()))
        raw_losses.append(float(raw_loss.detach().cpu()))
        for source_name, indices in source_indices.items():
            source_loss_logs.setdefault(source_name, []).extend(
                float(row_losses[index].detach().cpu()) for index in indices
            )

    equal_source_epoch_loss = (
        float(np.mean([np.mean(values) for values in source_loss_logs.values()]))
        if source_loss_logs
        and source_loss_reduction in {"equal", "per_source_mean"}
        else None
    )
    result = {
        "loss_norm": (
            equal_source_epoch_loss
            if equal_source_epoch_loss is not None
            else (float(np.mean(losses)) if losses else float("nan"))
        ),
        "loss_raw": float(np.mean(raw_losses)) if raw_losses else float("nan"),
        "position_loss": (
            float(np.mean(position_losses))
            if position_losses
            else 0.0
        ),
    }
    if truncated_rollout_horizon is not None:
        result["first_step_loss_norm"] = (
            float(np.mean(first_step_losses)) if first_step_losses else float("nan")
        )
        result[f"rollout_step_{truncated_rollout_horizon}_loss_norm"] = (
            float(np.mean(endpoint_losses)) if endpoint_losses else float("nan")
        )
    for source_name, values in source_loss_logs.items():
        source_key = "".join(
            character if character.isalnum() else "_"
            for character in source_name.lower()
        ).strip("_")
        result[f"source_{source_key}_loss_norm"] = float(np.mean(values))
        if source_name in source_first_step_logs:
            result[f"source_{source_key}_first_step_loss_norm"] = float(
                np.mean(source_first_step_logs[source_name])
            )
        if source_name in source_endpoint_logs:
            result[
                f"source_{source_key}_rollout_step_{truncated_rollout_horizon}_loss_norm"
            ] = float(np.mean(source_endpoint_logs[source_name]))
    return result


def epoch_recurrent_memory_propagator(
    model,
    sims,
    rows,
    stats: LatentNormalizer,
    *,
    batch_graphs: int,
    latent_cache,
    context_cache,
    fixed_observed_frames: tuple[int, int],
    use_static_context: bool = False,
    network_variation_weight: float = 0.0,
    frame_variation=None,
    optimizer=None,
    **_unused,
) -> dict[str, float]:
    """Teacher-force complete trajectories while carrying GRU memory forward."""

    is_train = optimizer is not None
    model.train(is_train)
    grouped: dict[int, list] = {}
    for row in rows:
        grouped.setdefault(int(row[0]), []).append(row)
    sim_ids = sorted(grouped)
    losses, raw_losses = [], []
    first_observed, last_observed = map(int, fixed_observed_frames)
    grad_context = torch.enable_grad if is_train else torch.no_grad
    with grad_context():
        for sim_batch in iter_batches(sim_ids, batch_graphs, shuffle=is_train):
            common_starts = sorted(
                set.intersection(
                    *[
                        {int(row[1]) for row in grouped[sim_idx]}
                        for sim_idx in sim_batch
                    ]
                )
            )
            if not common_starts:
                continue
            target_by_sim = {
                sim_idx: {
                    int(row[1]): int(row[2][0]) for row in grouped[sim_idx]
                }
                for sim_idx in sim_batch
            }
            context = None
            if use_static_context:
                context = torch.stack([context_cache[i] for i in sim_batch])
            memory = model.initial_memory(
                len(sim_batch),
                device=latent_cache[(sim_batch[0], 0)].device,
                dtype=latent_cache[(sim_batch[0], 0)].dtype,
            )
            previous = torch.stack(
                [latent_cache[(i, max(0, first_observed - 1))] for i in sim_batch]
            )
            for frame in range(first_observed, last_observed):
                current = torch.stack([latent_cache[(i, frame)] for i in sim_batch])
                _, memory = latent_step_recurrent_memory(
                    model, current, previous, memory, stats, context=context
                )
                previous = current
            chunk_losses, chunk_raw = [], []
            for start_frame in common_starts:
                current = torch.stack(
                    [latent_cache[(i, start_frame)] for i in sim_batch]
                )
                target_frames = [target_by_sim[i][start_frame] for i in sim_batch]
                target = torch.stack(
                    [latent_cache[(i, t)] for i, t in zip(sim_batch, target_frames)]
                )
                predicted, memory = latent_step_recurrent_memory(
                    model, current, previous, memory, stats, context=context
                )
                scale = stats.dz_std.to(current).clamp_min(1e-6)
                per_row = (
                    ((predicted - current) / scale - (target - current) / scale)
                    .square()
                    .mean(dim=-1)
                )
                if float(network_variation_weight) > 0:
                    variation = torch.stack(
                        [frame_variation[t].to(current) for t in target_frames]
                    ).clamp_min(1e-6)
                    per_row = per_row + float(network_variation_weight) * (
                        (predicted - target) / variation
                    ).square().mean(dim=-1)
                chunk_losses.append(per_row.mean())
                chunk_raw.append((predicted - target).square().mean())
                # The next teacher-forced input is ``target`` itself, so its
                # predecessor is the current state rather than the target.
                previous = current
                if len(chunk_losses) == 16 or start_frame == common_starts[-1]:
                    loss = torch.stack(chunk_losses).mean()
                    raw_loss = torch.stack(chunk_raw).mean()
                    if is_train:
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                    raw_losses.append(float(raw_loss.detach().cpu()))
                    memory = memory.detach()
                    chunk_losses, chunk_raw = [], []
    return {
        "loss_norm": float(np.mean(losses)) if losses else float("nan"),
        "loss_raw": float(np.mean(raw_losses)) if raw_losses else float("nan"),
        "position_loss": 0.0,
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
    max_progress_frame: int | None = None,
    unroll_curriculum=None,
    unroll_stage_epochs=None,
    truncated_rollout_horizon: int | None = None,
    mix_sources: bool = False,
    use_static_context: bool = False,
    context_include_temperature: bool = False,
    context_include_source_id: bool = False,
    rho_scale_mode: str | None = None,
    source_loss_reduction: str = "pooled",
    history_noise_std: float = 0.0,
    source_classification_weight: float = 0.0,
    frozen_latent_cache_dir: str | Path | None = None,
    use_pcgrad: bool = False,
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
        "context_include_source_id": context_include_source_id,
        "rho_scale_mode": rho_scale_mode,
        "max_progress_frame": max_progress_frame,
    }
    objective = str(objective).lower()

    if objective in {"one_step", "next_step"}:
        epoch_fn = epoch_propagator
        extra = {
            "loss_mode": loss_mode,
            "ae_target_mode": ae_target_mode,
            "physics_config": physics_config,
            "context_pool_mode": context_pool_mode,
            "source_loss_reduction": source_loss_reduction,
            "use_pcgrad": use_pcgrad,
            "mix_sources": mix_sources,
            "source_classification_weight": source_classification_weight,
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
        "recurrent_memory_one_step",
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
        if getattr(model, "uses_fixed_observed_state", False) or getattr(
            model, "uses_recurrent_memory", False
        ) or getattr(model, "uses_lagged_history", False):
            required_fixed_frames = (
                int(getattr(model, "fixed_history_size", 2))
                if getattr(model, "uses_fixed_window_history", False)
                else 2
            )
            if fixed_observed_frames is None or len(fixed_observed_frames) != required_fixed_frames:
                raise ValueError(
                    f"The fixed-history model requires exactly {required_fixed_frames} fixed_observed_frames."
                )
            fixed_observed_frames = tuple(int(frame) for frame in fixed_observed_frames)
            if fixed_observed_frames[0] < 0 or any(
                left >= right
                for left, right in zip(fixed_observed_frames, fixed_observed_frames[1:])
            ):
                raise ValueError(
                    "fixed_observed_frames must be increasing non-negative indices."
                )
            last_observed = fixed_observed_frames[-1]
            train_rows = [row for row in train_rows if int(row[1]) >= last_observed]
            val_rows = [row for row in val_rows if int(row[1]) >= last_observed]
        training_label = (
            "one-step history training"
            if max_horizon == 1
            else "closed-loop training"
        )
        print(f"preparing frozen AE latents for {training_label}")
        feature_dir = (
            Path(frozen_latent_cache_dir)
            if frozen_latent_cache_dir is not None
            else None
        )
        train_cache, train_context_cache = _load_or_precompute_frozen_latents(
            feature_dir / "train.pt" if feature_dir is not None else None,
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
            context_include_source_id=context_include_source_id,
            context_pool_mode=context_pool_mode,
            fixed_observed_frames=fixed_observed_frames,
        )
        val_cache, val_context_cache = _load_or_precompute_frozen_latents(
            feature_dir / "val.pt" if feature_dir is not None else None,
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
            context_include_source_id=context_include_source_id,
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
                "source_loss_reduction": source_loss_reduction,
                "history_noise_std": history_noise_std,
                "truncated_rollout_horizon": truncated_rollout_horizon,
            }

            def stage_epoch(sims, rows, cache, context_cache, optimizer=None):
                if getattr(model, "uses_recurrent_memory", False):
                    return epoch_recurrent_memory_propagator(
                        model,
                        sims,
                        rows,
                        stats,
                        optimizer=optimizer,
                        latent_cache=cache,
                        context_cache=context_cache,
                        **stage_common,
                    )
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
    edge_mode = canonical_edge_mode(getattr(ae_model, "edge_mode", "stored"))
    if edge_mode == "complete":
        edge_index, _, ref_edge_attr = undirected_complete_graph_edge_data(
            ref, ref, pos_dim=pos_dim, device=device
        )
    elif edge_mode in {"stored", "compact_stored", "compact_delta_stored"}:
        edge_index = ref.edge_index.to(device).long()
        if edge_mode == "compact_stored":
            ref_edge_attr = compact_reference_edge_features(ref, pos_dim=pos_dim, device=device)
        elif edge_mode == "compact_delta_stored":
            ref_edge_attr = compact_delta_reference_edge_features(ref, pos_dim=pos_dim, device=device)
        else:
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
        "normalized_delta_velocity_acceleration",
        "displacement_velocity_acceleration",
        "kinematic_state",
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
