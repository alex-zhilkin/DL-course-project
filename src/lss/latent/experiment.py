"""High-level latent simulator experiment workflow.

This module keeps notebook code focused on configuration and visualization.
It owns dataset resolution, deterministic splitting, model training, rollout
evaluation, and the tabular summaries shared by the latent-space notebooks.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Callable
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from graph_utils import calc_p_ratio_rollout_sides
from torch_geometric.data import Data

from ..data import load_dataset, resolve_dataset_splits, simulation_temperature
from ..graph import clone_graph
from .models import (
    NodeDeltaAttentionAutoEncoder,
    NodeDeltaDirectAttentionAutoEncoder,
    NodeDeltaMLPAutoEncoder,
    NodeDeltaPyramidMLPAutoEncoder,
    NodeDeltaSingleStageAttentionAutoEncoder,
    make_latent_propagator,
)
from .physics import PhysicsLossConfig
from .simulation import (
    filtered_frame_ids,
    fit_ae_target_stats,
    fit_edge_stats,
    fit_node_feature_stats,
    fit_reference_edge_stats,
    frame_for_filtered_step,
    make_frame_index,
    make_transition_index,
    pearson_r,
    r2_score,
    trajectory_p_ratio_sides_strain_gated,
    trajectory_p_ratio_sides_robust,
)
from .training import (
    LatentNormalizer,
    TrainingConfig,
    TrainingResult,
    decode_latent_to_graph,
    encode_frame_latent,
    encode_reference_context,
    fit_latent_step_stats,
    initial_structure_scale,
    latent_step,
    latent_step_fixed_history,
    latent_step_fixed_window,
    latent_step_history,
    latent_step_kinematic,
    latent_step_lagged_history,
    latent_step_recurrent_memory,
    latent_step_velocity,
    make_multistep_transition_index,
    make_velocity_transition_index,
    batch_delta_graphs,
    ae_target_tensor,
    train_autoencoder,
    train_propagator,
)

DEFAULT_PROPAGATOR_CHECKPOINT_METRIC = (
    "val_rollout_macro_source_endpoint_p_ratio_r2"
)

AE_CONFIG_KEY_MAP = {
    "model": "autoencoder_model",
    "target_mode": "ae_target_mode",
    "max_train_frames_per_sim": "ae_max_train_frames_per_sim",
    "train_frame_skip": "ae_train_frame_skip",
    "val_frame_skip": "ae_val_frame_skip",
    "max_val_frames_per_sim": "ae_max_val_frames_per_sim",
    "max_epochs": "ae_max_epochs",
    "patience": "ae_patience",
    "lr": "ae_lr",
    "weight_decay": "ae_weight_decay",
    "balance_sources": "ae_balance_sources",
    "mix_sources": "ae_mix_sources",
    "gradient_method": "ae_gradient_method",
    "nash_max_iter": "ae_nash_max_iter",
    "train_rows_per_source": "ae_train_rows_per_source",
    "coordinate_weights": "ae_coordinate_weights",
    "pratio_eval_every": "ae_pratio_eval_every",
    "pratio_eval_step": "ae_pratio_eval_step",
    "pratio_eval_steps": "ae_pratio_eval_steps",
    "checkpoint_metric": "ae_checkpoint_metric",
    "checkpoint_mode": "ae_checkpoint_mode",
}

PROPAGATOR_CONFIG_KEY_MAP = {
    "max_train_transitions_per_sim": "dyn_max_train_transitions_per_sim",
    "max_epochs": "dyn_max_epochs",
    "patience": "dyn_patience",
    "lr": "dyn_lr",
    "weight_decay": "dyn_weight_decay",
    "context_dim": "graph_context_dim",
    "initial_velocity": "initial_velocity",
    "fixed_observed_frames": "fixed_observed_frames",
    "rollout_history_frames": "rollout_history_frames",
}


def _expand_component_config(
    expanded: dict,
    component: dict | None,
    *,
    component_name: str,
    key_map: dict[str, str],
    prefix_unknown: bool,
) -> None:
    """Expand one concise component dictionary into the internal flat schema."""

    if component is None:
        return
    if not isinstance(component, dict):
        raise TypeError(f"{component_name}_config must be a dictionary.")
    for key, value in component.items():
        if key.startswith(("ae_", "dyn_", "propagator_")) or (
            component_name == "ae" and key.startswith("autoencoder_")
        ):
            raise ValueError(
                f"Use the short key {key!r} without a component prefix inside "
                f"{component_name}_config."
            )
        target = key_map.get(
            key,
            f"propagator_{key}" if prefix_unknown else key,
        )
        if target in expanded:
            raise ValueError(f"Duplicate {component_name.upper()} setting: {target}")
        expanded[target] = value


def _expand_component_configs(cfg: dict) -> dict:
    """Expand concise nested AE and propagator notebook configurations."""

    expanded = {
        key: value
        for key, value in cfg.items()
        if key not in {"ae_config", "propagator_config"}
    }
    _expand_component_config(
        expanded,
        cfg.get("ae_config"),
        component_name="ae",
        key_map=AE_CONFIG_KEY_MAP,
        prefix_unknown=False,
    )
    _expand_component_config(
        expanded,
        cfg.get("propagator_config"),
        component_name="propagator",
        key_map=PROPAGATOR_CONFIG_KEY_MAP,
        prefix_unknown=True,
    )
    return expanded

KINEMATIC_OBJECTIVES = {
    "kinematic_multistep",
    "kinematic",
    "anchored_multistep",
    "closed_loop",
    "history_one_step",
    "fixed_history_one_step",
    "recurrent_memory_one_step",
}

THREE_FRAME_INITIALIZATIONS = {
    "three_frames",
    "three_frame",
    "observed_three",
    "history3",
}


def _observed_position_graph(sim, frame_index: int, *, pos_dim: int):
    """Keep only observed positions and box metadata needed by p-ratio metrics."""

    return _p_ratio_position_graph(sim[frame_index], pos_dim=pos_dim)


def _p_ratio_position_graph(graph, *, pos_dim: int):
    """Create a lightweight graph for trajectory p-ratio evaluation.

    Retaining topology and edge features at every rollout frame can consume
    gigabytes even though the side-strain estimators use only positions and the
    reference box.
    """

    out = Data(x=graph.x[:, :pos_dim].detach().cpu().float().clone())
    if hasattr(graph, "box"):
        out.box = graph.box
    if hasattr(graph, "box_tensor") and isinstance(graph.box_tensor, torch.Tensor):
        out.box_tensor = graph.box_tensor.detach().cpu().clone()
    return out


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(value[key]) for key in sorted(value)}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _pretrained_ae_matches_requested_config(path, params: dict) -> bool:
    """Return whether a saved AE was trained with the requested AE recipe."""

    keys = tuple(params.get("pretrained_ae_config_keys", ()))
    if not keys:
        return True
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    saved = bundle.get("params", {})
    mismatches = {
        key: (saved.get(key), params.get(key))
        for key in keys
        if _json_safe(saved.get(key)) != _json_safe(params.get(key))
    }
    if mismatches:
        message = "saved AE recipe changed: " + ", ".join(sorted(mismatches))
        if bool(params.get("pretrained_ae_require_matching_config", False)):
            raise ValueError(message + "; refusing to retrain the frozen AE.")
        print(
            message + "; retraining AE",
            flush=True,
        )
        return False
    return True


def _rows_by_source(sims, rows) -> dict[str, list]:
    """Group indexed frame/transition rows by their trajectory source."""

    grouped: dict[str, list] = {}
    for row in rows:
        sim_index = int(row[0])
        source = str(getattr(sims[sim_index][0], "source_name", "unknown"))
        grouped.setdefault(source, []).append(row)
    return grouped


def _balance_rows_by_source(
    sims,
    rows,
    *,
    rows_per_source: int,
    seed: int,
) -> list:
    """Sample the same number of rows per source without discarding sources.

    Large sources are sampled without replacement. Small sources are repeated
    only after every available row has been used. The propagator never receives
    the source name; this only prevents a large dataset from dominating the
    optimization objective.
    """

    rows_per_source = int(rows_per_source)
    if rows_per_source < 1:
        raise ValueError("rows_per_source must be positive.")
    grouped = _rows_by_source(sims, rows)
    if len(grouped) <= 1:
        return list(rows)
    generator = torch.Generator().manual_seed(int(seed))
    selected = []
    for source in sorted(grouped):
        source_rows = grouped[source]
        if not source_rows:
            continue
        order = torch.randperm(len(source_rows), generator=generator).tolist()
        shuffled = [source_rows[index] for index in order]
        repeats, remainder = divmod(rows_per_source, len(shuffled))
        selected.extend(shuffled * repeats)
        selected.extend(shuffled[:remainder])
    if selected:
        order = torch.randperm(len(selected), generator=generator).tolist()
        selected = [selected[index] for index in order]
    return selected


def _limit_rows_to_source_trajectories(
    sims,
    rows,
    limits: dict[str, int] | None,
) -> list:
    """Keep rows from at most the requested number of real trajectories per source."""

    if not limits:
        return list(rows)
    normalized_limits = {str(source): int(limit) for source, limit in limits.items()}
    if any(limit < 1 for limit in normalized_limits.values()):
        raise ValueError("Per-source trajectory limits must be positive.")
    selected_simulations: dict[str, set[int]] = {
        source: set() for source in normalized_limits
    }
    for sim_idx, sim in enumerate(sims):
        source = str(getattr(sim[0], "source_name", "unknown"))
        if source not in normalized_limits:
            continue
        if len(selected_simulations[source]) < normalized_limits[source]:
            selected_simulations[source].add(sim_idx)
    missing = set(normalized_limits) - {
        source for source, indices in selected_simulations.items() if indices
    }
    if missing:
        raise ValueError(
            "Per-source trajectory limits did not match loaded sources: "
            f"{sorted(missing)}"
        )
    return [
        row
        for row in rows
        if int(row[0])
        in selected_simulations.get(
            str(getattr(sims[int(row[0])][0], "source_name", "unknown")),
            set(),
        )
    ]


def _limit_rows_per_trajectory(rows, limit: int) -> list:
    """Select up to ``limit`` distinct rows spanning each trajectory."""

    limit = int(limit)
    if limit < 1:
        raise ValueError("The per-trajectory transition limit must be positive.")
    grouped: dict[int, list] = {}
    for row in rows:
        grouped.setdefault(int(row[0]), []).append(row)
    selected = []
    for sim_idx in sorted(grouped):
        sim_rows = grouped[sim_idx]
        if len(sim_rows) <= limit:
            selected.extend(sim_rows)
            continue
        if limit == 1:
            selected.append(sim_rows[0])
            continue
        indices = [
            round(index * (len(sim_rows) - 1) / (limit - 1))
            for index in range(limit)
        ]
        selected.extend(sim_rows[index] for index in indices)
    return selected


def _autoencoder_class(model_type: str):
    model_type = str(model_type).lower()
    if model_type in {"mlp", "mean_mlp", "mean_pool"}:
        return NodeDeltaMLPAutoEncoder
    if model_type in {"pyramid_mlp", "mean_pyramid_mlp"}:
        return NodeDeltaPyramidMLPAutoEncoder
    if model_type in {"direct_attention", "attention_direct", "direct_attention_decoder"}:
        return NodeDeltaDirectAttentionAutoEncoder
    if model_type in {"single_stage_attention", "direct_latent_attention", "node_to_latent_attention"}:
        return NodeDeltaSingleStageAttentionAutoEncoder
    if model_type in {"attention", "attention_mlp"}:
        return NodeDeltaAttentionAutoEncoder
    raise ValueError(f"Unknown autoencoder_model: {model_type}")


def _make_ae_pratio_epoch_callback(
    *,
    params: dict,
    val_data,
    normalizers: dict[str, torch.Tensor],
    label: str,
    device,
):
    """Evaluate decoded validation deformation during AE training."""

    every = int(params.get("ae_pratio_eval_every", 0))
    if every <= 0:
        return None
    configured_steps = params.get("ae_pratio_eval_steps")
    steps = (
        sorted({int(step) for step in configured_steps})
        if configured_steps is not None
        else [int(params.get("ae_pratio_eval_step", 100))]
    )
    step_suffix = "_".join(str(step) for step in steps)

    def callback(epoch: int, active_model) -> dict[str, float]:
        if int(epoch) % every:
            return {}
        rows, _ = evaluate_autoencoder_reconstruction_horizons(
            active_model,
            val_data,
            cfg=params,
            normalizers=normalizers,
            dataset=label,
            split_name="val",
            rollout_steps=steps,
            device=device,
        )
        metrics = {}
        for step in steps:
            step_rows = rows.loc[rows["rollout_steps"] == step]
            metrics[f"val_ae_p_ratio_r2_step_{step}"] = r2_score(
                step_rows["true_p_ratio"], step_rows["pred_p_ratio"]
            )
        source_sums = []
        source_step_scores: dict[int, list[float]] = {int(step): [] for step in steps}
        for source_name, group in rows.groupby("source", sort=True):
            source_key = "".join(
                character if character.isalnum() else "_"
                for character in str(source_name).lower()
            ).strip("_")
            source_scores = []
            for step in steps:
                step_group = group.loc[group["rollout_steps"] == step]
                score = r2_score(step_group["true_p_ratio"], step_group["pred_p_ratio"])
                source_scores.append(score)
                if np.isfinite(score):
                    source_step_scores[int(step)].append(float(score))
                metrics[f"val_ae_source_{source_key}_p_ratio_r2_step_{step}"] = score
                metrics[f"val_ae_source_{source_key}_position_mse_step_{step}"] = float(
                    step_group["final_pos_mse"].mean()
                )
            source_sum = float(np.sum(source_scores)) if np.all(np.isfinite(source_scores)) else float("nan")
            metrics[f"val_ae_source_{source_key}_p_ratio_r2_sum_{step_suffix}"] = source_sum
            source_sums.append(source_sum)
        for step, scores in source_step_scores.items():
            if scores:
                metrics[f"val_ae_mean_source_p_ratio_r2_step_{step}"] = float(
                    np.mean(scores)
                )
        finite_source_sums = [value for value in source_sums if np.isfinite(value)]
        if finite_source_sums:
            metrics[f"val_ae_min_source_p_ratio_r2_sum_{step_suffix}"] = float(min(finite_source_sums))
        return metrics

    return callback


def initialize_displacement_pca_layers(
    model,
    sims,
    frame_rows,
    *,
    pos_dim: int,
    node_feature_mode: str,
    target_mode: str,
    normalizers: dict[str, torch.Tensor],
    device,
    max_samples: int = 200_000,
) -> dict:
    """Initialize displacement-facing AE weights from per-node PCA directions.

    Full-field PCA is undefined across variable-size graphs. This applies the
    shared PCA basis of normalized per-node displacement features instead.
    """

    node_parts, target_parts = [], []
    count = 0
    with torch.no_grad():
        for row in frame_rows:
            batch = batch_delta_graphs(
                sims, [row], pos_dim=pos_dim, device=device,
                node_feature_mode=node_feature_mode,
            )
            node = (batch["node_feature"] - normalizers["node_feature_mean"].to(device)) / normalizers[
                "node_feature_std"
            ].to(device)
            target = (ae_target_tensor(batch, target_mode) - normalizers["target_mean"].to(device)) / normalizers[
                "target_std"
            ].to(device)
            node_parts.append(node[:, :pos_dim].cpu())
            target_parts.append(target[:, :pos_dim].cpu())
            count += node.size(0)
            if count >= int(max_samples):
                break

    def basis(parts):
        values = torch.cat(parts, dim=0)[: int(max_samples)]
        centered = values - values.mean(dim=0, keepdim=True)
        _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
        explained = singular_values.square()
        explained = explained / explained.sum().clamp_min(1e-12)
        return vh, explained

    encoder_basis, encoder_explained = basis(node_parts)
    decoder_basis, decoder_explained = basis(target_parts)
    if not isinstance(model.node_in, torch.nn.Linear):
        raise TypeError("PCA initialization currently requires a linear node_in layer.")
    output_layer = model.node_decoder[-1]
    if not isinstance(output_layer, torch.nn.Linear):
        raise TypeError("PCA initialization requires a linear final decoder layer.")
    with torch.no_grad():
        # Preserve all unrelated random weights. Align the first pos_dim hidden
        # channels with PCA scores and initialize the output projection with
        # the inverse PCA orientation.
        model.node_in.weight[:pos_dim, :pos_dim].copy_(encoder_basis)
        output_layer.weight[:, :pos_dim].copy_(decoder_basis.transpose(0, 1))
    return {
        "encoder_explained_variance": encoder_explained.tolist(),
        "decoder_explained_variance": decoder_explained.tolist(),
        "samples": min(count, int(max_samples)),
    }


def latent_experiment_cache_key(source_spec: dict, cfg: dict) -> str:
    """Stable hash from source/training configuration, independent of notebook name."""

    cfg = _expand_component_configs(cfg)

    ignored_source = {
        "label",
        "source_name",
        "display_name",
        "experiment_name",
    }
    ignored_cfg = {
        "cache_dir",
        "cache_path",
        "force_train",
        "force_train_autoencoder",
        "device",
        "label",
        "source_name",
        "display_name",
        "experiment_name",
        "cache_require_matching_config",
    }
    payload = {
        "source_spec": _json_safe(
            {key: value for key, value in source_spec.items() if key not in ignored_source}
        ),
        "cfg": _json_safe({key: value for key, value in cfg.items() if key not in ignored_cfg}),
        "cache_version": 3,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _cache_path(source_spec: dict, cfg: dict) -> Path | None:
    explicit_path = cfg.get("cache_path")
    if explicit_path:
        return Path(explicit_path).expanduser()
    cache_dir = cfg.get("cache_dir")
    if not cache_dir:
        return None
    path = Path(cache_dir).expanduser()
    return path / f"latent_{latent_experiment_cache_key(source_spec, cfg)}.pt"


def _save_ae_cache(result: dict, source_spec: dict, cfg: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "ae_only",
            "cache_key": latent_experiment_cache_key(source_spec, cfg),
            "source_spec": dict(source_spec),
            "params": dict(result["params"]),
            "ae_state_dict": {
                key: value.detach().cpu().clone()
                for key, value in result["ae"].state_dict().items()
            },
            "stats": result["stats"],
            "ae_history": result["ae_history"],
        },
        path,
    )


def _load_ae_cache(path: Path, cfg: dict, *, device) -> dict:
    bundle = torch.load(path, map_location=device, weights_only=False)
    spec = dict(bundle["source_spec"])
    params = {**cfg, **bundle["params"], **spec}
    stats = bundle["stats"]
    normalizers = {
        key: stats[key].to(device)
        for key in (
            "target_mean",
            "target_std",
            "node_feature_mean",
            "node_feature_std",
            "edge_mean",
            "edge_std",
        )
    }
    for key in ("ref_edge_mean", "ref_edge_std"):
        if key in stats:
            normalizers[key] = stats[key].to(device)
    autoencoder_type = str(params.get("autoencoder_model", "attention")).lower()
    autoencoder_cls = _autoencoder_class(autoencoder_type)
    ae_model = autoencoder_cls(
        pos_dim=int(params["pos_dim"]),
        node_feature_dim=int(normalizers["node_feature_mean"].numel()),
        edge_dim=int(normalizers["edge_mean"].numel()),
        hidden_size=int(params["hidden_size"]),
        latent_dim=int(params["latent_dim"]),
        latent_tokens=int(params["latent_tokens"]),
        reconstruction_dim=int(normalizers["target_mean"].numel()),
    ).to(device)
    ae_model.edge_mode = str(params.get("edge_mode", "stored"))
    ae_model.load_state_dict(bundle["ae_state_dict"])
    ae_model.eval()
    for parameter in ae_model.parameters():
        parameter.requires_grad_(False)

    p_ratio_fn = lambda sim, idx=-1: ground_truth_p_ratio(
        sim,
        idx,
        dataset_name=params["dataset_name"],
        cfg=params,
    )
    train_data, val_data, test_data, split_info = resolve_train_val_test(
        spec,
        params,
        split_seed=params.get("split_seed"),
        p_ratio_fn=p_ratio_fn,
    )
    if not bool(params.get("static_context_use_physical_reference", True)):
        _use_normalized_reference_context(
            (train_data, val_data, test_data), pos_dim=int(params["pos_dim"])
        )
    return {
        "label": spec["label"],
        "params": params,
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
        "split_info": pd.DataFrame(split_info),
        "ae": ae_model,
        "normalizers": normalizers,
        "stats": stats,
        "ae_history": pd.DataFrame(bundle.get("ae_history", [])),
        "cache_path": str(path),
    }


def seed_everything(seed: int) -> int:
    """Seed Python, NumPy, and Torch and return the normalized seed."""

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the nearest parent containing ``pyproject.toml``."""

    root = Path.cwd().resolve() if start is None else Path(start).expanduser().resolve()
    while root != root.parent and not (root / "pyproject.toml").exists():
        root = root.parent
    if not (root / "pyproject.toml").exists():
        raise FileNotFoundError("Could not locate project root containing pyproject.toml.")
    return root


def resolve_existing_path(path_like: str | Path, *, project_root: str | Path | None = None) -> str:
    """Resolve a dataset path relative to the project and its data directory."""

    root = find_project_root(project_root)
    path = Path(path_like).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.extend((root / path, root / "data" / path.name))
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    tried = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Could not resolve dataset path {path_like!r}. Tried: {tried}")


def prepare_source_spec(
    dataset_name: str,
    dataset_specs: dict[str, dict],
    cfg: dict,
    *,
    seed: int,
    project_root: str | Path | None = None,
) -> dict:
    """Resolve one configured dataset and attach model parameters."""

    base = dict(dataset_specs[dataset_name])
    candidates = base.pop("path_candidates", [base.get("path")])
    errors = []
    for candidate in candidates:
        try:
            base["path"] = resolve_existing_path(candidate, project_root=project_root)
            break
        except FileNotFoundError as exc:
            errors.append(str(exc))
    else:
        raise FileNotFoundError(
            f"Dataset {dataset_name!r} was selected but none of its paths exist:\n"
            + "\n".join(errors)
        )

    base["source_name"] = base["label"]
    spec = {
        **base,
        "dataset_name": dataset_name,
        "target_mode": cfg["ae_target_mode"],
        "ae_target_mode": cfg["ae_target_mode"],
        "node_feature_mode": cfg["node_feature_mode"],
        "latent_dim": int(cfg["latent_dim"]),
        "repeat_idx": int(cfg.get("repeat_idx", 1)),
        "model_seed": int(cfg.get("model_seed", seed + 1009 * int(cfg["latent_dim"]))),
        "latent_tokens": int(cfg["latent_tokens"]),
        "hidden_size": int(cfg["hidden_size"]),
        "edge_feature_dim": int(cfg.get("edge_feature_dim", 0)),
        "ae_max_train_frames_per_sim": int(cfg["ae_max_train_frames_per_sim"]),
        "dyn_max_train_transitions_per_sim": int(cfg["dyn_max_train_transitions_per_sim"]),
        "ae_max_epochs": int(cfg["ae_max_epochs"]),
        "ae_patience": int(cfg["ae_patience"]),
        "ae_lr": float(cfg["ae_lr"]),
        "ae_weight_decay": float(cfg["ae_weight_decay"]),
        "dyn_max_epochs": int(cfg["dyn_max_epochs"]),
        "dyn_patience": int(cfg["dyn_patience"]),
        "dyn_lr": float(cfg["dyn_lr"]),
        "dyn_weight_decay": float(cfg["dyn_weight_decay"]),
    }
    spec["label"] = f"{base['label']} CV{spec['latent_dim']} {cfg['ae_target_mode']}"
    return spec


def is_temperature_dataset(dataset_name: str) -> bool:
    normalized = str(dataset_name).strip().lower().replace("-", "_")
    return normalized in {
        "depablo_10k",
        "depablo_10k_mix_temp",
        "depablo_mixed_temp",
        "lj_noisy",
        "lj_noisy_eps0.01_sigma1.0_cutoff1.122",
    }


TRAJECTORY_P_RATIO_ESTIMATORS = {
    "robust_trajectory",
    "trajectory_robust",
    "strain_gated_trajectory",
    "trajectory_strain_gated",
    "time_slope_trajectory",
}


def _uses_trajectory_p_ratio(cfg: dict | None) -> bool:
    return str((cfg or {}).get("p_ratio_estimator", "default")).lower() in (
        TRAJECTORY_P_RATIO_ESTIMATORS
    )


def _uses_endpoint_p_ratio(cfg: dict | None) -> bool:
    return str((cfg or {}).get("p_ratio_estimator", "default")).lower() in {
        "endpoint",
        "start_end",
        "two_frame",
    }


def temperature_p_ratio(trajectory, *, cfg: dict | None = None, last_index: int = -1) -> float:
    """Evaluate p-ratio from a trajectory prefix using the configured estimator."""

    cfg = cfg or {}
    estimator = str(cfg.get("p_ratio_estimator", "robust_trajectory")).lower()
    if estimator in {
        "strain_gated_trajectory",
        "trajectory_strain_gated",
        "time_slope_trajectory",
    }:
        return float(
            trajectory_p_ratio_sides_strain_gated(
                trajectory,
                last_index=last_index,
                min_fit_frames=int(cfg.get("p_ratio_min_fit_frames", 4)),
                min_driven_strain_range=float(
                    cfg.get("p_ratio_min_driven_strain_range", 1e-4)
                ),
                side_quantile=float(cfg.get("p_ratio_side_quantile", 0.10)),
                min_abs_strain=float(cfg.get("p_ratio_min_abs_strain", 1e-5)),
            )
        )
    return float(
        trajectory_p_ratio_sides_robust(
            trajectory,
            last_index=last_index,
            min_fit_frames=int(cfg.get("p_ratio_min_fit_frames", 8)),
            min_driven_strain_range=float(
                cfg.get("p_ratio_min_driven_strain_range", 1e-3)
            ),
            smooth_window=int(cfg.get("p_ratio_smooth_window", 5)),
            side_quantile=float(cfg.get("p_ratio_side_quantile", 0.10)),
            min_abs_strain=float(cfg.get("p_ratio_min_abs_strain", 1e-5)),
        )
    )


def ground_truth_p_ratio(
    sim,
    last_index: int = -1,
    *,
    dataset_name: str,
    cfg: dict | None = None,
) -> float:
    """Use trajectory fitting for noisy-temperature data and endpoint fitting otherwise."""

    if (
        not _uses_endpoint_p_ratio(cfg)
        and (is_temperature_dataset(dataset_name) or _uses_trajectory_p_ratio(cfg))
        and len(sim) > 2
    ):
        return temperature_p_ratio(sim, cfg=cfg, last_index=last_index)
    return float(calc_p_ratio_rollout_sides(sim, last_index))


def make_jump_transition_index(
    sims,
    *,
    step_stride: int,
    frame_skip: int = 1,
    max_starts_per_sim: int | None = None,
) -> list[tuple[int, int, int]]:
    """Create ``z(t) -> z(t+k)`` transition rows."""

    stride = max(1, int(step_stride))
    rows = []
    for sim_idx, sim in enumerate(sims):
        frame_ids = filtered_frame_ids(sim, frame_skip=frame_skip, include_last=True)
        starts = list(range(max(0, len(frame_ids) - stride)))
        if max_starts_per_sim is not None:
            starts = starts[: int(max_starts_per_sim)]
        rows.extend(
            (int(sim_idx), int(frame_ids[start]), int(frame_ids[start + stride]))
            for start in starts
        )
    return rows


def make_jump_velocity_transition_index(
    sims,
    *,
    step_stride: int,
    frame_skip: int = 1,
    max_starts_per_sim: int | None = None,
) -> list[tuple[int, int, int, int]]:
    """Create equally spaced ``z(t-k), z(t), z(t+k)`` rows."""

    stride = max(1, int(step_stride))
    rows = []
    for sim_idx, sim in enumerate(sims):
        frame_ids = filtered_frame_ids(sim, frame_skip=frame_skip, include_last=True)
        if len(frame_ids) <= stride:
            continue
        centers = [0] + list(range(stride, max(stride, len(frame_ids) - stride)))
        if max_starts_per_sim is not None:
            centers = centers[: int(max_starts_per_sim)]
        for center in centers:
            previous = max(0, center - stride)
            rows.append(
                (
                    int(sim_idx),
                    int(frame_ids[previous]),
                    int(frame_ids[center]),
                    int(frame_ids[center + stride]),
                )
            )
    return rows


def rollout_steps_for_sims(
    sims,
    requested_steps,
    *,
    frame_skip: int = 1,
) -> list[int]:
    """Clip explicit rollout horizons to the common available trajectory length."""

    if not sims:
        return []
    max_steps = min(
        max(len(filtered_frame_ids(sim, frame_skip=frame_skip, include_last=True)) - 1, 0)
        for sim in sims
    )
    if max_steps <= 0:
        return []
    return sorted(
        {
            min(int(step), max_steps)
            for step in requested_steps
            if min(int(step), max_steps) > 0
        }
    )


def resolve_train_val_test(
    source_spec: dict,
    params: dict,
    *,
    split_seed: int | None,
    p_ratio_fn: Callable | None = None,
):
    """Resolve a deterministic shuffled split, optionally filtering train p-ratio."""
    edge_multiplicity = int(
        source_spec.get("edge_multiplicity", params.get("edge_multiplicity", 1))
    )
    edge_vector_dim = int(
        source_spec.get("edge_vector_dim", params.get("edge_vector_dim", 2))
    )
    coordinate_normalization = source_spec.get(
        "coordinate_normalization", params.get("coordinate_normalization")
    )
    coordinate_pos_dim = int(params.get("pos_dim", 2))
    stiffness_length_exponent = source_spec.get(
        "edge_stiffness_length_exponent",
        params.get("edge_stiffness_length_exponent"),
    )
    append_lj_indicator = bool(
        source_spec.get(
            "append_lj_indicator", params.get("append_lj_indicator", False)
        )
    )
    add_lj_two_hop_edges = bool(
        source_spec.get(
            "add_lj_two_hop_edges", params.get("add_lj_two_hop_edges", False)
        )
    )

    if source_spec.get("dataset_mixture"):
        if params.get("min_train_p_ratio") is not None:
            raise ValueError("dataset_mixture cannot be combined with min_train_p_ratio.")
        if bool(params.get("split_stratify_temperature", False)):
            raise ValueError("dataset_mixture cannot be combined with temperature stratification.")
        return resolve_dataset_splits(
            source_spec.get("path", source_spec["dataset_mixture"][0]["path"]),
            train_count=0,
            val_count=0,
            dataset_mixture=source_spec["dataset_mixture"],
            split_seed=split_seed,
            shuffle_within_source=True,
            mix_holdout_across_sources=False,
            edge_multiplicity=edge_multiplicity,
            edge_vector_dim=edge_vector_dim,
            coordinate_normalization=coordinate_normalization,
            pos_dim=coordinate_pos_dim,
            edge_stiffness_length_exponent=stiffness_length_exponent,
            append_lj_indicator=append_lj_indicator,
            add_lj_two_hop_edges=add_lj_two_hop_edges,
        )

    min_train_p_ratio = params.get("min_train_p_ratio")
    stratify_temperature = bool(params.get("split_stratify_temperature", False))
    if stratify_temperature:
        if min_train_p_ratio is not None:
            raise ValueError(
                "Temperature-stratified splitting cannot currently be combined "
                "with min_train_p_ratio."
            )
        sims = load_dataset(
            source_spec["path"],
            edge_multiplicity=edge_multiplicity,
            edge_vector_dim=edge_vector_dim,
            coordinate_normalization=coordinate_normalization,
            pos_dim=coordinate_pos_dim,
            edge_stiffness_length_exponent=stiffness_length_exponent,
            append_lj_indicator=append_lj_indicator,
            add_lj_two_hop_edges=add_lj_two_hop_edges,
        )
        temperatures_by_sim = [simulation_temperature(sim) for sim in sims]
        if not temperatures_by_sim or not all(
            np.isfinite(temperature) for temperature in temperatures_by_sim
        ):
            warnings.warn(
                "Temperature stratification was requested, but this dataset has no "
                "complete temperature metadata. Using a normal shuffled split.",
                RuntimeWarning,
                stacklevel=2,
            )
            stratify_temperature = False

    if stratify_temperature:
        generator = torch.Generator()
        generator.manual_seed(0 if split_seed is None else int(split_seed))

        groups: dict[float, list] = {}
        for sim, temperature in zip(sims, temperatures_by_sim):
            groups.setdefault(temperature, []).append(sim)
        if not groups:
            raise ValueError("No temperature groups were found for stratified splitting.")

        temperatures = sorted(groups)
        for temperature in temperatures:
            group = groups[temperature]
            order = torch.randperm(len(group), generator=generator).tolist()
            groups[temperature] = [group[idx] for idx in order]

        def take_balanced(count: int) -> list:
            count = int(count)
            available = sum(len(group) for group in groups.values())
            if count > available:
                raise ValueError(
                    f"Requested {count} stratified samples but only {available} remain."
                )
            selected = []
            while len(selected) < count:
                active = [temperature for temperature in temperatures if groups[temperature]]
                if not active:
                    break
                order = torch.randperm(len(active), generator=generator).tolist()
                for idx in order:
                    if len(selected) >= count:
                        break
                    selected.append(groups[active[idx]].pop())
            return selected

        train_data = take_balanced(int(params["train_count"]))
        val_data = take_balanced(int(params["val_count"]))
        test_data = [sim for temperature in temperatures for sim in groups[temperature]]
        if test_data:
            order = torch.randperm(len(test_data), generator=generator).tolist()
            test_data = [test_data[idx] for idx in order]

        def temperature_counts(split) -> dict[str, int]:
            counts = {
                temperature: sum(
                    simulation_temperature(sim) == temperature for sim in split
                )
                for temperature in temperatures
            }
            return {
                f"temperature_{temperature:g}": int(count)
                for temperature, count in counts.items()
            }

        split_info = [
            {
                "source": Path(source_spec["path"]).stem,
                "path": str(source_spec["path"]),
                "total": len(sims),
                "train": len(train_data),
                "val": len(val_data),
                "test": len(test_data),
                "stratified_by": "temperature",
                **{
                    f"train_{key}": value
                    for key, value in temperature_counts(train_data).items()
                },
                **{
                    f"val_{key}": value
                    for key, value in temperature_counts(val_data).items()
                },
                **{
                    f"test_{key}": value
                    for key, value in temperature_counts(test_data).items()
                },
            }
        ]
        return train_data, val_data, test_data, split_info

    if min_train_p_ratio is None:
        return resolve_dataset_splits(
            source_spec["path"],
            train_count=params["train_count"],
            val_count=params["val_count"],
            split_seed=split_seed,
            shuffle_within_source=True,
            edge_multiplicity=edge_multiplicity,
            edge_vector_dim=edge_vector_dim,
            coordinate_normalization=coordinate_normalization,
            pos_dim=coordinate_pos_dim,
            edge_stiffness_length_exponent=stiffness_length_exponent,
            append_lj_indicator=append_lj_indicator,
            add_lj_two_hop_edges=add_lj_two_hop_edges,
        )

    if p_ratio_fn is None:
        raise ValueError("p_ratio_fn is required when min_train_p_ratio is set.")
    sims = load_dataset(
        source_spec["path"],
        edge_multiplicity=edge_multiplicity,
        edge_vector_dim=edge_vector_dim,
        coordinate_normalization=coordinate_normalization,
        pos_dim=coordinate_pos_dim,
        edge_stiffness_length_exponent=stiffness_length_exponent,
        append_lj_indicator=append_lj_indicator,
        add_lj_two_hop_edges=add_lj_two_hop_edges,
    )
    generator = torch.Generator()
    generator.manual_seed(0 if split_seed is None else int(split_seed))
    order = torch.randperm(len(sims), generator=generator).tolist()
    sims = [sims[i] for i in order]

    p_ratios = [float(p_ratio_fn(sim, -1)) for sim in sims]
    train_count = int(params["train_count"])
    val_count = int(params["val_count"])
    eligible = [idx for idx, value in enumerate(p_ratios) if value > float(min_train_p_ratio)]
    if len(eligible) < train_count:
        raise ValueError(
            f"Only {len(eligible)} networks have final p-ratio > {min_train_p_ratio}; "
            f"need train_count={train_count}."
        )

    train_indices = set(eligible[:train_count])
    train_data = [sim for idx, sim in enumerate(sims) if idx in train_indices]
    holdout = [sim for idx, sim in enumerate(sims) if idx not in train_indices]
    val_data = holdout[:val_count]
    test_data = holdout[val_count:]
    train_pr = [p_ratios[idx] for idx in train_indices]
    split_info = [
        {
            "source": Path(source_spec["path"]).stem,
            "path": str(source_spec["path"]),
            "total": len(sims),
            "eligible_train": len(eligible),
            "min_train_p_ratio": float(min_train_p_ratio),
            "train": len(train_data),
            "val": len(val_data),
            "test": len(test_data),
            "train_p_ratio_min": float(np.min(train_pr)) if train_pr else float("nan"),
            "train_p_ratio_max": float(np.max(train_pr)) if train_pr else float("nan"),
        }
    ]
    return train_data, val_data, test_data, split_info


def _normalizers_to_cpu(normalizers: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    def move(value):
        if torch.is_tensor(value):
            return value.detach().cpu()
        if isinstance(value, dict):
            return {key: move(item) for key, item in value.items()}
        return value
    return {key: move(value) for key, value in normalizers.items()}


def _use_normalized_reference_context(splits, *, pos_dim: int) -> None:
    """Replace preserved raw reference geometry with normalized frame-zero data."""

    for sims in splits:
        for simulation in sims:
            if not simulation:
                continue
            reference = simulation[0]
            reference.reference_context_positions = (
                reference.x[:, :pos_dim].detach().clone()
            )
            edge_attr = getattr(reference, "edge_attr", None)
            if isinstance(edge_attr, torch.Tensor):
                reference.reference_context_edge_attr = edge_attr.detach().clone()


def rollout_metrics(df: pd.DataFrame, *, dataset, split_name, rollout_steps) -> dict:
    """Aggregate position and p-ratio rollout metrics."""

    if df.empty:
        return {
            "dataset": dataset,
            "split": split_name,
            "rollout_steps": int(rollout_steps),
            "used": 0,
            "p_ratio_used": 0,
        }
    valid_p_ratio = np.isfinite(df["true_p_ratio"]) & np.isfinite(df["pred_p_ratio"])
    final_pos_mse = float(df["final_pos_mse"].mean())
    initial_to_target_mse = float(df["initial_to_target_mse"].mean())
    pred_to_initial_mse = float(df["pred_to_initial_mse"].mean())
    error_fraction = (
        final_pos_mse / initial_to_target_mse
        if initial_to_target_mse > 0
        else float("nan")
    )
    movement_fraction = (
        pred_to_initial_mse / initial_to_target_mse
        if initial_to_target_mse > 0
        else float("nan")
    )
    return {
        "dataset": dataset,
        "split": split_name,
        "rollout_steps": int(rollout_steps),
        "used": int(len(df)),
        "p_ratio_used": int(valid_p_ratio.sum()),
        "p_ratio_r2": r2_score(df["true_p_ratio"], df["pred_p_ratio"]),
        "p_ratio_pearson": pearson_r(df["true_p_ratio"], df["pred_p_ratio"]),
        "p_ratio_mse": (
            float(
                np.mean(
                    (
                        df.loc[valid_p_ratio, "true_p_ratio"]
                        - df.loc[valid_p_ratio, "pred_p_ratio"]
                    )
                    ** 2
                )
            )
            if int(valid_p_ratio.sum())
            else float("nan")
        ),
        "final_pos_mse": final_pos_mse,
        "initial_to_target_mse": initial_to_target_mse,
        "pred_to_initial_mse": pred_to_initial_mse,
        "movement_fraction_mse": movement_fraction,
        "rollout_error_fraction": error_fraction,
        "rollout_position_r2": (
            float(np.clip(1.0 - error_fraction, 0.0, 1.0))
            if np.isfinite(error_fraction)
            else float("nan")
        ),
    }


def evaluate_rollout(
    ae_model,
    dyn_model,
    sims,
    latent_stats: LatentNormalizer,
    *,
    cfg: dict,
    normalizers: dict[str, torch.Tensor],
    dataset: str,
    split_name: str,
    rollout_steps: int,
    device,
) -> tuple[pd.DataFrame, dict]:
    """Evaluate a stride-aware first- or second-order latent rollout."""

    rows = []
    ae_model.eval()
    dyn_model.eval()
    frame_skip = int(cfg.get("frame_skip", 1))
    pos_dim = int(cfg["pos_dim"])
    objective = str(cfg.get("propagator_objective", "one_step")).lower()
    stride = max(1, int(cfg.get("propagator_step_stride", 1)))
    initial_velocity = str(cfg.get("initial_velocity", "zero")).lower()
    rollout_history_frames = max(1, int(cfg.get("rollout_history_frames", 1)))
    fixed_observed_frames = tuple(
        int(frame) for frame in cfg.get("fixed_observed_frames", (1, 10))
    )
    fixed_history_end = max(0, int(cfg["ae_max_train_frames_per_sim"]) - 1)
    loss_mode = str(cfg.get("propagator_loss", "delta")).lower()
    use_static_context = bool(cfg.get("propagator_use_static_context", False))
    context_include_temperature = bool(
        cfg.get("propagator_context_include_temperature", False)
    )
    context_pool_mode = str(cfg.get("propagator_context_pool", "mean"))
    p_ratio_window = "full"
    trajectory_p_ratio_requested = _uses_trajectory_p_ratio(cfg)
    endpoint_p_ratio_requested = _uses_endpoint_p_ratio(cfg)
    rho_scale_mode = cfg.get("polar_rho_scale_mode")

    encode = lambda sim, t: encode_frame_latent(
        ae_model,
        sim,
        int(t),
        pos_dim=pos_dim,
        node_feature_mode=cfg["node_feature_mode"],
        normalizers=normalizers,
        device=device,
    )

    with torch.no_grad():
        for sim_idx, sim in enumerate(sims):
            sim_dataset_name = str(
                getattr(sim[0], "source_name", cfg["dataset_name"])
            )
            temperature_data = is_temperature_dataset(sim_dataset_name)
            use_trajectory_p_ratio = (
                not endpoint_p_ratio_requested
                and (temperature_data or trajectory_p_ratio_requested)
            )
            available = max(
                len(filtered_frame_ids(sim, frame_skip=frame_skip, include_last=True)) - 1,
                0,
            )
            filtered_steps = min(int(rollout_steps), available)
            if getattr(dyn_model, "uses_fixed_observed_state", False):
                filtered_steps = min(filtered_steps, fixed_history_end)
            target_index = frame_for_filtered_step(
                sim,
                filtered_steps,
                frame_skip=frame_skip,
            )
            if filtered_steps <= 0 or target_index <= 0:
                continue

            z0 = encode(sim, 0)
            z = z0.clone()
            z_previous = z0.clone()
            z_previous_previous = z0.clone()
            context = (
                encode_reference_context(
                    ae_model,
                    sim,
                    pos_dim=pos_dim,
                    normalizers=normalizers,
                    device=device,
                    include_temperature=context_include_temperature,
                    include_source_id=bool(
                        cfg.get("propagator_context_include_source_id", False)
                    ),
                    pool_mode=context_pool_mode,
                )
                if use_static_context
                else None
            )
            context_is_encoded = False
            if (
                context is not None
                and context_pool_mode.lower()
                in {"learned_attention", "attention", "set_attention"}
            ):
                context = dyn_model.context_projection(context)
                context_is_encoded = True
            rho_scale = initial_structure_scale(
                sim,
                mode=rho_scale_mode,
                pos_dim=pos_dim,
                device=device,
            )
            predicted_window = []
            ground_truth_window = []
            if isinstance(p_ratio_window, str) and p_ratio_window.lower() == "full":
                window_start_step = 0
            else:
                window_start_step = max(0, filtered_steps - max(2, int(p_ratio_window)) + 1)
            if use_trajectory_p_ratio and window_start_step == 0:
                predicted_window.append(clone_graph(sim[0]).cpu())
                ground_truth_window.append(clone_graph(sim[0]).cpu())

            start_step = stride
            prev_dz = None
            recurrent_memory = None
            lagged_history = None
            if (
                objective not in {"velocity", "second_order"}
                and objective not in KINEMATIC_OBJECTIVES
                and rollout_history_frames > 1
            ):
                observed_order = (rollout_history_frames - 1) * stride
                if filtered_steps < observed_order:
                    continue
                observed_index = frame_for_filtered_step(
                    sim, observed_order, frame_skip=frame_skip
                )
                z = encode(sim, observed_index)
                start_step = observed_order + stride
                if use_trajectory_p_ratio:
                    for order in range(stride, observed_order + 1, stride):
                        if order < window_start_step:
                            continue
                        frame_index = frame_for_filtered_step(
                            sim, order, frame_skip=frame_skip
                        )
                        predicted_window.append(
                            _observed_position_graph(
                                sim, frame_index, pos_dim=pos_dim
                            )
                        )
                        ground_truth_window.append(
                            clone_graph(sim[frame_index]).cpu()
                        )
            elif objective in {"velocity", "second_order"}:
                if initial_velocity == "zero" or filtered_steps <= 1:
                    prev_dz = torch.zeros_like(z)
                elif initial_velocity in {"first_step", "gt_first", "observed"}:
                    first_order = min(stride, filtered_steps)
                    first_index = frame_for_filtered_step(
                        sim,
                        first_order,
                        frame_skip=frame_skip,
                    )
                    z1 = encode(sim, first_index)
                    prev_dz = z1 - z0
                    z = z1.clone()
                    start_step = first_order + stride
                    if use_trajectory_p_ratio and first_order >= window_start_step:
                        predicted_window.append(clone_graph(sim[first_index]).cpu())
                        ground_truth_window.append(clone_graph(sim[first_index]).cpu())
                elif initial_velocity == "mean":
                    prev_dz = latent_stats.dz_mean.squeeze(0).to(device)
                else:
                    raise ValueError(f"Unknown initial_velocity: {initial_velocity}")
            elif objective in KINEMATIC_OBJECTIVES:
                if getattr(dyn_model, "uses_lagged_history", False):
                    first_observed, last_observed = fixed_observed_frames
                    if filtered_steps < last_observed:
                        continue
                    lagged_history = [
                        encode(
                            sim,
                            frame_for_filtered_step(
                                sim, order, frame_skip=frame_skip
                            ),
                        )
                        for order in range(
                            first_observed, last_observed + 1, stride
                        )
                    ]
                    z = lagged_history[-1].clone()
                    start_step = last_observed + stride
                    if use_trajectory_p_ratio:
                        for order in range(stride, last_observed + 1, stride):
                            frame_index = frame_for_filtered_step(
                                sim, order, frame_skip=frame_skip
                            )
                            if order >= window_start_step:
                                predicted_window.append(
                                    _observed_position_graph(
                                        sim, frame_index, pos_dim=pos_dim
                                    )
                                )
                                ground_truth_window.append(
                                    clone_graph(sim[frame_index]).cpu()
                                )
                elif getattr(dyn_model, "uses_recurrent_memory", False):
                    first_observed, last_observed = fixed_observed_frames
                    if filtered_steps < last_observed:
                        continue
                    recurrent_memory = dyn_model.initial_memory(
                        1, device=z.device, dtype=z.dtype
                    )
                    previous_index = frame_for_filtered_step(
                        sim,
                        max(0, first_observed - stride),
                        frame_skip=frame_skip,
                    )
                    z_previous = encode(sim, previous_index)
                    for order in range(first_observed, last_observed, stride):
                        observed_index = frame_for_filtered_step(
                            sim, order, frame_skip=frame_skip
                        )
                        observed_z = encode(sim, observed_index)
                        _, recurrent_memory = latent_step_recurrent_memory(
                            dyn_model,
                            observed_z,
                            z_previous,
                            recurrent_memory,
                            latent_stats,
                            context=context,
                        )
                        z_previous = observed_z
                    last_index = frame_for_filtered_step(
                        sim, last_observed, frame_skip=frame_skip
                    )
                    z = encode(sim, last_index)
                    start_step = last_observed + stride
                    if use_trajectory_p_ratio:
                        for order in range(stride, last_observed + 1, stride):
                            frame_index = frame_for_filtered_step(
                                sim, order, frame_skip=frame_skip
                            )
                            if order >= window_start_step:
                                predicted_window.append(
                                    _observed_position_graph(
                                        sim, frame_index, pos_dim=pos_dim
                                    )
                                )
                                ground_truth_window.append(
                                    clone_graph(sim[frame_index]).cpu()
                                )
                elif getattr(dyn_model, "uses_fixed_observed_state", False):
                    required_fixed_frames = int(
                        getattr(dyn_model, "fixed_history_size", 2)
                        if getattr(dyn_model, "uses_fixed_window_history", False)
                        else 2
                    )
                    if len(fixed_observed_frames) != required_fixed_frames:
                        raise ValueError(
                            f"fixed_observed_frames must contain exactly {required_fixed_frames} frames."
                        )
                    observed_order = fixed_observed_frames[-1]
                    if filtered_steps < observed_order:
                        continue
                    observed_indices = [
                        frame_for_filtered_step(sim, order, frame_skip=frame_skip)
                        for order in fixed_observed_frames
                    ]
                    fixed_observed_latents = [
                        encode(sim, index) for index in observed_indices
                    ]
                    z = fixed_observed_latents[-1].clone()
                    # Emit the final observed state itself, then propagate from
                    # the following step. This makes a requested z(5) horizon
                    # an evaluated initialization point rather than z(6).
                    start_step = observed_order
                    if use_trajectory_p_ratio:
                        for order in range(stride, observed_order + 1, stride):
                            frame_index = frame_for_filtered_step(
                                sim, order, frame_skip=frame_skip
                            )
                            if order >= window_start_step:
                                predicted_window.append(
                                    _observed_position_graph(
                                        sim, frame_index, pos_dim=pos_dim
                                    )
                                )
                                ground_truth_window.append(
                                    clone_graph(sim[frame_index]).cpu()
                                )
                elif getattr(dyn_model, "uses_history_state", False):
                    if initial_velocity not in THREE_FRAME_INITIALIZATIONS:
                        raise ValueError(
                            "The three-frame propagator requires "
                            "initial_velocity='three_frames'."
                        )
                    observed_order = max(2, rollout_history_frames - 1) * stride
                    if filtered_steps < observed_order:
                        continue
                    history_orders = [
                        observed_order - 2 * stride,
                        observed_order - stride,
                        observed_order,
                    ]
                    history_indices = [
                        frame_for_filtered_step(sim, order, frame_skip=frame_skip)
                        for order in history_orders
                    ]
                    z_history = [encode(sim, index) for index in history_indices]
                    z_previous_previous, z_previous, z = (
                        z_history[0].clone(),
                        z_history[1].clone(),
                        z_history[2].clone(),
                    )
                    start_step = observed_order + stride
                    if use_trajectory_p_ratio:
                        for order in range(stride, observed_order + 1, stride):
                            frame_index = frame_for_filtered_step(
                                sim, order, frame_skip=frame_skip
                            )
                            if order >= window_start_step:
                                predicted_window.append(
                                    _observed_position_graph(
                                        sim, frame_index, pos_dim=pos_dim
                                    )
                                )
                                ground_truth_window.append(
                                    clone_graph(sim[frame_index]).cpu()
                                )
                elif initial_velocity == "zero" or filtered_steps <= 1:
                    pass
                elif initial_velocity in {"first_step", "gt_first", "observed"}:
                    first_order = min(stride, filtered_steps)
                    first_index = frame_for_filtered_step(
                        sim,
                        first_order,
                        frame_skip=frame_skip,
                    )
                    z1 = encode(sim, first_index)
                    z_previous, z = z0.clone(), z1.clone()
                    start_step = first_order + stride
                    if use_trajectory_p_ratio and first_order >= window_start_step:
                        predicted_window.append(clone_graph(sim[first_index]).cpu())
                        ground_truth_window.append(clone_graph(sim[first_index]).cpu())
                elif initial_velocity == "mean":
                    z_previous = z - latent_stats.dz_mean.squeeze(0).to(device)
                else:
                    raise ValueError(f"Unknown initial_velocity: {initial_velocity}")

            step_orders = list(range(start_step, filtered_steps + 1, stride))
            for step in step_orders:
                fixed_observed_endpoint = (
                    getattr(dyn_model, "uses_fixed_observed_state", False)
                    and step == fixed_observed_frames[-1]
                )
                if fixed_observed_endpoint:
                    pass
                elif objective in {"velocity", "second_order"}:
                    z, prev_dz = latent_step_velocity(
                        dyn_model,
                        z,
                        prev_dz,
                        latent_stats,
                        context=context,
                    )
                elif objective in KINEMATIC_OBJECTIVES:
                    if getattr(dyn_model, "uses_lagged_history", False):
                        z_next = latent_step_lagged_history(
                            dyn_model,
                            z,
                            lagged_history[0],
                            z0,
                            latent_stats,
                            frame_gap=len(lagged_history) - 1,
                            context=context,
                        )
                    elif getattr(dyn_model, "uses_recurrent_memory", False):
                        z_next, recurrent_memory = latent_step_recurrent_memory(
                            dyn_model,
                            z,
                            z_previous,
                            recurrent_memory,
                            latent_stats,
                            context=context,
                        )
                    elif getattr(dyn_model, "uses_fixed_observed_state", False):
                        progress = (
                            float(step) / max(1, fixed_history_end)
                            if getattr(dyn_model, "include_progress", False)
                            else None
                        )
                        if getattr(dyn_model, "uses_fixed_window_history", False):
                            z_next = latent_step_fixed_window(
                                dyn_model, z, fixed_observed_latents, latent_stats,
                                context=context,
                                context_is_encoded=context_is_encoded,
                                progress=progress,
                                observed_frame_gap=(
                                    fixed_observed_frames[-1]
                                    - fixed_observed_frames[-2]
                                ),
                            )
                        else:
                            z_next = latent_step_fixed_history(
                                dyn_model, z, fixed_observed_latents[0], fixed_observed_latents[1],
                                latent_stats,
                                observed_frame_gap=(fixed_observed_frames[1] - fixed_observed_frames[0]),
                                context=context,
                                context_is_encoded=context_is_encoded,
                                progress=progress,
                            )
                    elif getattr(dyn_model, "uses_history_state", False):
                        z_next = latent_step_history(
                            dyn_model,
                            z,
                            z_previous,
                            z_previous_previous,
                            z0,
                            latent_stats,
                            context=context,
                        )
                    else:
                        z_next = latent_step_kinematic(
                            dyn_model,
                            z,
                            z_previous,
                            z0,
                            latent_stats,
                            progress=step / max(1, available),
                            context=context,
                        )
                    if getattr(dyn_model, "uses_fixed_observed_state", False):
                        z = z_next
                    elif getattr(dyn_model, "uses_lagged_history", False):
                        lagged_history = [*lagged_history[1:], z_next]
                        z = z_next
                    elif getattr(dyn_model, "uses_recurrent_memory", False):
                        z_previous, z = z, z_next
                    else:
                        z_previous_previous, z_previous, z = z_previous, z, z_next
                else:
                    z = latent_step(
                        dyn_model,
                        z,
                        latent_stats,
                        loss_mode=loss_mode,
                        context=context,
                        context_is_encoded=context_is_encoded,
                        rho_scale=rho_scale,
                    )
                if (
                    use_trajectory_p_ratio
                    and step >= window_start_step
                    and not fixed_observed_endpoint
                ):
                    step_index = frame_for_filtered_step(sim, step, frame_skip=frame_skip)
                    predicted_window.append(
                        decode_latent_to_graph(
                            ae_model,
                            sim,
                            z,
                            step_index,
                            pos_dim=pos_dim,
                            ae_target_mode=cfg["ae_target_mode"],
                            normalizers=normalizers,
                            device=device,
                        )
                    )
                    ground_truth_window.append(clone_graph(sim[step_index]).cpu())

            pred_graph = decode_latent_to_graph(
                ae_model,
                sim,
                z,
                target_index,
                pos_dim=pos_dim,
                ae_target_mode=cfg["ae_target_mode"],
                normalizers=normalizers,
                device=device,
            )
            if (
                use_trajectory_p_ratio
                and getattr(dyn_model, "uses_fixed_observed_state", False)
                and filtered_steps == fixed_observed_frames[-1]
                and predicted_window
            ):
                predicted_window[-1] = _p_ratio_position_graph(
                    pred_graph, pos_dim=pos_dim
                )
            if use_trajectory_p_ratio and len(predicted_window) >= 2:
                pred_pr = temperature_p_ratio(predicted_window, cfg=cfg, last_index=-1)
                true_pr = temperature_p_ratio(ground_truth_window, cfg=cfg, last_index=-1)
            else:
                pred_pr = float(
                    calc_p_ratio_rollout_sides([clone_graph(sim[0]).cpu(), pred_graph], -1)
                )
                true_pr = ground_truth_p_ratio(
                    sim,
                    target_index,
                    dataset_name=sim_dataset_name,
                    cfg=cfg,
                )

            initial_pos = sim[0].x[:, :pos_dim].cpu().float()
            target_pos = sim[target_index].x[:, :pos_dim].cpu().float()
            pred_pos = pred_graph.x[:, :pos_dim].cpu().float()
            initial_to_target_mse = float(F.mse_loss(initial_pos, target_pos).item())
            pred_to_initial_mse = float(F.mse_loss(pred_pos, initial_pos).item())
            final_pos_mse = float(F.mse_loss(pred_pos, target_pos).item())
            error_fraction = (
                final_pos_mse / initial_to_target_mse
                if initial_to_target_mse > 0
                else float("nan")
            )
            movement_fraction = (
                pred_to_initial_mse / initial_to_target_mse
                if initial_to_target_mse > 0
                else float("nan")
            )
            rows.append(
                {
                    "dataset": dataset,
                    "split": split_name,
                    "sim_idx": int(sim_idx),
                    "source": str(getattr(sim[0], "source_name", dataset)),
                    "target_index": int(target_index),
                    "rollout_steps": int(rollout_steps),
                    "filtered_steps": int(filtered_steps),
                    "temperature": float(getattr(sim[0], "temperature", np.nan)),
                    "rho_scale_mode": str(rho_scale_mode or "none"),
                    "rho_scale": (
                        float(rho_scale.detach().cpu().reshape(-1)[0])
                        if rho_scale is not None
                        else float("nan")
                    ),
                    "pred_p_ratio": pred_pr,
                    "true_p_ratio": true_pr,
                    "final_pos_mse": final_pos_mse,
                    "initial_to_target_mse": initial_to_target_mse,
                    "pred_to_initial_mse": pred_to_initial_mse,
                    "movement_fraction_mse": movement_fraction,
                    "rollout_error_fraction": error_fraction,
                    "rollout_position_r2": (
                        float(np.clip(1.0 - error_fraction, 0.0, 1.0))
                        if np.isfinite(error_fraction)
                        else float("nan")
                    ),
                }
            )

    frame = pd.DataFrame(rows)
    return frame, rollout_metrics(
        frame,
        dataset=dataset,
        split_name=split_name,
        rollout_steps=rollout_steps,
    )


def evaluate_rollout_horizons(
    ae_model,
    dyn_model,
    sims,
    latent_stats: LatentNormalizer,
    *,
    cfg: dict,
    normalizers: dict[str, torch.Tensor],
    dataset: str,
    split_name: str,
    rollout_steps,
    device,
    endpoint_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Roll out once per simulation and record metrics at requested horizons."""

    horizons = rollout_steps_for_sims(
        sims,
        rollout_steps,
        frame_skip=int(cfg.get("frame_skip", 1)),
    )
    if not horizons:
        return pd.DataFrame(), pd.DataFrame()

    frame_skip = int(cfg.get("frame_skip", 1))
    pos_dim = int(cfg["pos_dim"])
    objective = str(cfg.get("propagator_objective", "one_step")).lower()
    stride = max(1, int(cfg.get("propagator_step_stride", 1)))
    initial_velocity = str(cfg.get("initial_velocity", "zero")).lower()
    rollout_history_frames = max(1, int(cfg.get("rollout_history_frames", 1)))
    fixed_observed_frames = tuple(
        int(frame) for frame in cfg.get("fixed_observed_frames", (1, 10))
    )
    fixed_history_end = max(0, int(cfg["ae_max_train_frames_per_sim"]) - 1)
    loss_mode = str(cfg.get("propagator_loss", "delta")).lower()
    use_static_context = bool(cfg.get("propagator_use_static_context", False))
    context_include_temperature = bool(
        cfg.get("propagator_context_include_temperature", False)
    )
    context_pool_mode = str(cfg.get("propagator_context_pool", "mean"))
    p_ratio_window = "full"
    trajectory_p_ratio_requested = _uses_trajectory_p_ratio(cfg)
    endpoint_p_ratio_requested = _uses_endpoint_p_ratio(cfg)
    rho_scale_mode = cfg.get("polar_rho_scale_mode")

    if getattr(dyn_model, "uses_fixed_observed_state", False):
        horizons = [step for step in horizons if step <= fixed_history_end]
        if not horizons:
            return pd.DataFrame(), pd.DataFrame()

    if stride > 1:
        rollout_origin = (
            fixed_observed_frames[-1]
            if getattr(dyn_model, "uses_fixed_observed_state", False)
            else 0
        )
        horizons = sorted(
            {
                rollout_origin
                + max(0, (int(step) - rollout_origin) // stride) * stride
                for step in horizons
            }
        )
        horizons = [step for step in horizons if step > 0]
        if not horizons:
            return pd.DataFrame(), pd.DataFrame()

    rows = []
    horizon_set = set(horizons)
    max_horizon = max(horizons)
    ae_model.eval()
    dyn_model.eval()

    def encode(sim, frame_idx):
        return encode_frame_latent(
            ae_model,
            sim,
            int(frame_idx),
            pos_dim=pos_dim,
            node_feature_mode=cfg["node_feature_mode"],
            normalizers=normalizers,
            device=device,
        )

    def decode(sim, z, target_index):
        return decode_latent_to_graph(
            ae_model,
            sim,
            z,
            target_index,
            pos_dim=pos_dim,
            ae_target_mode=cfg["ae_target_mode"],
            normalizers=normalizers,
            device=device,
        )

    with torch.no_grad():
        for sim_idx, sim in enumerate(sims):
            sim_dataset_name = str(
                getattr(sim[0], "source_name", cfg["dataset_name"])
            )
            temperature_data = is_temperature_dataset(sim_dataset_name)
            use_trajectory_p_ratio = (
                not endpoint_p_ratio_requested
                and (temperature_data or trajectory_p_ratio_requested)
            ) and not endpoint_only
            available = max(
                len(filtered_frame_ids(sim, frame_skip=frame_skip, include_last=True)) - 1,
                0,
            )
            sim_horizons = [step for step in horizons if step <= available]
            if not sim_horizons:
                continue

            z0 = encode(sim, 0)
            z = z0.clone()
            z_previous = z0.clone()
            z_previous_previous = z0.clone()
            context = (
                encode_reference_context(
                    ae_model,
                    sim,
                    pos_dim=pos_dim,
                    normalizers=normalizers,
                    device=device,
                    include_temperature=context_include_temperature,
                    include_source_id=bool(
                        cfg.get("propagator_context_include_source_id", False)
                    ),
                    pool_mode=context_pool_mode,
                )
                if use_static_context
                else None
            )
            context_is_encoded = False
            if (
                context is not None
                and context_pool_mode.lower()
                in {"learned_attention", "attention", "set_attention"}
            ):
                context = dyn_model.context_projection(context)
                context_is_encoded = True
            rho_scale = initial_structure_scale(
                sim,
                mode=rho_scale_mode,
                pos_dim=pos_dim,
                device=device,
            )
            prev_dz = None
            recurrent_memory = None
            lagged_history = None
            start_step = stride
            predicted_path = [_p_ratio_position_graph(sim[0], pos_dim=pos_dim)] if use_trajectory_p_ratio else []
            ground_truth_path = [_p_ratio_position_graph(sim[0], pos_dim=pos_dim)] if use_trajectory_p_ratio else []

            if (
                objective not in {"velocity", "second_order"}
                and objective not in KINEMATIC_OBJECTIVES
                and rollout_history_frames > 1
            ):
                observed_order = (rollout_history_frames - 1) * stride
                if available < observed_order:
                    continue
                observed_index = frame_for_filtered_step(
                    sim, observed_order, frame_skip=frame_skip
                )
                z = encode(sim, observed_index)
                start_step = observed_order + stride
                if use_trajectory_p_ratio:
                    for order in range(stride, observed_order + 1, stride):
                        frame_index = frame_for_filtered_step(
                            sim, order, frame_skip=frame_skip
                        )
                        predicted_path.append(
                            _observed_position_graph(
                                sim, frame_index, pos_dim=pos_dim
                            )
                        )
                        ground_truth_path.append(
                            _p_ratio_position_graph(sim[frame_index], pos_dim=pos_dim)
                        )
            elif objective in {"velocity", "second_order"}:
                if initial_velocity == "zero":
                    prev_dz = torch.zeros_like(z)
                elif initial_velocity in {"first_step", "gt_first", "observed"}:
                    first_index = frame_for_filtered_step(sim, stride, frame_skip=frame_skip)
                    z1 = encode(sim, first_index)
                    prev_dz = z1 - z0
                    z = z1.clone()
                    start_step = 2 * stride
                    if use_trajectory_p_ratio:
                        predicted_path.append(_p_ratio_position_graph(sim[first_index], pos_dim=pos_dim))
                        ground_truth_path.append(_p_ratio_position_graph(sim[first_index], pos_dim=pos_dim))
                elif initial_velocity == "mean":
                    prev_dz = latent_stats.dz_mean.squeeze(0).to(device)
                else:
                    raise ValueError(f"Unknown initial_velocity: {initial_velocity}")
            elif objective in KINEMATIC_OBJECTIVES:
                if getattr(dyn_model, "uses_lagged_history", False):
                    first_observed, last_observed = fixed_observed_frames
                    if available < last_observed:
                        continue
                    lagged_history = [
                        encode(
                            sim,
                            frame_for_filtered_step(
                                sim, order, frame_skip=frame_skip
                            ),
                        )
                        for order in range(
                            first_observed, last_observed + 1, stride
                        )
                    ]
                    z = lagged_history[-1].clone()
                    start_step = last_observed + stride
                    if use_trajectory_p_ratio:
                        for order in range(stride, last_observed + 1, stride):
                            frame_index = frame_for_filtered_step(
                                sim, order, frame_skip=frame_skip
                            )
                            predicted_path.append(
                                _observed_position_graph(
                                    sim, frame_index, pos_dim=pos_dim
                                )
                            )
                            ground_truth_path.append(
                                _p_ratio_position_graph(
                                    sim[frame_index], pos_dim=pos_dim
                                )
                            )
                elif getattr(dyn_model, "uses_recurrent_memory", False):
                    first_observed, last_observed = fixed_observed_frames
                    if available < last_observed:
                        continue
                    recurrent_memory = dyn_model.initial_memory(
                        1, device=z.device, dtype=z.dtype
                    )
                    previous_index = frame_for_filtered_step(
                        sim,
                        max(0, first_observed - stride),
                        frame_skip=frame_skip,
                    )
                    z_previous = encode(sim, previous_index)
                    for order in range(first_observed, last_observed, stride):
                        observed_index = frame_for_filtered_step(
                            sim, order, frame_skip=frame_skip
                        )
                        observed_z = encode(sim, observed_index)
                        _, recurrent_memory = latent_step_recurrent_memory(
                            dyn_model,
                            observed_z,
                            z_previous,
                            recurrent_memory,
                            latent_stats,
                            context=context,
                        )
                        z_previous = observed_z
                    last_index = frame_for_filtered_step(
                        sim, last_observed, frame_skip=frame_skip
                    )
                    z = encode(sim, last_index)
                    start_step = last_observed + stride
                    if use_trajectory_p_ratio:
                        for order in range(stride, last_observed + 1, stride):
                            frame_index = frame_for_filtered_step(
                                sim, order, frame_skip=frame_skip
                            )
                            predicted_path.append(
                                _observed_position_graph(
                                    sim, frame_index, pos_dim=pos_dim
                                )
                            )
                            ground_truth_path.append(
                                _p_ratio_position_graph(
                                    sim[frame_index], pos_dim=pos_dim
                                )
                            )
                elif getattr(dyn_model, "uses_fixed_observed_state", False):
                    required_fixed_frames = int(
                        getattr(dyn_model, "fixed_history_size", 2)
                        if getattr(dyn_model, "uses_fixed_window_history", False)
                        else 2
                    )
                    if len(fixed_observed_frames) != required_fixed_frames:
                        raise ValueError(
                            f"fixed_observed_frames must contain exactly {required_fixed_frames} frames."
                        )
                    observed_order = fixed_observed_frames[-1]
                    if available < observed_order:
                        continue
                    observed_indices = [
                        frame_for_filtered_step(sim, order, frame_skip=frame_skip)
                        for order in fixed_observed_frames
                    ]
                    fixed_observed_latents = [
                        encode(sim, index) for index in observed_indices
                    ]
                    z = fixed_observed_latents[-1].clone()
                    # The second fixed latent is observed and reported as such;
                    # autoregressive prediction begins at the following frame.
                    start_step = observed_order
                    if use_trajectory_p_ratio:
                        for order in range(stride, observed_order + 1, stride):
                            frame_index = frame_for_filtered_step(
                                sim, order, frame_skip=frame_skip
                            )
                            predicted_path.append(
                                _observed_position_graph(
                                    sim, frame_index, pos_dim=pos_dim
                                )
                            )
                            ground_truth_path.append(
                                _p_ratio_position_graph(sim[frame_index], pos_dim=pos_dim)
                            )
                elif getattr(dyn_model, "uses_history_state", False):
                    if initial_velocity not in THREE_FRAME_INITIALIZATIONS:
                        raise ValueError(
                            "The three-frame propagator requires "
                            "initial_velocity='three_frames'."
                        )
                    observed_order = max(2, rollout_history_frames - 1) * stride
                    if available < observed_order:
                        continue
                    history_orders = [
                        observed_order - 2 * stride,
                        observed_order - stride,
                        observed_order,
                    ]
                    history_indices = [
                        frame_for_filtered_step(sim, order, frame_skip=frame_skip)
                        for order in history_orders
                    ]
                    z_history = [encode(sim, index) for index in history_indices]
                    z_previous_previous, z_previous, z = (
                        z_history[0].clone(),
                        z_history[1].clone(),
                        z_history[2].clone(),
                    )
                    start_step = observed_order + stride
                    if use_trajectory_p_ratio:
                        for order in range(stride, observed_order + 1, stride):
                            frame_index = frame_for_filtered_step(
                                sim, order, frame_skip=frame_skip
                            )
                            predicted_path.append(
                                _observed_position_graph(
                                    sim, frame_index, pos_dim=pos_dim
                                )
                            )
                            ground_truth_path.append(
                                _p_ratio_position_graph(sim[frame_index], pos_dim=pos_dim)
                            )
                elif initial_velocity == "zero":
                    pass
                elif initial_velocity in {"first_step", "gt_first", "observed"}:
                    first_index = frame_for_filtered_step(
                        sim, stride, frame_skip=frame_skip
                    )
                    z1 = encode(sim, first_index)
                    z_previous, z = z0.clone(), z1.clone()
                    start_step = 2 * stride
                    if use_trajectory_p_ratio:
                        predicted_path.append(_p_ratio_position_graph(sim[first_index], pos_dim=pos_dim))
                        ground_truth_path.append(_p_ratio_position_graph(sim[first_index], pos_dim=pos_dim))
                elif initial_velocity == "mean":
                    z_previous = z - latent_stats.dz_mean.squeeze(0).to(device)
                else:
                    raise ValueError(f"Unknown initial_velocity: {initial_velocity}")

            for step in range(start_step, min(max_horizon, available) + 1, stride):
                fixed_observed_endpoint = (
                    getattr(dyn_model, "uses_fixed_observed_state", False)
                    and step == fixed_observed_frames[-1]
                )
                if fixed_observed_endpoint:
                    pass
                elif objective in {"velocity", "second_order"}:
                    z, prev_dz = latent_step_velocity(
                        dyn_model,
                        z,
                        prev_dz,
                        latent_stats,
                        context=context,
                    )
                elif objective in KINEMATIC_OBJECTIVES:
                    if getattr(dyn_model, "uses_lagged_history", False):
                        z_next = latent_step_lagged_history(
                            dyn_model,
                            z,
                            lagged_history[0],
                            z0,
                            latent_stats,
                            frame_gap=len(lagged_history) - 1,
                            context=context,
                        )
                    elif getattr(dyn_model, "uses_recurrent_memory", False):
                        z_next, recurrent_memory = latent_step_recurrent_memory(
                            dyn_model,
                            z,
                            z_previous,
                            recurrent_memory,
                            latent_stats,
                            context=context,
                        )
                    elif getattr(dyn_model, "uses_fixed_observed_state", False):
                        progress = (
                            float(step) / max(1, fixed_history_end)
                            if getattr(dyn_model, "include_progress", False)
                            else None
                        )
                        if getattr(dyn_model, "uses_fixed_window_history", False):
                            z_next = latent_step_fixed_window(
                                dyn_model, z, fixed_observed_latents, latent_stats,
                                context=context,
                                context_is_encoded=context_is_encoded,
                                progress=progress,
                                observed_frame_gap=(
                                    fixed_observed_frames[-1]
                                    - fixed_observed_frames[-2]
                                ),
                            )
                        else:
                            z_next = latent_step_fixed_history(
                                dyn_model, z, fixed_observed_latents[0], fixed_observed_latents[1],
                                latent_stats,
                                observed_frame_gap=(fixed_observed_frames[1] - fixed_observed_frames[0]),
                                context=context,
                                context_is_encoded=context_is_encoded,
                                progress=progress,
                            )
                    elif getattr(dyn_model, "uses_history_state", False):
                        z_next = latent_step_history(
                            dyn_model,
                            z,
                            z_previous,
                            z_previous_previous,
                            z0,
                            latent_stats,
                            context=context,
                        )
                    else:
                        z_next = latent_step_kinematic(
                            dyn_model,
                            z,
                            z_previous,
                            z0,
                            latent_stats,
                            progress=step / max(1, available),
                            context=context,
                        )
                    if getattr(dyn_model, "uses_fixed_observed_state", False):
                        z = z_next
                    elif getattr(dyn_model, "uses_lagged_history", False):
                        lagged_history = [*lagged_history[1:], z_next]
                        z = z_next
                    elif getattr(dyn_model, "uses_recurrent_memory", False):
                        z_previous, z = z, z_next
                    else:
                        z_previous_previous, z_previous, z = z_previous, z, z_next
                else:
                    z = latent_step(
                        dyn_model,
                        z,
                        latent_stats,
                        loss_mode=loss_mode,
                        context=context,
                        context_is_encoded=context_is_encoded,
                        rho_scale=rho_scale,
                    )

                target_index = frame_for_filtered_step(sim, step, frame_skip=frame_skip)
                pred_graph = None
                if use_trajectory_p_ratio or step in horizon_set:
                    pred_graph = decode(sim, z, target_index)
                if use_trajectory_p_ratio:
                    if fixed_observed_endpoint:
                        predicted_path[-1] = _p_ratio_position_graph(
                            pred_graph, pos_dim=pos_dim
                        )
                    else:
                        predicted_path.append(_p_ratio_position_graph(pred_graph, pos_dim=pos_dim))
                        ground_truth_path.append(_p_ratio_position_graph(sim[target_index], pos_dim=pos_dim))
                if step not in horizon_set:
                    continue

                if use_trajectory_p_ratio:
                    if isinstance(p_ratio_window, str) and p_ratio_window.lower() == "full":
                        predicted_window = predicted_path
                        ground_truth_window = ground_truth_path
                    else:
                        window_size = max(2, int(p_ratio_window))
                        predicted_window = predicted_path[-window_size:]
                        ground_truth_window = ground_truth_path[-window_size:]
                    pred_pr = temperature_p_ratio(predicted_window, cfg=cfg, last_index=-1)
                    true_pr = temperature_p_ratio(ground_truth_window, cfg=cfg, last_index=-1)
                    endpoint_pred_pr = float(
                        calc_p_ratio_rollout_sides(
                            [clone_graph(sim[0]).cpu(), pred_graph],
                            -1,
                        )
                    )
                    endpoint_true_pr = float(calc_p_ratio_rollout_sides(sim, target_index))
                else:
                    pred_pr = float(
                        calc_p_ratio_rollout_sides(
                            [clone_graph(sim[0]).cpu(), pred_graph],
                            -1,
                        )
                    )
                    true_pr = ground_truth_p_ratio(
                        sim,
                        target_index,
                        dataset_name=sim_dataset_name,
                        cfg=cfg,
                    )
                    endpoint_pred_pr = pred_pr
                    endpoint_true_pr = true_pr

                initial_pos = sim[0].x[:, :pos_dim].cpu().float()
                target_pos = sim[target_index].x[:, :pos_dim].cpu().float()
                pred_pos = pred_graph.x[:, :pos_dim].cpu().float()
                initial_to_target_mse = float(F.mse_loss(initial_pos, target_pos).item())
                pred_to_initial_mse = float(F.mse_loss(pred_pos, initial_pos).item())
                final_pos_mse = float(F.mse_loss(pred_pos, target_pos).item())
                error_fraction = (
                    final_pos_mse / initial_to_target_mse
                    if initial_to_target_mse > 0
                    else float("nan")
                )
                movement_fraction = (
                    pred_to_initial_mse / initial_to_target_mse
                    if initial_to_target_mse > 0
                    else float("nan")
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "split": split_name,
                        "sim_idx": int(sim_idx),
                        "source": str(getattr(sim[0], "source_name", dataset)),
                        "target_index": int(target_index),
                        "rollout_steps": int(step),
                        "filtered_steps": int(step),
                        "temperature": float(getattr(sim[0], "temperature", np.nan)),
                        "rho_scale_mode": str(rho_scale_mode or "none"),
                        "rho_scale": (
                            float(rho_scale.detach().cpu().reshape(-1)[0])
                            if rho_scale is not None
                            else float("nan")
                        ),
                        "pred_p_ratio": pred_pr,
                        "true_p_ratio": true_pr,
                        "endpoint_pred_p_ratio": endpoint_pred_pr,
                        "endpoint_true_p_ratio": endpoint_true_pr,
                        "final_pos_mse": final_pos_mse,
                        "initial_to_target_mse": initial_to_target_mse,
                        "pred_to_initial_mse": pred_to_initial_mse,
                        "movement_fraction_mse": movement_fraction,
                        "rollout_error_fraction": error_fraction,
                        "rollout_position_r2": (
                            float(np.clip(1.0 - error_fraction, 0.0, 1.0))
                            if np.isfinite(error_fraction)
                            else float("nan")
                        ),
                    }
                )

    raw = pd.DataFrame(rows)
    if raw.empty:
        return (
            pd.DataFrame(
                columns=[
                    "dataset", "split", "sim_idx", "source", "rollout_steps",
                    "true_p_ratio", "pred_p_ratio",
                ]
            ),
            pd.DataFrame(),
        )
    stats = pd.DataFrame(
        [
            rollout_metrics(
                group,
                dataset=dataset,
                split_name=split_name,
                rollout_steps=int(step),
            )
            for step, group in raw.groupby("rollout_steps", sort=True)
        ]
    )
    return raw, stats


def evaluate_autoencoder_reconstruction_horizons(
    ae_model,
    sims,
    *,
    cfg: dict,
    normalizers: dict[str, torch.Tensor],
    dataset: str,
    split_name: str,
    rollout_steps,
    device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Measure reconstruction quality without introducing propagator error."""

    horizons = rollout_steps_for_sims(
        sims,
        rollout_steps,
        frame_skip=int(cfg.get("frame_skip", 1)),
    )
    if not horizons:
        return pd.DataFrame(), pd.DataFrame()

    frame_skip = int(cfg.get("frame_skip", 1))
    pos_dim = int(cfg["pos_dim"])
    trajectory_p_ratio_requested = _uses_trajectory_p_ratio(cfg)
    endpoint_p_ratio_requested = _uses_endpoint_p_ratio(cfg)
    horizon_set = set(horizons)
    max_horizon = max(horizons)
    rows = []
    ae_model.eval()

    with torch.no_grad():
        for sim_idx, sim in enumerate(sims):
            sim_dataset_name = str(
                getattr(sim[0], "source_name", cfg["dataset_name"])
            )
            temperature_data = is_temperature_dataset(sim_dataset_name)
            use_trajectory_p_ratio = (
                not endpoint_p_ratio_requested
                and (temperature_data or trajectory_p_ratio_requested)
            )
            available = max(
                len(filtered_frame_ids(sim, frame_skip=frame_skip, include_last=True)) - 1,
                0,
            )
            sim_horizons = [step for step in horizons if step <= available]
            if not sim_horizons:
                continue

            pred_path = [_p_ratio_position_graph(sim[0], pos_dim=pos_dim)]
            true_path = [_p_ratio_position_graph(sim[0], pos_dim=pos_dim)]
            for step in range(1, min(max_horizon, available) + 1):
                target_index = frame_for_filtered_step(sim, step, frame_skip=frame_skip)
                z = encode_frame_latent(
                    ae_model,
                    sim,
                    target_index,
                    pos_dim=pos_dim,
                    node_feature_mode=cfg["node_feature_mode"],
                    normalizers=normalizers,
                    device=device,
                )
                pred_graph = decode_latent_to_graph(
                    ae_model,
                    sim,
                    z,
                    target_index,
                    pos_dim=pos_dim,
                    ae_target_mode=cfg["ae_target_mode"],
                    normalizers=normalizers,
                    device=device,
                )
                pred_path.append(_p_ratio_position_graph(pred_graph, pos_dim=pos_dim))
                true_path.append(_p_ratio_position_graph(sim[target_index], pos_dim=pos_dim))
                if step not in horizon_set:
                    continue

                if use_trajectory_p_ratio:
                    pred_pr = temperature_p_ratio(pred_path, cfg=cfg, last_index=-1)
                    true_pr = temperature_p_ratio(true_path, cfg=cfg, last_index=-1)
                    endpoint_pred_pr = float(
                        calc_p_ratio_rollout_sides(
                            [clone_graph(sim[0]).cpu(), pred_graph],
                            -1,
                        )
                    )
                    endpoint_true_pr = float(calc_p_ratio_rollout_sides(sim, target_index))
                else:
                    pred_pr = float(
                        calc_p_ratio_rollout_sides(
                            [clone_graph(sim[0]).cpu(), pred_graph],
                            -1,
                        )
                    )
                    true_pr = ground_truth_p_ratio(
                        sim,
                        target_index,
                        dataset_name=sim_dataset_name,
                        cfg=cfg,
                    )
                    endpoint_pred_pr = pred_pr
                    endpoint_true_pr = true_pr

                initial_pos = sim[0].x[:, :pos_dim].cpu().float()
                target_pos = sim[target_index].x[:, :pos_dim].cpu().float()
                pred_pos = pred_graph.x[:, :pos_dim].cpu().float()
                initial_to_target_mse = float(F.mse_loss(initial_pos, target_pos).item())
                final_pos_mse = float(F.mse_loss(pred_pos, target_pos).item())
                transverse_mse = float(
                    F.mse_loss(pred_pos[:, 1], target_pos[:, 1]).item()
                )
                rows.append(
                    {
                        "dataset": dataset,
                        "split": split_name,
                        "sim_idx": int(sim_idx),
                        "source": str(getattr(sim[0], "source_name", dataset)),
                        "target_index": int(target_index),
                        "rollout_steps": int(step),
                        "pred_p_ratio": pred_pr,
                        "true_p_ratio": true_pr,
                        "endpoint_pred_p_ratio": endpoint_pred_pr,
                        "endpoint_true_p_ratio": endpoint_true_pr,
                        "final_pos_mse": final_pos_mse,
                        "transverse_pos_mse": transverse_mse,
                        "initial_to_target_mse": initial_to_target_mse,
                        "pred_to_initial_mse": float(
                            F.mse_loss(pred_pos, initial_pos).item()
                        ),
                    }
                )

    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw, pd.DataFrame()
    stats_rows = []
    for step, group in raw.groupby("rollout_steps"):
        metric = rollout_metrics(
            group,
            dataset=dataset,
            split_name=split_name,
            rollout_steps=int(step),
        )
        metric["transverse_pos_mse"] = float(group["transverse_pos_mse"].mean())
        stats_rows.append(metric)
    return raw, pd.DataFrame(stats_rows)


def train_latent_autoencoder_experiment(source_spec: dict, cfg: dict, *, device) -> dict:
    """Train only the latent autoencoder and return splits for analysis notebooks."""

    params = {**cfg, **source_spec}
    label = source_spec["label"]
    dataset_name = params["dataset_name"]
    p_ratio_fn = lambda sim, idx=-1: ground_truth_p_ratio(
        sim,
        idx,
        dataset_name=dataset_name,
        cfg=cfg,
    )
    train_data, val_data, test_data, split_info = resolve_train_val_test(
        source_spec,
        params,
        split_seed=cfg.get("split_seed"),
        p_ratio_fn=p_ratio_fn,
    )
    if not bool(params.get("static_context_use_physical_reference", True)):
        _use_normalized_reference_context(
            (train_data, val_data, test_data), pos_dim=int(params["pos_dim"])
        )
    if not train_data:
        raise ValueError("Autoencoder training split is empty.")
    if not val_data:
        raise ValueError(
            "Autoencoder validation split is empty. Reduce the requested training "
            "trajectory count or reserve at least one validation trajectory."
        )
    if bool(params.get("temperature_pratio_use_training_prior", False)) and is_temperature_dataset(
        dataset_name
    ):
        prior_cfg = dict(params)
        prior_cfg.pop("temperature_pratio_prior", None)
        prior_values = np.asarray(
            [temperature_p_ratio(sim, cfg=prior_cfg, last_index=-1) for sim in train_data],
            dtype=float,
        )
        finite_prior_values = prior_values[np.isfinite(prior_values)]
        if len(finite_prior_values):
            params["temperature_pratio_prior"] = float(np.mean(finite_prior_values))
    print(f"\n=== {label} ===")
    print(pd.DataFrame(split_info).to_string(index=False))

    pos_dim = int(params["pos_dim"])
    batch_graphs = int(params["batch_graphs"])
    frame_skip = int(params.get("frame_skip", 1))
    ae_train_frame_skip = int(params.get("ae_train_frame_skip", frame_skip))
    ae_frame_budget = int(params["ae_max_train_frames_per_sim"])
    ae_val_frame_skip = int(params.get("ae_val_frame_skip", frame_skip))
    ae_val_frame_budget = int(
        params.get("ae_max_val_frames_per_sim", ae_frame_budget)
    )
    node_mode = params["node_feature_mode"]
    target_mode = params["ae_target_mode"]
    edge_mode = str(params.get("edge_mode", "stored"))
    pretrained_ae_path = params.get("pretrained_ae_cache_path")
    use_pretrained_ae = bool(
        pretrained_ae_path
        and Path(pretrained_ae_path).expanduser().exists()
        and not bool(params.get("force_train_autoencoder", False))
        and _pretrained_ae_matches_requested_config(pretrained_ae_path, params)
    )

    train_frames = make_frame_index(
        train_data,
        frame_skip=ae_train_frame_skip,
        max_frames_per_sim=ae_frame_budget,
        include_last=True,
        start_frame_order=int(params.get("train_frame_start_order", 0)),
    )
    val_frames = make_frame_index(
        val_data,
        frame_skip=ae_val_frame_skip,
        max_frames_per_sim=ae_val_frame_budget,
        include_last=True,
        start_frame_order=int(params.get("train_frame_start_order", 0)),
    )
    if bool(params.get("ae_balance_sources", False)):
        train_rows_per_source = int(
            params.get("ae_train_rows_per_source", ae_frame_budget)
        )
        train_frames = _balance_rows_by_source(
            train_data,
            train_frames,
            rows_per_source=train_rows_per_source,
            seed=int(params.get("model_seed", 0)) + 101,
        )
        val_groups = _rows_by_source(val_data, val_frames)
        if len(val_groups) > 1:
            val_frames = _balance_rows_by_source(
                val_data,
                val_frames,
                rows_per_source=min(len(rows) for rows in val_groups.values()),
                seed=int(params.get("model_seed", 0)) + 102,
            )
        print(
            "source-balanced AE rows:",
            {key: len(value) for key, value in _rows_by_source(train_data, train_frames).items()},
        )
    if bool(params.get("ae_mix_sources", False)):
        print(
            "source-mixed AE coverage:",
            {
                source: {
                    "trajectories": len({int(row[0]) for row in source_rows}),
                    "frames": len(source_rows),
                }
                for source, source_rows in _rows_by_source(
                    train_data, train_frames
                ).items()
            },
        )
    use_cached_normalizers = bool(
        use_pretrained_ae
        and params.get("pretrained_ae_skip_stat_fitting", False)
    )
    if use_cached_normalizers:
        if bool(params.get("pretrained_ae_require_matching_normalizers", True)):
            raise ValueError(
                "pretrained_ae_skip_stat_fitting requires "
                "pretrained_ae_require_matching_normalizers=False."
            )
        bundle = torch.load(pretrained_ae_path, map_location=device, weights_only=False)
        saved_stats = bundle.get("stats", bundle.get("normalizers"))
        if saved_stats is None:
            raise ValueError(
                f"Checkpoint at {pretrained_ae_path} has no AE normalizers."
            )
        normalizers = {}
        for key in (
            "target_mean", "target_std", "node_feature_mean", "node_feature_std",
            "edge_mean", "edge_std", "ref_edge_mean", "ref_edge_std",
        ):
            fallback_key = {"ref_edge_mean": "edge_mean", "ref_edge_std": "edge_std"}.get(key, key)
            value = saved_stats.get(key, saved_stats.get(fallback_key))
            if value is None:
                raise ValueError(
                    f"Checkpoint at {pretrained_ae_path} has no normalizer {key!r}."
                )
            normalizers[key] = value.to(device)
        target_mean, target_std = normalizers["target_mean"], normalizers["target_std"]
        node_mean, node_std = normalizers["node_feature_mean"], normalizers["node_feature_std"]
        edge_mean, edge_std = normalizers["edge_mean"], normalizers["edge_std"]
        ref_edge_mean, ref_edge_std = normalizers["ref_edge_mean"], normalizers["ref_edge_std"]
        print(f"loaded pretrained AE normalizers: {pretrained_ae_path}")
    else:
        target_mean, target_std = fit_ae_target_stats(
            train_data,
            train_frames,
            pos_dim=pos_dim,
            batch_graphs=batch_graphs,
            device=device,
            target_mode=target_mode,
            node_feature_mode=node_mode,
        )
        node_mean, node_std = fit_node_feature_stats(
            train_data,
            train_frames,
            pos_dim=pos_dim,
            batch_graphs=batch_graphs,
            device=device,
            node_feature_mode=node_mode,
        )
        edge_mean, edge_std = fit_edge_stats(
            train_data,
            train_frames,
            pos_dim=pos_dim,
            batch_graphs=batch_graphs,
            device=device,
            edge_mode=edge_mode,
        )
        ref_edge_mean, ref_edge_std = fit_reference_edge_stats(
            train_data,
            train_frames,
            pos_dim=pos_dim,
            batch_graphs=batch_graphs,
            device=device,
            edge_mode=edge_mode,
        )
        normalizers = {
            "target_mean": target_mean,
            "target_std": target_std,
            "node_feature_mean": node_mean,
            "node_feature_std": node_std,
            "edge_mean": edge_mean,
            "edge_std": edge_std,
            "ref_edge_mean": ref_edge_mean,
            "ref_edge_std": ref_edge_std,
        }
    params["node_feature_dim"] = int(node_mean.numel())
    params["edge_feature_dim"] = int(edge_mean.numel())

    autoencoder_type = str(params.get("autoencoder_model", "attention")).lower()
    autoencoder_cls = _autoencoder_class(autoencoder_type)
    ae_model = autoencoder_cls(
        pos_dim=pos_dim,
        node_feature_dim=int(node_mean.numel()),
        edge_dim=params["edge_feature_dim"],
        hidden_size=int(params["hidden_size"]),
        latent_dim=int(params["latent_dim"]),
        latent_tokens=int(params["latent_tokens"]),
        reconstruction_dim=int(target_mean.numel()),
    ).to(device)
    ae_model.edge_mode = edge_mode
    pretrained_ae_path = params.get("pretrained_ae_cache_path")
    use_pretrained_ae = bool(
        pretrained_ae_path
        and Path(pretrained_ae_path).expanduser().exists()
        and not bool(params.get("force_train_autoencoder", False))
        and _pretrained_ae_matches_requested_config(pretrained_ae_path, params)
    )
    if use_pretrained_ae and bool(params.get("pca_initialize_displacement_layers", False)):
        raise ValueError("Cannot combine pretrained_ae_cache_path with PCA initialization.")
    if bool(params.get("pca_initialize_displacement_layers", False)):
        params["pca_initialization"] = initialize_displacement_pca_layers(
            ae_model, train_data, train_frames, pos_dim=pos_dim,
            node_feature_mode=node_mode, target_mode=target_mode,
            normalizers=normalizers, device=device,
        )
        print(f"PCA initialization: {params['pca_initialization']}")
    ae_epoch_callback = _make_ae_pratio_epoch_callback(
        params=params,
        val_data=val_data,
        normalizers=normalizers,
        label=label,
        device=device,
    )
    print("autoencoder")
    if use_pretrained_ae:
        bundle = torch.load(pretrained_ae_path, map_location=device, weights_only=False)
        if "ae_state_dict" not in bundle:
            raise ValueError(
                f"Checkpoint at {pretrained_ae_path} does not contain an autoencoder."
            )
        saved_stats = bundle.get("stats", bundle.get("normalizers"))
        if saved_stats is None:
            raise ValueError(
                f"Checkpoint at {pretrained_ae_path} has no AE normalizers."
            )
        require_matching_stats = bool(
            params.get("pretrained_ae_require_matching_normalizers", True)
        )
        for key, current in normalizers.items():
            fallback_key = {
                "ref_edge_mean": "edge_mean",
                "ref_edge_std": "edge_std",
            }.get(key, key)
            cached = saved_stats.get(key, saved_stats.get(fallback_key))
            if cached is None:
                raise ValueError(
                    f"Checkpoint at {pretrained_ae_path} has no normalizer {key!r}."
                )
            cached = cached.to(device)
            if require_matching_stats and not torch.allclose(
                current, cached, rtol=1e-5, atol=1e-7
            ):
                raise ValueError(
                    f"Pretrained AE normalizer {key!r} does not match the current data split."
                )
            normalizers[key] = cached
        ae_state = dict(bundle["ae_state_dict"])
        # Checkpoints created before reference edges received a separate
        # projection used the dynamic edge projection for both paths.
        for suffix in ("weight", "bias"):
            ref_key = f"ref_edge_in.{suffix}"
            edge_key = f"edge_in.{suffix}"
            if ref_key not in ae_state and edge_key in ae_state:
                ae_state[ref_key] = ae_state[edge_key].clone()
        ae_model.load_state_dict(ae_state)
        saved_history = pd.DataFrame(bundle.get("ae_history", [])).rename(
            columns={
                "train_objective": "train_loss",
                "val_objective": "val_loss",
                "train_mse_norm": "train_reconstruction",
                "val_mse_norm": "val_reconstruction",
            }
        )
        if len(saved_history) and "val_loss" in saved_history:
            finite_history = saved_history[
                np.isfinite(saved_history["val_loss"].to_numpy(float))
            ]
        else:
            finite_history = pd.DataFrame()
        if finite_history.empty:
            saved_history = pd.DataFrame(
                [{"epoch": 0, "train_loss": np.nan, "val_loss": np.nan}]
            )
            best_val_loss, best_epoch = float("nan"), 0
        else:
            best_row = finite_history.loc[finite_history["val_loss"].idxmin()]
            best_val_loss = float(best_row["val_loss"])
            best_epoch = int(best_row["epoch"])
        ae_result = TrainingResult(
            model=ae_model,
            history=saved_history,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
        )
        print(f"loaded pretrained AE: {pretrained_ae_path}")
    else:
        ae_result = train_autoencoder(
            ae_model,
            train_data,
            val_data,
            train_frames,
            val_frames,
            batch_graphs=batch_graphs,
            pos_dim=pos_dim,
            node_feature_mode=node_mode,
            ae_target_mode=target_mode,
            normalizers=normalizers,
            device=device,
            edge_mode=edge_mode,
            config=TrainingConfig(
                max_epochs=int(params["ae_max_epochs"]),
                patience=int(params["ae_patience"]),
                learning_rate=float(params["ae_lr"]),
                weight_decay=float(params["ae_weight_decay"]),
                min_delta=float(params["early_stop_min_delta"]),
                log_every=10,
            ),
            coordinate_weights=params.get("ae_coordinate_weights"),
            mix_sources=bool(params.get("ae_mix_sources", False)),
            gradient_method=str(params.get("ae_gradient_method", "mean")),
            nash_max_iter=int(params.get("ae_nash_max_iter", 50)),
            epoch_callback=ae_epoch_callback,
            selection_metric_key=params.get("ae_checkpoint_metric"),
            selection_mode=str(params.get("ae_checkpoint_mode", "min")),
        )
    ae_model = ae_result.model
    for parameter in ae_model.parameters():
        parameter.requires_grad_(False)

    ae_history = ae_result.history.rename(
        columns={
            "train_loss": "train_objective",
            "val_loss": "val_objective",
            "train_reconstruction": "train_mse_norm",
            "val_reconstruction": "val_mse_norm",
        }
    ).assign(
        dataset=label,
        latent_dim=int(params["latent_dim"]),
        target_mode=target_mode,
        hidden_size=int(params["hidden_size"]),
    )
    stats = {
        **_normalizers_to_cpu(normalizers),
        "delta_mean": target_mean.detach().cpu(),
        "delta_std": target_std.detach().cpu(),
        "ae_target_mode": target_mode,
    }
    return {
        "label": label,
        "params": params,
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
        "split_info": pd.DataFrame(split_info),
        "ae": ae_model,
        "normalizers": normalizers,
        "stats": stats,
        "ae_history": ae_history,
    }


def run_latent_experiment(source_spec: dict, cfg: dict, *, device) -> dict:
    """Run either AE-only analysis training or full rollout training from cfg flags."""

    cfg = _expand_component_configs(cfg)

    should_rollout = bool(cfg.get("should_rollout", True))
    should_train_propagator = bool(
        cfg.get("should_train_propagator", should_rollout)
    )
    path = _cache_path(source_spec, cfg)
    expected_cache_key = latent_experiment_cache_key(source_spec, cfg)
    cache_is_current = False
    if path is not None and path.exists() and not bool(cfg.get("force_train", False)):
        cached = torch.load(path, map_location="cpu", weights_only=False)
        require_matching_cache = bool(cfg.get("cache_require_matching_config", False))
        cache_is_current = (
            cached.get("cache_key") == expected_cache_key
            if require_matching_cache
            else True
        )
        if not cache_is_current:
            print(
                f"ignoring stale latent cache (configuration changed): {path}",
                flush=True,
            )
    if cache_is_current:
        print(f"loading cached latent experiment: {path}")
        if should_rollout or should_train_propagator:
            from .capacity import load_experiment_bundle

            result = load_experiment_bundle(path, cfg=cfg, device=device)
            result["cache_path"] = str(path)
            return result
        return _load_ae_cache(path, cfg, device=device)

    if should_rollout or should_train_propagator:
        result = train_latent_experiment(source_spec, cfg, device=device)
        if path is not None:
            from .capacity import save_experiment_bundle

            save_experiment_bundle(
                result,
                source_spec,
                path,
                cache_key=expected_cache_key,
            )
            result["cache_path"] = str(path)
        return result

    result = train_latent_autoencoder_experiment(source_spec, cfg, device=device)
    if path is not None:
        _save_ae_cache(result, source_spec, cfg, path)
        result["cache_path"] = str(path)
    return result


def train_latent_experiment(source_spec: dict, cfg: dict, *, device) -> dict:
    """Train one autoencoder and propagator, then evaluate all three splits."""

    params = {**cfg, **source_spec}
    label = source_spec["label"]
    dataset_name = params["dataset_name"]
    p_ratio_fn = lambda sim, idx=-1: ground_truth_p_ratio(
        sim,
        idx,
        dataset_name=dataset_name,
        cfg=cfg,
    )
    train_data, val_data, test_data, split_info = resolve_train_val_test(
        source_spec,
        params,
        split_seed=cfg.get("split_seed"),
        p_ratio_fn=p_ratio_fn,
    )
    if not bool(params.get("static_context_use_physical_reference", True)):
        _use_normalized_reference_context(
            (train_data, val_data, test_data), pos_dim=int(params["pos_dim"])
        )
    if not train_data:
        raise ValueError("Latent-model training split is empty.")
    if not val_data:
        raise ValueError(
            "Latent-model validation split is empty. Reduce the requested training "
            "trajectory count or reserve at least one validation trajectory."
        )
    if bool(params.get("temperature_pratio_use_training_prior", False)) and is_temperature_dataset(
        dataset_name
    ):
        prior_cfg = dict(params)
        prior_cfg.pop("temperature_pratio_prior", None)
        prior_values = np.asarray(
            [temperature_p_ratio(sim, cfg=prior_cfg, last_index=-1) for sim in train_data],
            dtype=float,
        )
        finite_prior_values = prior_values[np.isfinite(prior_values)]
        if len(finite_prior_values):
            params["temperature_pratio_prior"] = float(np.mean(finite_prior_values))
    print(f"\n=== {label} ===")
    print(pd.DataFrame(split_info).to_string(index=False))

    pos_dim = int(params["pos_dim"])
    batch_graphs = int(params["batch_graphs"])
    frame_skip = int(params.get("frame_skip", 1))
    ae_train_frame_skip = int(params.get("ae_train_frame_skip", frame_skip))
    ae_frame_budget = int(params["ae_max_train_frames_per_sim"])
    ae_val_frame_skip = int(params.get("ae_val_frame_skip", frame_skip))
    ae_val_frame_budget = int(
        params.get("ae_max_val_frames_per_sim", ae_frame_budget)
    )
    dyn_budget = int(params["dyn_max_train_transitions_per_sim"])
    node_mode = params["node_feature_mode"]
    target_mode = params["ae_target_mode"]
    edge_mode = str(params.get("edge_mode", "stored"))
    pretrained_ae_path = params.get("pretrained_ae_cache_path")
    use_pretrained_ae = bool(
        pretrained_ae_path
        and Path(pretrained_ae_path).expanduser().exists()
        and not bool(params.get("force_train_autoencoder", False))
        and _pretrained_ae_matches_requested_config(pretrained_ae_path, params)
    )

    train_frames = make_frame_index(
        train_data,
        frame_skip=ae_train_frame_skip,
        max_frames_per_sim=ae_frame_budget,
        include_last=True,
        start_frame_order=int(params.get("train_frame_start_order", 0)),
    )
    val_frames = make_frame_index(
        val_data,
        frame_skip=ae_val_frame_skip,
        max_frames_per_sim=ae_val_frame_budget,
        include_last=True,
        start_frame_order=int(params.get("train_frame_start_order", 0)),
    )
    if bool(params.get("ae_balance_sources", False)):
        train_rows_per_source = int(
            params.get("ae_train_rows_per_source", ae_frame_budget)
        )
        train_frames = _balance_rows_by_source(
            train_data,
            train_frames,
            rows_per_source=train_rows_per_source,
            seed=int(params.get("model_seed", 0)) + 101,
        )
        val_groups = _rows_by_source(val_data, val_frames)
        if len(val_groups) > 1:
            val_frames = _balance_rows_by_source(
                val_data,
                val_frames,
                rows_per_source=min(len(rows) for rows in val_groups.values()),
                seed=int(params.get("model_seed", 0)) + 102,
            )
        print(
            "source-balanced AE rows:",
            {key: len(value) for key, value in _rows_by_source(train_data, train_frames).items()},
        )
    if bool(params.get("ae_mix_sources", False)):
        print(
            "source-mixed AE coverage:",
            {
                source: {
                    "trajectories": len({int(row[0]) for row in source_rows}),
                    "frames": len(source_rows),
                }
                for source, source_rows in _rows_by_source(
                    train_data, train_frames
                ).items()
            },
        )
    target_mean, target_std = fit_ae_target_stats(
        train_data,
        train_frames,
        pos_dim=pos_dim,
        batch_graphs=batch_graphs,
        device=device,
        target_mode=target_mode,
        node_feature_mode=node_mode,
    )
    node_mean, node_std = fit_node_feature_stats(
        train_data,
        train_frames,
        pos_dim=pos_dim,
        batch_graphs=batch_graphs,
        device=device,
        node_feature_mode=node_mode,
    )
    params["node_feature_dim"] = int(node_mean.numel())
    edge_mean, edge_std = fit_edge_stats(
        train_data,
        train_frames,
        pos_dim=pos_dim,
        batch_graphs=batch_graphs,
        device=device,
        edge_mode=edge_mode,
    )
    ref_edge_mean, ref_edge_std = fit_reference_edge_stats(
        train_data,
        train_frames,
        pos_dim=pos_dim,
        batch_graphs=batch_graphs,
        device=device,
        edge_mode=edge_mode,
    )
    normalizers = {
        "target_mean": target_mean,
        "target_std": target_std,
        "node_feature_mean": node_mean,
        "node_feature_std": node_std,
        "edge_mean": edge_mean,
        "edge_std": edge_std,
        "ref_edge_mean": ref_edge_mean,
        "ref_edge_std": ref_edge_std,
    }
    params["edge_feature_dim"] = int(edge_mean.numel())

    autoencoder_type = str(params.get("autoencoder_model", "attention")).lower()
    autoencoder_cls = _autoencoder_class(autoencoder_type)
    ae_model = autoencoder_cls(
        pos_dim=pos_dim,
        node_feature_dim=int(node_mean.numel()),
        edge_dim=params["edge_feature_dim"],
        hidden_size=int(params["hidden_size"]),
        latent_dim=int(params["latent_dim"]),
        latent_tokens=int(params["latent_tokens"]),
        reconstruction_dim=int(target_mean.numel()),
    ).to(device)
    ae_model.edge_mode = edge_mode
    if use_pretrained_ae and bool(params.get("pca_initialize_displacement_layers", False)):
        raise ValueError("Cannot combine pretrained_ae_cache_path with PCA initialization.")
    if bool(params.get("pca_initialize_displacement_layers", False)):
        params["pca_initialization"] = initialize_displacement_pca_layers(
            ae_model, train_data, train_frames, pos_dim=pos_dim,
            node_feature_mode=node_mode, target_mode=target_mode,
            normalizers=normalizers, device=device,
        )
        print(f"PCA initialization: {params['pca_initialization']}")
    ae_epoch_callback = _make_ae_pratio_epoch_callback(
        params=params,
        val_data=val_data,
        normalizers=normalizers,
        label=label,
        device=device,
    )
    print("autoencoder")
    if use_pretrained_ae:
        bundle = torch.load(pretrained_ae_path, map_location=device, weights_only=False)
        if "ae_state_dict" not in bundle:
            raise ValueError(
                f"Checkpoint at {pretrained_ae_path} does not contain an autoencoder."
            )
        saved_stats = bundle.get("stats", bundle.get("normalizers"))
        if saved_stats is None:
            raise ValueError(
                f"Checkpoint at {pretrained_ae_path} has no AE normalizers."
            )
        require_matching_stats = bool(
            params.get("pretrained_ae_require_matching_normalizers", True)
        )
        for key, current in normalizers.items():
            fallback_key = {
                "ref_edge_mean": "edge_mean",
                "ref_edge_std": "edge_std",
            }.get(key, key)
            cached = saved_stats.get(key, saved_stats.get(fallback_key))
            if cached is None:
                raise ValueError(
                    f"Checkpoint at {pretrained_ae_path} has no normalizer {key!r}."
                )
            cached = cached.to(device)
            if require_matching_stats and not torch.allclose(
                current, cached, rtol=1e-5, atol=1e-7
            ):
                raise ValueError(
                    f"Pretrained AE normalizer {key!r} does not match the current data split."
                )
            normalizers[key] = cached
        ae_state = dict(bundle["ae_state_dict"])
        # Checkpoints created before reference edges received a separate
        # projection used the dynamic edge projection for both paths.
        for suffix in ("weight", "bias"):
            ref_key = f"ref_edge_in.{suffix}"
            edge_key = f"edge_in.{suffix}"
            if ref_key not in ae_state and edge_key in ae_state:
                ae_state[ref_key] = ae_state[edge_key].clone()
        ae_model.load_state_dict(ae_state)
        saved_history = pd.DataFrame(bundle.get("ae_history", [])).rename(
            columns={
                "train_objective": "train_loss",
                "val_objective": "val_loss",
                "train_mse_norm": "train_reconstruction",
                "val_mse_norm": "val_reconstruction",
            }
        )
        if len(saved_history) and "val_loss" in saved_history:
            finite_history = saved_history[
                np.isfinite(saved_history["val_loss"].to_numpy(float))
            ]
        else:
            finite_history = pd.DataFrame()
        if finite_history.empty:
            saved_history = pd.DataFrame(
                [{"epoch": 0, "train_loss": np.nan, "val_loss": np.nan}]
            )
            best_val_loss, best_epoch = float("nan"), 0
        else:
            best_row = finite_history.loc[finite_history["val_loss"].idxmin()]
            best_val_loss = float(best_row["val_loss"])
            best_epoch = int(best_row["epoch"])
        ae_result = TrainingResult(
            model=ae_model,
            history=saved_history,
            best_val_loss=best_val_loss,
            best_epoch=best_epoch,
        )
        print(f"loaded pretrained AE: {pretrained_ae_path}")
    else:
        ae_result = train_autoencoder(
            ae_model,
            train_data,
            val_data,
            train_frames,
            val_frames,
            batch_graphs=batch_graphs,
            pos_dim=pos_dim,
            node_feature_mode=node_mode,
            ae_target_mode=target_mode,
            normalizers=normalizers,
            device=device,
            edge_mode=edge_mode,
            config=TrainingConfig(
                max_epochs=int(params["ae_max_epochs"]),
                patience=int(params["ae_patience"]),
                learning_rate=float(params["ae_lr"]),
                weight_decay=float(params["ae_weight_decay"]),
                min_delta=float(params["early_stop_min_delta"]),
                log_every=10,
            ),
            coordinate_weights=params.get("ae_coordinate_weights"),
            mix_sources=bool(params.get("ae_mix_sources", False)),
            gradient_method=str(params.get("ae_gradient_method", "mean")),
            nash_max_iter=int(params.get("ae_nash_max_iter", 50)),
            epoch_callback=ae_epoch_callback,
            selection_metric_key=params.get("ae_checkpoint_metric"),
            selection_mode=str(params.get("ae_checkpoint_mode", "min")),
        )
    ae_model = ae_result.model
    for parameter in ae_model.parameters():
        parameter.requires_grad_(False)

    objective = str(params.get("propagator_objective", "one_step")).lower()
    stride = max(1, int(params.get("propagator_step_stride", 1)))
    # Fixed-history rows are filtered after their observation window is known,
    # then sampled across the usable trajectory. Other objectives retain the
    # existing leading-frame budget behavior.
    transition_index_budget = (
        None
        if objective in {
            "fixed_history_one_step",
            "history_one_step",
            "recurrent_memory_one_step",
        }
        else dyn_budget
    )
    if objective in {"velocity", "second_order"}:
        if stride > 1:
            train_dyn_rows = make_jump_velocity_transition_index(
                train_data,
                step_stride=stride,
                frame_skip=frame_skip,
                max_starts_per_sim=dyn_budget,
            )
            val_dyn_rows = make_jump_velocity_transition_index(
                val_data,
                step_stride=stride,
                frame_skip=frame_skip,
                max_starts_per_sim=dyn_budget,
            )
        else:
            train_dyn_rows = make_velocity_transition_index(
                train_data,
                frame_skip=frame_skip,
                max_frames_per_sim=dyn_budget,
            )
            val_dyn_rows = make_velocity_transition_index(
                val_data,
                frame_skip=frame_skip,
                max_frames_per_sim=dyn_budget,
            )
        latent_stat_rows = [
            (sim_idx, t0, t1)
            for sim_idx, _t_prev, t0, t1 in train_dyn_rows
            ]
    else:
        fixed_history_objective = objective in {
            "history_one_step",
            "fixed_history_one_step",
            "recurrent_memory_one_step",
        }
        if fixed_history_objective and stride > 1:
            # A fixed-history one-step model can predict a coarse physical
            # transition directly.  It is still one model evaluation per row,
            # not a stride-many autoregressive unroll: represent the jump
            # target as the sole target in the existing kinematic row format.
            train_jump_rows = make_jump_transition_index(
                train_data,
                step_stride=stride,
                frame_skip=frame_skip,
                max_starts_per_sim=transition_index_budget,
            )
            val_jump_rows = make_jump_transition_index(
                val_data,
                step_stride=stride,
                frame_skip=frame_skip,
                max_starts_per_sim=transition_index_budget,
            )
            train_dyn_rows = [
                (sim_idx, start_frame, [target_frame])
                for sim_idx, start_frame, target_frame in train_jump_rows
            ]
            val_dyn_rows = [
                (sim_idx, start_frame, [target_frame])
                for sim_idx, start_frame, target_frame in val_jump_rows
            ]
            latent_stat_rows = train_jump_rows
            # `horizons` counts model unroll calls, rather than physical frame
            # distance.  There is exactly one direct coarse transition here.
            params["propagator_multistep_horizons"] = [1]
        elif objective in {"multistep", "multi_step"} | KINEMATIC_OBJECTIVES:
            if stride != 1:
                raise ValueError("multistep propagator training currently requires propagator_step_stride=1.")
            if objective in KINEMATIC_OBJECTIVES:
                configured_horizons = params.get("propagator_multistep_horizons")
                if configured_horizons is not None:
                    multistep_horizons = sorted(
                        {int(horizon) for horizon in configured_horizons}
                    )
                else:
                    max_horizon = (
                        1
                        if objective in {
                            "history_one_step",
                            "fixed_history_one_step",
                            "recurrent_memory_one_step",
                        }
                        else 16
                    )
                    multistep_horizons = list(range(1, max_horizon + 1))
                params["propagator_multistep_horizons"] = multistep_horizons
            else:
                multistep_horizons = [
                    int(horizon)
                    for horizon in params.get(
                        "propagator_multistep_horizons", list(range(1, 9))
                    )
                ]
            train_dyn_rows = make_multistep_transition_index(
                train_data,
                horizons=multistep_horizons,
                frame_skip=frame_skip,
                max_starts_per_sim=transition_index_budget,
            )
            val_dyn_rows = make_multistep_transition_index(
                val_data,
                horizons=multistep_horizons,
                frame_skip=frame_skip,
                max_starts_per_sim=transition_index_budget,
            )
            latent_stat_rows = make_transition_index(
                train_data,
                frame_skip=frame_skip,
                max_frames_per_sim=transition_index_budget,
            )
        elif stride > 1:
            train_dyn_rows = make_jump_transition_index(
                train_data,
                step_stride=stride,
                frame_skip=frame_skip,
                max_starts_per_sim=dyn_budget,
            )
            val_dyn_rows = make_jump_transition_index(
                val_data,
                step_stride=stride,
                frame_skip=frame_skip,
                max_starts_per_sim=dyn_budget,
            )
            latent_stat_rows = train_dyn_rows
        else:
            train_dyn_rows = make_transition_index(
                train_data,
                frame_skip=frame_skip,
                max_frames_per_sim=dyn_budget,
            )
            val_dyn_rows = make_transition_index(
                val_data,
                frame_skip=frame_skip,
                max_frames_per_sim=dyn_budget,
            )
            latent_stat_rows = train_dyn_rows

    if objective in {
        "fixed_history_one_step",
        "history_one_step",
        "recurrent_memory_one_step",
    }:
        fixed_observed_frames = tuple(
            int(frame)
            for frame in params.get("fixed_observed_frames", (1, 10))
        )
        fixed_model_name = str(params.get("propagator_model", "")).lower()
        required_fixed_frames = (
            int(params.get("propagator_fixed_history_size", 4))
            if (
                "fixed_window" in fixed_model_name
                or "fixed_four_frame" in fixed_model_name
                or "learned_motion_context" in fixed_model_name
            )
            else 2
        )
        if (
            len(fixed_observed_frames) != required_fixed_frames
            or fixed_observed_frames[0] < 0
            or any(left >= right for left, right in zip(fixed_observed_frames, fixed_observed_frames[1:]))
        ):
            raise ValueError(
                f"History propagation requires {required_fixed_frames} increasing "
                "fixed_observed_frames."
            )
        last_observed = fixed_observed_frames[-1]
        # The AE sees filtered frame orders 0..(budget - 1). Keep propagator
        # supervision inside precisely that same temporal support. A row's
        # final element is its target order (or target-order list).
        fixed_history_end = max(0, ae_frame_budget - 1)

        def within_fixed_history_window(row):
            targets = row[2]
            final_target = max(targets) if isinstance(targets, (list, tuple)) else targets
            return int(row[1]) >= last_observed and int(final_target) <= fixed_history_end

        train_dyn_rows = [
            row for row in train_dyn_rows if within_fixed_history_window(row)
        ]
        val_dyn_rows = [
            row for row in val_dyn_rows if within_fixed_history_window(row)
        ]
        latent_stat_rows = [
            row for row in latent_stat_rows if within_fixed_history_window(row)
        ]
        train_dyn_rows = _limit_rows_per_trajectory(train_dyn_rows, dyn_budget)
        val_dyn_rows = _limit_rows_per_trajectory(val_dyn_rows, dyn_budget)
        latent_stat_rows = _limit_rows_per_trajectory(latent_stat_rows, dyn_budget)

    train_trajectory_limits = params.get(
        "propagator_train_trajectories_per_source"
    )
    val_trajectory_limits = params.get(
        "propagator_val_trajectories_per_source"
    )
    if train_trajectory_limits:
        train_dyn_rows = _limit_rows_to_source_trajectories(
            train_data,
            train_dyn_rows,
            train_trajectory_limits,
        )
        latent_stat_rows = _limit_rows_to_source_trajectories(
            train_data,
            latent_stat_rows,
            train_trajectory_limits,
        )
    if val_trajectory_limits:
        val_dyn_rows = _limit_rows_to_source_trajectories(
            val_data,
            val_dyn_rows,
            val_trajectory_limits,
        )

    if bool(params.get("propagator_balance_sources", False)):
        train_rows_per_source = int(
            params.get("propagator_train_rows_per_source", dyn_budget)
        )
        balance_seed = int(params.get("model_seed", 0)) + 201
        train_dyn_rows = _balance_rows_by_source(
            train_data,
            train_dyn_rows,
            rows_per_source=train_rows_per_source,
            seed=balance_seed,
        )
        latent_stat_rows = _balance_rows_by_source(
            train_data,
            latent_stat_rows,
            rows_per_source=train_rows_per_source,
            seed=balance_seed + 1,
        )
        val_groups = _rows_by_source(val_data, val_dyn_rows)
        if len(val_groups) > 1:
            val_dyn_rows = _balance_rows_by_source(
                val_data,
                val_dyn_rows,
                rows_per_source=min(len(rows) for rows in val_groups.values()),
                seed=balance_seed + 2,
            )
        print(
            "source-balanced propagator rows:",
            {key: len(value) for key, value in _rows_by_source(train_data, train_dyn_rows).items()},
        )

    train_groups = _rows_by_source(train_data, train_dyn_rows)
    print(
        "propagator training coverage:",
        {
            source: {
                "trajectories": len({int(row[0]) for row in source_rows}),
                "transitions": len(source_rows),
            }
            for source, source_rows in train_groups.items()
        },
    )

    print(
        "latent propagator:",
        f"objective={objective}",
        f"step_stride={stride}",
        f"horizons={params.get('propagator_multistep_horizons', None)}",
        f"train_rows={len(train_dyn_rows)}",
        f"val_rows={len(val_dyn_rows)}",
    )
    latent_stats = fit_latent_step_stats(
        ae_model,
        train_data,
        latent_stat_rows,
        batch_graphs=batch_graphs,
        pos_dim=pos_dim,
        node_feature_mode=node_mode,
        normalizers=normalizers,
        device=device,
        use_static_context=bool(params.get("propagator_use_static_context", False)),
        context_include_temperature=bool(
            params.get("propagator_context_include_temperature", False)
        ),
        context_include_source_id=bool(
            params.get("propagator_context_include_source_id", False)
        ),
        context_pool_mode=str(params.get("propagator_context_pool", "mean")),
        rho_scale_mode=params.get("polar_rho_scale_mode"),
    )
    loss_mode = str(params.get("propagator_loss", "delta")).lower()
    default_model = (
        "velocity_mlp"
        if objective in {"velocity", "second_order"}
        else (
            "kinematic_mlp"
            if objective in KINEMATIC_OBJECTIVES
            else (
                "direct_mlp"
                if loss_mode in {"next_z", "jepa", "next_embedding"}
                else "residual_mlp"
            )
        )
    )
    dyn_model = make_latent_propagator(
        int(params["latent_dim"]),
        int(params.get("propagator_hidden_size", params["hidden_size"])),
        model_type=params.get("propagator_model") or default_model,
        context_dim=(
            (
                (
                    4
                    if str(params.get("propagator_context_pool", "mean")).lower()
                    in {"source_id", "source", "dataset_id"}
                    else int(params["hidden_size"])
                    * (
                        4
                        if str(params.get("propagator_context_pool", "mean")).lower()
                        in {"moments", "distribution", "mean_std_min_max"}
                        else 1
                    )
                )
                + int(bool(params.get("propagator_context_include_temperature", False)))
                + 4
                * int(
                    bool(params.get("propagator_context_include_source_id", False))
                    and str(params.get("propagator_context_pool", "mean")).lower()
                    not in {"source_id", "source", "dataset_id"}
                )
            )
            if params.get("propagator_use_static_context", False)
            else 0
        ),
        graph_context_dim=(
            int(params["graph_context_dim"])
            if params.get("propagator_use_static_context", False)
            and params.get("graph_context_dim") is not None
            else None
        ),
        context_include_temperature=bool(
            params.get("propagator_context_include_temperature", False)
        ),
        context_pool_mode=str(params.get("propagator_context_pool", "mean")),
        fixed_history_include_progress=bool(
            params.get("propagator_fixed_history_include_progress", False)
        ),
        fixed_history_size=int(params.get("propagator_fixed_history_size", 4)),
        fixed_history_motion_context_dim=int(
            params.get("propagator_fixed_history_motion_context_dim", 6)
        ),
        history_depth=int(params.get("propagator_history_depth", 3)),
        history_activation=str(params.get("propagator_history_activation", "gelu")),
        history_dropout=float(params.get("propagator_history_dropout", 0.0)),
        history_attention_heads=int(params.get("propagator_history_attention_heads", 2)),
        history_attention_layers=int(params.get("propagator_history_attention_layers", 1)),
        source_names=sorted({str(getattr(sim[0], "source_name", "unknown")) for sim in train_data}),
    ).to(device)

    propagator_epoch_callback = None
    rollout_checkpoint_metric = params.get(
        "propagator_checkpoint_metric",
        DEFAULT_PROPAGATOR_CHECKPOINT_METRIC,
    )
    rollout_checkpoint_mode = str(
        params.get(
            "propagator_checkpoint_mode",
            "min" if rollout_checkpoint_metric is None else "max",
        )
    ).lower()
    if bool(params.get("propagator_rollout_eval_every_epoch", False)) or rollout_checkpoint_metric:
        rollout_eval_interval = max(
            1, int(params.get("propagator_rollout_eval_interval", 1))
        )
        configured_horizons = params.get("propagator_rollout_eval_horizons")
        if configured_horizons is None:
            eval_horizons = [
                int(
                    params.get(
                        "propagator_rollout_eval_horizon",
                        max(params.get("rollout_steps_grid", [100])),
                    )
                )
            ]
        else:
            eval_horizons = sorted({int(step) for step in configured_horizons})
            if not eval_horizons or min(eval_horizons) < 1:
                raise ValueError(
                    "propagator_rollout_eval_horizons must contain positive steps."
                )
        eval_source = params.get("propagator_rollout_eval_source")
        eval_sources = params.get("propagator_rollout_eval_sources")
        if eval_source is not None and eval_sources is not None:
            raise ValueError(
                "Specify only one of propagator_rollout_eval_source or "
                "propagator_rollout_eval_sources."
            )
        eval_candidates = val_data
        if eval_source is not None:
            eval_source = str(eval_source)
            eval_candidates = [
                sim
                for sim in val_data
                if str(getattr(sim[0], "source_name", label)) == eval_source
            ]
            if not eval_candidates:
                raise ValueError(
                    "propagator_rollout_eval_source did not match any validation "
                    f"simulation: {eval_source!r}"
                )
        elif eval_sources is not None:
            selected_sources = {str(source) for source in eval_sources}
            if not selected_sources:
                raise ValueError("propagator_rollout_eval_sources cannot be empty.")
            eval_candidates = [
                sim
                for sim in val_data
                if str(getattr(sim[0], "source_name", label)) in selected_sources
            ]
            missing_sources = selected_sources - {
                str(getattr(sim[0], "source_name", label))
                for sim in eval_candidates
            }
            if missing_sources:
                raise ValueError(
                    "propagator_rollout_eval_sources did not match validation "
                    f"simulations: {sorted(missing_sources)!r}"
                )
        eval_per_source = params.get("propagator_rollout_eval_sims_per_source")
        if eval_per_source is not None:
            per_source = max(1, int(eval_per_source))
            source_counts: dict[str, int] = {}
            epoch_eval_sims = []
            for sim in eval_candidates:
                source_name = str(getattr(sim[0], "source_name", label))
                count = source_counts.get(source_name, 0)
                if count >= per_source:
                    continue
                epoch_eval_sims.append(sim)
                source_counts[source_name] = count + 1
        else:
            eval_count = params.get("propagator_rollout_eval_max_sims", 20)
            epoch_eval_sims = (
                eval_candidates
                if eval_count is None
                else eval_candidates[: max(1, int(eval_count))]
            )

        def propagator_epoch_callback(epoch, active_model):
            if int(epoch) % rollout_eval_interval != 0:
                return None
            epoch_rows, epoch_stats = evaluate_rollout_horizons(
                ae_model,
                active_model,
                epoch_eval_sims,
                latent_stats,
                cfg=params,
                normalizers=normalizers,
                dataset=label,
                split_name="val_epoch",
                rollout_steps=eval_horizons,
                device=device,
                endpoint_only=True,
            )
            if epoch_stats.empty:
                return {
                    "val_rollout_p_ratio_r2": float("nan"),
                    "val_rollout_p_ratio_pearson": float("nan"),
                    "val_rollout_used": 0.0,
                }
            finite_r2 = epoch_stats["p_ratio_r2"].to_numpy(float)
            finite_r2 = finite_r2[np.isfinite(finite_r2)]
            finite_pearson = epoch_stats["p_ratio_pearson"].to_numpy(float)
            finite_pearson = finite_pearson[np.isfinite(finite_pearson)]
            metrics = {
                "val_rollout_p_ratio_r2": (
                    float(np.mean(finite_r2)) if len(finite_r2) else float("nan")
                ),
                "val_rollout_p_ratio_pearson": (
                    float(np.mean(finite_pearson))
                    if len(finite_pearson)
                    else float("nan")
                ),
                "val_rollout_used": float(epoch_stats["used"].min()),
            }
            if "source" in epoch_rows and not epoch_rows.empty:
                source_scores = []
                source_scores_by_name: dict[str, list[float]] = {}
                endpoint_scores = []
                endpoint_scores_by_name: dict[str, list[float]] = {}
                for (_step, _source), source_group in epoch_rows.groupby(
                    ["rollout_steps", "source"], sort=False
                ):
                    score = r2_score(
                        source_group["true_p_ratio"],
                        source_group["pred_p_ratio"],
                    )
                    if np.isfinite(score):
                        source_scores.append(float(score))
                        source_scores_by_name.setdefault(str(_source), []).append(float(score))
                    if {
                        "endpoint_true_p_ratio",
                        "endpoint_pred_p_ratio",
                    }.issubset(source_group.columns):
                        endpoint_score = r2_score(
                            source_group["endpoint_true_p_ratio"],
                            source_group["endpoint_pred_p_ratio"],
                        )
                        if np.isfinite(endpoint_score):
                            endpoint_scores.append(float(endpoint_score))
                            endpoint_scores_by_name.setdefault(str(_source), []).append(
                                float(endpoint_score)
                            )
                metrics["val_rollout_macro_source_p_ratio_r2"] = (
                    float(np.mean(source_scores)) if source_scores else float("nan")
                )
                metrics["val_rollout_min_source_p_ratio_r2"] = (
                    float(np.min(source_scores)) if source_scores else float("nan")
                )
                metrics["val_rollout_macro_source_endpoint_p_ratio_r2"] = (
                    float(np.mean(endpoint_scores))
                    if endpoint_scores
                    else float("nan")
                )
                metrics["val_rollout_min_source_endpoint_p_ratio_r2"] = (
                    float(np.min(endpoint_scores))
                    if endpoint_scores
                    else float("nan")
                )
                for source_name, scores in source_scores_by_name.items():
                    source_key = "".join(
                        character if character.isalnum() else "_"
                        for character in source_name.lower()
                    ).strip("_")
                    metrics[f"val_rollout_source_{source_key}_p_ratio_r2"] = float(
                        np.mean(scores)
                    )
                for source_name, scores in endpoint_scores_by_name.items():
                    source_key = "".join(
                        character if character.isalnum() else "_"
                        for character in source_name.lower()
                    ).strip("_")
                    metrics[
                        f"val_rollout_source_{source_key}_endpoint_p_ratio_r2"
                    ] = float(np.mean(scores))
            for _, horizon_metric in epoch_stats.iterrows():
                step = int(horizon_metric["rollout_steps"])
                metrics[f"val_rollout_p_ratio_r2_step_{step}"] = horizon_metric.get(
                    "p_ratio_r2", float("nan")
                )
            return metrics

    dyn_result = train_propagator(
        dyn_model,
        ae_model,
        train_data,
        val_data,
        train_dyn_rows,
        val_dyn_rows,
        latent_stats,
        batch_graphs=batch_graphs,
        pos_dim=pos_dim,
        node_feature_mode=node_mode,
        ae_target_mode=target_mode,
        normalizers=normalizers,
        device=device,
        loss_mode=loss_mode,
        objective=objective,
        horizons=params.get("propagator_multistep_horizons"),
        frame_skip=frame_skip,
        context_pool_mode=str(params.get("propagator_context_pool", "mean")),
        position_loss_weight=float(
            params.get("propagator_position_loss_weight", 0.0)
        ),
        position_boundary_weight=float(
            params.get("propagator_position_boundary_weight", 1.0)
        ),
        position_boundary_fraction=float(
            params.get("propagator_position_boundary_fraction", 0.10)
        ),
        position_coordinate_weights=params.get(
            "propagator_position_coordinate_weights"
        ),
        network_variation_weight=float(
            params.get("propagator_network_variation_weight", 0.0)
        ),
        network_variation_floor_fraction=float(
            params.get("propagator_network_variation_floor_fraction", 0.05)
        ),
        fixed_observed_frames=params.get("fixed_observed_frames"),
        max_progress_frame=int(params.get("ae_max_train_frames_per_sim", 1)) - 1,
        unroll_curriculum=params.get("propagator_unroll_curriculum"),
        unroll_stage_epochs=params.get("propagator_unroll_stage_epochs"),
        truncated_rollout_horizon=params.get("propagator_truncated_rollout_horizon"),
        mix_sources=bool(params.get("propagator_mix_sources", False)),
        use_static_context=bool(params.get("propagator_use_static_context", False)),
        context_include_temperature=bool(
            params.get("propagator_context_include_temperature", False)
        ),
        context_include_source_id=bool(
            params.get("propagator_context_include_source_id", False)
        ),
        rho_scale_mode=params.get("polar_rho_scale_mode"),
        source_loss_reduction=str(
            params.get("propagator_source_loss_reduction", "pooled")
        ),
        history_noise_std=float(
            params.get("propagator_history_noise_std", 0.0)
        ),
        source_classification_weight=float(
            params.get("propagator_source_classification_weight", 0.0)
        ),
        frozen_latent_cache_dir=params.get("propagator_frozen_latent_cache_dir"),
        use_pcgrad=bool(params.get("propagator_use_pcgrad", False)),
        physics_config=(
            PhysicsLossConfig(
                lambda_phys=float(params.get("physics_lambda", 0.0)),
                lambda_mse=float(params.get("physics_mse_lambda", 1.0)),
                inertial_weight=float(params.get("physics_inertial_weight", 1.0)),
                spring_weight=float(params.get("physics_spring_weight", 1.0)),
                external_weight=float(params.get("physics_external_weight", 1.0)),
                boundary_weight=float(params.get("physics_boundary_weight", 1.0)),
                box_weight=float(params.get("physics_box_weight", 0.0)),
                spring_strain_margin=float(params.get("physics_spring_strain_margin", 0.0)),
                default_mass=float(params.get("physics_default_mass", 1.0)),
                dt=params.get("physics_dt", 1.0),
                normalize_by_speed=bool(params.get("physics_normalize_by_speed", False)),
                speed_epsilon=float(params.get("physics_speed_epsilon", 1e-3)),
                latent_noise_std=float(params.get("physics_latent_noise_std", 0.0)),
            )
            if bool(params.get("physics_loss_enabled", False))
            else None
        ),
        epoch_callback=propagator_epoch_callback,
        selection_metric_key=rollout_checkpoint_metric,
        selection_mode=rollout_checkpoint_mode,
        config=TrainingConfig(
            max_epochs=int(params["dyn_max_epochs"]),
            patience=int(params["dyn_patience"]),
            learning_rate=float(params["dyn_lr"]),
            weight_decay=float(params["dyn_weight_decay"]),
            min_delta=float(params["early_stop_min_delta"]),
            log_every=10,
        ),
    )
    dyn_model = dyn_result.model

    ae_history = ae_result.history.rename(
        columns={
            "train_loss": "train_objective",
            "val_loss": "val_objective",
            "train_reconstruction": "train_mse_norm",
            "val_reconstruction": "val_mse_norm",
        }
    ).assign(
        dataset=label,
        latent_dim=int(params["latent_dim"]),
        target_mode=target_mode,
        hidden_size=int(params["hidden_size"]),
    )
    dyn_history = dyn_result.history.rename(
        columns={
            "train_loss_norm": "train_dz_mse_norm",
            "val_loss_norm": "val_dz_mse_norm",
            "val_loss_raw": "val_dz_mse_raw",
        }
    ).assign(
        dataset=label,
        latent_dim=int(params["latent_dim"]),
        target_mode=target_mode,
        hidden_size=int(params["hidden_size"]),
        propagator_loss=loss_mode,
        propagator_objective=objective,
        propagator_multistep_horizons=str(params.get("propagator_multistep_horizons", "")),
        propagator_multistep_loss=(
            "all_steps_plus_optional_decoded_endpoint"
            if objective in KINEMATIC_OBJECTIVES
            else "final_horizon"
        ),
        propagator_step_stride=stride,
        initial_velocity=params.get("initial_velocity", "zero"),
    )

    rollout_parts = []
    rollout_stats_parts = []
    ae_reconstruction_parts = []
    ae_reconstruction_stats_parts = []
    rollout_eval_splits = {
        str(split_name)
        for split_name in params.get(
            "rollout_eval_splits",
            ("train", "val", "test"),
        )
    }
    for split_name, sims in {
        "train": train_data,
        "val": val_data,
        "test": test_data,
    }.items():
        if split_name not in rollout_eval_splits:
            continue
        rollout_eval_source = params.get("rollout_eval_source")
        if rollout_eval_source is not None:
            rollout_eval_source = str(rollout_eval_source)
            sims = [
                sim
                for sim in sims
                if str(getattr(sim[0], "source_name", label))
                == rollout_eval_source
            ]
        final_per_source = params.get("rollout_final_eval_sims_per_source")
        if final_per_source is not None:
            per_source_limit = int(final_per_source)
            if per_source_limit < 1:
                raise ValueError(
                    "rollout_final_eval_sims_per_source must be positive."
                )
            source_counts: dict[str, int] = {}
            balanced_sims = []
            for sim in sims:
                source_name = str(
                    getattr(sim[0], "source_name", label)
                )
                if source_counts.get(source_name, 0) >= per_source_limit:
                    continue
                balanced_sims.append(sim)
                source_counts[source_name] = source_counts.get(source_name, 0) + 1
            sims = balanced_sims
        max_eval_by_split = params.get("rollout_eval_max_sims_by_split", {})
        max_eval_sims = max_eval_by_split.get(
            split_name,
            params.get("rollout_eval_max_sims_per_split"),
        )
        if max_eval_sims is not None:
            sims = sims[: int(max_eval_sims)]
        steps_grid = rollout_steps_for_sims(
            sims,
            params["rollout_steps_grid"],
            frame_skip=frame_skip,
        )
        print(
            f"rollout eval split={split_name}: sims={len(sims)} "
            f"horizons={len(steps_grid)} "
            f"up to {max(steps_grid) if steps_grid else 0}"
        )
        frame, stats = evaluate_rollout_horizons(
            ae_model,
            dyn_model,
            sims,
            latent_stats,
            cfg=params,
            normalizers=normalizers,
            dataset=label,
            split_name=split_name,
            rollout_steps=steps_grid,
            device=device,
        )
        rollout_parts.append(frame)
        rollout_stats_parts.append(stats)
        if split_name in {"val", "test"}:
            ae_frame, ae_stats = evaluate_autoencoder_reconstruction_horizons(
                ae_model,
                sims,
                cfg=params,
                normalizers=normalizers,
                dataset=label,
                split_name=split_name,
                rollout_steps=steps_grid,
                device=device,
            )
            ae_reconstruction_parts.append(ae_frame)
            ae_reconstruction_stats_parts.append(ae_stats)

    stats = {
        **_normalizers_to_cpu(normalizers),
        "delta_mean": target_mean.detach().cpu(),
        "delta_std": target_std.detach().cpu(),
        "ae_target_mode": target_mode,
        **_normalizers_to_cpu(latent_stats.as_dict()),
    }
    return {
        "label": label,
        "params": params,
        "train_data": train_data,
        "val_data": val_data,
        "test_data": test_data,
        "split_info": pd.DataFrame(split_info),
        "ae": ae_model,
        "dyn": dyn_model,
        "latent_stats": latent_stats,
        "normalizers": normalizers,
        "stats": stats,
        "ae_history": ae_history,
        "dyn_history": dyn_history,
        "rollout_rows": (
            pd.concat(rollout_parts, ignore_index=True)
            if rollout_parts
            else pd.DataFrame()
        ),
        "rollout_stats": (
            pd.concat(rollout_stats_parts, ignore_index=True)
            if rollout_stats_parts
            else pd.DataFrame()
        ),
        "ae_reconstruction_rows": (
            pd.concat(ae_reconstruction_parts, ignore_index=True)
            if ae_reconstruction_parts
            else pd.DataFrame()
        ),
        "ae_reconstruction_stats": (
            pd.concat(ae_reconstruction_stats_parts, ignore_index=True)
            if ae_reconstruction_stats_parts
            else pd.DataFrame()
        ),
    }


def build_autoencoder(params: dict, *, edge_dim: int, device):
    """Construct the shared graph autoencoder from experiment parameters."""

    autoencoder_cls = _autoencoder_class(params.get("autoencoder_model", "attention"))
    return autoencoder_cls(
        pos_dim=int(params["pos_dim"]),
        node_feature_dim=int(
            params.get("node_feature_dim", params["pos_dim"])
        ),
        edge_dim=int(edge_dim),
        hidden_size=int(params["hidden_size"]),
        latent_dim=int(params["latent_dim"]),
        latent_tokens=int(params["latent_tokens"]),
    ).to(device)


def add_result_metadata(frame: pd.DataFrame, result: dict) -> pd.DataFrame:
    frame = frame.copy()
    frame["latent_dim"] = int(result["params"]["latent_dim"])
    frame["repeat_idx"] = int(result["params"].get("repeat_idx", 1))
    frame["model_seed"] = int(result["params"]["model_seed"])
    frame["run_label"] = result["label"]
    return frame


def result_tables(results: list[dict]) -> dict[str, pd.DataFrame]:
    """Combine histories and rollout data from one or more experiment results."""

    def combine(key):
        parts = [add_result_metadata(result[key], result) for result in results]
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    return {
        "ae_history": combine("ae_history"),
        "dyn_history": combine("dyn_history"),
        "rollout_rows": combine("rollout_rows"),
        "rollout_stats": combine("rollout_stats"),
        "ae_reconstruction_rows": combine("ae_reconstruction_rows"),
        "ae_reconstruction_stats": combine("ae_reconstruction_stats"),
    }


def save_result_tables(tables: dict[str, pd.DataFrame], output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)


def initial_latent_table(
    result: dict,
    sims,
    split_name: str,
    *,
    device,
) -> pd.DataFrame:
    """Encode the initial frame and attach network-level metadata."""

    rows = []
    params = result["params"]
    ae_model = result["ae"]
    ae_model.eval()
    with torch.no_grad():
        for sim_idx, sim in enumerate(sims):
            z = encode_frame_latent(
                ae_model,
                sim,
                0,
                pos_dim=int(params["pos_dim"]),
                node_feature_mode=params["node_feature_mode"],
                normalizers=result["normalizers"],
                device=device,
            ).detach().cpu().numpy().reshape(-1)
            row = {
                "split": split_name,
                "latent_dim": int(params["latent_dim"]),
                "repeat_idx": int(params.get("repeat_idx", 1)),
                "run_label": result["label"],
                "sim_idx": int(sim_idx),
                "source": str(getattr(sim[0], "source_name", result.get("source_name", result["label"]))),
                "frames": int(len(sim)),
                "nodes": int(sim[0].x.shape[0]),
                "temperature": float(getattr(sim[0], "temperature", np.nan)),
                "final_p_ratio": ground_truth_p_ratio(
                    sim,
                    -1,
                    dataset_name=params["dataset_name"],
                    cfg=params,
                ),
            }
            row.update({f"z{idx}": float(value) for idx, value in enumerate(z)})
            rows.append(row)
    return pd.DataFrame(rows)


def initial_latent_analysis(results: list[dict], *, device) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build initial-latent tables and per-coordinate p-ratio correlations."""

    latent_parts = []
    for result in results:
        latent_parts.extend(
            initial_latent_table(result, result[f"{split}_data"], split, device=device)
            for split in ("train", "val", "test")
        )
    latent_df = pd.concat(latent_parts, ignore_index=True) if latent_parts else pd.DataFrame()

    rows = []
    for (latent_dim, split_name), group in latent_df.groupby(["latent_dim", "split"]):
        for z_col in [
            col for col in group.columns if col.startswith("z") and group[col].notna().any()
        ]:
            clean = group[[z_col, "final_p_ratio"]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(clean) < 3 or clean[z_col].std(ddof=0) <= 0:
                continue
            slope, intercept = np.polyfit(clean[z_col], clean["final_p_ratio"], deg=1)
            fitted = slope * clean[z_col].to_numpy(float) + intercept
            rows.append(
                {
                    "latent_dim": int(latent_dim),
                    "split": split_name,
                    "latent": z_col,
                    "target": "final_p_ratio",
                    "r2": r2_score(clean["final_p_ratio"], fitted),
                    "pearson_r": float(clean[z_col].corr(clean["final_p_ratio"])),
                    "spearman_r": float(
                        clean[z_col].corr(clean["final_p_ratio"], method="spearman")
                    ),
                    "n": int(len(clean)),
                }
            )
    corr_df = pd.DataFrame(rows)
    if not corr_df.empty:
        corr_df["abs_pearson_r"] = corr_df["pearson_r"].abs()
        corr_df = corr_df.sort_values(
            ["split", "latent_dim", "abs_pearson_r"],
            ascending=[True, True, False],
        )
    return latent_df, corr_df


def rollout_curve_summary(
    rollout_rows: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize raw rollout rows by repeat and across repeats."""

    if rollout_rows.empty:
        return pd.DataFrame(), pd.DataFrame()
    frame = rollout_rows.copy()
    frame["p_ratio_sq_error"] = (frame["pred_p_ratio"] - frame["true_p_ratio"]) ** 2
    by_repeat = frame.groupby(
        ["latent_dim", "repeat_idx", "split", "rollout_steps"],
        as_index=False,
    ).agg(
        final_pos_mse=("final_pos_mse", "mean"),
        p_ratio_mse=("p_ratio_sq_error", "mean"),
        used=("sim_idx", "count"),
    )
    summary = frame.groupby(
        ["latent_dim", "split", "rollout_steps"],
        as_index=False,
    ).agg(
        final_pos_mse=("final_pos_mse", "mean"),
        p_ratio_mse=("p_ratio_sq_error", "mean"),
        used=("sim_idx", "count"),
    )
    metric_rows = []
    for (latent_dim, split_name, steps), group in frame.groupby(
        ["latent_dim", "split", "rollout_steps"]
    ):
        metric_rows.append(
            {
                "latent_dim": int(latent_dim),
                "split": split_name,
                "rollout_steps": int(steps),
                "p_ratio_r2": r2_score(group["true_p_ratio"], group["pred_p_ratio"]),
                "p_ratio_pearson": pearson_r(
                    group["true_p_ratio"],
                    group["pred_p_ratio"],
                ),
            }
        )
    summary = summary.merge(
        pd.DataFrame(metric_rows),
        on=["latent_dim", "split", "rollout_steps"],
        how="left",
    )
    return by_repeat, summary


__all__ = [
    "add_result_metadata",
    "build_autoencoder",
    "evaluate_rollout",
    "evaluate_rollout_horizons",
    "find_project_root",
    "ground_truth_p_ratio",
    "initial_latent_analysis",
    "initial_latent_table",
    "is_temperature_dataset",
    "make_jump_transition_index",
    "make_jump_velocity_transition_index",
    "prepare_source_spec",
    "resolve_existing_path",
    "resolve_train_val_test",
    "result_tables",
    "rollout_curve_summary",
    "rollout_metrics",
    "rollout_steps_for_sims",
    "run_latent_experiment",
    "save_result_tables",
    "seed_everything",
    "temperature_p_ratio",
    "train_latent_autoencoder_experiment",
    "train_latent_experiment",
]
