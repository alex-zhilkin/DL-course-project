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
    frame_for_filtered_step,
    make_frame_index,
    make_transition_index,
    pearson_r,
    r2_score,
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
    latent_step_history,
    latent_step_kinematic,
    latent_step_velocity,
    make_multistep_transition_index,
    make_velocity_transition_index,
    batch_delta_graphs,
    ae_target_tensor,
    train_autoencoder,
    train_propagator,
)

KINEMATIC_OBJECTIVES = {
    "kinematic_multistep",
    "kinematic",
    "anchored_multistep",
    "closed_loop",
    "history_one_step",
}

THREE_FRAME_INITIALIZATIONS = {
    "three_frames",
    "three_frame",
    "observed_three",
    "history3",
}


def _observed_position_graph(sim, frame_index: int, *, pos_dim: int):
    """Copy observed positions onto the reference graph's static metadata."""

    graph = clone_graph(sim[0])
    graph.x = graph.x.clone().float()
    graph.x[:, :pos_dim] = sim[frame_index].x[:, :pos_dim].cpu().float()
    return graph.cpu()


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
        "device",
        "label",
        "source_name",
        "display_name",
        "experiment_name",
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
    autoencoder_type = str(params.get("autoencoder_model", "attention")).lower()
    autoencoder_cls = _autoencoder_class(autoencoder_type)
    ae_model = autoencoder_cls(
        pos_dim=int(params["pos_dim"]),
        node_feature_dim=int(normalizers["node_feature_mean"].numel()),
        edge_dim=int(normalizers["edge_mean"].numel()),
        hidden_size=int(params["hidden_size"]),
        latent_dim=int(params["latent_dim"]),
        latent_tokens=int(params["latent_tokens"]),
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


def temperature_p_ratio(trajectory, *, cfg: dict | None = None, last_index: int = -1) -> float:
    """Fixed robust estimator used only for mixed-temperature dePablo data."""

    del cfg
    return float(
        trajectory_p_ratio_sides_robust(
            trajectory,
            last_index=last_index,
            min_fit_frames=8,
            min_driven_strain_range=1e-3,
            smooth_window=5,
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

    if is_temperature_dataset(dataset_name) and len(sim) > 2:
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
        )

    if p_ratio_fn is None:
        raise ValueError("p_ratio_fn is required when min_train_p_ratio is set.")
    sims = load_dataset(
        source_spec["path"],
        edge_multiplicity=edge_multiplicity,
        edge_vector_dim=edge_vector_dim,
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
    return {
        key: value.detach().cpu() if torch.is_tensor(value) else value
        for key, value in normalizers.items()
    }


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
    loss_mode = str(cfg.get("propagator_loss", "delta")).lower()
    use_static_context = bool(cfg.get("propagator_use_static_context", False))
    context_include_temperature = bool(
        cfg.get("propagator_context_include_temperature", False)
    )
    context_pool_mode = str(cfg.get("propagator_context_pool", "mean"))
    p_ratio_window = cfg.get("temperature_pratio_window", "full")
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
            available = max(
                len(filtered_frame_ids(sim, frame_skip=frame_skip, include_last=True)) - 1,
                0,
            )
            filtered_steps = min(int(rollout_steps), available)
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
                    pool_mode=context_pool_mode,
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
            predicted_window = []
            ground_truth_window = []
            if isinstance(p_ratio_window, str) and p_ratio_window.lower() == "full":
                window_start_step = 0
            else:
                window_start_step = max(0, filtered_steps - max(2, int(p_ratio_window)) + 1)
            if temperature_data and window_start_step == 0:
                predicted_window.append(clone_graph(sim[0]).cpu())
                ground_truth_window.append(clone_graph(sim[0]).cpu())

            start_step = stride
            prev_dz = None
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
                if temperature_data:
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
                    if temperature_data and first_order >= window_start_step:
                        predicted_window.append(clone_graph(sim[first_index]).cpu())
                        ground_truth_window.append(clone_graph(sim[first_index]).cpu())
                elif initial_velocity == "mean":
                    prev_dz = latent_stats.dz_mean.squeeze(0).to(device)
                else:
                    raise ValueError(f"Unknown initial_velocity: {initial_velocity}")
            elif objective in KINEMATIC_OBJECTIVES:
                if getattr(dyn_model, "uses_history_state", False):
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
                    if temperature_data:
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
                    if temperature_data and first_order >= window_start_step:
                        predicted_window.append(clone_graph(sim[first_index]).cpu())
                        ground_truth_window.append(clone_graph(sim[first_index]).cpu())
                elif initial_velocity == "mean":
                    z_previous = z - latent_stats.dz_mean.squeeze(0).to(device)
                else:
                    raise ValueError(f"Unknown initial_velocity: {initial_velocity}")

            step_orders = list(range(start_step, filtered_steps + 1, stride))
            for step in step_orders:
                if objective in {"velocity", "second_order"}:
                    z, prev_dz = latent_step_velocity(
                        dyn_model,
                        z,
                        prev_dz,
                        latent_stats,
                        context=context,
                    )
                elif objective in KINEMATIC_OBJECTIVES:
                    if getattr(dyn_model, "uses_history_state", False):
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
                    z_previous_previous, z_previous, z = z_previous, z, z_next
                else:
                    z = latent_step(
                        dyn_model,
                        z,
                        latent_stats,
                        loss_mode=loss_mode,
                        context=context,
                        rho_scale=rho_scale,
                    )
                if temperature_data and step >= window_start_step:
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
            if temperature_data and len(predicted_window) >= 2:
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
    loss_mode = str(cfg.get("propagator_loss", "delta")).lower()
    use_static_context = bool(cfg.get("propagator_use_static_context", False))
    context_include_temperature = bool(
        cfg.get("propagator_context_include_temperature", False)
    )
    context_pool_mode = str(cfg.get("propagator_context_pool", "mean"))
    p_ratio_window = cfg.get("temperature_pratio_window", "full")
    rho_scale_mode = cfg.get("polar_rho_scale_mode")

    if stride > 1:
        horizons = sorted({int(step) - (int(step) % stride) for step in horizons})
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
                    pool_mode=context_pool_mode,
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
            prev_dz = None
            start_step = stride
            predicted_path = [clone_graph(sim[0]).cpu()] if temperature_data else []
            ground_truth_path = [clone_graph(sim[0]).cpu()] if temperature_data else []

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
                if temperature_data:
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
                            clone_graph(sim[frame_index]).cpu()
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
                    if temperature_data:
                        predicted_path.append(clone_graph(sim[first_index]).cpu())
                        ground_truth_path.append(clone_graph(sim[first_index]).cpu())
                elif initial_velocity == "mean":
                    prev_dz = latent_stats.dz_mean.squeeze(0).to(device)
                else:
                    raise ValueError(f"Unknown initial_velocity: {initial_velocity}")
            elif objective in KINEMATIC_OBJECTIVES:
                if getattr(dyn_model, "uses_history_state", False):
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
                    if temperature_data:
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
                                clone_graph(sim[frame_index]).cpu()
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
                    if temperature_data:
                        predicted_path.append(clone_graph(sim[first_index]).cpu())
                        ground_truth_path.append(clone_graph(sim[first_index]).cpu())
                elif initial_velocity == "mean":
                    z_previous = z - latent_stats.dz_mean.squeeze(0).to(device)
                else:
                    raise ValueError(f"Unknown initial_velocity: {initial_velocity}")

            for step in range(start_step, min(max_horizon, available) + 1, stride):
                if objective in {"velocity", "second_order"}:
                    z, prev_dz = latent_step_velocity(
                        dyn_model,
                        z,
                        prev_dz,
                        latent_stats,
                        context=context,
                    )
                elif objective in KINEMATIC_OBJECTIVES:
                    if getattr(dyn_model, "uses_history_state", False):
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
                    z_previous_previous, z_previous, z = z_previous, z, z_next
                else:
                    z = latent_step(
                        dyn_model,
                        z,
                        latent_stats,
                        loss_mode=loss_mode,
                        context=context,
                        rho_scale=rho_scale,
                    )

                target_index = frame_for_filtered_step(sim, step, frame_skip=frame_skip)
                pred_graph = None
                if temperature_data or step in horizon_set:
                    pred_graph = decode(sim, z, target_index)
                if temperature_data:
                    predicted_path.append(pred_graph)
                    ground_truth_path.append(clone_graph(sim[target_index]).cpu())
                if step not in horizon_set:
                    continue

                if temperature_data:
                    if isinstance(p_ratio_window, str) and p_ratio_window.lower() == "full":
                        predicted_window = predicted_path
                        ground_truth_window = ground_truth_path
                    else:
                        window_size = max(2, int(p_ratio_window))
                        predicted_window = predicted_path[-window_size:]
                        ground_truth_window = ground_truth_path[-window_size:]
                    pred_pr = temperature_p_ratio(predicted_window, cfg=cfg, last_index=-1)
                    true_pr = temperature_p_ratio(ground_truth_window, cfg=cfg, last_index=-1)
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
            available = max(
                len(filtered_frame_ids(sim, frame_skip=frame_skip, include_last=True)) - 1,
                0,
            )
            sim_horizons = [step for step in horizons if step <= available]
            if not sim_horizons:
                continue

            pred_path = [clone_graph(sim[0]).cpu()]
            true_path = [clone_graph(sim[0]).cpu()]
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
                pred_path.append(pred_graph)
                true_path.append(clone_graph(sim[target_index]).cpu())
                if step not in horizon_set:
                    continue

                if temperature_data:
                    pred_pr = temperature_p_ratio(pred_path, cfg=cfg, last_index=-1)
                    true_pr = temperature_p_ratio(true_path, cfg=cfg, last_index=-1)
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
    ae_frame_budget = int(params["ae_max_train_frames_per_sim"])
    ae_val_frame_skip = int(params.get("ae_val_frame_skip", frame_skip))
    ae_val_frame_budget = int(
        params.get("ae_max_val_frames_per_sim", ae_frame_budget)
    )
    node_mode = params["node_feature_mode"]
    target_mode = params["ae_target_mode"]
    edge_mode = str(params.get("edge_mode", "stored"))

    train_frames = make_frame_index(
        train_data,
        frame_skip=frame_skip,
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
    normalizers = {
        "target_mean": target_mean,
        "target_std": target_std,
        "node_feature_mean": node_mean,
        "node_feature_std": node_std,
        "edge_mean": edge_mean,
        "edge_std": edge_std,
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
    pretrained_ae_path = params.get("pretrained_ae_cache_path")
    if pretrained_ae_path and bool(params.get("pca_initialize_displacement_layers", False)):
        raise ValueError("Cannot combine pretrained_ae_cache_path with PCA initialization.")
    if bool(params.get("pca_initialize_displacement_layers", False)):
        params["pca_initialization"] = initialize_displacement_pca_layers(
            ae_model, train_data, train_frames, pos_dim=pos_dim,
            node_feature_mode=node_mode, target_mode=target_mode,
            normalizers=normalizers, device=device,
        )
        print(f"PCA initialization: {params['pca_initialization']}")
    print("autoencoder")
    if pretrained_ae_path:
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
            cached = saved_stats[key].to(device)
            if require_matching_stats and not torch.allclose(
                current, cached, rtol=1e-5, atol=1e-7
            ):
                raise ValueError(
                    f"Pretrained AE normalizer {key!r} does not match the current data split."
                )
            normalizers[key] = cached
        ae_model.load_state_dict(bundle["ae_state_dict"])
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

    should_rollout = bool(cfg.get("should_rollout", True))
    should_train_propagator = bool(
        cfg.get("should_train_propagator", should_rollout)
    )
    path = _cache_path(source_spec, cfg)
    if path is not None and path.exists() and not bool(cfg.get("force_train", False)):
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

            save_experiment_bundle(result, source_spec, path)
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

    train_frames = make_frame_index(
        train_data,
        frame_skip=frame_skip,
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
    normalizers = {
        "target_mean": target_mean,
        "target_std": target_std,
        "node_feature_mean": node_mean,
        "node_feature_std": node_std,
        "edge_mean": edge_mean,
        "edge_std": edge_std,
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
    if bool(params.get("pca_initialize_displacement_layers", False)):
        params["pca_initialization"] = initialize_displacement_pca_layers(
            ae_model, train_data, train_frames, pos_dim=pos_dim,
            node_feature_mode=node_mode, target_mode=target_mode,
            normalizers=normalizers, device=device,
        )
        print(f"PCA initialization: {params['pca_initialization']}")
    print("autoencoder")
    if pretrained_ae_path:
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
            cached = saved_stats[key].to(device)
            if require_matching_stats and not torch.allclose(
                current, cached, rtol=1e-5, atol=1e-7
            ):
                raise ValueError(
                    f"Pretrained AE normalizer {key!r} does not match the current data split."
                )
            normalizers[key] = cached
        ae_model.load_state_dict(bundle["ae_state_dict"])
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
        )
    ae_model = ae_result.model
    for parameter in ae_model.parameters():
        parameter.requires_grad_(False)

    objective = str(params.get("propagator_objective", "one_step")).lower()
    stride = max(1, int(params.get("propagator_step_stride", 1)))
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
        if objective in {"multistep", "multi_step"} | KINEMATIC_OBJECTIVES:
            if stride != 1:
                raise ValueError("multistep propagator training currently requires propagator_step_stride=1.")
            if objective in KINEMATIC_OBJECTIVES:
                max_horizon = 1 if objective == "history_one_step" else 16
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
                max_starts_per_sim=dyn_budget,
            )
            val_dyn_rows = make_multistep_transition_index(
                val_data,
                horizons=multistep_horizons,
                frame_skip=frame_skip,
                max_starts_per_sim=dyn_budget,
            )
            latent_stat_rows = make_transition_index(
                train_data,
                frame_skip=frame_skip,
                max_frames_per_sim=dyn_budget,
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
        context_pool_mode=str(params.get("propagator_context_pool", "mean")),
        rho_scale_mode=params.get("polar_rho_scale_mode"),
    )
    latent_stats.z_mean = torch.zeros_like(latent_stats.z_mean)
    latent_stats.z_std = torch.ones_like(latent_stats.z_std)
    if latent_stats.z_next_mean is not None:
        latent_stats.z_next_mean = torch.zeros_like(latent_stats.z_next_mean)
    if latent_stats.z_next_std is not None:
        latent_stats.z_next_std = torch.ones_like(latent_stats.z_next_std)
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
            int(params["hidden_size"])
            * (
                4
                if str(params.get("propagator_context_pool", "mean")).lower()
                in {"moments", "distribution", "mean_std_min_max"}
                else 1
            )
            + int(bool(params.get("propagator_context_include_temperature", False)))
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
            params.get("propagator_use_static_context", False)
            and params.get("propagator_context_include_temperature", False)
        ),
        context_pool_mode=str(params.get("propagator_context_pool", "mean")),
    ).to(device)

    propagator_epoch_callback = None
    rollout_checkpoint_metric = params.get("propagator_checkpoint_metric")
    if bool(params.get("propagator_rollout_eval_every_epoch", False)) or rollout_checkpoint_metric:
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
        eval_per_source = params.get("propagator_rollout_eval_sims_per_source")
        if eval_per_source is not None:
            per_source = max(1, int(eval_per_source))
            source_counts: dict[str, int] = {}
            epoch_eval_sims = []
            for sim in val_data:
                source_name = str(getattr(sim[0], "source_name", label))
                count = source_counts.get(source_name, 0)
                if count >= per_source:
                    continue
                epoch_eval_sims.append(sim)
                source_counts[source_name] = count + 1
        else:
            eval_count = params.get("propagator_rollout_eval_max_sims", 20)
            epoch_eval_sims = (
                val_data
                if eval_count is None
                else val_data[: max(1, int(eval_count))]
            )

        def propagator_epoch_callback(epoch, active_model):
            del epoch
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
                for (_step, _source), source_group in epoch_rows.groupby(
                    ["rollout_steps", "source"], sort=False
                ):
                    score = r2_score(
                        source_group["true_p_ratio"],
                        source_group["pred_p_ratio"],
                    )
                    if np.isfinite(score):
                        source_scores.append(float(score))
                metrics["val_rollout_macro_source_p_ratio_r2"] = (
                    float(np.mean(source_scores)) if source_scores else float("nan")
                )
                metrics["val_rollout_min_source_p_ratio_r2"] = (
                    float(np.min(source_scores)) if source_scores else float("nan")
                )
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
        use_static_context=bool(params.get("propagator_use_static_context", False)),
        context_include_temperature=bool(
            params.get("propagator_context_include_temperature", False)
        ),
        rho_scale_mode=params.get("polar_rho_scale_mode"),
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
        selection_mode=str(params.get("propagator_checkpoint_mode", "min")),
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
    for split_name, sims in {
        "train": train_data,
        "val": val_data,
        "test": test_data,
    }.items():
        max_eval_sims = params.get("rollout_eval_max_sims_per_split")
        if max_eval_sims is not None:
            sims = sims[: int(max_eval_sims)]
        steps_grid = rollout_steps_for_sims(
            sims,
            params["rollout_steps_grid"],
            frame_skip=frame_skip,
        )
        print(
            f"rollout eval split={split_name}: {len(steps_grid)} horizons "
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
