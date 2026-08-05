"""Capacity-sweep orchestration built on the shared latent experiment workflow."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .experiment import (
    add_result_metadata,
    evaluate_rollout_horizons,
    initial_latent_table,
    prepare_source_spec,
    resolve_existing_path,
    resolve_train_val_test,
    rollout_steps_for_sims,
    seed_everything,
    train_latent_experiment,
)
from .models import (
    NodeDeltaAttentionAutoEncoder,
    NodeDeltaMLPAutoEncoder,
    NodeDeltaPyramidMLPAutoEncoder,
    NodeDeltaSingleStageAttentionAutoEncoder,
    make_latent_propagator,
)
from .simulation import r2_score
from .training import LatentNormalizer


def experiment_config_fingerprint(config: dict, *, length: int = 12) -> str:
    """Return a stable cache tag for a complete experiment configuration."""

    runtime_only = {"cache_path", "force_train", "device"}
    identity = {
        str(key): value
        for key, value in config.items()
        if str(key) not in runtime_only
    }
    payload = json.dumps(identity, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[: int(length)]


FINAL_05_2D_PROPAGATORS = {
    "depablo_low_temp": {
        "cache_name": "final_2d_single_stage_h64_one_step.pt",
        "ae_cache_name": "ae_2d_single_stage_h64_frames100.pt",
        "seed_offset": 67,
        "autoencoder_model": "single_stage_attention",
        "hidden_size": 64,
        "latent_tokens": 16,
        "batch_graphs": 32,
        "ae_max_train_frames_per_sim": 100,
        "ae_max_epochs": 40,
        "ae_patience": 6,
        "ae_lr": 2e-4,
        "ae_weight_decay": 1e-5,
        "propagator_objective": "one_step",
        "propagator_model": "delta_mlp",
        "propagator_loss": "delta",
        "propagator_hidden_size": 64,
        "graph_context_dim": 16,
        "dyn_lr": 1e-4,
        "dyn_weight_decay": 1e-5,
        "dyn_max_epochs": 40,
        "dyn_patience": 6,
        "propagator_rollout_eval_horizon": 100,
        "propagator_rollout_eval_horizons": [50, 100, 150, 199],
    },
    "reid": {
        "cache_name": "final_2d_single_stage_h64_train30_ae200_prop32_dyn100_one_step.pt",
        "ae_cache_name": "ae_2d_single_stage_h64_train30_frames200.pt",
        "seed_offset": 109,
        "train_count": 30,
        "val_count": 20,
        "autoencoder_model": "single_stage_attention",
        "hidden_size": 64,
        "latent_tokens": 16,
        "batch_graphs": 32,
        # The AE learns the complete state manifold; the propagator below still
        # fits only the first 100 transitions and is evaluated beyond them.
        "ae_max_train_frames_per_sim": 200,
        "ae_max_epochs": 40,
        "ae_patience": 6,
        "ae_lr": 2e-4,
        "ae_weight_decay": 5e-5,
        "propagator_objective": "one_step",
        "propagator_model": "delta_mlp",
        "propagator_loss": "delta",
        "propagator_hidden_size": 32,
        "graph_context_dim": 16,
        "dyn_lr": 1e-4,
        "dyn_weight_decay": 1e-5,
        "dyn_max_epochs": 30,
        "dyn_patience": 6,
        "propagator_rollout_eval_horizon": 100,
        "propagator_rollout_eval_horizons": [50, 100, 150, 199],
    },
    "depablo_mixed_temp": {
        "cache_name": "../../depablo_mixed_temp/latent_rollout_attention_cv2_train20_frames100_epochs80_pat8_dyn_epochs80_dynpat8_seed20261013.pt",
        "seed_offset": 0,
        "propagator_objective": "one_step",
        "propagator_model": "delta_mlp",
        "propagator_loss": "delta",
        "propagator_hidden_size": 96,
        "graph_context_dim": 16,
        "dyn_lr": 3e-5,
        "dyn_weight_decay": 1e-4,
        "dyn_max_epochs": 80,
        "dyn_patience": 8,
        "propagator_rollout_eval_horizon": 100,
    },
}


def final_05_2d_propagator_config(dataset_name: str) -> dict | None:
    """Return the validation-selected final 2D propagator settings for 05b/05c."""

    config = FINAL_05_2D_PROPAGATORS.get(str(dataset_name))
    return dict(config) if config is not None else None


def build_capacity_specs(
    dataset_names,
    dataset_specs,
    cfg,
    *,
    latent_dims,
    train_counts,
    train_frame_counts,
    target_modes,
    repeats,
    seed,
) -> list[dict]:
    """Build resolved sweep specifications."""

    specs = []
    for dataset_name, latent_dim, train_count, train_frames, target_mode, repeat_idx in itertools.product(
        dataset_names,
        latent_dims,
        train_counts,
        train_frame_counts,
        target_modes,
        range(1, int(repeats) + 1),
    ):
        local_cfg = {
            **cfg,
            "dataset_name": dataset_name,
            "latent_dim": int(latent_dim),
            "ae_target_mode": target_mode,
            "node_feature_mode": target_mode,
            "ae_max_train_frames_per_sim": int(train_frames),
            "dyn_max_train_transitions_per_sim": int(train_frames),
            "repeat_idx": int(repeat_idx),
            "model_seed": int(
                seed
                + 1009 * int(latent_dim)
                + 9176 * int(repeat_idx)
                + 104729 * int(train_count)
                + 13007 * int(train_frames)
            ),
        }
        source = prepare_source_spec(
            dataset_name,
            dataset_specs,
            local_cfg,
            seed=seed,
        )
        source.update(
            {
                "train_count": int(train_count),
                "sweep_target_mode": target_mode,
                "sweep_latent_dim": int(latent_dim),
                "sweep_train_count": int(train_count),
                "sweep_train_frames": int(train_frames),
                "sweep_repeat": int(repeat_idx),
                "sweep_model_seed": int(local_cfg["model_seed"]),
                "graph_context_dim": local_cfg.get("graph_context_dim"),
                "dataset_label": dataset_specs[dataset_name]["label"],
            }
        )
        specs.append(source)
    return specs


def capacity_model_name(spec: dict) -> str:
    context_suffix = (
        f"_ctx{int(spec['graph_context_dim']):03d}"
        if spec.get("graph_context_dim") is not None
        else ""
    )
    return (
        f"{spec['dataset_name']}_{spec['sweep_target_mode']}"
        f"_cv{int(spec['sweep_latent_dim']):02d}"
        f"_nets{int(spec['sweep_train_count']):03d}"
        f"_frames{int(spec['sweep_train_frames']):03d}"
        f"{context_suffix}"
        f"_rep{int(spec['sweep_repeat']):02d}.pt"
    )


def save_experiment_bundle(result: dict, spec: dict, path: str | Path) -> Path:
    """Save a trained experiment without serializing trajectory objects."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "spec": dict(spec),
        "params": dict(result["params"]),
        "ae_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in result["ae"].state_dict().items()
        },
        "dyn_state_dict": {
            key: value.detach().cpu().clone()
            for key, value in result["dyn"].state_dict().items()
        },
        "stats": result["stats"],
        "ae_history": result["ae_history"],
        "dyn_history": result["dyn_history"],
        "rollout_rows": result.get("rollout_rows", pd.DataFrame()),
        "rollout_stats": result.get("rollout_stats", pd.DataFrame()),
        "ae_reconstruction_rows": result.get("ae_reconstruction_rows", pd.DataFrame()),
        "ae_reconstruction_stats": result.get("ae_reconstruction_stats", pd.DataFrame()),
    }
    torch.save(bundle, path)
    return path


def load_experiment_bundle(
    path: str | Path,
    *,
    cfg: dict,
    device,
    split_cache: dict | None = None,
) -> dict:
    """Restore models, normalizers, and split data from a saved bundle."""

    bundle = torch.load(path, map_location=device, weights_only=False)
    spec = dict(bundle["spec"])
    spec["path"] = resolve_existing_path(spec["path"])
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
    latent_stats = LatentNormalizer.from_dict(
        {
            "z_mean": stats["z_mean"].to(device),
            "z_std": stats["z_std"].to(device),
            "dz_mean": stats["dz_mean"].to(device),
            "dz_std": stats["dz_std"].to(device),
            "z_next_mean": stats.get("z_next_mean", stats["z_mean"]).to(device),
            "z_next_std": stats.get("z_next_std", stats["z_std"]).to(device),
            "context_mean": (
                stats["context_mean"].to(device)
                if "context_mean" in stats
                else None
            ),
            "context_std": (
                stats["context_std"].to(device)
                if "context_std" in stats
                else None
            ),
        }
    )
    autoencoder_type = str(params.get("autoencoder_model", "attention")).lower()
    if autoencoder_type in {"mlp", "mean_mlp", "mean_pool"}:
        autoencoder_cls = NodeDeltaMLPAutoEncoder
    elif autoencoder_type in {"pyramid_mlp", "mean_pyramid_mlp"}:
        autoencoder_cls = NodeDeltaPyramidMLPAutoEncoder
    elif autoencoder_type in {"single_stage_attention", "direct_latent_attention", "node_to_latent_attention"}:
        autoencoder_cls = NodeDeltaSingleStageAttentionAutoEncoder
    else:
        autoencoder_cls = NodeDeltaAttentionAutoEncoder
    ae_model = autoencoder_cls(
        pos_dim=int(params["pos_dim"]),
        node_feature_dim=int(
            params.get(
                "node_feature_dim",
                normalizers["node_feature_mean"].numel(),
            )
        ),
        edge_dim=int(normalizers["edge_mean"].numel()),
        hidden_size=int(params["hidden_size"]),
        latent_dim=int(params["latent_dim"]),
        latent_tokens=int(params["latent_tokens"]),
    ).to(device)
    ae_model.edge_mode = str(params.get("edge_mode", "stored"))
    ae_model.load_state_dict(bundle["ae_state_dict"])
    ae_model.eval()

    objective = str(params.get("propagator_objective", "one_step")).lower()
    loss_mode = str(params.get("propagator_loss", "delta")).lower()
    kinematic_objectives = {
        "kinematic_multistep",
        "kinematic",
        "anchored_multistep",
        "closed_loop",
    }
    default_model = (
        "velocity_mlp"
        if objective in {"velocity", "second_order"}
        else (
            "kinematic_mlp"
            if objective in kinematic_objectives
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
    dyn_model.load_state_dict(bundle["dyn_state_dict"])
    dyn_model.eval()

    mixture = params.get("dataset_mixture") or []
    train_count = params.get("train_count")
    val_count = params.get("val_count")
    if train_count is None and mixture:
        train_count = sum(int(item.get("train_count", 0)) for item in mixture)
    if val_count is None and mixture:
        val_count = sum(int(item.get("val_count", 0)) for item in mixture)
    holdout_train_count = sum(
        int(item.get("holdout_train_count", item.get("train_count", 0)))
        for item in mixture
    )
    if train_count is None or val_count is None:
        raise KeyError(
            "Cached latent experiment requires train_count/val_count or dataset_mixture counts"
        )

    split_key = (
        spec["path"],
        int(train_count),
        int(holdout_train_count),
        int(val_count),
        params.get("split_seed"),
        params.get("min_train_p_ratio"),
        bool(params.get("split_stratify_temperature", False)),
    )
    split_data = split_cache.get(split_key) if split_cache is not None else None
    if split_data is None:
        split_data = resolve_train_val_test(
            spec,
            params,
            split_seed=params.get("split_seed"),
        )
        if split_cache is not None:
            split_cache[split_key] = split_data
    train_data, val_data, test_data, split_info = split_data
    return {
        "label": spec["label"],
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
        "ae_history": pd.DataFrame(bundle.get("ae_history", [])),
        "dyn_history": pd.DataFrame(bundle.get("dyn_history", [])),
        "rollout_rows": pd.DataFrame(bundle.get("rollout_rows", [])),
        "rollout_stats": pd.DataFrame(bundle.get("rollout_stats", [])),
        "ae_reconstruction_rows": pd.DataFrame(bundle.get("ae_reconstruction_rows", [])),
        "ae_reconstruction_stats": pd.DataFrame(bundle.get("ae_reconstruction_stats", [])),
    }


def evaluate_experiment(result: dict, cfg: dict, *, device) -> dict:
    """Evaluate every configured rollout horizon on the test split."""

    params = result["params"]
    sims = result["test_data"]
    max_eval_sims = cfg.get(
        "rollout_eval_max_sims_per_split",
        params.get("rollout_eval_max_sims_per_split"),
    )
    if max_eval_sims is not None:
        sims = sims[: int(max_eval_sims)]
    steps = rollout_steps_for_sims(
        sims,
        cfg["rollout_steps_grid"],
        frame_skip=int(params.get("frame_skip", 1)),
    )
    result["rollout_rows"], result["rollout_stats"] = evaluate_rollout_horizons(
        result["ae"],
        result["dyn"],
        sims,
        result["latent_stats"],
        cfg=params,
        normalizers=result["normalizers"],
        dataset=result["label"],
        split_name="test",
        rollout_steps=steps,
        device=device,
    )
    return result


def _metadata(frame: pd.DataFrame, spec: dict) -> pd.DataFrame:
    frame = frame.copy()
    frame["dataset_name"] = spec["dataset_name"]
    frame["dataset_label"] = spec["dataset_label"]
    frame["target_mode"] = spec["sweep_target_mode"]
    frame["latent_dim"] = int(spec["sweep_latent_dim"])
    frame["train_networks"] = int(spec["sweep_train_count"])
    frame["train_frames_per_network"] = int(spec["sweep_train_frames"])
    frame["repeat_idx"] = int(spec["sweep_repeat"])
    frame["model_seed"] = int(spec["sweep_model_seed"])
    return frame


def _summary_row(result: dict, spec: dict, model_path: Path) -> dict:
    stats = result["rollout_stats"]
    target_step = max(stats["rollout_steps"]) if not stats.empty else np.nan
    row = stats[stats["rollout_steps"].eq(target_step)].tail(1)
    return {
        "dataset": result["label"],
        "dataset_name": spec["dataset_name"],
        "dataset_label": spec["dataset_label"],
        "target_mode": spec["sweep_target_mode"],
        "repeat_idx": int(spec["sweep_repeat"]),
        "model_seed": int(spec["sweep_model_seed"]),
        "latent_dim": int(spec["sweep_latent_dim"]),
        "train_networks": int(spec["sweep_train_count"]),
        "train_frames_per_network": int(spec["sweep_train_frames"]),
        "summary_rollout_step": int(target_step) if np.isfinite(target_step) else np.nan,
        "test_rollout_final_pos_mse": (
            float(row["final_pos_mse"].iloc[0]) if not row.empty else np.nan
        ),
        "test_rollout_position_r2": (
            float(row["rollout_position_r2"].iloc[0]) if not row.empty else np.nan
        ),
        "test_rollout_p_ratio_r2": (
            float(row["p_ratio_r2"].iloc[0]) if not row.empty else np.nan
        ),
        "ae_best_val_mse_norm": float(result["ae_history"]["val_mse_norm"].min()),
        "dyn_best_val_dz_mse_norm": float(result["dyn_history"]["val_dz_mse_norm"].min()),
        "model_path": str(model_path),
    }


def aggregate_repeats(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    keys = [
        "dataset_name",
        "target_mode",
        "latent_dim",
        "train_networks",
        "train_frames_per_network",
    ]
    metrics = [
        col
        for col in summary.select_dtypes(include=[np.number]).columns
        if col not in {*keys, "repeat_idx", "model_seed"}
    ]
    mean = summary.groupby(keys, as_index=False)[metrics].mean()
    repeats = (
        summary.groupby(keys, as_index=False)["repeat_idx"]
        .nunique()
        .rename(columns={"repeat_idx": "n_repeats"})
    )
    out = mean.merge(repeats, on=keys, how="left")
    for metric in (
        "test_rollout_final_pos_mse",
        "test_rollout_position_r2",
        "test_rollout_p_ratio_r2",
        "ae_best_val_mse_norm",
        "dyn_best_val_dz_mse_norm",
    ):
        std = (
            summary.groupby(keys, as_index=False)[metric]
            .std()
            .rename(columns={metric: f"{metric}_std"})
        )
        out = out.merge(std, on=keys, how="left")
        out[f"{metric}_std"] = out[f"{metric}_std"].fillna(0.0)
    return out


def run_capacity_sweep(
    specs: list[dict],
    cfg: dict,
    *,
    device,
    output_dir: str | Path,
    force_training: bool = False,
    collect_initial_latents: bool = True,
) -> dict[str, pd.DataFrame]:
    """Train or restore every sweep run and return all paper-facing tables."""

    output_dir = Path(output_dir)
    model_dir = output_dir / "models"
    export_dir = output_dir / "exports"
    model_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    rollout_stats = []
    rollout_raw = []
    ae_histories = []
    dyn_histories = []
    latent_raw = []
    manifests = []
    split_cache = {}

    for run_idx, spec in enumerate(specs, start=1):
        path = model_dir / capacity_model_name(spec)
        print(f"[{run_idx}/{len(specs)}] {path.name}")
        local_cfg = {
            **cfg,
            "dataset_name": spec["dataset_name"],
            "latent_dim": int(spec["sweep_latent_dim"]),
            "ae_target_mode": spec["sweep_target_mode"],
            "node_feature_mode": spec["sweep_target_mode"],
            "ae_max_train_frames_per_sim": int(spec["sweep_train_frames"]),
            "dyn_max_train_transitions_per_sim": int(spec["sweep_train_frames"]),
            "model_seed": int(spec["sweep_model_seed"]),
            "repeat_idx": int(spec["sweep_repeat"]),
        }
        seed_everything(spec["sweep_model_seed"])
        if path.exists() and not force_training:
            result = load_experiment_bundle(
                path,
                cfg=local_cfg,
                device=device,
                split_cache=split_cache,
            )
            result = evaluate_experiment(result, local_cfg, device=device)
        else:
            result = train_latent_experiment(spec, local_cfg, device=device)
            save_experiment_bundle(result, spec, path)

        summaries.append(_summary_row(result, spec, path))
        rollout_stats.append(_metadata(result["rollout_stats"], spec))
        rollout_raw.append(_metadata(result["rollout_rows"], spec))
        ae_histories.append(_metadata(result["ae_history"], spec))
        dyn_histories.append(_metadata(result["dyn_history"], spec))
        if collect_initial_latents:
            latent_raw.extend(
                _metadata(
                    initial_latent_table(result, result[f"{split}_data"], split, device=device),
                    spec,
                )
                for split in ("val", "test")
            )
        manifests.append(
            {
                "model_path": str(path),
                "dataset_name": spec["dataset_name"],
                "latent_dim": int(spec["sweep_latent_dim"]),
                "train_networks": int(spec["sweep_train_count"]),
                "train_frames_per_network": int(spec["sweep_train_frames"]),
                "repeat_idx": int(spec["sweep_repeat"]),
            }
        )

    tables = {
        "summary": pd.DataFrame(summaries),
        "rollout_stats": pd.concat(rollout_stats, ignore_index=True),
        "rollout_raw": pd.concat(rollout_raw, ignore_index=True),
        "ae_history": pd.concat(ae_histories, ignore_index=True),
        "dyn_history": pd.concat(dyn_histories, ignore_index=True),
        "latent_raw": (
            pd.concat(latent_raw, ignore_index=True)
            if latent_raw
            else pd.DataFrame()
        ),
        "model_manifest": pd.DataFrame(manifests),
    }
    tables["summary_mean"] = aggregate_repeats(tables["summary"])
    for name, frame in tables.items():
        frame.to_csv(export_dir / f"{name}.csv", index=False)
    return tables


def fit_initial_latent_readouts(
    latent_raw: pd.DataFrame,
    *,
    ridge_alpha: float = 1e-3,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit readouts on validation latents and report only held-out test scores."""

    keys = [
        "dataset_name",
        "target_mode",
        "latent_dim",
        "train_networks",
        "train_frames_per_network",
        "repeat_idx",
    ]
    metrics = []
    predictions = []
    for key, group in latent_raw.groupby(keys, sort=False):
        val = group[group["split"].eq("val")]
        test = group[group["split"].eq("test")]
        latent_dim = int(key[2])
        z_cols = [f"z{i}" for i in range(latent_dim)]
        if val.empty or test.empty or not z_cols:
            continue

        single_candidates = []
        for z_col in z_cols:
            x_val_single = val[[z_col]].to_numpy(float)
            single_mean = x_val_single.mean(axis=0, keepdims=True)
            single_std = x_val_single.std(axis=0, keepdims=True) + 1e-12
            design_single = np.column_stack(
                [np.ones(len(val)), (x_val_single - single_mean) / single_std]
            )
            penalty_single = np.eye(design_single.shape[1])
            penalty_single[0, 0] = 0
            coef_single = np.linalg.solve(
                design_single.T @ design_single
                + float(ridge_alpha) * penalty_single,
                design_single.T @ val["final_p_ratio"].to_numpy(float),
            )
            val_pred = design_single @ coef_single
            single_candidates.append(
                (
                    r2_score(val["final_p_ratio"], val_pred),
                    z_col,
                    single_mean,
                    single_std,
                    coef_single,
                )
            )
        _, best_coordinate, single_mean, single_std, single_coef = max(
            single_candidates,
            key=lambda item: item[0] if np.isfinite(item[0]) else -np.inf,
        )
        single_test_design = np.column_stack(
            [
                np.ones(len(test)),
                (test[[best_coordinate]].to_numpy(float) - single_mean) / single_std,
            ]
        )
        single_pred = single_test_design @ single_coef

        x_mean = val[z_cols].to_numpy(float).mean(axis=0, keepdims=True)
        x_std = val[z_cols].to_numpy(float).std(axis=0, keepdims=True) + 1e-12
        x_val = (val[z_cols].to_numpy(float) - x_mean) / x_std
        y_val = val["final_p_ratio"].to_numpy(float)
        design = np.column_stack([np.ones(len(x_val)), x_val])
        penalty = np.eye(design.shape[1])
        penalty[0, 0] = 0
        coef = np.linalg.solve(
            design.T @ design + float(ridge_alpha) * penalty,
            design.T @ y_val,
        )
        x_test = (test[z_cols].to_numpy(float) - x_mean) / x_std
        pred = np.column_stack([np.ones(len(x_test)), x_test]) @ coef
        true = test["final_p_ratio"].to_numpy(float)
        base = dict(zip(keys, key, strict=False))
        metrics.append(
            {
                **base,
                "best_single_coordinate": best_coordinate,
                "single_test_r2": r2_score(true, single_pred),
                "single_test_rmse": float(np.sqrt(np.mean((true - single_pred) ** 2))),
                "combo_test_r2": r2_score(true, pred),
                "combo_test_rmse": float(np.sqrt(np.mean((true - pred) ** 2))),
                "n_val": int(len(val)),
                "n_test": int(len(test)),
            }
        )
        predictions.append(
            pd.DataFrame(
                {
                    **base,
                    "split": "test",
                    "sim_idx": test["sim_idx"].to_numpy(int),
                    "true_p_ratio": true,
                    "best_single_coordinate": best_coordinate,
                    "best_single_pred_p_ratio": single_pred,
                    "linear_combo_pred_p_ratio": pred,
                }
            )
        )

    metrics_df = pd.DataFrame(metrics)
    predictions_df = (
        pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    )
    summary = (
        metrics_df.groupby(keys[:-1], as_index=False)
        .agg(
            single_test_r2_mean=("single_test_r2", "mean"),
            single_test_r2_std=("single_test_r2", "std"),
            combo_test_r2_mean=("combo_test_r2", "mean"),
            combo_test_r2_std=("combo_test_r2", "std"),
            n_repeats=("repeat_idx", "nunique"),
        )
        if not metrics_df.empty
        else pd.DataFrame()
    )
    if not summary.empty:
        summary[["single_test_r2_std", "combo_test_r2_std"]] = summary[
            ["single_test_r2_std", "combo_test_r2_std"]
        ].fillna(0.0)
    return metrics_df, predictions_df, summary


__all__ = [
    "aggregate_repeats",
    "build_capacity_specs",
    "capacity_model_name",
    "evaluate_experiment",
    "fit_initial_latent_readouts",
    "load_experiment_bundle",
    "run_capacity_sweep",
    "save_experiment_bundle",
]
