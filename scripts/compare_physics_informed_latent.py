"""Compare normal, hybrid-energy, and physics-only 04a latent simulators."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from lss.latent.experiment import (
    evaluate_rollout_horizons,
    find_project_root,
    rollout_metrics,
    run_latent_experiment,
    seed_everything,
)
from lss.utils import resolve_device


SEED = 20260705
HORIZONS = (50, 100, 150, 199)
VARIANTS = {
    "normal_latent_mse": {"enabled": False},
    "physics_only": {
        "enabled": True, "lambda_phys": 1.0, "lambda_mse": 0.0,
        "noise": 0.01,
    },
}


def _spec(root: Path, output: Path, name: str, variant: dict, *, force_train: bool, device):
    dataset_path = str(root / "data" / "depablo-10k-mix-temp.pt")
    cfg = {
        "dataset_name": "depablo_mixed_temp", "split_seed": 20260623,
        "split_stratify_temperature": False, "min_train_p_ratio": None,
        "device": str(device), "pos_dim": 2, "batch_graphs": 4, "frame_skip": 1,
        "train_frame_start_order": 0, "latent_dim": 2, "latent_tokens": 32,
        "hidden_size": 96, "autoencoder_model": "attention", "edge_feature_dim": 12,
        "ae_max_train_frames_per_sim": 100, "dyn_max_train_transitions_per_sim": 100,
        "ae_target_mode": "normalized_delta", "node_feature_mode": "normalized_delta",
        "ae_max_epochs": 60, "ae_patience": 6, "ae_lr": 5e-5, "ae_weight_decay": 1e-5,
        "dyn_max_epochs": 60, "dyn_patience": 6, "dyn_lr": 3e-5,
        "dyn_weight_decay": 1e-4, "propagator_use_static_context": True,
        "graph_context_dim": 16, "propagator_context_include_temperature": False,
        "propagator_step_stride": 1, "initial_velocity": "zero",
        "early_stop_min_delta": 1e-5, "rollout_steps_grid": list(HORIZONS),
        "rollout_eval_max_sims_per_split": 20, "temperature_pratio_window": "full",
        "temperature_pratio_estimator": "robust", "temperature_pratio_min_fit_frames": 8,
        "temperature_pratio_min_driven_strain_range": 1e-3,
        "temperature_pratio_smooth_window": 5, "should_rollout": True,
        "should_train_propagator": True, "force_train": bool(force_train),
        "cache_path": str(output / "model.pt"), "propagator_objective": "one_step",
        "propagator_model": "delta_mlp", "propagator_loss": "delta",
        "propagator_standardize_latent": False, "model_seed": SEED, "repeat_idx": 1,
        "physics_loss_enabled": bool(variant["enabled"]),
        "physics_lambda": float(variant.get("lambda_phys", 0.0)),
        "physics_mse_lambda": float(variant.get("lambda_mse", 1.0)),
        "physics_latent_noise_std": float(variant.get("noise", 0.0)),
        "physics_dt": 1.0, "physics_default_mass": 1.0,
        "physics_normalize_by_speed": False,
    }
    source = {
        "dataset_name": "depablo_mixed_temp", "source_name": "dePablo mixed temperature",
        "label": f"dePablo mixed temperature {name}", "path": dataset_path,
        "dataset_mixture": [{"name": "depablo_mixed_temp",
            "label": "dePablo mixed temperature", "path": dataset_path,
            "train_count": 20, "holdout_train_count": 20, "val_count": 20}],
        "target_mode": "normalized_delta", "ae_target_mode": "normalized_delta",
        "node_feature_mode": "normalized_delta", "latent_dim": 2, "repeat_idx": 1,
        "model_seed": SEED, "latent_tokens": 32, "hidden_size": 96,
        "autoencoder_model": "attention", "edge_feature_dim": 12,
        "ae_max_train_frames_per_sim": 100, "dyn_max_train_transitions_per_sim": 100,
        "ae_max_epochs": 60, "ae_patience": 6, "ae_lr": 5e-5,
        "ae_weight_decay": 1e-5, "dyn_max_epochs": 60, "dyn_patience": 6,
        "dyn_lr": 3e-5, "dyn_weight_decay": 1e-4,
    }
    return cfg, source


def run_physics_comparison(*, force_train=False, device_name="auto"):
    root = find_project_root()
    output = root / "notebooks" / "results" / "04a_physics_informed_latent"
    output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device_name)
    metric_rows, history_parts = [], []
    for name, variant in VARIANTS.items():
        variant_output = output / name
        variant_output.mkdir(parents=True, exist_ok=True)
        cfg, source = _spec(root, variant_output, name, variant,
                            force_train=force_train, device=device)
        seed_everything(SEED)
        result = run_latent_experiment(source, cfg, device=device)
        history = result["dyn_history"].copy(); history["variant"] = name
        history_parts.append(history)
        parts = []
        for start in range(0, len(result["test_data"]), 20):
            sims = result["test_data"][start:start + 20]
            raw, _ = evaluate_rollout_horizons(
                result["ae"], result["dyn"], sims, result["latent_stats"],
                cfg=result["params"], normalizers=result["normalizers"],
                dataset=result["label"], split_name="test",
                rollout_steps=list(HORIZONS), device=device,
            )
            raw = raw.copy(); raw["sim_idx"] += start; raw["variant"] = name
            parts.append(raw)
            print(f"{name}: evaluated {start + len(sims)}/{len(result['test_data'])}", flush=True)
        raw = pd.concat(parts, ignore_index=True)
        raw.to_csv(variant_output / "all_test_rollout_rows.csv", index=False)
        for horizon, group in raw.groupby("rollout_steps", sort=True):
            row = rollout_metrics(group, dataset=result["label"], split_name="test",
                                  rollout_steps=int(horizon))
            row["variant"] = name
            metric_rows.append(row)
    metrics = pd.DataFrame(metric_rows).sort_values(["rollout_steps", "variant"])
    histories = pd.concat(history_parts, ignore_index=True)
    metrics.to_csv(output / "comparison.csv", index=False)
    histories.to_csv(output / "training_history.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for variant, group in metrics.groupby("variant"):
        axes[0].plot(group["rollout_steps"], group["rollout_position_r2"], "o-", label=variant)
        axes[1].plot(group["rollout_steps"], group["p_ratio_r2"].clip(lower=0), "o-", label=variant)
    axes[0].set(xlabel="frame", ylabel="position R²", title="Position rollout")
    axes[1].set(xlabel="frame", ylabel="p-ratio R² (clipped at 0)", title="P-ratio rollout")
    for axis in axes: axis.legend(frameon=False)
    fig.savefig(output / "physics_loss_rollout_comparison.png", dpi=220)
    plt.close(fig)
    print(metrics[["variant", "rollout_steps", "rollout_position_r2",
                   "final_pos_mse", "p_ratio_r2"]].to_string(index=False))
    return metrics


if __name__ == "__main__":
    run_physics_comparison()
