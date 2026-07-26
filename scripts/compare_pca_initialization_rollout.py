"""Controlled PCA-initialization ablation on mixed-temperature dePablo data."""

from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import pandas as pd
import torch

from lss.latent.experiment import (
    evaluate_autoencoder_reconstruction_horizons,
    evaluate_rollout_horizons,
    find_project_root,
    rollout_metrics,
    run_latent_experiment,
    seed_everything,
)
from lss.utils import resolve_device


BASE_SEED = 20260705


def _all_test_metrics(result, *, device, horizon=199):
    rollout_parts, oracle_parts = [], []
    for start in range(0, len(result["test_data"]), 20):
        sims = result["test_data"][start:start + 20]
        rollout, _ = evaluate_rollout_horizons(
            result["ae"], result["dyn"], sims, result["latent_stats"],
            cfg=result["params"], normalizers=result["normalizers"],
            dataset=result["label"], split_name="test", rollout_steps=[horizon], device=device,
        )
        oracle, _ = evaluate_autoencoder_reconstruction_horizons(
            result["ae"], sims, cfg=result["params"], normalizers=result["normalizers"],
            dataset=result["label"], split_name="test", rollout_steps=[horizon], device=device,
        )
        rollout = rollout.copy(); oracle = oracle.copy()
        rollout["sim_idx"] += start; oracle["sim_idx"] += start
        rollout_parts.append(rollout); oracle_parts.append(oracle)
        print(f"evaluated {start + len(sims)}/{len(result['test_data'])}", flush=True)
    rollout_rows = pd.concat(rollout_parts, ignore_index=True)
    oracle_rows = pd.concat(oracle_parts, ignore_index=True)
    rollout_stats = rollout_metrics(
        rollout_rows, dataset=result["label"], split_name="test", rollout_steps=horizon
    )
    oracle_stats = rollout_metrics(
        oracle_rows.rename(columns={"reconstructed_p_ratio": "pred_p_ratio"})
        if "reconstructed_p_ratio" in oracle_rows else oracle_rows,
        dataset=result["label"], split_name="test", rollout_steps=horizon,
    ) if {"true_p_ratio", "pred_p_ratio"}.issubset(oracle_rows.columns) else {
        "final_pos_mse": float(oracle_rows["final_pos_mse"].mean())
    }
    return rollout_rows, oracle_rows, rollout_stats, oracle_stats


def run_variant(name: str, *, pca_init: bool, force_train: bool, device, seed=BASE_SEED,
                output=None):
    root = find_project_root()
    output = Path(output) if output is not None else (
        root / "notebooks" / "results" / "04k_pca_initialization" / name
    )
    output.mkdir(parents=True, exist_ok=True)
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
        "dyn_max_epochs": 60, "dyn_patience": 6, "dyn_lr": 3e-5, "dyn_weight_decay": 1e-4,
        "propagator_use_static_context": True, "graph_context_dim": 16,
        "propagator_context_include_temperature": False, "propagator_step_stride": 1,
        "initial_velocity": "zero", "early_stop_min_delta": 1e-5,
        "rollout_steps_grid": [199], "rollout_eval_max_sims_per_split": 20,
        "temperature_pratio_window": "full", "temperature_pratio_estimator": "robust",
        "temperature_pratio_min_fit_frames": 8,
        "temperature_pratio_min_driven_strain_range": 1e-3,
        "temperature_pratio_smooth_window": 5, "should_rollout": True,
        "should_train_propagator": True, "force_train": bool(force_train),
        "cache_path": str(output / "model.pt"), "propagator_objective": "one_step",
        "propagator_model": "delta_mlp", "propagator_loss": "delta",
        "propagator_standardize_latent": False, "model_seed": int(seed), "repeat_idx": 1,
        "pca_initialize_displacement_layers": bool(pca_init),
    }
    source = {
        "dataset_name": "depablo_mixed_temp", "source_name": "dePablo mixed temperature",
        "label": f"dePablo mixed temperature {name}", "path": dataset_path,
        "dataset_mixture": [{"name": "depablo_mixed_temp", "label": "dePablo mixed temperature",
                             "path": dataset_path, "train_count": 20,
                             "holdout_train_count": 20, "val_count": 20}],
        "target_mode": "normalized_delta", "ae_target_mode": "normalized_delta",
        "node_feature_mode": "normalized_delta", "latent_dim": 2, "repeat_idx": 1,
        "model_seed": int(seed), "latent_tokens": 32, "hidden_size": 96,
        "autoencoder_model": "attention", "edge_feature_dim": 12,
        "ae_max_train_frames_per_sim": 100, "dyn_max_train_transitions_per_sim": 100,
        "ae_max_epochs": 60, "ae_patience": 6, "ae_lr": 5e-5, "ae_weight_decay": 1e-5,
        "dyn_max_epochs": 60, "dyn_patience": 6, "dyn_lr": 3e-5, "dyn_weight_decay": 1e-4,
        "pca_initialize_displacement_layers": bool(pca_init),
    }
    seed_everything(seed)
    started = time.perf_counter()
    result = run_latent_experiment(source, cfg, device=device)
    rollout_rows, oracle_rows, rollout, oracle = _all_test_metrics(result, device=device)
    rollout_rows.to_csv(output / "rollout_rows_199.csv", index=False)
    oracle_rows.to_csv(output / "oracle_rows_199.csv", index=False)
    row = {
        "variant": name, "seed": int(seed), "pca_initialization": bool(pca_init),
        "test_networks": len(result["test_data"]), "rollout_step": 199,
        "rollout_position_mse": rollout.get("final_pos_mse"),
        "rollout_position_r2": rollout.get("rollout_position_r2"),
        "rollout_pratio_r2": rollout.get("p_ratio_r2"),
        "rollout_pratio_pearson": rollout.get("p_ratio_pearson"),
        "oracle_position_mse": oracle.get("final_pos_mse"),
        "ae_best_val_mse": float(result["ae_history"]["val_mse_norm"].min()),
        "dyn_best_val_mse": float(result["dyn_history"]["val_dz_mse_norm"].min()),
        "elapsed_seconds": time.perf_counter() - started,
    }
    return row


def run_comparison(*, force_train=False, device_name="auto"):
    root = find_project_root()
    output = root / "notebooks" / "results" / "04k_pca_initialization"
    output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device_name)
    rows = [
        run_variant("random_initialization", pca_init=False, force_train=force_train, device=device),
        run_variant("pca_initialization", pca_init=True, force_train=force_train, device=device),
    ]
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output / "comparison.csv", index=False)
    print(comparison.to_string(index=False))
    return comparison


def run_repeated_comparison(*, repeats=5, force_train=False, device_name="auto"):
    root = find_project_root()
    output = root / "notebooks" / "results" / "04k_pca_initialization"
    repeated = output / "five_seeds"
    repeated.mkdir(parents=True, exist_ok=True)
    device = resolve_device(device_name)
    seeds = [BASE_SEED + offset for offset in range(int(repeats))]
    rows = []
    for seed in seeds:
        for name, pca_init in (("random_initialization", False), ("pca_initialization", True)):
            # Reuse the already completed matched run for the first seed.
            variant_output = output / name if seed == BASE_SEED else repeated / f"seed_{seed}" / name
            rows.append(run_variant(
                name, pca_init=pca_init, force_train=force_train,
                device=device, seed=seed, output=variant_output,
            ))
            pd.DataFrame(rows).to_csv(repeated / "runs.csv", index=False)
    runs = pd.DataFrame(rows)
    metrics = [
        "rollout_position_mse", "rollout_position_r2", "rollout_pratio_r2",
        "rollout_pratio_pearson", "oracle_position_mse", "ae_best_val_mse",
        "dyn_best_val_mse",
    ]
    summary = runs.groupby("variant")[metrics].agg(["mean", "std"])
    summary.to_csv(repeated / "summary.csv")
    print(summary.to_string())
    return runs, summary


if __name__ == "__main__":
    run_repeated_comparison()
