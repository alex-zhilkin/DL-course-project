"""Quick controlled attention-vs-mean-MLP latent simulator ablation."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from lss.latent.experiment import (
    evaluate_rollout_horizons,
    find_project_root,
    rollout_metrics,
    seed_everything,
    train_latent_experiment,
)
from lss.utils import resolve_device


SEEDS = (20260623, 20260624, 20260625)


def run_variant(
    name: str,
    *,
    seed: int,
    project_root: Path,
    device,
    standardize_latent: bool = True,
) -> tuple[dict, pd.DataFrame]:
    dataset_path = str(project_root / "data" / "depablo-10k-mix-temp.pt")
    cfg = {
        "dataset_name": "depablo_mixed_temp",
        "split_seed": int(seed),
        "split_stratify_temperature": False,
        "device": str(device),
        "pos_dim": 2,
        "batch_graphs": 4,
        "frame_skip": 1,
        "train_frame_start_order": 0,
        "latent_dim": 2,
        "latent_tokens": 32,
        "hidden_size": 96,
        "edge_feature_dim": 12,
        "autoencoder_model": name,
        "ae_max_train_frames_per_sim": 30,
        "dyn_max_train_transitions_per_sim": 30,
        "ae_target_mode": "normalized_delta",
        "node_feature_mode": "normalized_delta",
        "ae_max_epochs": 12,
        "ae_patience": 4,
        "ae_lr": 5e-5,
        "ae_weight_decay": 1e-5,
        "dyn_max_epochs": 12,
        "dyn_patience": 4,
        "dyn_lr": 3e-5,
        "dyn_weight_decay": 1e-4,
        "propagator_use_static_context": True,
        "graph_context_dim": 16,
        "propagator_context_include_temperature": False,
        "propagator_step_stride": 1,
        "initial_velocity": "zero",
        "early_stop_min_delta": 1e-5,
        "rollout_steps_grid": [20, 50, 100, 150, 199],
        "rollout_eval_max_sims_per_split": 20,
        "temperature_pratio_window": "full",
        "temperature_pratio_estimator": "robust",
        "temperature_pratio_min_fit_frames": 8,
        "temperature_pratio_min_driven_strain_range": 1e-3,
        "temperature_pratio_smooth_window": 5,
        "should_rollout": True,
        "should_train_propagator": True,
        "propagator_objective": "one_step",
        "propagator_model": "residual_mlp",
        "propagator_loss": "delta",
        "propagator_standardize_latent": bool(standardize_latent),
        "model_seed": int(seed) + 201,
        "repeat_idx": 1,
    }
    source_spec = {
        "dataset_name": "depablo_mixed_temp",
        "source_name": "dePablo mixed temperature",
        "label": f"dePablo mixed temperature CV2 {name}",
        "path": dataset_path,
        "dataset_mixture": [
            {
                "name": "depablo_mixed_temp",
                "label": "dePablo mixed temperature",
                "path": dataset_path,
                "train_count": 20,
                "val_count": 10,
            }
        ],
        "target_mode": "normalized_delta",
        "ae_target_mode": "normalized_delta",
        "node_feature_mode": "normalized_delta",
        "latent_dim": 2,
        "repeat_idx": 1,
        "model_seed": int(seed) + 201,
        "latent_tokens": 32,
        "hidden_size": 96,
        "edge_feature_dim": 12,
        "ae_max_train_frames_per_sim": 30,
        "dyn_max_train_transitions_per_sim": 30,
        "ae_max_epochs": 12,
        "ae_patience": 4,
        "ae_lr": 5e-5,
        "ae_weight_decay": 1e-5,
        "dyn_max_epochs": 12,
        "dyn_patience": 4,
        "dyn_lr": 3e-5,
        "dyn_weight_decay": 1e-4,
        "autoencoder_model": name,
    }
    seed_everything(int(seed) + 201)
    started = time.perf_counter()
    result = train_latent_experiment(source_spec, cfg, device=device)
    test_parts = []
    test_sims = result["test_data"][:100]
    for start in range(0, len(test_sims), 20):
        chunk = test_sims[start : start + 20]
        chunk_rows, _ = evaluate_rollout_horizons(
            result["ae"],
            result["dyn"],
            chunk,
            result["latent_stats"],
            cfg=result["params"],
            normalizers=result["normalizers"],
            dataset=result["label"],
            split_name="test",
            rollout_steps=cfg["rollout_steps_grid"],
            device=device,
        )
        chunk_rows = chunk_rows.copy()
        chunk_rows["sim_idx"] += start
        test_parts.append(chunk_rows)
        print(f"{name} seed={seed}: evaluated {start + len(chunk)}/100 test networks", flush=True)
    test_rows = pd.concat(test_parts, ignore_index=True)
    test = pd.DataFrame(
        [
            rollout_metrics(
                group,
                dataset=result["label"],
                split_name="test",
                rollout_steps=int(step),
            )
            for step, group in test_rows.groupby("rollout_steps", sort=True)
        ]
    )
    elapsed = time.perf_counter() - started
    test["autoencoder"] = name
    test["latent_standardization"] = bool(standardize_latent)
    test["seed"] = int(seed)
    test["elapsed_seconds"] = elapsed
    test["ae_best_val_mse"] = float(result["ae_history"]["val_mse_norm"].min())
    test["dyn_best_val_mse"] = float(result["dyn_history"]["val_dz_mse_norm"].min())
    return result, test


def main() -> None:
    project_root = find_project_root()
    output_dir = project_root / "notebooks" / "results" / "04a_attention_vs_mlp_3seed"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device("auto")
    rows = []
    run_info = {}
    for seed in SEEDS:
        for name in ("attention", "mean_mlp"):
            print(f"\n### seed={seed} {name} ###", flush=True)
            result, stats = run_variant(
                name,
                seed=seed,
                project_root=project_root,
                device=device,
            )
            rows.append(stats)
            run_info[f"{seed}_{name}"] = {
                "ae_parameters": sum(parameter.numel() for parameter in result["ae"].parameters()),
                "propagator_parameters": sum(parameter.numel() for parameter in result["dyn"].parameters()),
            }
            del result
    comparison = pd.concat(rows, ignore_index=True)
    comparison.to_csv(output_dir / "rollout_comparison_by_seed.csv", index=False)
    summary = (
        comparison.groupby(["autoencoder", "rollout_steps"], as_index=False)
        .agg(
            position_r2_mean=("rollout_position_r2", "mean"),
            position_r2_std=("rollout_position_r2", "std"),
            final_pos_mse_mean=("final_pos_mse", "mean"),
            final_pos_mse_std=("final_pos_mse", "std"),
            p_ratio_r2_mean=("p_ratio_r2", "mean"),
            p_ratio_r2_std=("p_ratio_r2", "std"),
            p_ratio_used_mean=("p_ratio_used", "mean"),
            elapsed_seconds_mean=("elapsed_seconds", "mean"),
        )
    )
    summary.to_csv(output_dir / "rollout_comparison_summary.csv", index=False)
    (output_dir / "run_info.json").write_text(json.dumps(run_info, indent=2))
    print("\nTEST COMPARISON BY SEED", flush=True)
    print(
        comparison[
            [
                "autoencoder",
                "seed",
                "rollout_steps",
                "rollout_position_r2",
                "final_pos_mse",
                "p_ratio_r2",
                "ae_best_val_mse",
                "dyn_best_val_mse",
                "elapsed_seconds",
            ]
        ].to_string(index=False),
        flush=True,
    )
    print("\nTHREE-SEED SUMMARY", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"saved: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
