"""Train and evaluate LJ latent simulators with one versus twenty networks."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

from lss.latent.analysis import framewise_latent_descriptor_sweep
from lss.latent.capacity import evaluate_experiment
from lss.latent.experiment import run_latent_experiment, seed_everything
from lss.utils import resolve_device


DATA_PATH = PROJECT_ROOT / "data" / "lj-noisy-eps0.01-sigma1.0-cutoff1.122_200sims_200frames.pt"
SPLIT_SEED = 20260716
MODEL_SEED = 20260716


def residualize(values: np.ndarray, progress: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(progress)), progress])
    return values - design @ np.linalg.lstsq(design, values, rcond=None)[0]


def latent_metrics(frame: pd.DataFrame) -> dict[str, float]:
    test = frame[frame["split"].eq("test")].copy()
    z = test[["z0", "z1"]].to_numpy(float)
    z_standard = (z - z.mean(axis=0)) / (z.std(axis=0) + 1e-12)
    progress = test["frame_progress"].to_numpy(float)
    residual = residualize(z_standard, progress)
    singular = np.linalg.svd(z_standard, compute_uv=False) ** 2
    initial = test.sort_values("frame_idx").groupby("sim_idx", as_index=False).first()
    return {
        "test_frames": len(test),
        "test_networks": int(test["sim_idx"].nunique()),
        "framewise_z0_z1_r": float(np.corrcoef(z_standard.T)[0, 1]),
        "partial_z0_z1_r_given_progress": float(np.corrcoef(residual.T)[0, 1]),
        "initial_z0_z1_r": float(np.corrcoef(initial[["z0", "z1"]].to_numpy(float).T)[0, 1]),
        "pc2_fraction": float(singular[1] / singular.sum()),
        "z0_pratio_r": float(np.corrcoef(test["z0"], test["side_final_trajectory_p_ratio"])[0, 1]),
        "z1_pratio_r": float(np.corrcoef(test["z1"], test["side_final_trajectory_p_ratio"])[0, 1]),
    }


def specs(train_count: int, *, output_dir: Path, device, max_epochs: int, force: bool):
    cache = output_dir / "models" / f"lj_noisy_train{train_count}_seed{MODEL_SEED}.pt"
    cache.parent.mkdir(parents=True, exist_ok=True)
    cfg = {
        "dataset_name": "lj_noisy",
        "split_seed": SPLIT_SEED,
        "device": str(device),
        "pos_dim": 2,
        "batch_graphs": 8,
        "frame_skip": 1,
        "train_frame_start_order": 0,
        "latent_dim": 2,
        "latent_tokens": 32,
        "hidden_size": 96,
        "autoencoder_model": "attention",
        "edge_feature_dim": 12,
        "ae_target_mode": "normalized_delta",
        "node_feature_mode": "normalized_delta",
        "ae_max_train_frames_per_sim": 100,
        "dyn_max_train_transitions_per_sim": 100,
        "ae_max_epochs": max_epochs,
        "ae_patience": 8,
        "ae_lr": 5e-5,
        "ae_weight_decay": 1e-5,
        "dyn_max_epochs": max_epochs,
        "dyn_patience": 8,
        "dyn_lr": 3e-5,
        "dyn_weight_decay": 1e-4,
        "propagator_use_static_context": True,
        "graph_context_dim": 16,
        "propagator_context_include_temperature": False,
        "propagator_step_stride": 1,
        "initial_velocity": "zero",
        "propagator_objective": "one_step",
        "propagator_model": "delta_mlp",
        "propagator_loss": "delta",
        "propagator_standardize_latent": False,
        "early_stop_min_delta": 1e-5,
        "rollout_steps_grid": [10, 20, 50, 100, 150],
        "rollout_eval_max_sims_per_split": 30,
        "temperature_pratio_window": "full",
        "temperature_pratio_estimator": "robust",
        "temperature_pratio_min_fit_frames": 8,
        "temperature_pratio_min_driven_strain_range": 1e-3,
        "temperature_pratio_smooth_window": 5,
        "should_rollout": True,
        "should_train_propagator": True,
        "force_train": force,
        "cache_path": str(cache),
        "model_seed": MODEL_SEED,
        "repeat_idx": 1,
    }
    source = {
        "dataset_name": "lj_noisy",
        "source_name": "LJ noisy",
        "label": f"LJ noisy CV2, train={train_count}",
        "path": str(DATA_PATH),
        "dataset_mixture": [
            {
                "name": "lj_noisy",
                "label": "LJ noisy",
                "path": str(DATA_PATH),
                "train_count": train_count,
                "val_count": 20,
            }
        ],
        "target_mode": "normalized_delta",
        "ae_target_mode": "normalized_delta",
        "node_feature_mode": "normalized_delta",
        "latent_dim": 2,
        "latent_tokens": 32,
        "hidden_size": 96,
        "autoencoder_model": "attention",
        "edge_feature_dim": 12,
        "ae_max_train_frames_per_sim": 100,
        "dyn_max_train_transitions_per_sim": 100,
        "ae_max_epochs": max_epochs,
        "ae_patience": 8,
    }
    return source, cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-counts", nargs="+", type=int, default=[1, 20])
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Compile the LJ dataset first: {DATA_PATH}")
    output_dir = PROJECT_ROOT / "notebooks" / "results" / "08_lj_train1_vs20"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    latent_rows = []
    rollout_parts = []

    for train_count in args.train_counts:
        print(f"\n### LJ noisy: train_count={train_count}", flush=True)
        seed_everything(MODEL_SEED)
        source, cfg = specs(
            train_count,
            output_dir=output_dir,
            device=device,
            max_epochs=args.max_epochs,
            force=args.force,
        )
        result = run_latent_experiment(source, cfg, device=device)
        if result["rollout_stats"].empty:
            result = evaluate_experiment(result, cfg, device=device)
        frame, *_ = framewise_latent_descriptor_sweep(
            result, device=device, frame_stride=4, max_frames_per_sim=60
        )
        frame.insert(0, "train_count", train_count)
        frame.to_csv(output_dir / f"framewise_train{train_count}.csv", index=False)
        best_ae = result["ae_history"].loc[result["ae_history"]["val_objective"].idxmin()]
        latent_rows.append(
            {
                "train_count": train_count,
                **latent_metrics(frame),
                "best_val_reconstruction": float(best_ae["val_mse_norm"]),
            }
        )
        rollout = result["rollout_stats"].copy()
        rollout.insert(0, "train_count", train_count)
        rollout_parts.append(rollout)
        pd.DataFrame(latent_rows).to_csv(output_dir / "latent_correlation_summary.csv", index=False)
        pd.concat(rollout_parts, ignore_index=True).to_csv(
            output_dir / "rollout_by_train_count.csv", index=False
        )

    latent_summary = pd.DataFrame(latent_rows)
    rollout_all = pd.concat(rollout_parts, ignore_index=True)
    rollout_summary = rollout_all[rollout_all["split"].eq("test")][
        [
            "train_count",
            "rollout_steps",
            "used",
            "rollout_position_r2",
            "p_ratio_r2",
            "p_ratio_pearson",
            "final_pos_mse",
        ]
    ].sort_values(["train_count", "rollout_steps"])
    rollout_summary.to_csv(output_dir / "test_rollout_summary.csv", index=False)
    print("\nLatent summary\n", latent_summary.to_string(index=False))
    print("\nHeld-out rollout summary\n", rollout_summary.to_string(index=False))


if __name__ == "__main__":
    main()
