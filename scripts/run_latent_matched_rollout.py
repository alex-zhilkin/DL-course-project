"""Matched validation-only rollout experiments on a frozen shared four-source AE."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np
import pandas as pd
import torch


ROOT = Path(os.environ.get("LSS_PROJECT_ROOT", Path(__file__).resolve().parents[1]))
CODE = Path(os.environ.get("LSS_CODE_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(CODE / "src"))

from lss.latent.experiment import run_latent_experiment, seed_everything


RESULTS = Path(os.environ.get("LSS_ROLLOUT_RESULTS", ROOT / "notebooks/results/latent_matched_rollouts"))
AE_PATH = Path(os.environ.get("LSS_AE_PATH", ROOT / "notebooks/results/latent_matched_study/shared4_d8_s123/ae.pt"))
MANIFEST_PATH = ROOT / "notebooks" / "results" / "latent_matched_study" / "split_manifest.json"
SOURCES = ("reid", "depablo_low_temp", "depablo_mixed_temp", "lj_noisy")
VARIANTS = {
    "acceleration_all4": {
        "model": "history_mlp", "objective": "history_one_step",
        "fixed_observed_frames": (1, 5), "fixed_history_size": 2,
        "multistep_horizons": [1], "context_include_source_id": False,
        "initial_velocity": "three_frames",
    },
    "acceleration_transfer3": {
        "model": "history_mlp", "objective": "history_one_step",
        "fixed_observed_frames": (1, 5), "fixed_history_size": 2,
        "multistep_horizons": [1], "context_include_source_id": False,
        "initial_velocity": "three_frames", "exclude_mixed_training": True,
    },
    "velocity_transfer3": {
        "model": "fixed_velocity_residual_mlp",
        "fixed_observed_frames": (1, 5), "fixed_history_size": 2,
        "multistep_horizons": [1], "context_include_source_id": False,
        "exclude_mixed_training": True,
    },
    "delta_transfer3": {
        "model": "delta_mlp",
        "fixed_observed_frames": (1, 5), "fixed_history_size": 2,
        "multistep_horizons": [1], "context_include_source_id": False,
        "exclude_mixed_training": True, "objective": "one_step",
    },
    "delta_all4": {
        "model": "delta_mlp",
        "fixed_observed_frames": (1, 5), "fixed_history_size": 2,
        "multistep_horizons": [1], "context_include_source_id": False,
        "objective": "one_step",
    },
    "velocity_one": {
        "model": "fixed_velocity_residual_mlp",
        "fixed_observed_frames": (1, 5),
        "fixed_history_size": 2,
        "multistep_horizons": [1],
        "context_include_source_id": False,
    },
    "velocity_source_id_one": {
        "model": "fixed_velocity_residual_mlp",
        "fixed_observed_frames": (1, 5),
        "fixed_history_size": 2,
        "multistep_horizons": [1],
        "context_include_source_id": True,
        "context_include_temperature": False,
    },
    "velocity_temperature_one": {
        "model": "fixed_velocity_residual_mlp",
        "fixed_observed_frames": (1, 5),
        "fixed_history_size": 2,
        "multistep_horizons": [1],
        "context_include_source_id": False,
        "context_include_temperature": True,
    },
    "velocity_source_temperature_one": {
        "model": "fixed_velocity_residual_mlp",
        "fixed_observed_frames": (1, 5),
        "fixed_history_size": 2,
        "multistep_horizons": [1],
        "context_include_source_id": True,
        "context_include_temperature": True,
    },
    "velocity_multistep8": {
        "model": "fixed_velocity_residual_mlp",
        "fixed_observed_frames": (1, 5),
        "fixed_history_size": 2,
        "multistep_horizons": list(range(1, 9)),
        "context_include_source_id": False,
    },
    "window_one": {
        "model": "fixed_window_mlp",
        "fixed_observed_frames": (0, 1, 3, 5),
        "fixed_history_size": 4,
        "multistep_horizons": [1],
        "context_include_source_id": False,
    },
    "window_multistep8": {
        "model": "fixed_window_mlp",
        "fixed_observed_frames": (0, 1, 3, 5),
        "fixed_history_size": 4,
        "multistep_horizons": list(range(1, 9)),
        "context_include_source_id": False,
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def r2(true: pd.Series, pred: pd.Series) -> float:
    y = true.to_numpy(float)
    yhat = pred.to_numpy(float)
    valid = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[valid], yhat[valid]
    denominator = np.square(y - y.mean()).sum() if len(y) else 0.0
    return float(1.0 - np.square(y - yhat).sum() / denominator) if denominator > 0 else float("nan")


def source_summary(rows: pd.DataFrame, *, metric_prefix: str = "") -> pd.DataFrame:
    summaries = []
    for (source, step), group in rows.groupby(["source", "rollout_steps"], sort=False):
        true_column = f"{metric_prefix}true_p_ratio"
        pred_column = f"{metric_prefix}pred_p_ratio"
        valid = np.isfinite(group[true_column]) & np.isfinite(group[pred_column])
        summaries.append(
            {
                "source": source,
                "rollout_steps": int(step),
                "used": int(valid.sum()),
                "total": int(len(group)),
                "p_ratio_r2": r2(group[true_column], group[pred_column]),
                "p_ratio_mae": float(
                    np.abs(group.loc[valid, true_column] - group.loc[valid, pred_column]).mean()
                ) if valid.any() else float("nan"),
                "target_variance": float(np.var(group.loc[valid, true_column])) if valid.any() else float("nan"),
            }
        )
    return pd.DataFrame(summaries)


def build_mixture(manifest: dict, selected: tuple[str, ...]) -> list[dict]:
    mixture = []
    for name in selected:
        spec = manifest["sources"][name]
        if sha256(Path(spec["path"])) != spec["sha256"]:
            raise ValueError(f"Dataset changed since split freeze: {name}")
        indices = {**spec["split_indices"], "test": []}
        mixture.append(
            {
                "name": name,
                "label": name,
                "path": spec["path"],
                "train_count": 30,
                "val_count": 20,
                "split_indices": indices,
                "append_lj_indicator": True,
                "lj_max_graph_distance": 3 if name == "lj_noisy" else None,
            }
        )
    return mixture


def build_config(
    *, variant: str, scope: str, seed: int, output: Path, mixture: list[dict]
) -> dict:
    recipe = VARIANTS[variant]
    selected = tuple(item["name"] for item in mixture)
    trained = tuple(s for s in selected if not (
        recipe.get("exclude_mixed_training", False) and s == "depablo_mixed_temp"
    ))
    if scope != "shared4" and (
        recipe["context_include_source_id"] or recipe.get("context_include_temperature", False)
    ):
        raise ValueError("Conditioning comparisons are defined only for shared4 dynamics.")
    rows_per_source = 30 * 95
    propagator = {
        "max_train_transitions_per_sim": 95,
        "max_epochs": 30,
        "patience": 6,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "hidden_size": 96,
        "objective": recipe.get("objective", "fixed_history_one_step"),
        "model": recipe["model"],
        "loss": "delta",
        "fixed_observed_frames": recipe["fixed_observed_frames"],
        "fixed_history_size": recipe["fixed_history_size"],
        "multistep_horizons": recipe["multistep_horizons"],
        "mix_sources": scope == "shared4",
        "balance_sources": scope == "shared4",
        "train_rows_per_source": rows_per_source,
        "source_loss_reduction": "equal" if scope == "shared4" else "pooled",
        "train_trajectories_per_source": {source: 30 for source in trained},
        "val_trajectories_per_source": {source: 20 for source in trained},
        "rollout_eval_sources": list(trained),
        "rollout_history_frames": 6,
        "initial_velocity": recipe.get("initial_velocity", "zero"),
        "use_static_context": True,
        "context_pool": "mean",
        "context_dim": 16,
        "context_include_source_id": recipe["context_include_source_id"],
        "context_include_temperature": recipe.get("context_include_temperature", False),
        "fixed_history_include_progress": False,
        "rollout_eval_every_epoch": True,
        "rollout_eval_interval": 2,
        "rollout_eval_horizons": [100],
        "rollout_eval_sims_per_source": 20,
        "checkpoint_metric": "val_rollout_min_source_endpoint_p_ratio_r2",
        "checkpoint_mode": "max",
        "frozen_latent_cache_dir": str(output / "frozen_latents"),
    }
    return {
        "dataset_name": "latent_matched_rollouts",
        "split_seed": 123,
        "model_seed": seed,
        "pos_dim": 2,
        "batch_graphs": 512,
        "frame_skip": 1,
        "coordinate_normalization": "position_normalization",
        "edge_mode": "compact_stored",
        "static_context_use_physical_reference": True,
        "ae_config": {
            "model": "attention",
            "latent_dim": int(os.environ.get("LSS_LATENT_DIM", "8")),
            "latent_tokens": 32,
            "hidden_size": 128,
            "target_mode": "normalized_delta",
            "node_feature_mode": "normalized_delta",
            "edge_feature_dim": 5,
            "max_train_frames_per_sim": 101,
            "max_val_frames_per_sim": 101,
            "max_epochs": 40,
            "patience": 8,
            "lr": 1e-4,
            "weight_decay": 1e-5,
        },
        "propagator_config": propagator,
        "pretrained_ae_cache_path": str(AE_PATH),
        "pretrained_ae_config_keys": [
            "autoencoder_model", "latent_dim", "latent_tokens", "hidden_size",
            "ae_target_mode", "node_feature_mode", "edge_mode",
        ],
        "pretrained_ae_require_matching_config": True,
        "pretrained_ae_require_matching_normalizers": False,
        "pretrained_ae_skip_stat_fitting": True,
        "should_rollout": True,
        "should_train_propagator": True,
        "force_train": True,
        "force_train_autoencoder": False,
        "cache_path": str(output / "bundle.pt"),
        "cache_require_matching_config": True,
        "dataset_mixture": mixture,
        "early_stop_min_delta": 1e-5,
        "p_ratio_estimator": "endpoint",
        "rollout_steps_grid": [10, 25, 50, 75, 100],
        "rollout_eval_splits": ["val"],
        "rollout_eval_max_sims_by_split": {},
        "rollout_final_eval_sims_per_source": None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--scope", choices=("shared4", *SOURCES), required=True)
    parser.add_argument("--seed", type=int, choices=(123, 456, 786), required=True)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.scope != "shared4" and args.variant != "velocity_one":
        raise ValueError("Source-specific dynamics are the matched velocity_one comparison.")
    if not AE_PATH.is_file():
        raise FileNotFoundError(AE_PATH)
    manifest = json.loads(MANIFEST_PATH.read_text())
    selected = SOURCES if args.scope == "shared4" else (args.scope,)
    mixture = build_mixture(manifest, selected)
    output = RESULTS / f"{args.variant}_{args.scope}_s{args.seed}"
    cfg = build_config(
        variant=args.variant, scope=args.scope, seed=args.seed, output=output, mixture=mixture
    )
    if args.prepare_only:
        print(json.dumps({"output": str(output), "selected": selected, "config": cfg}, indent=2))
        return
    if (output / "completed.json").is_file():
        print(f"already completed: {output}")
        return
    if output.exists():
        raise FileExistsError(f"Partial output exists; inspect before retrying: {output}")
    output.mkdir(parents=True)
    started = time.time()
    source = {
        "dataset_name": "latent_matched_rollouts",
        "source_name": args.scope,
        "label": args.scope,
        "path": mixture[0]["path"],
        "dataset_mixture": mixture,
    }
    (output / "recipe.json").write_text(
        json.dumps(
            {
                "source": source,
                "config": cfg,
                "variant_definition": VARIANTS[args.variant],
                "manifest_sha256": sha256(MANIFEST_PATH),
                "ae_sha256": sha256(AE_PATH),
                "code_root": str(CODE),
                "job_id": os.environ.get("PBS_JOBID"),
                "host": platform.node(),
                "torch": torch.__version__,
            },
            indent=2,
        ) + "\n"
    )
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    seed_everything(args.seed)
    result = run_latent_experiment(source, cfg, device=torch.device("cpu"))
    if result["test_data"]:
        raise AssertionError("Dynamics screens must not expose reserved test trajectories.")
    result["dyn_history"].to_csv(output / "dynamics_history.csv", index=False)
    rollout_rows = result["rollout_rows"].copy()
    rollout_rows.to_csv(output / "validation_rollout_rows.csv", index=False)
    source_summary(rollout_rows).to_csv(output / "validation_rollout_summary.csv", index=False)
    ae_rows = result["ae_reconstruction_rows"].copy()
    ae_rows.to_csv(output / "validation_ae_rows.csv", index=False)
    source_summary(ae_rows).to_csv(output / "validation_ae_summary.csv", index=False)
    (output / "completed.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "split": "validation",
                "seconds": time.time() - started,
                "job_id": os.environ.get("PBS_JOBID"),
            },
            indent=2,
        ) + "\n"
    )
    endpoint = pd.read_csv(output / "validation_rollout_summary.csv").query("rollout_steps == 100")
    print(endpoint.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
