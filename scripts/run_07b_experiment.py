#!/usr/bin/env python
"""Run the current 07b shared-rollout experiment without a notebook kernel."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lss.latent.experiment import run_latent_experiment, seed_everything


MODEL_KEY = "D_history_gated_graph_context_unbalanced_lj40"
SEED = 123
TRAIN_COUNTS = {
    "reid": 20,
    "depablo_low_temp": 20,
    "depablo_mixed_temp": 20,
    "lj_noisy": 40,
}
VAL_PER_SOURCE = 15
TRAIN_STEPS_PER_SIM = 150
LATENT_DIM = 10
OUTPUT = PROJECT_ROOT / "notebooks" / "results" / "07b_mixed_reid_depablo_lj_context_ablation"

DATASETS = {
    "reid": {"label": "Reid", "path": PROJECT_ROOT / "data" / "reid_200_frames.pt"},
    "depablo_low_temp": {"label": "dePablo low-T", "path": PROJECT_ROOT / "data" / "depablo-near-zero-temp.pt"},
    "depablo_mixed_temp": {"label": "dePablo mixed-T", "path": PROJECT_ROOT / "data" / "depablo-10k-mix-temp.pt"},
    "lj_noisy": {"label": "noisy LJ", "path": PROJECT_ROOT / "data" / "lj-noisy-eps0.01-sigma1.0-cutoff1.122_200sims_200frames.pt"},
}


def build_config(device: torch.device) -> tuple[dict, Path]:
    for spec in DATASETS.values():
        if not spec["path"].is_file():
            raise FileNotFoundError(spec["path"])
    mixture = [
        {
            "name": key,
            "label": spec["label"],
            "path": str(spec["path"]),
            "train_count": TRAIN_COUNTS[key],
            "val_count": VAL_PER_SOURCE,
            "edge_vector_dim": 2,
        }
        for key, spec in DATASETS.items()
    ]
    train_tag = "_".join(f"{key}{count}" for key, count in TRAIN_COUNTS.items())
    cache = OUTPUT / (
        f"mixed4source_{MODEL_KEY}_latent{LATENT_DIM}_attention_stored13_"
        f"unbalanced_{train_tag}_100frames.pt"
    )
    ae_config = {
        "latent_dim": LATENT_DIM,
        "latent_tokens": 32,
        "hidden_size": 96,
        "model": "attention",
        "edge_feature_dim": 13,
        "target_mode": "normalized_delta",
        "node_feature_mode": "normalized_delta",
        "max_train_frames_per_sim": TRAIN_STEPS_PER_SIM + 1,
        "max_val_frames_per_sim": TRAIN_STEPS_PER_SIM + 1,
        "max_epochs": 50,
        "patience": 5,
        "lr": 2e-4,
        "weight_decay": 1e-5,
        "balance_sources": False,
        "mix_sources": True,
    }
    propagator_config = {
        "max_train_transitions_per_sim": TRAIN_STEPS_PER_SIM,
        "max_epochs": 40,
        "patience": 5,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "hidden_size": 64,
        "objective": "fixed_history_one_step",
        "model": "fixed_window_history_gated_context_mlp",
        "loss": "delta",
        "step_stride": 1,
        "fixed_observed_frames": (0, 1, 2, 3),
        "fixed_history_size": 4,
        "fixed_history_motion_context_dim": 6,
        "multistep_horizons": [1],
        "mix_sources": True,
        "balance_sources": False,
        "train_trajectories_per_source": TRAIN_COUNTS,
        "val_trajectories_per_source": {key: VAL_PER_SOURCE for key in DATASETS},
        "use_static_context": True,
        "context_pool": "mean",
        "context_dim": 8,
        "context_include_source_id": True,
        "fixed_history_include_progress": False,
        "rollout_eval_every_epoch": True,
        "rollout_eval_interval": 5,
        "rollout_eval_horizons": [100],
        "rollout_eval_sims_per_source": VAL_PER_SOURCE,
        "checkpoint_metric": "val_rollout_min_source_endpoint_p_ratio_r2",
        "checkpoint_mode": "max",
    }
    cfg = {
        "dataset_name": "mixed_reid_depablo_lj",
        "split_seed": SEED,
        "model_seed": SEED,
        "repeat_idx": 1,
        "split_stratify_temperature": False,
        "min_train_p_ratio": None,
        "device": str(device),
        "pos_dim": 2,
        "batch_graphs": 2048,
        "frame_skip": 1,
        "train_frame_start_order": 0,
        "coordinate_normalization": "position_normalization",
        "edge_feature_schema": "physical_static_normalized_edge_changes_v3",
        "edge_mode": "stored",
        "ae_config": ae_config,
        "propagator_config": propagator_config,
        "rollout_steps_grid": [10, 25, 50, 75, 100],
        "rollout_eval_splits": ["test"],
        "rollout_eval_max_sims_by_split": {},
        "rollout_eval_source": None,
        "rollout_final_eval_sims_per_source": None,
        "early_stop_min_delta": 1e-5,
        "should_rollout": True,
        "should_train_propagator": True,
        "force_train": True,
        "force_train_autoencoder": True,
        "cache_path": str(cache),
        "cache_require_matching_config": True,
        "dataset_mixture": mixture,
    }
    return cfg, cache


def r2(true: np.ndarray, pred: np.ndarray) -> float:
    denominator = float(np.square(true - true.mean()).sum())
    return float("nan") if denominator <= 0 else 1.0 - float(np.square(true - pred).sum()) / denominator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not visible to this process.")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cfg, cache = build_config(device)
    source = {
        "dataset_name": "mixed_reid_depablo_lj",
        "source_name": "mixed_reid_depablo_lj",
        "label": "Reid + dePablo + noisy LJ",
        "path": cfg["dataset_mixture"][0]["path"],
        "dataset_mixture": cfg["dataset_mixture"],
    }
    print({"device": str(device), "cache": str(cache), "train_counts": TRAIN_COUNTS})
    seed_everything(SEED)
    result = run_latent_experiment(source, cfg, device=device)
    rows = result["rollout_rows"].query("split == 'test'").copy()
    rows["source"] = rows["source"].map({key: spec["label"] for key, spec in DATASETS.items()})
    summaries = []
    for (source_name, step), group in rows.groupby(["source", "rollout_steps"], sort=False):
        valid = group[["true_p_ratio", "pred_p_ratio"]].replace([np.inf, -np.inf], np.nan).dropna()
        summaries.append({
            "source": source_name,
            "rollout_steps": int(step),
            "n": len(valid),
            "p_ratio_r2": r2(valid.true_p_ratio.to_numpy(float), valid.pred_p_ratio.to_numpy(float)) if len(valid) > 1 else float("nan"),
        })
    summary = pd.DataFrame(summaries)
    summary_path = OUTPUT / f"{MODEL_KEY}_rollout_p_ratio_r2.csv"
    endpoint_path = OUTPUT / f"{MODEL_KEY}_endpoint_p_ratio_r2.csv"
    summary.to_csv(summary_path, index=False)
    endpoints = summary.query("rollout_steps == 100").pivot(index="source", columns="rollout_steps", values="p_ratio_r2")
    endpoints.to_csv(endpoint_path)
    print(endpoints.round(4))
    print({"summary": str(summary_path), "endpoint": str(endpoint_path)})


if __name__ == "__main__":
    main()
