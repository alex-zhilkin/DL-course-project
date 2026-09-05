#!/usr/bin/env python
"""First validation-only propagator screen using the shared 16D AE cache."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lss.latent.experiment import run_latent_experiment, seed_everything

SEED = 123
MODEL_KEY = "S_ae16_velocity_mean_context_source_id_screen50"
OUTPUT = PROJECT_ROOT / "notebooks" / "results" / "07b_mixed_reid_depablo_lj_context_ablation"
AE_CACHE = OUTPUT / "mixed4source_S_ae16_unbalanced_lj30_screen50.pt"
TRAIN_COUNTS = {"reid": 12, "depablo_low_temp": 12, "depablo_mixed_temp": 12, "lj_noisy": 30}
VAL_PER_SOURCE = 8
DATASETS = {
    "reid": ("Reid", PROJECT_ROOT / "data" / "reid_200_frames.pt"),
    "depablo_low_temp": ("dePablo low-T", PROJECT_ROOT / "data" / "depablo-near-zero-temp.pt"),
    "depablo_mixed_temp": ("dePablo mixed-T", PROJECT_ROOT / "data" / "depablo-10k-mix-temp.pt"),
    "lj_noisy": ("noisy LJ", PROJECT_ROOT / "data" / "lj-noisy-eps0.01-sigma1.0-cutoff1.122_200sims_200frames.pt"),
}


def r2(true: np.ndarray, pred: np.ndarray) -> float:
    denom = float(np.square(true - true.mean()).sum())
    return float("nan") if denom <= 0 else 1.0 - float(np.square(true - pred).sum()) / denom


def main() -> None:
    if not AE_CACHE.is_file():
        raise FileNotFoundError(f"Shared AE is not ready: {AE_CACHE}")
    mixture = [
        {"name": key, "label": label, "path": str(path), "train_count": TRAIN_COUNTS[key], "val_count": VAL_PER_SOURCE, "edge_vector_dim": 2}
        for key, (label, path) in DATASETS.items()
    ]
    source = {"dataset_name": "mixed_reid_depablo_lj", "source_name": "mixed_reid_depablo_lj", "label": "Reid + dePablo + noisy LJ", "path": mixture[0]["path"], "dataset_mixture": mixture}
    cfg = {
        "dataset_name": "mixed_reid_depablo_lj", "split_seed": SEED, "model_seed": SEED,
        "pos_dim": 2, "batch_graphs": 128, "frame_skip": 1,
        "coordinate_normalization": "position_normalization",
        "edge_feature_schema": "physical_static_normalized_edge_changes_v3", "edge_mode": "stored",
        "ae_config": {"latent_dim": 16, "latent_tokens": 32, "hidden_size": 128, "model": "attention", "edge_feature_dim": 13, "target_mode": "normalized_delta", "node_feature_mode": "normalized_delta", "max_train_frames_per_sim": 51, "max_val_frames_per_sim": 51, "max_epochs": 20, "patience": 5, "lr": 2e-4, "weight_decay": 1e-5, "balance_sources": False, "mix_sources": True},
        "propagator_config": {
            "max_train_transitions_per_sim": 30, "max_epochs": 12, "patience": 4,
            "lr": 2e-4, "weight_decay": 1e-4, "hidden_size": 64,
            "objective": "fixed_history_one_step", "model": "fixed_velocity_residual_mlp", "loss": "delta",
            "fixed_observed_frames": (1, 5), "mix_sources": True, "balance_sources": False,
            "train_trajectories_per_source": TRAIN_COUNTS,
            "val_trajectories_per_source": {key: VAL_PER_SOURCE for key in DATASETS},
            "use_static_context": True, "context_pool": "mean", "context_dim": 8,
            "context_include_source_id": True, "rollout_eval_every_epoch": True,
            "rollout_eval_interval": 2, "rollout_eval_horizons": [50],
            "rollout_eval_sims_per_source": VAL_PER_SOURCE,
            "checkpoint_metric": "val_rollout_min_source_endpoint_p_ratio_r2", "checkpoint_mode": "max",
        },
        "pretrained_ae_cache_path": str(AE_CACHE),
        "pretrained_ae_require_matching_normalizers": False,
        "pretrained_ae_skip_stat_fitting": True,
        "should_rollout": True, "should_train_propagator": True,
        "force_train": True, "force_train_autoencoder": False,
        "cache_path": str(OUTPUT / f"mixed4source_{MODEL_KEY}.pt"),
        "cache_require_matching_config": True, "dataset_mixture": mixture,
        "early_stop_min_delta": 1e-5, "rollout_steps_grid": [10, 25, 50],
        "rollout_eval_splits": ["val"], "rollout_eval_max_sims_by_split": {},
        "rollout_final_eval_sims_per_source": None,
    }
    print("velocity screen: loading frozen shared AE", flush=True)
    seed_everything(SEED)
    result = run_latent_experiment(source, cfg, device=torch.device("cpu"))
    rows = result["rollout_rows"].query("split == 'val' and rollout_steps == 50").copy()
    rows["source"] = rows["source"].map({key: label for key, (label, _) in DATASETS.items()})
    report = []
    for source_name, group in rows.groupby("source", sort=False):
        group = group[["true_p_ratio", "pred_p_ratio"]].replace([np.inf, -np.inf], np.nan).dropna()
        report.append({"source": source_name, "n": len(group), "p_ratio_r2": r2(group.true_p_ratio.to_numpy(float), group.pred_p_ratio.to_numpy(float))})
    report = pd.DataFrame(report)
    path = OUTPUT / f"{MODEL_KEY}_val_step50_p_ratio_r2.csv"
    report.to_csv(path, index=False)
    print(report.round(4).to_string(index=False))
    print(f"report: {path}")


if __name__ == "__main__":
    main()
