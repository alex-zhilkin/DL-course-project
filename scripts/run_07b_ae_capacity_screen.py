#!/usr/bin/env python
"""Validation-only shared-AE capacity screen for notebook 07b.

This intentionally trains no propagator: a rollout cannot recover information
that the autoencoder has discarded.  The screen uses an unbalanced, mixed
source training set (with additional noisy-LJ trajectories) and reports only
source-wise validation reconstruction p-ratio R² at step 50.
"""

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

from lss.latent.experiment import (
    evaluate_autoencoder_reconstruction_horizons,
    run_latent_experiment,
    seed_everything,
)


SEED = 123
MODEL_KEY = "S_ae16_unbalanced_lj30_screen50"
OUTPUT = PROJECT_ROOT / "notebooks" / "results" / "07b_mixed_reid_depablo_lj_context_ablation"
TRAIN_COUNTS = {
    "reid": 12,
    "depablo_low_temp": 12,
    "depablo_mixed_temp": 12,
    "lj_noisy": 30,
}
VAL_PER_SOURCE = 8
DATASETS = {
    "reid": ("Reid", PROJECT_ROOT / "data" / "reid_200_frames.pt"),
    "depablo_low_temp": ("dePablo low-T", PROJECT_ROOT / "data" / "depablo-near-zero-temp.pt"),
    "depablo_mixed_temp": ("dePablo mixed-T", PROJECT_ROOT / "data" / "depablo-10k-mix-temp.pt"),
    "lj_noisy": ("noisy LJ", PROJECT_ROOT / "data" / "lj-noisy-eps0.01-sigma1.0-cutoff1.122_200sims_200frames.pt"),
}


def r2(true: np.ndarray, pred: np.ndarray) -> float:
    denominator = float(np.square(true - true.mean()).sum())
    return float("nan") if denominator <= 0 else 1.0 - float(np.square(true - pred).sum()) / denominator


def main() -> None:
    mixture = [
        {
            "name": key,
            "label": label,
            "path": str(path),
            "train_count": TRAIN_COUNTS[key],
            "val_count": VAL_PER_SOURCE,
            "edge_vector_dim": 2,
        }
        for key, (label, path) in DATASETS.items()
    ]
    source = {
        "dataset_name": "mixed_reid_depablo_lj",
        "source_name": "mixed_reid_depablo_lj",
        "label": "Reid + dePablo + noisy LJ",
        "path": mixture[0]["path"],
        "dataset_mixture": mixture,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cache = OUTPUT / f"mixed4source_{MODEL_KEY}.pt"
    cfg = {
        "dataset_name": "mixed_reid_depablo_lj",
        "split_seed": SEED,
        "model_seed": SEED,
        "pos_dim": 2,
        # Attention batches scale with the total nodes and edges, not just the
        # number of trajectory frames.  This keeps the CPU screen responsive.
        "batch_graphs": 256,
        "frame_skip": 1,
        "coordinate_normalization": "position_normalization",
        "edge_feature_schema": "physical_static_normalized_edge_changes_v3",
        "edge_mode": "stored",
        "ae_config": {
            "latent_dim": 16,
            "latent_tokens": 32,
            "hidden_size": 128,
            "model": "attention",
            "edge_feature_dim": 13,
            "target_mode": "normalized_delta",
            "node_feature_mode": "normalized_delta",
            "max_train_frames_per_sim": 51,
            "max_val_frames_per_sim": 51,
            "max_epochs": 20,
            "patience": 5,
            "lr": 2e-4,
            "weight_decay": 1e-5,
            "balance_sources": False,
            "mix_sources": True,
        },
        "should_rollout": False,
        "should_train_propagator": False,
        "force_train": True,
        "force_train_autoencoder": True,
        "cache_path": str(cache),
        "cache_require_matching_config": True,
        "dataset_mixture": mixture,
        "early_stop_min_delta": 1e-5,
    }
    print("AE screen: loading datasets, then training one shared 16D AE", flush=True)
    seed_everything(SEED)
    result = run_latent_experiment(source, cfg, device=torch.device("cpu"))
    rows, _ = evaluate_autoencoder_reconstruction_horizons(
        result["ae"],
        result["val_data"],
        cfg=result["params"],
        normalizers=result["normalizers"],
        dataset=result["label"],
        split_name="val",
        rollout_steps=[50],
        device=torch.device("cpu"),
    )
    rows = rows.loc[rows["rollout_steps"] == 50].copy()
    rows["source"] = rows["source"].map({key: label for key, (label, _) in DATASETS.items()})
    report = []
    for source_name, group in rows.groupby("source", sort=False):
        valid = group[["true_p_ratio", "pred_p_ratio"]].replace([np.inf, -np.inf], np.nan).dropna()
        report.append({"source": source_name, "n": len(valid), "p_ratio_r2": r2(valid.true_p_ratio.to_numpy(float), valid.pred_p_ratio.to_numpy(float))})
    report = pd.DataFrame(report)
    path = OUTPUT / f"{MODEL_KEY}_val_step50_ae_p_ratio_r2.csv"
    report.to_csv(path, index=False)
    print(report.round(4).to_string(index=False))
    print(f"report: {path}")


if __name__ == "__main__":
    main()
