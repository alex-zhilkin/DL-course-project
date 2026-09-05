#!/usr/bin/env python
"""Small validation-only screen for the proven shared velocity propagator."""

from __future__ import annotations

from pathlib import Path
import sys
import os

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lss.latent.experiment import run_latent_experiment, seed_everything


SEED = 123
MODEL_KEY = "S_velocity_mean_context_screen50"
OUTPUT = PROJECT_ROOT / "notebooks" / "results" / "07b_mixed_reid_depablo_lj_context_ablation"
PRETRAINED_AE = OUTPUT / (
    "mixed4source_C_history_gated_graph_context_unbalanced_lj70_latent10_"
    "attention_stored13_unbalanced_reid20_low20_mixed20_lj70_100frames.pt"
)
TRAIN_COUNTS = {key: 5 for key in ("reid", "depablo_low_temp", "depablo_mixed_temp", "lj_noisy")}
VAL_PER_SOURCE = 5
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
    if not PRETRAINED_AE.is_file():
        raise FileNotFoundError(PRETRAINED_AE)
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
    cache = OUTPUT / f"mixed4source_{MODEL_KEY}_latent10_pretrained_ae.pt"
    cfg = {
        "dataset_name": "mixed_reid_depablo_lj",
        "split_seed": SEED,
        "model_seed": SEED,
        "device": "cpu",
        "pos_dim": 2,
        "batch_graphs": 128,
        "frame_skip": 1,
        "coordinate_normalization": "position_normalization",
        "edge_feature_schema": "physical_static_normalized_edge_changes_v3",
        "edge_mode": "stored",
        "ae_config": {
            "latent_dim": 10,
            "latent_tokens": 32,
            "hidden_size": 96,
            "model": "attention",
            "edge_feature_dim": 13,
            "target_mode": "normalized_delta",
            "node_feature_mode": "normalized_delta",
            "max_train_frames_per_sim": 51,
            "max_val_frames_per_sim": 51,
            "max_epochs": 1,
            "patience": 1,
            "lr": 2e-4,
            "weight_decay": 1e-5,
            "balance_sources": False,
            "mix_sources": True,
        },
        "propagator_config": {
            "max_train_transitions_per_sim": 30,
            "max_epochs": 8,
            "patience": 4,
            "lr": 2e-4,
            "weight_decay": 1e-4,
            "hidden_size": 64,
            "objective": "fixed_history_one_step",
            "model": "fixed_velocity_residual_mlp",
            "loss": "delta",
            "step_stride": 1,
            "fixed_observed_frames": (1, 5),
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
            "rollout_eval_interval": 2,
            "rollout_eval_horizons": [50],
            "rollout_eval_sims_per_source": 5,
            "checkpoint_metric": "val_rollout_min_source_endpoint_p_ratio_r2",
            "checkpoint_mode": "max",
        },
        "pretrained_ae_cache_path": str(PRETRAINED_AE),
        "pretrained_ae_require_matching_normalizers": False,
        "pretrained_ae_skip_stat_fitting": True,
        "rollout_steps_grid": [10, 25, 50],
        "rollout_eval_splits": ["val"],
        "rollout_eval_max_sims_by_split": {},
        "rollout_final_eval_sims_per_source": None,
        "early_stop_min_delta": 1e-5,
        "force_train": True,
        "force_train_autoencoder": False,
        "should_rollout": True,
        "should_train_propagator": True,
        "cache_path": str(cache),
        "cache_require_matching_config": True,
        "dataset_mixture": mixture,
    }
    source = {
        "dataset_name": "mixed_reid_depablo_lj",
        "source_name": "mixed_reid_depablo_lj",
        "label": "Reid + dePablo + noisy LJ",
        "path": mixture[0]["path"],
        "dataset_mixture": mixture,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    print("screen: loading source datasets and cached AE", flush=True)
    seed_everything(SEED)
    result = run_latent_experiment(source, cfg, device=torch.device("cpu"))
    rows = result["rollout_rows"].query("split == 'val' and rollout_steps == 50").copy()
    rows["source"] = rows["source"].map(
        {key: label for key, (label, _) in DATASETS.items()}
    )
    report = []
    for source_name, group in rows.groupby("source", sort=False):
        valid = group[["true_p_ratio", "pred_p_ratio"]].replace([np.inf, -np.inf], np.nan).dropna()
        report.append({"source": source_name, "n": len(valid), "p_ratio_r2": r2(valid.true_p_ratio.to_numpy(float), valid.pred_p_ratio.to_numpy(float))})
    report = pd.DataFrame(report)
    path = OUTPUT / f"{MODEL_KEY}_val_step50_p_ratio_r2.csv"
    report.to_csv(path, index=False)
    print(report.round(4).to_string(index=False))
    print(f"report: {path}")


if __name__ == "__main__":
    main()
