from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from lss.latent.experiment import run_latent_experiment, seed_everything
from lss.utils import resolve_device


SEED = 20260806
DEVICE = resolve_device("auto")
OUTPUT = ROOT / "notebooks" / "results" / "06b_mixed_dataset_shared_latent_rollout" / "propagator_tuning"
OUTPUT.mkdir(parents=True, exist_ok=True)
PRETRAINED = ROOT / "notebooks" / "results" / "06b_mixed_dataset_shared_latent_rollout" / "shared_no_history_cv2_a632e3b47291.pt"

DATASETS = {
    "reid": ("Reid", ROOT / "data" / "reid_200_frames.pt"),
    "depablo_low_temp": ("de Pablo low-T", ROOT / "data" / "depablo-near-zero-temp.pt"),
    "depablo_mixed_temp": ("de Pablo mixed-T", ROOT / "data" / "depablo-10k-mix-temp.pt"),
}

TRIALS = [
    {"name": "small64", "hidden": 64, "lr": 1.0e-4, "weight_decay": 1.0e-4},
    {"name": "small96", "hidden": 96, "lr": 1.5e-4, "weight_decay": 5.0e-5},
    {"name": "base128", "hidden": 128, "lr": 2.0e-4, "weight_decay": 1.0e-5},
]


def run_trial(trial: dict) -> dict:
    cfg = {
        "dataset_name": "mixed_sources_shared",
        "split_seed": SEED,
        "split_stratify_temperature": False,
        "min_train_p_ratio": None,
        "device": str(DEVICE),
        "pos_dim": 2,
        "batch_graphs": 32,
        "frame_skip": 1,
        "train_frame_start_order": 0,
        "latent_dim": 2,
        "latent_tokens": 32,
        "hidden_size": 96,
        "autoencoder_model": "attention",
        "edge_feature_dim": 12,
        "ae_target_mode": "normalized_delta",
        "node_feature_mode": "normalized_delta",
        "ae_max_train_frames_per_sim": 200,
        "ae_val_frame_skip": 4,
        "ae_max_val_frames_per_sim": 200,
        "ae_max_epochs": 1,
        "ae_patience": 1,
        "ae_lr": 1e-4,
        "ae_weight_decay": 1e-5,
        "pretrained_ae_cache_path": str(PRETRAINED),
        "pretrained_ae_require_matching_normalizers": True,
        "dyn_max_train_transitions_per_sim": 100,
        "dyn_max_epochs": 45,
        "dyn_patience": 8,
        "dyn_lr": trial["lr"],
        "dyn_weight_decay": trial["weight_decay"],
        "propagator_hidden_size": trial["hidden"],
        "propagator_objective": "one_step",
        "propagator_model": "delta_mlp",
        "propagator_loss": "delta",
        "propagator_step_stride": 1,
        "propagator_use_static_context": True,
        "graph_context_dim": 16,
        "propagator_context_include_temperature": False,
        "propagator_context_pool": "mean",
        "propagator_rollout_eval_every_epoch": True,
        "propagator_rollout_eval_horizons": [100],
        "propagator_rollout_eval_sims_per_source": 5,
        "propagator_checkpoint_metric": "val_rollout_min_source_p_ratio_r2",
        "propagator_checkpoint_mode": "max",
        "temperature_pratio_window": "full",
        "rollout_steps_grid": [100],
        "rollout_eval_max_sims_per_split": 10,
        "early_stop_min_delta": 1e-5,
        "should_rollout": True,
        "should_train_propagator": True,
        "force_train": True,
        "cache_path": str(OUTPUT / f"{trial['name']}.pt"),
        "model_seed": SEED,
        "repeat_idx": 1,
    }
    mixture = [
        {"name": key, "label": label, "path": str(path), "train_count": 10, "val_count": 20}
        for key, (label, path) in DATASETS.items()
    ]
    source = {
        "dataset_name": cfg["dataset_name"],
        "source_name": "three-source mixture",
        "label": f"06b tuning {trial['name']}",
        "path": str(DATASETS["reid"][1]),
        "dataset_mixture": mixture,
        "target_mode": cfg["ae_target_mode"],
        "ae_target_mode": cfg["ae_target_mode"],
        "node_feature_mode": cfg["node_feature_mode"],
        **{
            key: cfg[key]
            for key in [
                "latent_dim", "latent_tokens", "hidden_size", "autoencoder_model",
                "edge_feature_dim", "ae_max_train_frames_per_sim", "ae_max_epochs",
                "ae_patience", "ae_lr", "ae_weight_decay",
                "dyn_max_train_transitions_per_sim", "dyn_max_epochs", "dyn_patience",
                "dyn_lr", "dyn_weight_decay", "model_seed", "repeat_idx",
                "pretrained_ae_cache_path", "pretrained_ae_require_matching_normalizers",
            ]
        },
    }
    seed_everything(SEED)
    result = run_latent_experiment(source, cfg, device=DEVICE)
    history = result["dyn_history"].copy()
    history.to_csv(OUTPUT / f"{trial['name']}_history.csv", index=False)
    metric = "val_rollout_min_source_p_ratio_r2"
    best_row = history.loc[history[metric].idxmax()]
    return {
        **trial,
        "best_epoch": int(best_row["epoch"]),
        "min_source_r2": float(best_row[metric]),
        "macro_source_r2": float(best_row["val_rollout_macro_source_p_ratio_r2"]),
        "pooled_r2": float(best_row["val_rollout_p_ratio_r2"]),
    }


def main() -> None:
    if not PRETRAINED.exists():
        raise FileNotFoundError(PRETRAINED)
    rows = []
    for trial in TRIALS:
        print(f"\n=== {trial['name']} ===", flush=True)
        rows.append(run_trial(trial))
        pd.DataFrame(rows).to_csv(OUTPUT / "summary.csv", index=False)
        print(json.dumps(rows[-1], indent=2), flush=True)
    summary = pd.DataFrame(rows).sort_values(
        ["min_source_r2", "macro_source_r2"], ascending=False
    )
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
