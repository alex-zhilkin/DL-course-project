"""Sweep joint acceleration propagators on the fixed notebook-08 AE features.

The AE is never trained here.  The first trial encodes every train/validation
frame and stores it under ``frozen_latent_features``; all later trials load
the same latent tensors and static reference-graph contexts.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lss.latent.experiment import run_latent_experiment, seed_everything
from lss.latent.simulation import r2_score
from lss.utils import resolve_device


AE_HASH = "fd62241d7a192d4c"
SEED = 456456
OUTPUT = ROOT / "notebooks/results/08_four_source_nash_mtl_autoencoder"
AE_CACHE = OUTPUT / "cache" / f"latent_{AE_HASH}.pt"
FEATURE_CACHE = OUTPUT / "frozen_latent_features" / f"ae_{AE_HASH}_seed_{SEED}_mean"
TRIAL_CACHE = OUTPUT / "acceleration_sweep"


def source_spec() -> dict:
    datasets = {
        "reid": ("Reid", ROOT / "data/reid_200_frames.pt", 30),
        "depablo_low_temp": (
            "de Pablo low-T", ROOT / "data/depablo-near-zero-temp.pt", 30
        ),
        "lj_noisy": (
            "noisy LJ",
            ROOT / "data/lj-noisy-eps0.01-sigma1.0-cutoff1.122_200sims_200frames.pt",
            60,
        ),
    }
    return {
        "dataset_name": "08 frozen-AE acceleration sweep",
        "source_name": "08 frozen-AE acceleration sweep",
        "label": "08 frozen-AE acceleration sweep",
        "path": str(datasets["reid"][1]),
        "dataset_mixture": [
            {
                "name": key,
                "label": label,
                "path": str(path),
                "train_count": train_count,
                "val_count": 30,
                "append_lj_indicator": True,
                "add_lj_two_hop_edges": key == "lj_noisy",
            }
            for key, (label, path, train_count) in datasets.items()
        ],
    }


AE_CONFIG = {
    "model": "attention",
    "latent_dim": 6,
    "latent_tokens": 32,
    "hidden_size": 128,
    "target_mode": "normalized_delta",
    "node_feature_mode": "normalized_delta",
    "edge_feature_dim": 5,
    "max_train_frames_per_sim": 150,
    "max_val_frames_per_sim": 150,
    "train_frame_skip": 1,
    "val_frame_skip": 1,
    "max_epochs": 50,
    "patience": 5,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "mix_sources": True,
    "balance_sources": False,
    "gradient_method": "nash_mtl",
    "nash_max_iter": 50,
    "pratio_eval_every": 1,
    "pratio_eval_steps": [10, 50, 100],
}


def trial_config(args, *, hidden_size: int, lr: float, weight_decay: float) -> dict:
    propagator_config = {
        "model": "history_mlp",
        "hidden_size": hidden_size,
        "objective": "kinematic_multistep",
        "loss": "delta",
        "step_stride": 1,
        "initial_velocity": "three_frames",
        "rollout_history_frames": 3,
        # Truncated five-step acceleration loss: no p-ratio and no BPTT.
        "multistep_horizons": [1, 2, 3, 4, 5],
        "truncated_rollout_horizon": 5,
        "max_train_transitions_per_sim": args.max_transitions,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "lr": lr,
        "weight_decay": weight_decay,
        "mix_sources": True,
        "source_loss_reduction": "pooled",
        "train_trajectories_per_source": {
            "reid": 30,
            "depablo_low_temp": 30,
            "lj_noisy": 60,
        },
        "val_trajectories_per_source": {
            "reid": 30,
            "depablo_low_temp": 30,
            "lj_noisy": 30,
        },
        # The only conditioning is the frozen AE reference-graph encoding.
        "use_static_context": True,
        "context_pool": "mean",
        "context_dim": 16,
        "context_include_temperature": False,
        "context_include_source_id": False,
        "frozen_latent_cache_dir": str(FEATURE_CACHE),
        # No p-ratio computation or use until final held-out evaluation.
        "rollout_eval_every_epoch": False,
        "checkpoint_metric": None,
        "checkpoint_mode": "min",
    }
    return {
        "ae_config": AE_CONFIG,
        "propagator_config": propagator_config,
        "dataset_name": "08 frozen-AE acceleration sweep",
        "split_seed": SEED,
        "model_seed": SEED,
        "device": str(resolve_device("auto")),
        "pos_dim": 2,
        "batch_graphs": 32,
        "frame_skip": 1,
        "coordinate_normalization": "position_normalization",
        "edge_mode": "compact_stored",
        "static_context_use_physical_reference": True,
        "rollout_steps_grid": [10, 50, 100],
        "rollout_final_eval_sims_per_source": 30,
        "should_rollout": True,
        "should_train_propagator": True,
        "pretrained_ae_cache_path": str(AE_CACHE),
        "pretrained_ae_config_keys": [
            "split_seed", "coordinate_normalization", "edge_mode", "pos_dim",
            "latent_dim", "latent_tokens", "hidden_size", "autoencoder_model",
            "edge_feature_dim", "node_feature_mode", "ae_target_mode",
            "ae_max_train_frames_per_sim", "ae_max_val_frames_per_sim",
        ],
        "pretrained_ae_require_matching_config": True,
        "pretrained_ae_require_matching_normalizers": True,
        "force_train_autoencoder": False,
        "force_train": args.force,
        "cache_require_matching_config": True,
        "cache_dir": str(TRIAL_CACHE),
        "early_stop_min_delta": 1e-5,
        # Evaluation only; this is never passed to propagator training.
        "p_ratio_estimator": "endpoint",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="retrain cached trial recipes")
    parser.add_argument("--max-epochs", type=int, default=45)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--max-transitions", type=int, default=100)
    args = parser.parse_args()
    if not AE_CACHE.is_file():
        raise FileNotFoundError(f"Frozen AE cache is missing: {AE_CACHE}")

    # Small, interpretable screen: capacity × optimizer scale × regularization.
    recipes = list(itertools.product(
        (96, 128),
        (5e-5, 1e-4),
        (1e-6, 1e-5),
    ))
    source = source_spec()
    rows = []
    for index, (hidden_size, lr, weight_decay) in enumerate(recipes, start=1):
        config = trial_config(
            args, hidden_size=hidden_size, lr=lr, weight_decay=weight_decay
        )
        print(
            f"\n[{index}/{len(recipes)}] acceleration: hidden={hidden_size}, "
            f"lr={lr:g}, wd={weight_decay:g}",
            flush=True,
        )
        seed_everything(SEED)
        result = run_latent_experiment(source, config, device=resolve_device("auto"))
        heldout = result["rollout_rows"].query(
            "split == 'test' and rollout_steps == 100"
        )
        scores = {
            name: r2_score(group.true_p_ratio, group.pred_p_ratio)
            for name, group in heldout.groupby("source")
        }
        row = {"hidden_size": hidden_size, "lr": lr, "weight_decay": weight_decay}
        row.update({f"test_step100_{source}_p_ratio_r2": value for source, value in scores.items()})
        rows.append(row)
        summary = pd.DataFrame(rows).sort_values(
            "test_step100_lj_noisy_p_ratio_r2", ascending=False
        )
        TRIAL_CACHE.mkdir(parents=True, exist_ok=True)
        summary.to_csv(TRIAL_CACHE / "sweep_summary.csv", index=False)
        print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
