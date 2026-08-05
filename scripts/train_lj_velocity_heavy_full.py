"""Full-scale velocity-heavy AE and matched z(3) propagator comparison."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from lss.latent.experiment import run_latent_experiment, seed_everything
from scripts.compare_lj_ae_history_z3 import (
    DATA,
    SEED,
    TARGET_STEP,
    fit_propagator,
    p_ratio_metrics,
    rollout,
)
from scripts.quick_lj_frozen_ae_propagator_sweep import encode_latent_table


OUTPUT = (
    ROOT
    / "notebooks/results/08_history_aware_latent_rollout/lj_noisy"
    / "velocity_heavy_full"
)
AE_CHECKPOINT = OUTPUT / "velocity_heavy_ae.pt"
LATENT_CACHE = OUTPUT / "encoded_latents_frame100.pt"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = {
        "dataset_name": "lj_noisy",
        "split_seed": SEED,
        "model_seed": SEED,
        "device": str(device),
        "train_count": 300,
        "val_count": 50,
        "split_stratify_temperature": False,
        "min_train_p_ratio": None,
        "pos_dim": 2,
        "batch_graphs": 50,
        "frame_skip": 1,
        "train_frame_start_order": 3,
        "edge_multiplicity": 1,
        "edge_vector_dim": 8,
        "edge_mode": "stored",
        "latent_dim": 32,
        "latent_tokens": 32,
        "hidden_size": 64,
        "autoencoder_model": "attention",
        "ae_target_mode": "modular_history3",
        "node_feature_mode": "modular_history3",
        "ae_coordinate_weights": [0.25, 0.25, 1, 1, 1, 1, 1, 1],
        "ae_max_train_frames_per_sim": 200,
        "ae_max_epochs": 15,
        "ae_patience": 4,
        "ae_lr": 1e-4,
        "ae_weight_decay": 1e-5,
        "early_stop_min_delta": 1e-5,
        "should_rollout": False,
        "should_train_propagator": False,
        "cache_path": str(AE_CHECKPOINT),
        "force_train": True,
    }
    source = {
        **cfg,
        "source_name": "Noisy LJ",
        "label": "Noisy LJ | full velocity-heavy history AE",
        "path": str(DATA),
    }
    print(
        json.dumps(
            {
                "device": str(device),
                "train_networks": 300,
                "frames": 200,
                "coordinate_weights": cfg["ae_coordinate_weights"],
            },
            indent=2,
        ),
        flush=True,
    )
    seed_everything(SEED)
    result = run_latent_experiment(source, cfg, device=device)

    prop_train_sims = result["train_data"][:200]
    prop_val_sims = result["val_data"][:40]
    prop_test_sims = result["test_data"][:80]
    sims = prop_train_sims + prop_val_sims + prop_test_sims
    if LATENT_CACHE.exists():
        z = torch.load(LATENT_CACHE, map_location="cpu", weights_only=True)
    else:
        z = encode_latent_table(
            result["ae"],
            result["normalizers"],
            sims,
            max_frame=TARGET_STEP,
            batch_size=24,
            device=device,
            node_feature_mode="modular_history3",
        )
        torch.save(z, LATENT_CACHE)
    train_z, val_z, test_z = z[:200], z[200:240], z[240:]

    ceiling = p_ratio_metrics(
        result,
        prop_test_sims,
        test_z[:, TARGET_STEP],
        TARGET_STEP,
        device,
    )
    rows = []
    for use_z3 in (False, True):
        model, stats, best = fit_propagator(
            train_z,
            val_z,
            use_anchor=use_z3,
            device=device,
            seed=SEED + int(use_z3),
        )
        predicted = rollout(
            model,
            test_z,
            stats,
            use_anchor=use_z3,
            device=device,
        ).cpu()
        metrics = p_ratio_metrics(
            result,
            prop_test_sims,
            predicted,
            TARGET_STEP,
            device,
        )
        row = {
            "ae_variant": "velocity_heavy_full",
            "use_z3": use_z3,
            "ae_test_p_ratio_r2": ceiling["p_ratio_r2"],
            "ae_test_p_ratio_pearson": ceiling["p_ratio_pearson"],
            "rollout_p_ratio_r2": metrics["p_ratio_r2"],
            "rollout_p_ratio_pearson": metrics["p_ratio_pearson"],
            "rollout_pred_to_true_std": metrics["pred_to_true_std"],
            "val_terminal_latent_mse": best["loss"],
            "best_epoch": best["epoch"],
        }
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "comparison.csv", index=False)
    print(frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
