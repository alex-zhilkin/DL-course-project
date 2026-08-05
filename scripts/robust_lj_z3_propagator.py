"""Check whether the best frozen-AE noisy-LJ propagator is seed-robust."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.quick_lj_frozen_ae_propagator_sweep import evaluate, restore_ae
from scripts.tune_lj_z3_delta_propagator import fit_config, rollout


SEED = 657567
TARGET_STEP = 100
TRAIN, VAL, TEST = 200, 40, 80
AE_CHECKPOINT = (
    ROOT
    / "notebooks/results/08_history_aware_latent_rollout/lj_noisy"
    / "history_aware_ae.pt"
)
DATA = (
    ROOT
    / "data/lj-noisy-eps0.01-sigma1.0-cutoff1.122_500sims_200frames.pt"
)
LATENTS = (
    ROOT
    / "notebooks/results/08_history_aware_latent_rollout/lj_noisy"
    / "z3_reproduction/encoded_latents_frame100.pt"
)
OUTPUT = (
    ROOT
    / "notebooks/results/08_history_aware_latent_rollout/lj_noisy"
    / "propagator_robustness"
)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae, normalizers, params = restore_ae(AE_CHECKPOINT, device)
    all_sims = torch.load(DATA, map_location="cpu", weights_only=False)
    generator = torch.Generator().manual_seed(int(params["split_seed"]))
    order = torch.randperm(len(all_sims), generator=generator).tolist()
    val_sims = [all_sims[index] for index in order[300 : 300 + VAL]]
    test_sims = [all_sims[index] for index in order[350 : 350 + TEST]]

    z = torch.load(LATENTS, map_location="cpu", weights_only=True)
    expected = (TRAIN + VAL + TEST, TARGET_STEP + 1, int(params["latent_dim"]))
    if tuple(z.shape) != expected:
        raise ValueError(f"Latent cache has shape {tuple(z.shape)}, expected {expected}.")
    train_z, val_z, test_z = z[:TRAIN], z[TRAIN : TRAIN + VAL], z[TRAIN + VAL :]

    configurations = {
        "notebook_baseline": {
            "include_progress": True,
            "hidden_size": 64,
            "depth": 2,
            "learning_rate": 3e-4,
            "weight_decay": 1e-5,
        },
        "best_progress": {
            "include_progress": True,
            "hidden_size": 128,
            "depth": 3,
            "learning_rate": 1e-4,
            "weight_decay": 1e-5,
        },
        "best_no_progress": {
            "include_progress": False,
            "hidden_size": 256,
            "depth": 3,
            "learning_rate": 1e-4,
            "weight_decay": 0.0,
        },
    }
    rows = []
    for name, configuration in configurations.items():
        for repeat in range(5):
            model_seed = SEED + 1000 * (list(configurations).index(name) + 1) + repeat
            model, stats = fit_config(
                train_z,
                val_z,
                **configuration,
                target_step=TARGET_STEP,
                device=device,
                seed=model_seed,
            )
            common = {
                "target_step": TARGET_STEP,
                "include_progress": configuration["include_progress"],
                "x_mean": stats["x_mean"],
                "x_std": stats["x_std"],
                "y_mean": stats["y_mean"],
                "y_std": stats["y_std"],
                "device": device,
            }
            predicted_val = rollout(model, val_z, **common).cpu()
            predicted_test = rollout(model, test_z, **common).cpu()
            val_metrics = evaluate(
                ae, normalizers, val_sims, predicted_val, TARGET_STEP, device
            )
            test_metrics = evaluate(
                ae, normalizers, test_sims, predicted_test, TARGET_STEP, device
            )
            row = {
                "configuration": name,
                "repeat": repeat,
                "model_seed": model_seed,
                **configuration,
                "best_epoch": stats["best_epoch"],
                "val_terminal_latent_mse": stats["val_terminal_latent_mse"],
                "val_p_ratio_r2": val_metrics["p_ratio_r2"],
                "val_p_ratio_pearson": val_metrics["p_ratio_pearson"],
                "test_p_ratio_r2": test_metrics["p_ratio_r2"],
                "test_p_ratio_pearson": test_metrics["p_ratio_pearson"],
                "test_pred_to_true_std": test_metrics["pred_to_true_std"],
            }
            rows.append(row)
            print(json.dumps(row, indent=2), flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "seed_results.csv", index=False)
    summary = (
        frame.groupby("configuration")
        .agg(
            test_r2_mean=("test_p_ratio_r2", "mean"),
            test_r2_std=("test_p_ratio_r2", "std"),
            test_r2_min=("test_p_ratio_r2", "min"),
            test_r2_max=("test_p_ratio_r2", "max"),
            val_r2_mean=("val_p_ratio_r2", "mean"),
            latent_mse_mean=("val_terminal_latent_mse", "mean"),
            spread_mean=("test_pred_to_true_std", "mean"),
        )
        .sort_values("test_r2_mean", ascending=False)
    )
    summary.to_csv(OUTPUT / "summary.csv")
    print("\n" + summary.to_string(), flush=True)


if __name__ == "__main__":
    main()
