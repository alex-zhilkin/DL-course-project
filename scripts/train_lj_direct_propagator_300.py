"""Train the selected direct noisy-LJ latent propagator on all 300 AE networks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.compare_lj_direct_latent_propagator import fit, predict_terminal
from scripts.quick_lj_frozen_ae_propagator_sweep import (
    encode_latent_table,
    evaluate,
    restore_ae,
)


SEED = 657567
TARGET_STEP = 100
TRAIN, VAL, TEST = 300, 40, 80
BASE = ROOT / "notebooks/results/08_history_aware_latent_rollout/lj_noisy"
AE_CHECKPOINT = BASE / "history_aware_ae.pt"
DATA = ROOT / "data/lj-noisy-eps0.01-sigma1.0-cutoff1.122_500sims_200frames.pt"
OUTPUT = BASE / "direct_latent_propagator_300"
LATENTS = OUTPUT / "encoded_latents_frame100.pt"


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae, normalizers, params = restore_ae(AE_CHECKPOINT, device)
    all_sims = torch.load(DATA, map_location="cpu", weights_only=False)
    generator = torch.Generator().manual_seed(int(params["split_seed"]))
    order = torch.randperm(len(all_sims), generator=generator).tolist()
    train_sims = [all_sims[index] for index in order[:TRAIN]]
    val_sims = [all_sims[index] for index in order[300 : 300 + VAL]]
    test_sims = [all_sims[index] for index in order[350 : 350 + TEST]]
    sims = train_sims + val_sims + test_sims
    if LATENTS.exists():
        z = torch.load(LATENTS, map_location="cpu", weights_only=True)
    else:
        z = encode_latent_table(
            ae,
            normalizers,
            sims,
            max_frame=TARGET_STEP,
            batch_size=24,
            device=device,
        )
        torch.save(z, LATENTS)
    train_z = z[:TRAIN]
    val_z = z[TRAIN : TRAIN + VAL]
    test_z = z[TRAIN + VAL :]

    rows = []
    for repeat in range(5):
        model, stats, best = fit(
            train_z,
            val_z,
            "z3_velocity",
            hidden_size=128,
            depth=3,
            seed=SEED + 500 + repeat,
            device=device,
        )
        val_prediction = predict_terminal(
            model, stats, val_z, "z3_velocity", device
        ).cpu()
        test_prediction = predict_terminal(
            model, stats, test_z, "z3_velocity", device
        ).cpu()
        val_metrics = evaluate(
            ae, normalizers, val_sims, val_prediction, TARGET_STEP, device
        )
        test_metrics = evaluate(
            ae, normalizers, test_sims, test_prediction, TARGET_STEP, device
        )
        row = {
            "repeat": repeat,
            "best_epoch": best["epoch"],
            "val_terminal_latent_mse": best["loss"],
            "val_p_ratio_r2": val_metrics["p_ratio_r2"],
            "test_p_ratio_r2": test_metrics["p_ratio_r2"],
            "test_p_ratio_pearson": test_metrics["p_ratio_pearson"],
            "test_pred_to_true_std": test_metrics["pred_to_true_std"],
        }
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "seed_results.csv", index=False)
    summary = frame.agg(
        {
            "test_p_ratio_r2": ["mean", "std", "min", "max"],
            "val_p_ratio_r2": ["mean", "std", "min", "max"],
            "test_pred_to_true_std": ["mean", "std", "min", "max"],
        }
    )
    summary.to_csv(OUTPUT / "summary.csv")
    print("\n" + summary.to_string(), flush=True)


if __name__ == "__main__":
    main()
