"""Select the direct latent propagator by decoded validation-field error."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from lss.latent.training import decode_latent_to_graph
from scripts.compare_lj_direct_latent_propagator import (
    feature,
    predict_terminal,
    table,
)
from scripts.quick_lj_frozen_ae_propagator_sweep import evaluate, restore_ae
from scripts.tune_lj_z3_delta_propagator import DeltaMLP


SEED = 657567
TARGET_STEP = 100
TRAIN, VAL, TEST = 200, 40, 80
BASE = ROOT / "notebooks/results/08_history_aware_latent_rollout/lj_noisy"
AE_CHECKPOINT = BASE / "history_aware_ae.pt"
DATA = ROOT / "data/lj-noisy-eps0.01-sigma1.0-cutoff1.122_500sims_200frames.pt"
LATENTS = BASE / "z3_reproduction/encoded_latents_frame100.pt"
OUTPUT = BASE / "direct_latent_propagator_field_selected"


def decoded_field_mse(
    ae,
    normalizers,
    sims,
    predicted_z,
    device,
    target_step: int = TARGET_STEP,
    ae_target_mode: str = "modular_history3",
):
    target_step = int(target_step)
    total, count = 0.0, 0
    with torch.no_grad():
        for sim, latent in zip(sims, predicted_z):
            graph = decode_latent_to_graph(
                ae,
                sim,
                latent.to(device),
                target_step,
                pos_dim=2,
                ae_target_mode=ae_target_mode,
                normalizers=normalizers,
                device=device,
            )
            reference = sim[0].x[:, :2].to(device).float()
            true = sim[target_step].x[:, :2].to(device).float()
            scale = (reference.amax(0) - reference.amin(0)).clamp_min(1e-6)
            predicted = graph.x[:, :2].to(device).float()
            error = ((predicted - true) / scale).square()
            total += float(error.sum().cpu())
            count += error.numel()
    return total / max(count, 1)


def fit(
    ae,
    normalizers,
    train_z,
    val_z,
    val_sims,
    seed,
    device,
):
    history = "z3_velocity"
    train_x, train_y = table(train_z, history)
    stats = {
        "x_mean": train_x.mean(0).to(device),
        "x_std": train_x.std(0, unbiased=False).clamp_min(1e-6).to(device),
        "y_mean": train_y.mean(0).to(device),
        "y_std": train_y.std(0, unbiased=False).clamp_min(1e-6).to(device),
    }
    train_x = ((train_x - stats["x_mean"].cpu()) / stats["x_std"].cpu()).to(device)
    train_y = ((train_y - stats["y_mean"].cpu()) / stats["y_std"].cpu()).to(device)
    torch.manual_seed(seed)
    model = DeltaMLP(train_x.size(1), train_y.size(1), 128, 3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    generator = torch.Generator().manual_seed(seed)
    best, stale = None, 0
    for epoch in range(1, 81):
        model.train()
        order = torch.randperm(len(train_x), generator=generator)
        for start in range(0, len(order), 512):
            indices = order[start : start + 512].to(device)
            loss = nn.functional.mse_loss(model(train_x[indices]), train_y[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        predicted = predict_terminal(
            model, stats, val_z, history, device
        )
        field_mse = decoded_field_mse(
            ae, normalizers, val_sims, predicted, device
        )
        if best is None or field_mse < best["loss"] - 1e-8:
            best = {
                "loss": field_mse,
                "epoch": epoch,
                "state": deepcopy(
                    {key: value.detach().cpu() for key, value in model.state_dict().items()}
                ),
            }
            stale = 0
        else:
            stale += 1
        if stale >= 8:
            break
    model.load_state_dict(best["state"])
    return model, stats, best


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ae, normalizers, params = restore_ae(AE_CHECKPOINT, device)
    all_sims = torch.load(DATA, map_location="cpu", weights_only=False)
    generator = torch.Generator().manual_seed(int(params["split_seed"]))
    order = torch.randperm(len(all_sims), generator=generator).tolist()
    val_sims = [all_sims[index] for index in order[300 : 300 + VAL]]
    test_sims = [all_sims[index] for index in order[350 : 350 + TEST]]
    z = torch.load(LATENTS, map_location="cpu", weights_only=True)
    train_z, val_z, test_z = z[:TRAIN], z[TRAIN : TRAIN + VAL], z[TRAIN + VAL :]

    rows = []
    for repeat in range(5):
        model, stats, best = fit(
            ae,
            normalizers,
            train_z,
            val_z,
            val_sims,
            SEED + 800 + repeat,
            device,
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
            "val_decoded_field_mse": best["loss"],
            "val_p_ratio_r2": val_metrics["p_ratio_r2"],
            "test_p_ratio_r2": test_metrics["p_ratio_r2"],
            "test_p_ratio_pearson": test_metrics["p_ratio_pearson"],
            "test_pred_to_true_std": test_metrics["pred_to_true_std"],
        }
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "seed_results.csv", index=False)
    print(
        "\n"
        + frame[
            ["val_p_ratio_r2", "test_p_ratio_r2", "test_pred_to_true_std"]
        ].agg(["mean", "std", "min", "max"]).to_string(),
        flush=True,
    )


if __name__ == "__main__":
    main()
