"""Refine the robust direct noisy-LJ latent propagator without p-ratio loss."""

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

from scripts.quick_lj_frozen_ae_propagator_sweep import evaluate, restore_ae
from scripts.tune_lj_z3_delta_propagator import DeltaMLP


SEED = 657567
TARGET_STEP = 100
TRAIN, VAL, TEST = 200, 40, 80
BASE = ROOT / "notebooks/results/08_history_aware_latent_rollout/lj_noisy"
AE_CHECKPOINT = BASE / "history_aware_ae.pt"
LATENTS = BASE / "z3_reproduction/encoded_latents_frame100.pt"
DATA = ROOT / "data/lj-noisy-eps0.01-sigma1.0-cutoff1.122_500sims_200frames.pt"
OUTPUT = BASE / "direct_latent_propagator_refined"


def history_feature(z, kind):
    v1 = z[:, 1] - z[:, 0]
    v2 = z[:, 2] - z[:, 1]
    v3 = z[:, 3] - z[:, 2]
    if kind == "velocity":
        return torch.cat([z[:, 3], v3], dim=-1)
    if kind == "velocity_acceleration":
        return torch.cat([z[:, 3], v3, v3 - v2], dim=-1)
    if kind == "three_velocities":
        return torch.cat([z[:, 3], v1, v2, v3], dim=-1)
    raise ValueError(kind)


def feature(z, frame, kind):
    progress = torch.full(
        (len(z), 1), frame / TARGET_STEP, dtype=z.dtype, device=z.device
    )
    return torch.cat([history_feature(z, kind), progress], dim=-1)


def target(z, frame, mode):
    offset = z[:, frame] - z[:, 3]
    if mode == "offset":
        return offset
    if mode == "trajectory_rate":
        phase = (frame - 3) / (TARGET_STEP - 3)
        return offset / phase
    raise ValueError(mode)


def table(z, variant):
    xs, ys = [], []
    for frame in range(variant["frame_start"], TARGET_STEP + 1):
        xs.append(feature(z, frame, variant["history"]))
        ys.append(target(z, frame, variant["target"]))
    return torch.cat(xs), torch.cat(ys)


def predict(model, stats, z, variant, device):
    raw = feature(z, TARGET_STEP, variant["history"]).to(device)
    with torch.no_grad():
        value = model((raw - stats["x_mean"]) / stats["x_std"])
        value = value * stats["y_std"] + stats["y_mean"]
    # Both targets equal the full offset at normalized phase 1.
    return z[:, 3].to(device) + value


def fit(train_z, val_z, variant, seed, device):
    train_x, train_y = table(train_z, variant)
    stats = {
        "x_mean": train_x.mean(0).to(device),
        "x_std": train_x.std(0, unbiased=False).clamp_min(1e-6).to(device),
        "y_mean": train_y.mean(0).to(device),
        "y_std": train_y.std(0, unbiased=False).clamp_min(1e-6).to(device),
    }
    train_x = ((train_x - stats["x_mean"].cpu()) / stats["x_std"].cpu()).to(device)
    train_y = ((train_y - stats["y_mean"].cpu()) / stats["y_std"].cpu()).to(device)
    latent_scale = train_z.std((0, 1), unbiased=False).clamp_min(1e-6).to(device)
    torch.manual_seed(seed)
    model = DeltaMLP(train_x.size(1), train_y.size(1), 128, 3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-5)
    generator = torch.Generator().manual_seed(seed)
    best, stale = None, 0
    for epoch in range(1, 101):
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
        val_prediction = predict(model, stats, val_z, variant, device)
        val_loss = float(
            (((val_prediction - val_z[:, TARGET_STEP].to(device)) / latent_scale) ** 2)
            .mean()
            .cpu()
        )
        if best is None or val_loss < best["loss"] - 1e-5:
            best = {
                "loss": val_loss,
                "epoch": epoch,
                "state": deepcopy(
                    {key: value.detach().cpu() for key, value in model.state_dict().items()}
                ),
            }
            stale = 0
        else:
            stale += 1
        if stale >= 10:
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

    variants = [
        {"name": "velocity_all_offset", "history": "velocity", "frame_start": 4, "target": "offset"},
        {"name": "velocity_late_offset", "history": "velocity", "frame_start": 20, "target": "offset"},
        {"name": "velocity_late_rate", "history": "velocity", "frame_start": 20, "target": "trajectory_rate"},
        {"name": "velocity_accel_late_offset", "history": "velocity_acceleration", "frame_start": 20, "target": "offset"},
        {"name": "three_velocities_late_offset", "history": "three_velocities", "frame_start": 20, "target": "offset"},
    ]
    rows = []
    for variant_index, variant in enumerate(variants):
        for repeat in range(3):
            model, stats, best = fit(
                train_z,
                val_z,
                variant,
                SEED + 100 * (variant_index + 1) + repeat,
                device,
            )
            val_prediction = predict(model, stats, val_z, variant, device).cpu()
            test_prediction = predict(model, stats, test_z, variant, device).cpu()
            val_metrics = evaluate(
                ae, normalizers, val_sims, val_prediction, TARGET_STEP, device
            )
            test_metrics = evaluate(
                ae, normalizers, test_sims, test_prediction, TARGET_STEP, device
            )
            row = {
                **variant,
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
    summary = (
        frame.groupby("name")
        .agg(
            test_r2_mean=("test_p_ratio_r2", "mean"),
            test_r2_std=("test_p_ratio_r2", "std"),
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
