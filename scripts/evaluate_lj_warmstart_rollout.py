"""Warm-start LJ rollout after estimating network-specific velocity from observations."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import torch

from lss.latent.capacity import load_experiment_bundle
from lss.latent.experiment import ground_truth_p_ratio, temperature_p_ratio
from lss.latent.training import decode_latent_to_graph, encode_frame_latent
from lss.latent.simulation import pearson_r, r2_score
from lss.utils import resolve_device


RESULTS = PROJECT_ROOT / "notebooks" / "results"
BASELINE = RESULTS / "08_lj_train1_vs20" / "models" / "lj_noisy_train20_seed20260716.pt"
OUTPUT = RESULTS / "09_lj_rollout_optimization"


def fit_ridge(x, y, ridge=0.1):
    mean, std = x.mean(0), x.std(0) + 1e-12
    z = (x - mean) / std
    design = np.column_stack([np.ones(len(z)), z])
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return mean, std, coef


def predict_ridge(x, fitted):
    mean, std, coef = fitted
    return np.column_stack([np.ones(len(x)), (x - mean) / std]) @ coef


def latent_slope(sequence):
    return np.polyfit(np.arange(len(sequence), dtype=float), sequence, 1)[0]


def main():
    device = resolve_device("auto")
    bundle = torch.load(BASELINE, map_location="cpu", weights_only=False)
    cfg = dict(bundle["params"])
    result = load_experiment_bundle(BASELINE, cfg=cfg, device=device)
    encoded = torch.load(OUTPUT / "encoded_latents.pt", map_location="cpu", weights_only=False)
    train_latents = encoded["train_latents"].numpy()
    warm_steps = 50
    horizons = [100, 150]

    train_x = np.stack([latent_slope(z[: warm_steps + 1]) for z in train_latents])
    train_y = np.asarray(
        [ground_truth_p_ratio(sim, dataset_name="lj_noisy", cfg=cfg) for sim in result["train_data"]]
    )
    readout = fit_ridge(train_x, train_y)

    rows = []
    for sim_idx, sim in enumerate(result["test_data"]):
        with torch.no_grad():
            observed_z = torch.stack(
                [
                    encode_frame_latent(
                        result["ae"],
                        sim,
                        frame,
                        pos_dim=2,
                        node_feature_mode="normalized_delta",
                        normalizers=result["normalizers"],
                        device=device,
                    )
                    for frame in range(warm_steps + 1)
                ]
            )
        slope = torch.as_tensor(
            latent_slope(observed_z.cpu().numpy()), dtype=observed_z.dtype, device=device
        )
        latent_pratio = float(predict_ridge(slope.cpu().numpy()[None, :], readout)[0])
        true_final_pratio = ground_truth_p_ratio(sim, dataset_name="lj_noisy", cfg=cfg)

        for horizon in horizons:
            predicted_path = [frame.clone().cpu() for frame in sim[: warm_steps + 1]]
            z_warm = observed_z[-1]
            with torch.no_grad():
                for frame in range(warm_steps + 1, horizon + 1):
                    z = z_warm + slope * float(frame - warm_steps)
                    predicted_path.append(
                        decode_latent_to_graph(
                            result["ae"],
                            sim,
                            z,
                            frame,
                            pos_dim=2,
                            ae_target_mode="normalized_delta",
                            normalizers=result["normalizers"],
                            device=device,
                        ).cpu()
                    )
            pred_graph = predicted_path[-1]
            target = sim[horizon]
            final_pos_mse = float(
                torch.mean((pred_graph.x[:, :2] - target.x[:, :2].cpu()).square())
            )
            initial_to_target_mse = float(
                torch.mean((sim[0].x[:, :2].cpu() - target.x[:, :2].cpu()).square())
            )
            decoded_pratio = temperature_p_ratio(predicted_path, cfg=cfg)
            true_horizon_pratio = ground_truth_p_ratio(
                sim, horizon, dataset_name="lj_noisy", cfg=cfg
            )
            rows.append(
                {
                    "sim_idx": sim_idx,
                    "warm_steps": warm_steps,
                    "rollout_steps": horizon,
                    "predicted_steps": horizon - warm_steps,
                    "final_pos_mse": final_pos_mse,
                    "initial_to_target_mse": initial_to_target_mse,
                    "decoded_p_ratio": decoded_pratio,
                    "true_horizon_p_ratio": true_horizon_pratio,
                    "latent_readout_p_ratio": latent_pratio,
                    "true_final_p_ratio": true_final_pratio,
                }
            )
        if sim_idx == 0 or (sim_idx + 1) % 20 == 0:
            print(f"warm-start evaluated {sim_idx + 1}/{len(result['test_data'])}", flush=True)

    raw = pd.DataFrame(rows)
    summaries = []
    for horizon, group in raw.groupby("rollout_steps"):
        movement_fraction = group["final_pos_mse"].mean() / group["initial_to_target_mse"].mean()
        summaries.append(
            {
                "warm_steps": warm_steps,
                "rollout_steps": horizon,
                "predicted_steps": horizon - warm_steps,
                "networks": len(group),
                "rollout_position_r2": float(np.clip(1 - movement_fraction, 0, 1)),
                "decoded_p_ratio_r2": r2_score(
                    group["true_horizon_p_ratio"], group["decoded_p_ratio"]
                ),
                "decoded_p_ratio_pearson": pearson_r(
                    group["true_horizon_p_ratio"], group["decoded_p_ratio"]
                ),
                "latent_readout_final_p_ratio_r2": r2_score(
                    group["true_final_p_ratio"], group["latent_readout_p_ratio"]
                ),
                "latent_readout_final_p_ratio_pearson": pearson_r(
                    group["true_final_p_ratio"], group["latent_readout_p_ratio"]
                ),
            }
        )
    summary = pd.DataFrame(summaries)
    raw.to_csv(OUTPUT / "warmstart_rollout_rows.csv", index=False)
    summary.to_csv(OUTPUT / "warmstart_rollout_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
