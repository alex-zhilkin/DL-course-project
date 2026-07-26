"""Evaluate Poisson ratio directly from true and rolled-out latent trajectory slopes."""

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
from lss.latent.experiment import ground_truth_p_ratio
from lss.latent.models import make_latent_propagator
from lss.latent.training import (
    LatentNormalizer,
    encode_frame_latent,
    encode_reference_context,
    latent_step,
)
from lss.latent.simulation import pearson_r, r2_score
from lss.utils import resolve_device


ROOT = PROJECT_ROOT / "notebooks" / "results"
BASELINE = ROOT / "08_lj_train1_vs20" / "models" / "lj_noisy_train20_seed20260716.pt"
OPT = ROOT / "09_lj_rollout_optimization"
HORIZONS = [20, 50, 100, 150]


def slope(sequence: np.ndarray) -> np.ndarray:
    return np.polyfit(np.arange(len(sequence), dtype=float), sequence, 1)[0]


def fit_ridge(x: np.ndarray, y: np.ndarray, ridge: float = 0.1):
    mean, std = x.mean(0), x.std(0) + 1e-12
    xz = (x - mean) / std
    design = np.column_stack([np.ones(len(xz)), xz])
    penalty = np.eye(design.shape[1]) * ridge
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return mean, std, coef


def predict_ridge(x, fitted):
    mean, std, coef = fitted
    return np.column_stack([np.ones(len(x)), (x - mean) / std]) @ coef


def load_variant(name: str, result: dict, device):
    if name == "baseline_raw":
        return result["dyn"], result["latent_stats"], 16
    path = OPT / f"{name}.pt"
    bundle = torch.load(path, map_location=device, weights_only=False)
    spec = bundle["spec"]
    model = make_latent_propagator(
        2,
        int(spec["hidden_size"]),
        model_type="delta_mlp",
        context_dim=96,
        graph_context_dim=int(spec["graph_context_dim"]),
    ).to(device)
    model.load_state_dict(bundle["state_dict"])
    stats = LatentNormalizer.from_dict(
        {key: value.to(device) for key, value in bundle["stats"].items()}
    )
    return model.eval(), stats, int(spec["graph_context_dim"])


def rollout_latents(model, stats, result, sims, horizon, device):
    sequences = []
    with torch.no_grad():
        for sim in sims:
            z = encode_frame_latent(
                result["ae"],
                sim,
                0,
                pos_dim=2,
                node_feature_mode="normalized_delta",
                normalizers=result["normalizers"],
                device=device,
            )
            context = encode_reference_context(
                result["ae"],
                sim,
                pos_dim=2,
                normalizers=result["normalizers"],
                device=device,
            )
            seq = [z.detach().cpu().numpy()]
            for _ in range(horizon):
                z = latent_step(model, z, stats, loss_mode="delta", context=context)
                seq.append(z.detach().cpu().numpy())
            sequences.append(np.stack(seq))
    return np.stack(sequences)


def main():
    device = resolve_device("auto")
    baseline_bundle = torch.load(BASELINE, map_location="cpu", weights_only=False)
    cfg = dict(baseline_bundle["params"])
    result = load_experiment_bundle(BASELINE, cfg=cfg, device=device)
    encoded = torch.load(OPT / "encoded_latents.pt", map_location="cpu", weights_only=False)
    train_latents = encoded["train_latents"].numpy()
    train_sims = result["train_data"]
    test_sims = result["test_data"][:30]
    variants = [
        "baseline_raw",
        "standardized_ctx16",
        "standardized_ctx96",
        "standardized_ctx16_multistep20",
    ]
    rows = []
    for horizon in HORIZONS:
        train_x = np.stack([slope(z[: horizon + 1]) for z in train_latents])
        train_y = np.asarray(
            [ground_truth_p_ratio(sim, horizon, dataset_name="lj_noisy", cfg=cfg) for sim in train_sims]
        )
        train_mask = np.isfinite(train_y) & np.isfinite(train_x).all(1)
        fitted = fit_ridge(train_x[train_mask], train_y[train_mask])
        test_y = np.asarray(
            [ground_truth_p_ratio(sim, horizon, dataset_name="lj_noisy", cfg=cfg) for sim in test_sims]
        )
        for name in variants:
            model, stats, _ = load_variant(name, result, device)
            predicted_sequences = rollout_latents(
                model, stats, result, test_sims, horizon, device
            )
            test_x = np.stack([slope(z) for z in predicted_sequences])
            prediction = predict_ridge(test_x, fitted)
            mask = np.isfinite(test_y) & np.isfinite(prediction)
            rows.append(
                {
                    "variant": name,
                    "rollout_steps": horizon,
                    "train_used": int(train_mask.sum()),
                    "test_used": int(mask.sum()),
                    "latent_readout_p_ratio_r2": r2_score(test_y[mask], prediction[mask]),
                    "latent_readout_p_ratio_pearson": pearson_r(test_y[mask], prediction[mask]),
                    "prediction_std": float(np.std(prediction[mask])),
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(OPT / "latent_slope_readout_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
