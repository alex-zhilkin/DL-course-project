"""Evaluate p-ratio readouts from reconstruction-only LJ latent trajectories."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch

from lss.latent.capacity import load_experiment_bundle
from lss.latent.experiment import ground_truth_p_ratio
from lss.latent.simulation import pearson_r, r2_score
from lss.latent.training import encode_frame_latent
from lss.utils import resolve_device


EXPERIMENT = ROOT / "notebooks" / "results" / "13_lj_reconstruction_only"
OUT = ROOT / "notebooks" / "results" / "14_lj_latent_pratio"
COUNTS = (20, 200, 1000)
RIDGES = np.logspace(-5, 5, 41)


def encode_sims(result, sims, device):
    sequences = []
    with torch.no_grad():
        for index, sim in enumerate(sims):
            sequences.append(
                torch.stack(
                    [
                        encode_frame_latent(
                            result["ae"], sim, frame, pos_dim=2,
                            node_feature_mode="normalized_delta",
                            normalizers=result["normalizers"], device=device,
                        ).cpu()
                        for frame in range(len(sim))
                    ]
                )
            )
            if index == 0 or (index + 1) % 200 == 0:
                print(f"encoded {index + 1}/{len(sims)}", flush=True)
    return torch.stack(sequences).numpy()


def latent_features(sequences: np.ndarray) -> dict[str, np.ndarray]:
    time = np.linspace(0.0, 1.0, sequences.shape[1])
    centered_time = time - time.mean()
    slope = np.einsum("t,ntd->nd", centered_time, sequences) / np.sum(centered_time ** 2)
    delta_path = sequences[:, 1:] - sequences[:, :1]
    return {
        "initial_z": sequences[:, 0],
        "midpoint_z": sequences[:, sequences.shape[1] // 2],
        "final_z": sequences[:, -1],
        "endpoint_delta_z": sequences[:, -1] - sequences[:, 0],
        "latent_slope": slope,
        "full_delta_path": delta_path.reshape(len(sequences), -1),
    }


def quadratic(features: np.ndarray) -> np.ndarray:
    columns = [features]
    for i in range(features.shape[1]):
        for j in range(i, features.shape[1]):
            columns.append((features[:, i] * features[:, j])[:, None])
    return np.concatenate(columns, axis=1)


def fit_ridge(train_x, train_y, val_x, val_y):
    mean = train_x.mean(0)
    std = train_x.std(0)
    keep = std > 1e-10
    train_z = (train_x[:, keep] - mean[keep]) / std[keep]
    val_z = (val_x[:, keep] - mean[keep]) / std[keep]
    train_design = np.column_stack([np.ones(len(train_z)), train_z])
    val_design = np.column_stack([np.ones(len(val_z)), val_z])
    best = None
    for ridge in RIDGES:
        penalty = np.eye(train_design.shape[1]) * ridge
        penalty[0, 0] = 0.0
        coefficient = np.linalg.solve(
            train_design.T @ train_design + penalty,
            train_design.T @ train_y,
        )
        prediction = val_design @ coefficient
        mse = float(np.mean((prediction - val_y) ** 2))
        if best is None or mse < best[0]:
            best = (mse, float(ridge), coefficient)
    return {"mean": mean, "std": std, "keep": keep, "ridge": best[1], "coefficient": best[2]}


def predict(features, model):
    z = (features[:, model["keep"]] - model["mean"][model["keep"]]) / model["std"][model["keep"]]
    return np.column_stack([np.ones(len(z)), z]) @ model["coefficient"]


def labels(sims, cfg):
    trajectory = np.asarray([
        ground_truth_p_ratio(sim, dataset_name="lj_noisy", cfg=cfg) for sim in sims
    ])
    registry = np.asarray([float(sim[0].registry_poisson_ratio) for sim in sims])
    return {"trajectory_p_ratio": trajectory, "registry_p_ratio": registry}


def main():
    device = resolve_device("auto")
    OUT.mkdir(parents=True, exist_ok=True)
    summary_rows, prediction_parts, readout_models = [], [], {}
    for count in COUNTS:
        print(f"\n=== latent -> p-ratio, train={count} ===", flush=True)
        checkpoint = EXPERIMENT / "models" / f"lj_reconstruction_train{count}_seed20260716.pt"
        raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
        cfg = dict(raw["params"])
        result = load_experiment_bundle(checkpoint, cfg=cfg, device=device)
        cache = OUT / f"latents_train{count}.pt"
        if cache.exists():
            encoded = torch.load(cache, map_location="cpu", weights_only=False)
        else:
            encoded = {
                split: encode_sims(result, result[f"{split}_data"], device)
                for split in ("train", "val", "test")
            }
            torch.save(encoded, cache)
        split_features = {split: latent_features(encoded[split]) for split in encoded}
        split_labels = {
            split: labels(result[f"{split}_data"], cfg) for split in encoded
        }
        for target_name in ("trajectory_p_ratio", "registry_p_ratio"):
            train_y = split_labels["train"][target_name]
            val_y = split_labels["val"][target_name]
            test_y = split_labels["test"][target_name]
            for feature_name in split_features["train"]:
                for degree in (1, 2):
                    train_x = split_features["train"][feature_name]
                    val_x = split_features["val"][feature_name]
                    test_x = split_features["test"][feature_name]
                    if degree == 2:
                        train_x, val_x, test_x = map(quadratic, (train_x, val_x, test_x))
                    finite_train = np.isfinite(train_y) & np.isfinite(train_x).all(1)
                    finite_val = np.isfinite(val_y) & np.isfinite(val_x).all(1)
                    finite_test = np.isfinite(test_y) & np.isfinite(test_x).all(1)
                    model = fit_ridge(
                        train_x[finite_train], train_y[finite_train],
                        val_x[finite_val], val_y[finite_val],
                    )
                    prediction = predict(test_x[finite_test], model)
                    readout_models[
                        f"train{count}/{target_name}/{feature_name}/degree{degree}"
                    ] = model
                    row = {
                        "train_networks": count,
                        "target": target_name,
                        "feature": feature_name,
                        "degree": degree,
                        "feature_dim": train_x.shape[1],
                        "ridge": model["ridge"],
                        "test_used": int(finite_test.sum()),
                        "test_r2": r2_score(test_y[finite_test], prediction),
                        "test_pearson": pearson_r(test_y[finite_test], prediction),
                        "prediction_std": float(np.std(prediction)),
                    }
                    summary_rows.append(row)
                    prediction_parts.append(pd.DataFrame({
                        "train_networks": count, "target": target_name,
                        "feature": feature_name, "degree": degree,
                        "sim_index": np.flatnonzero(finite_test),
                        "true_p_ratio": test_y[finite_test], "pred_p_ratio": prediction,
                    }))
                    print(row, flush=True)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["target", "train_networks", "test_r2"], ascending=[True, True, False]
    )
    summary.to_csv(OUT / "summary.csv", index=False)
    pd.concat(prediction_parts, ignore_index=True).to_csv(OUT / "predictions.csv", index=False)
    torch.save(readout_models, OUT / "readout_models.pt")
    best = summary.groupby(["train_networks", "target"], as_index=False).first()
    best.to_csv(OUT / "best_by_train_count.csv", index=False)
    print("\nBest readouts\n", best.to_string(index=False))


if __name__ == "__main__":
    main()
