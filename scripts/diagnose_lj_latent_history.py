"""Controlled diagnostic for history-aware latent rollouts on noisy LJ."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lss.latent.experiment import run_latent_experiment, seed_everything
from lss.latent.training import (
    encode_frame_latent,
    encode_reference_context,
    latent_step,
    latent_step_history,
)


DATA = ROOT / "data/lj-noisy-eps0.01-sigma1.0-cutoff1.122_200sims_200frames.pt"
OUTPUT = ROOT / "notebooks/results/08_history_aware_latent_rollout/diagnostic"
SEED = 20261114
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual = np.square(y_true - y_pred).sum()
    total = np.square(y_true - y_true.mean(axis=0, keepdims=True)).sum()
    return float(1.0 - residual / max(float(total), 1e-12))


def corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    a, b = y_true.reshape(-1), y_pred.reshape(-1)
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan


def ridge_fit(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    x_mean, y_mean = x.mean(0), y.mean(0)
    xc, yc = x - x_mean, y - y_mean
    weights = np.linalg.solve(
        xc.T @ xc + alpha * np.eye(x.shape[1], dtype=x.dtype),
        xc.T @ yc,
    )
    return weights, y_mean - x_mean @ weights


def base_config(
    name: str,
    node_mode: str,
    history_frames: int,
    *,
    standardize_latent: bool = False,
    pretrained_ae: str | None = None,
    true_history_model: bool = False,
    p_ratio_weight: float = 0.0,
) -> tuple[dict, dict]:
    mixture = [{
        "name": "lj_noisy",
        "label": "Noisy LJ",
        "path": str(DATA),
        "train_count": 20,
        "val_count": 20,
        "edge_multiplicity": 1,
    }]
    cfg = {
        "dataset_name": "lj_noisy",
        "dataset_mixture": mixture,
        "split_seed": SEED,
        "model_seed": SEED,
        "device": str(DEVICE),
        "split_stratify_temperature": False,
        "min_train_p_ratio": None,
        "pos_dim": 2,
        "batch_graphs": 4,
        "frame_skip": 1,
        "train_frame_start_order": 0,
        "edge_multiplicity": 1,
        "edge_vector_dim": 2,
        "edge_mode": "stored",
        "latent_dim": 6,
        "latent_tokens": 32,
        "hidden_size": 96,
        "autoencoder_model": "attention",
        "ae_target_mode": "normalized_delta",
        "node_feature_mode": node_mode,
        "node_feature_dim": 4 if node_mode == "normalized_delta_velocity" else 2,
        "ae_max_train_frames_per_sim": 100,
        "ae_max_epochs": 50,
        "ae_patience": 7,
        "ae_lr": 5e-5,
        "ae_weight_decay": 1e-5,
        "ae_strain_loss_weight": 0.0,
        "ae_p_ratio_loss_weight": 0.0,
        "dyn_max_train_transitions_per_sim": 100,
        "dyn_max_epochs": 50,
        "dyn_patience": 7,
        "dyn_lr": 3e-5,
        "dyn_weight_decay": 1e-4,
        "propagator_hidden_size": 96,
        "propagator_objective": "one_step",
        "propagator_model": "delta_mlp",
        "propagator_loss": "delta",
        "propagator_standardize_latent": standardize_latent,
        "propagator_use_static_context": True,
        "propagator_context_pool": "mean",
        "graph_context_dim": 16,
        "rollout_history_frames": history_frames,
        "propagator_position_loss_weight": 0.0,
        "propagator_p_ratio_loss_weight": p_ratio_weight,
        "propagator_p_ratio_minimum_driven_strain": 1e-3,
        "propagator_p_ratio_boundary_fraction": 0.10,
        "early_stop_min_delta": 1e-5,
        "rollout_steps_grid": [10, 20, 50, 100, 150, 199],
        "rollout_eval_max_sims_per_split": 30,
        "temperature_pratio_window": "full",
        "temperature_pratio_estimator": "robust",
        "temperature_pratio_min_fit_frames": 8,
        "temperature_pratio_min_driven_strain_range": 1e-3,
        "temperature_pratio_smooth_window": 5,
        "should_rollout": True,
        "should_train_propagator": True,
        "force_train": False,
        "cache_path": str(OUTPUT / f"{name}.pt"),
    }
    if pretrained_ae is not None:
        cfg["pretrained_ae_cache_path"] = str(OUTPUT / pretrained_ae)
    if true_history_model:
        cfg.update({
            "propagator_objective": "history_one_step",
            "propagator_model": "history_mlp",
            "propagator_curriculum_horizons": [1],
            "propagator_curriculum_epochs": [50],
            "initial_velocity": "three_frames",
        })
    source = {
        **cfg,
        "source_name": f"Noisy LJ {name}",
        "label": f"Noisy LJ {name}",
        "path": str(DATA),
    }
    return source, cfg


def collect_velocity_probe(result: dict, split: str, max_networks: int = 30):
    features, velocities = [], []
    sims = result[f"{split}_data"][:max_networks]
    mode = result["params"]["node_feature_mode"]
    ae = result["ae"].eval()
    with torch.no_grad():
        for sim in sims:
            ref = sim[0].x[:, :2].float()
            scale = (ref.amax(0) - ref.amin(0)).clamp_min(1e-6)
            for frame in range(1, min(100, len(sim) - 1), 5):
                z = encode_frame_latent(
                    ae, sim, frame, pos_dim=2, node_feature_mode=mode,
                    normalizers=result["normalizers"], device=DEVICE,
                )
                z_next = encode_frame_latent(
                    ae, sim, frame + 1, pos_dim=2, node_feature_mode=mode,
                    normalizers=result["normalizers"], device=DEVICE,
                )
                velocity = (
                    sim[frame].x[:, :2].float() - sim[frame - 1].x[:, :2].float()
                ) / scale.reshape(1, -1)
                z_np = z.cpu().numpy()
                ref_centered = ref - ref.mean(0, keepdim=True)
                ref_scaled = (ref_centered / scale.reshape(1, -1)).numpy()
                z_nodes = np.broadcast_to(z_np, (len(ref_scaled), len(z_np)))
                interactions = (
                    z_nodes[:, :, None] * ref_scaled[:, None, :]
                ).reshape(len(ref_scaled), -1)
                features.append(
                    np.concatenate([z_nodes, ref_scaled, interactions], axis=1)
                )
                velocities.append(velocity.numpy())
    return np.concatenate(features), np.concatenate(velocities)


def collect_latent_transitions(result: dict, split: str, max_networks: int = 30):
    rows = []
    sims = result[f"{split}_data"][:max_networks]
    mode = result["params"]["node_feature_mode"]
    ae = result["ae"].eval()
    with torch.no_grad():
        for sim in sims:
            for frame in range(1, min(100, len(sim) - 1), 5):
                z = encode_frame_latent(
                    ae, sim, frame, pos_dim=2, node_feature_mode=mode,
                    normalizers=result["normalizers"], device=DEVICE,
                )
                z_next = encode_frame_latent(
                    ae, sim, frame + 1, pos_dim=2, node_feature_mode=mode,
                    normalizers=result["normalizers"], device=DEVICE,
                )
                rows.append((z.cpu().numpy(), z_next.cpu().numpy()))
    return rows


def velocity_probe(result: dict) -> dict:
    train_z, train_v = collect_velocity_probe(result, "train", 20)
    val_z, val_v = collect_velocity_probe(result, "val", 20)
    test_z, test_v = collect_velocity_probe(result, "test", 30)
    best = None
    for alpha in np.logspace(-6, 3, 10):
        weights, intercept = ridge_fit(train_z, train_v, float(alpha))
        score = r2(val_v, val_z @ weights + intercept)
        if best is None or score > best[0]:
            best = (score, float(alpha), weights, intercept)
    _, alpha, weights, intercept = best
    prediction = test_z @ weights + intercept
    return {
        "velocity_probe_alpha": alpha,
        "velocity_probe_test_r2": r2(test_v, prediction),
        "velocity_probe_test_pearson": corr(test_v, prediction),
    }


def latent_step_probe(result: dict) -> dict:
    true_delta, predicted_delta = [], []
    mode = result["params"]["node_feature_mode"]
    ae, propagator = result["ae"].eval(), result["dyn"].eval()
    with torch.no_grad():
        for sim in result["test_data"][:30]:
            context = encode_reference_context(
                ae, sim, pos_dim=2, normalizers=result["normalizers"],
                device=DEVICE, pool_mode=result["params"]["propagator_context_pool"],
            )
            start_frame = 2 if getattr(propagator, "uses_history_state", False) else 1
            z_reference = encode_frame_latent(
                ae, sim, 0, pos_dim=2, node_feature_mode=mode,
                normalizers=result["normalizers"], device=DEVICE,
            )
            for frame in range(start_frame, min(100, len(sim) - 1), 5):
                z = encode_frame_latent(
                    ae, sim, frame, pos_dim=2, node_feature_mode=mode,
                    normalizers=result["normalizers"], device=DEVICE,
                )
                z_next = encode_frame_latent(
                    ae, sim, frame + 1, pos_dim=2, node_feature_mode=mode,
                    normalizers=result["normalizers"], device=DEVICE,
                )
                if getattr(propagator, "uses_history_state", False):
                    z_previous = encode_frame_latent(
                        ae, sim, frame - 1, pos_dim=2, node_feature_mode=mode,
                        normalizers=result["normalizers"], device=DEVICE,
                    )
                    z_previous_previous = encode_frame_latent(
                        ae, sim, frame - 2, pos_dim=2, node_feature_mode=mode,
                        normalizers=result["normalizers"], device=DEVICE,
                    )
                    prediction = latent_step_history(
                        propagator, z, z_previous, z_previous_previous,
                        z_reference, result["latent_stats"], context=context,
                    )
                else:
                    prediction = latent_step(
                        propagator, z, result["latent_stats"],
                        loss_mode="delta", context=context,
                    )
                true_delta.append((z_next - z).cpu().numpy())
                predicted_delta.append((prediction - z).cpu().numpy())
    true_delta = np.asarray(true_delta)
    predicted_delta = np.asarray(predicted_delta)
    return {
        "latent_delta_test_r2": r2(true_delta, predicted_delta),
        "latent_delta_test_pearson": corr(true_delta, predicted_delta),
        "latent_delta_pred_to_true_std": float(
            predicted_delta.std() / max(float(true_delta.std()), 1e-12)
        ),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = [
        ("displacement_only", "normalized_delta", 1, False, None, False, 0.0),
        ("displacement_velocity", "normalized_delta_velocity", 3, False, None, False, 0.0),
        (
            "displacement_velocity_standardized",
            "normalized_delta_velocity",
            3,
            True,
            "displacement_velocity.pt",
            False,
            0.0,
        ),
        (
            "three_latent_history",
            "normalized_delta_velocity",
            3,
            True,
            "displacement_velocity.pt",
            True,
            0.0,
        ),
        (
            "three_latent_history_pratio_w1",
            "normalized_delta_velocity",
            3,
            True,
            "displacement_velocity.pt",
            True,
            1.0,
        ),
        (
            "three_latent_history_pratio_w10",
            "normalized_delta_velocity",
            3,
            True,
            "displacement_velocity.pt",
            True,
            10.0,
        ),
    ]
    summaries = []
    for (
        name,
        node_mode,
        history_frames,
        standardize_latent,
        pretrained_ae,
        true_history_model,
        p_ratio_weight,
    ) in cases:
        print(f"\n===== {name} on {DEVICE} =====", flush=True)
        source, cfg = base_config(
            name,
            node_mode,
            history_frames,
            standardize_latent=standardize_latent,
            pretrained_ae=pretrained_ae,
            true_history_model=true_history_model,
            p_ratio_weight=p_ratio_weight,
        )
        seed_everything(SEED)
        result = run_latent_experiment(source, cfg, device=DEVICE)
        result["ae_reconstruction_stats"].to_csv(
            OUTPUT / f"{name}_ae_reconstruction_stats.csv", index=False
        )
        result["rollout_stats"].to_csv(
            OUTPUT / f"{name}_rollout_stats.csv", index=False
        )
        diagnostics = {**velocity_probe(result), **latent_step_probe(result)}
        test_rollout = result["rollout_stats"]
        if "split" in test_rollout:
            test_rollout = test_rollout[test_rollout["split"].eq("test")]
        for step in (100, 150, 199):
            rows = test_rollout[test_rollout["rollout_steps"].eq(step)]
            if not rows.empty:
                diagnostics[f"rollout_pratio_r2_step_{step}"] = float(rows.iloc[0]["p_ratio_r2"])
                diagnostics[f"rollout_position_r2_step_{step}"] = float(
                    rows.iloc[0]["rollout_position_r2"]
                )
        diagnostics["case"] = name
        summaries.append(diagnostics)
        print(pd.Series(diagnostics).to_string(), flush=True)
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUTPUT / "diagnostic_summary.csv", index=False)
    print("\n===== FINAL COMPARISON =====")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
