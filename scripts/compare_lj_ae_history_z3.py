"""Controlled AE-history and z(3)-anchor comparison for noisy LJ."""

from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from graph_utils import calc_p_ratio_rollout_sides

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from lss.graph import clone_graph
from lss.latent.experiment import run_latent_experiment, seed_everything
from lss.latent.simulation import r2_score
from lss.latent.training import decode_latent_to_graph
from scripts.quick_lj_frozen_ae_propagator_sweep import encode_latent_table
from scripts.tune_lj_z3_delta_propagator import DeltaMLP


DATA = ROOT / "data/lj-noisy-eps0.01-sigma1.0-cutoff1.122_500sims_200frames.pt"
OUTPUT = ROOT / "notebooks/results/08_history_aware_latent_rollout/lj_noisy/history_ablation"
SEED = 657567
TRAIN, VAL, TEST = 60, 15, 30
TARGET_STEP = 100


def ae_config(variant: dict, device: str) -> dict:
    return {
        "dataset_name": "lj_noisy",
        "split_seed": SEED,
        "model_seed": SEED,
        "device": device,
        "train_count": TRAIN,
        "val_count": VAL,
        "split_stratify_temperature": False,
        "min_train_p_ratio": None,
        "pos_dim": 2,
        "batch_graphs": 30,
        "frame_skip": 1,
        "train_frame_start_order": 3,
        "edge_multiplicity": 1,
        "edge_vector_dim": 8,
        "edge_mode": "stored",
        "latent_dim": 32,
        "latent_tokens": 32,
        "hidden_size": 64,
        "autoencoder_model": "attention",
        "ae_target_mode": variant["mode"],
        "node_feature_mode": variant["mode"],
        "ae_coordinate_weights": variant.get("weights"),
        "ae_max_train_frames_per_sim": 101,
        "ae_max_epochs": 6,
        "ae_patience": 3,
        "ae_lr": 2e-4,
        "ae_weight_decay": 1e-5,
        "early_stop_min_delta": 1e-5,
        "should_rollout": False,
        "should_train_propagator": False,
        "force_train": True,
    }


def propagator_feature(current, anchor, progress, *, use_anchor):
    pieces = [current]
    if use_anchor:
        pieces.append(anchor)
    pieces.append(
        torch.full(
            (len(current), 1),
            float(progress),
            dtype=current.dtype,
            device=current.device,
        )
    )
    return torch.cat(pieces, dim=-1)


def propagator_table(z, *, use_anchor):
    features, targets = [], []
    horizon = z.size(1) - 1
    anchor = z[:, 3]
    for frame in range(3, horizon):
        features.append(
            propagator_feature(
                z[:, frame], anchor, frame / horizon, use_anchor=use_anchor
            )
        )
        targets.append(z[:, frame + 1] - z[:, frame])
    return torch.cat(features), torch.cat(targets)


def rollout(model, z, stats, *, use_anchor, device):
    current = z[:, 3].to(device)
    anchor = current.clone()
    with torch.no_grad():
        for frame in range(3, TARGET_STEP):
            raw = propagator_feature(
                current,
                anchor,
                frame / TARGET_STEP,
                use_anchor=use_anchor,
            )
            prediction = model((raw - stats["x_mean"]) / stats["x_std"])
            current = current + prediction * stats["y_std"] + stats["y_mean"]
    return current


def fit_propagator(train_z, val_z, *, use_anchor, device, seed):
    train_x, train_y = propagator_table(train_z, use_anchor=use_anchor)
    stats = {
        "x_mean": train_x.mean(0).to(device),
        "x_std": train_x.std(0, unbiased=False).clamp_min(1e-6).to(device),
        "y_mean": train_y.mean(0).to(device),
        "y_std": train_y.std(0, unbiased=False).clamp_min(1e-6).to(device),
    }
    train_x = ((train_x - stats["x_mean"].cpu()) / stats["x_std"].cpu()).to(device)
    train_y = ((train_y - stats["y_mean"].cpu()) / stats["y_std"].cpu()).to(device)
    val_scale = train_z.std((0, 1), unbiased=False).clamp_min(1e-6).to(device)

    torch.manual_seed(seed)
    model = DeltaMLP(train_x.size(1), train_y.size(1), 64, 2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    generator = torch.Generator().manual_seed(seed)
    best, stale = None, 0
    for epoch in range(1, 51):
        model.train()
        order = torch.randperm(len(train_x), generator=generator)
        for start in range(0, len(order), 512):
            idx = order[start : start + 512].to(device)
            loss = torch.nn.functional.mse_loss(model(train_x[idx]), train_y[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        pred = rollout(model, val_z, stats, use_anchor=use_anchor, device=device)
        terminal = float(
            (((pred - val_z[:, TARGET_STEP].to(device)) / val_scale).square().mean()).cpu()
        )
        if best is None or terminal < best["loss"] - 1e-5:
            best = {
                "loss": terminal,
                "epoch": epoch,
                "state": deepcopy(
                    {key: value.detach().cpu() for key, value in model.state_dict().items()}
                ),
            }
            stale = 0
        else:
            stale += 1
        if stale >= 7:
            break
    model.load_state_dict(best["state"])
    return model, stats, best


def p_ratio_metrics(ae_result, sims, predicted_z, target_step, device):
    true_values, predicted_values = [], []
    with torch.no_grad():
        for sim, z in zip(sims, predicted_z):
            graph = decode_latent_to_graph(
                ae_result["ae"],
                sim,
                z.to(device),
                target_step,
                pos_dim=2,
                ae_target_mode=ae_result["params"]["ae_target_mode"],
                normalizers=ae_result["normalizers"],
                device=device,
            )
            true_values.append(float(calc_p_ratio_rollout_sides(sim, target_step)))
            predicted_values.append(
                float(
                    calc_p_ratio_rollout_sides(
                        [clone_graph(sim[0]).cpu(), graph], -1
                    )
                )
            )
    return {
        "p_ratio_r2": r2_score(true_values, predicted_values),
        "p_ratio_pearson": float(np.corrcoef(true_values, predicted_values)[0, 1]),
        "pred_to_true_std": float(
            np.std(predicted_values) / max(np.std(true_values), 1e-12)
        ),
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variants = [
        {"name": "displacement_only", "mode": "normalized_delta"},
        {"name": "history_equal", "mode": "modular_history3"},
        {
            "name": "history_velocity_heavy",
            "mode": "modular_history3",
            "weights": [0.25, 0.25, 1, 1, 1, 1, 1, 1],
        },
    ]
    rows = []
    for variant_index, variant in enumerate(variants):
        print(f"\n=== {variant['name']} ===", flush=True)
        cfg = ae_config(variant, str(device))
        source = {
            **cfg,
            "source_name": "Noisy LJ",
            "label": f"Noisy LJ | {variant['name']}",
            "path": str(DATA),
        }
        seed_everything(SEED + variant_index)
        result = run_latent_experiment(source, cfg, device=device)
        sims = result["train_data"] + result["val_data"] + result["test_data"][:TEST]
        z = encode_latent_table(
            result["ae"],
            result["normalizers"],
            sims,
            max_frame=TARGET_STEP,
            batch_size=24,
            device=device,
            node_feature_mode=variant["mode"],
        )
        train_z = z[:TRAIN]
        val_z = z[TRAIN : TRAIN + VAL]
        test_z = z[TRAIN + VAL :]
        test_sims = result["test_data"][:TEST]
        ceiling = p_ratio_metrics(
            result, test_sims, test_z[:, TARGET_STEP], TARGET_STEP, device
        )
        for use_anchor in (False, True):
            model, stats, best = fit_propagator(
                train_z,
                val_z,
                use_anchor=use_anchor,
                device=device,
                seed=SEED + 100 * variant_index + int(use_anchor),
            )
            predicted = rollout(
                model, test_z, stats, use_anchor=use_anchor, device=device
            ).cpu()
            metrics = p_ratio_metrics(
                result, test_sims, predicted, TARGET_STEP, device
            )
            row = {
                "ae_variant": variant["name"],
                "use_z3": use_anchor,
                "ae_test_p_ratio_r2": ceiling["p_ratio_r2"],
                "rollout_p_ratio_r2": metrics["p_ratio_r2"],
                "rollout_p_ratio_pearson": metrics["p_ratio_pearson"],
                "rollout_pred_to_true_std": metrics["pred_to_true_std"],
                "val_terminal_latent_mse": best["loss"],
                "best_epoch": best["epoch"],
            }
            rows.append(row)
            print(json.dumps(row, indent=2), flush=True)
    frame = pd.DataFrame(rows).sort_values(
        "rollout_p_ratio_r2", ascending=False
    )
    frame.to_csv(OUTPUT / "comparison.csv", index=False)
    print("\n" + frame.to_string(index=False))
    print(f"saved {OUTPUT/'comparison.csv'}")


if __name__ == "__main__":
    main()
