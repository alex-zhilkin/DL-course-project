"""Four-step-trained autoregressive latent propagator for noisy LJ."""

from __future__ import annotations

import json
import math
import sys
import time
from copy import deepcopy
from pathlib import Path

import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.quick_lj_frozen_ae_propagator_sweep import evaluate, restore_ae
from scripts.select_lj_direct_propagator_by_field import decoded_field_mse
from scripts.tune_lj_z3_delta_propagator import DeltaMLP


SEED = 657567
TARGET_STEP = 100
UNROLL = 4
TRAIN, VAL, TEST = 200, 40, 80
BASE = ROOT / "notebooks/results/08_history_aware_latent_rollout/lj_noisy"
AE_CHECKPOINT = BASE / "history_aware_ae.pt"
DATA = ROOT / "data/lj-noisy-eps0.01-sigma1.0-cutoff1.122_500sims_200frames.pt"
LATENTS = BASE / "z3_reproduction/encoded_latents_frame100.pt"
OUTPUT = BASE / "four_step_latent_propagator"


def raw_feature(
    current,
    second,
    third,
    frame,
    target_step: int = TARGET_STEP,
    persistent_anchor=None,
):
    progress = torch.full(
        (len(current), 1),
        frame / int(target_step),
        dtype=current.dtype,
        device=current.device,
    )
    pieces = [current, second, third]
    if persistent_anchor is not None:
        pieces.append(persistent_anchor)
    pieces.append(progress)
    return torch.cat(pieces, dim=-1)


def teacher_stats(
    z,
    target_step: int = TARGET_STEP,
    history_mode: str = "anchor_velocity",
    observed_frames=(3, 6, 10),
):
    observed_frames = tuple(int(frame) for frame in observed_frames)
    xs, ys = [], []
    fixed_history_modes = {"fixed_latent_frames", "fixed_latent_frames3", "fixed_velocity_residual"}
    start_frame = max(observed_frames) if history_mode in fixed_history_modes else (10 if history_mode == "fixed_spaced_latents3" else 3)
    for frame in range(start_frame, int(target_step)):
        if history_mode in {"rolling_latents3", "rolling_latents3_anchor"}:
            second, third = z[:, frame - 1], z[:, frame - 2]
        elif history_mode == "fixed_observed_latents3":
            second, third = z[:, 1], z[:, 2]
        elif history_mode == "fixed_spaced_latents3":
            second, third = z[:, 3], z[:, 6]
        elif history_mode in {"fixed_latent_frames", "fixed_latent_frames3"}:
            second, third = z[:, observed_frames[0]], z[:, observed_frames[1]]
        elif history_mode == "fixed_velocity_residual":
            second = z[:, start_frame]
            frame_gap = max(observed_frames[1] - observed_frames[0], 1)
            third = (z[:, observed_frames[1]] - z[:, observed_frames[0]]) / frame_gap
        elif history_mode == "anchor_velocity":
            second, third = z[:, 3], z[:, 3] - z[:, 2]
        else:
            raise ValueError(f"Unknown history_mode: {history_mode}")
        anchor = z[:, observed_frames[2]] if history_mode in {"fixed_latent_frames", "fixed_latent_frames3"} and len(observed_frames) == 3 else (z[:, 10] if history_mode == "fixed_spaced_latents3" else (
            z[:, 3] if history_mode in {
            "rolling_latents3_anchor", "fixed_observed_latents3"
            } else None
        ))
        xs.append(raw_feature(
            z[:, frame], second, third, frame, target_step,
            persistent_anchor=anchor,
        ))
        target_delta = z[:, frame + 1] - z[:, frame]
        if history_mode == "fixed_velocity_residual":
            target_delta = target_delta - third
        ys.append(target_delta)
    x, y = torch.cat(xs), torch.cat(ys)
    return {
        "x_mean": x.mean(0),
        "x_std": x.std(0, unbiased=False).clamp_min(1e-6),
        "y_mean": y.mean(0),
        "y_std": y.std(0, unbiased=False).clamp_min(1e-6),
    }


def rollout(
    model,
    stats,
    z,
    device,
    target_step: int = TARGET_STEP,
    history_mode: str = "anchor_velocity",
    progress_scale: int | None = None,
    observed_frames=(3, 6, 10),
):
    # The rollout horizon may change for evaluation, but the time feature must
    # retain the convention used during training.
    if progress_scale is None:
        progress_scale = int(target_step)
    observed_frames = tuple(int(frame) for frame in observed_frames)
    fixed_history_modes = {"fixed_latent_frames", "fixed_latent_frames3", "fixed_velocity_residual"}
    start_frame = max(observed_frames) if history_mode in fixed_history_modes else (10 if history_mode == "fixed_spaced_latents3" else 3)
    current = z[:, start_frame].to(device)
    if history_mode in {"rolling_latents3", "rolling_latents3_anchor"}:
        second = z[:, 2].to(device)
        third = z[:, 1].to(device)
    elif history_mode == "fixed_observed_latents3":
        second = z[:, 1].to(device)
        third = z[:, 2].to(device)
    elif history_mode == "fixed_spaced_latents3":
        second = z[:, 3].to(device)
        third = z[:, 6].to(device)
    elif history_mode in {"fixed_latent_frames", "fixed_latent_frames3"}:
        second = z[:, observed_frames[0]].to(device)
        third = z[:, observed_frames[1]].to(device)
    elif history_mode == "fixed_velocity_residual":
        second = current.clone()
        frame_gap = max(observed_frames[1] - observed_frames[0], 1)
        third = ((z[:, observed_frames[1]] - z[:, observed_frames[0]]) / frame_gap).to(device)
    elif history_mode == "anchor_velocity":
        second = current.clone()
        third = (z[:, 3] - z[:, 2]).to(device)
    else:
        raise ValueError(f"Unknown history_mode: {history_mode}")
    persistent_anchor = (
        z[:, start_frame].to(device)
        if history_mode in {"rolling_latents3_anchor", "fixed_observed_latents3", "fixed_spaced_latents3", "fixed_latent_frames3"}
        or (history_mode == "fixed_latent_frames" and len(observed_frames) == 3)
        else None
    )
    with torch.no_grad():
        for frame in range(start_frame, int(target_step)):
            x = raw_feature(
                current, second, third, frame, progress_scale,
                persistent_anchor=persistent_anchor,
            )
            delta = (
                model((x - stats["x_mean"]) / stats["x_std"]) * stats["y_std"]
                + stats["y_mean"]
            )
            if history_mode == "fixed_velocity_residual":
                delta = delta + third
            next_current = current + delta
            if history_mode in {"rolling_latents3", "rolling_latents3_anchor"}:
                third, second = second, current
            current = next_current
    return current


def fit(
    ae,
    normalizers,
    train_z,
    val_z,
    val_sims,
    seed,
    device,
    rollout_eval_every: int = 2,
    hidden_size: int = 128,
    depth: int = 3,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    unroll_steps: int = UNROLL,
    target_step: int = TARGET_STEP,
    max_epochs: int = 60,
    early_stop_patience: int = 8,
    batch_size: int = 512,
    early_stop_min_delta: float = 1e-8,
    history_mode: str = "anchor_velocity",
    ae_target_mode: str = "modular_history3",
    observed_frames=(3, 6, 10),
    checkpoint_metric: str = "terminal_latent_mse",
    rollout_eval_horizons=None,
):
    unroll_steps = int(unroll_steps)
    target_step = int(target_step)
    max_epochs = int(max_epochs)
    early_stop_patience = int(early_stop_patience)
    batch_size = int(batch_size)
    if unroll_steps < 1:
        raise ValueError("unroll_steps must be at least one.")
    if target_step < 4 or unroll_steps > target_step - 3:
        raise ValueError("unroll_steps must leave at least one valid start frame from frame 3.")
    if max_epochs < 1 or early_stop_patience < 1 or batch_size < 1:
        raise ValueError("max_epochs, early_stop_patience, and batch_size must be at least one.")
    if target_step > train_z.size(1) - 1 or target_step > val_z.size(1) - 1:
        raise ValueError("target_step exceeds the encoded latent trajectory length.")
    stats = {
        key: value.to(device)
        for key, value in teacher_stats(train_z, target_step, history_mode, observed_frames).items()
    }
    latent_scale = train_z.std((0, 1), unbiased=False).clamp_min(1e-6).to(device)
    torch.manual_seed(seed)
    if history_mode in {"fixed_latent_frames", "fixed_latent_frames3"}:
        feature_latents = 1 + len(observed_frames)
    elif history_mode == "fixed_velocity_residual":
        feature_latents = 3
    else:
        feature_latents = 4 if history_mode in {
            "rolling_latents3_anchor", "fixed_observed_latents3", "fixed_spaced_latents3"
        } else 3
    model = DeltaMLP(
        feature_latents * train_z.size(-1) + 1,
        train_z.size(-1),
        hidden_size,
        depth,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    generator = torch.Generator().manual_seed(seed)
    observed_frames = tuple(int(frame) for frame in observed_frames)
    checkpoint_metric = str(checkpoint_metric).lower()
    if checkpoint_metric not in {"terminal_latent_mse", "val_rollout_p_ratio_r2"}:
        raise ValueError(
            "checkpoint_metric must be 'terminal_latent_mse' or "
            "'val_rollout_p_ratio_r2'."
        )
    if history_mode in {"fixed_latent_frames", "fixed_latent_frames3"} and len(observed_frames) not in {2, 3}:
        raise ValueError("fixed_latent_frames expects two or three observed frame indices.")
    if history_mode == "fixed_velocity_residual" and len(observed_frames) != 2:
        raise ValueError("fixed_velocity_residual expects exactly two observed frame indices.")
    fixed_history_modes = {"fixed_latent_frames", "fixed_latent_frames3", "fixed_velocity_residual"}
    start_frame = max(observed_frames) if history_mode in fixed_history_modes else (10 if history_mode == "fixed_spaced_latents3" else 3)
    if rollout_eval_horizons is None:
        rollout_eval_horizons = (target_step,)
    rollout_eval_horizons = tuple(sorted({int(step) for step in rollout_eval_horizons}))
    if not rollout_eval_horizons or rollout_eval_horizons[-1] > target_step or rollout_eval_horizons[0] <= start_frame:
        raise ValueError("rollout_eval_horizons must lie after the observed history and at or before target_step.")
    starts_per_sim = target_step - unroll_steps - start_frame + 1
    sample_count = len(train_z) * starts_per_sim
    updates_per_epoch = (sample_count + batch_size - 1) // batch_size
    best, stale, history = None, 0, []
    print(
        f"latent propagator: train_networks={len(train_z)}, val_networks={len(val_z)}, "
        f"start_frames={start_frame}..{target_step - unroll_steps}, unroll={unroll_steps}, "
        f"windows={sample_count}, updates_per_epoch={updates_per_epoch}, "
        f"latent_dim={train_z.size(-1)}, hidden={hidden_size}, depth={depth}, "
        f"lr={learning_rate:g}, target_step={target_step}, max_epochs={max_epochs}, "
        f"patience={early_stop_patience}, batch_size={batch_size}, "
        f"history_mode={history_mode}, ae_target_mode={ae_target_mode}",
        flush=True,
    )
    for epoch in range(1, max_epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_started = time.perf_counter()
        train_loss_sum = 0.0
        model.train()
        order = torch.randperm(sample_count, generator=generator)
        for start in range(0, sample_count, batch_size):
            indices = order[start : start + batch_size]
            sim_index = torch.div(indices, starts_per_sim, rounding_mode="floor")
            frame = indices.remainder(starts_per_sim) + start_frame
            current = train_z[sim_index, frame].to(device)
            if history_mode in {"rolling_latents3", "rolling_latents3_anchor"}:
                second = train_z[sim_index, frame - 1].to(device)
                third = train_z[sim_index, frame - 2].to(device)
            elif history_mode == "fixed_observed_latents3":
                second = train_z[sim_index, 1].to(device)
                third = train_z[sim_index, 2].to(device)
            elif history_mode == "fixed_spaced_latents3":
                second = train_z[sim_index, 3].to(device)
                third = train_z[sim_index, 6].to(device)
            elif history_mode in {"fixed_latent_frames", "fixed_latent_frames3"}:
                second = train_z[sim_index, observed_frames[0]].to(device)
                third = train_z[sim_index, observed_frames[1]].to(device)
            elif history_mode == "fixed_velocity_residual":
                second = train_z[sim_index, start_frame].to(device)
                frame_gap = max(observed_frames[1] - observed_frames[0], 1)
                third = (
                    (train_z[sim_index, observed_frames[1]] - train_z[sim_index, observed_frames[0]])
                    / frame_gap
                ).to(device)
            elif history_mode == "anchor_velocity":
                second = train_z[sim_index, 3].to(device)
                third = (train_z[sim_index, 3] - train_z[sim_index, 2]).to(device)
            else:
                raise ValueError(f"Unknown history_mode: {history_mode}")
            persistent_anchor = (
                train_z[sim_index, start_frame].to(device)
                if history_mode in {"rolling_latents3_anchor", "fixed_observed_latents3", "fixed_spaced_latents3", "fixed_latent_frames3"}
                or (history_mode == "fixed_latent_frames" and len(observed_frames) == 3)
                else None
            )
            loss = 0.0
            for offset in range(unroll_steps):
                step = frame + offset
                progress = step.to(device, dtype=current.dtype).unsqueeze(-1) / target_step
                pieces = [current, second, third]
                if persistent_anchor is not None:
                    pieces.append(persistent_anchor)
                x = torch.cat([*pieces, progress], dim=-1)
                delta = (
                    model((x - stats["x_mean"]) / stats["x_std"]) * stats["y_std"]
                    + stats["y_mean"]
                )
                if history_mode == "fixed_velocity_residual":
                    delta = delta + third
                next_current = current + delta
                truth = train_z[sim_index, frame + offset + 1].to(device)
                loss = loss + (((next_current - truth) / latent_scale) ** 2).mean()
                if history_mode in {"rolling_latents3", "rolling_latents3_anchor"}:
                    third, second = second, current
                current = next_current
            loss = loss / unroll_steps
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += float(loss.detach().cpu()) * len(indices)
        train_loss = train_loss_sum / sample_count
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_seconds = time.perf_counter() - epoch_started
        model.eval()
        predicted = rollout(
            model, stats, val_z, device, target_step, history_mode,
            observed_frames=observed_frames,
        )
        terminal_latent_mse = float(
            (((predicted - val_z[:, target_step].to(device)) / latent_scale) ** 2)
            .mean()
            .cpu()
        )
        field_mse = decoded_field_mse(
            ae, normalizers, val_sims, predicted, device, target_step, ae_target_mode
        )
        rollout_interval = max(int(rollout_eval_every), 1)
        evaluate_rollout = epoch % rollout_interval == 0 or epoch == max_epochs
        rollout_metrics = None
        rollout_metrics_by_horizon = {}
        if evaluate_rollout:
            for horizon in rollout_eval_horizons:
                horizon_prediction = predicted if horizon == target_step else rollout(
                    model, stats, val_z, device, horizon, history_mode,
                    progress_scale=target_step, observed_frames=observed_frames,
                )
                rollout_metrics_by_horizon[horizon] = evaluate(
                    ae, normalizers, val_sims, horizon_prediction.cpu(), horizon,
                    device, ae_target_mode=ae_target_mode,
                )
            finite_scores = [
                float(metrics["p_ratio_r2"])
                for metrics in rollout_metrics_by_horizon.values()
                if math.isfinite(float(metrics["p_ratio_r2"]))
            ]
            rollout_metrics = {
                "p_ratio_r2": sum(finite_scores) / len(finite_scores) if finite_scores else float("nan"),
                "p_ratio_pearson": sum(float(metrics["p_ratio_pearson"]) for metrics in rollout_metrics_by_horizon.values()) / len(rollout_metrics_by_horizon),
            }
        checkpoint_evaluated = (
            checkpoint_metric != "val_rollout_p_ratio_r2" or evaluate_rollout
        )
        selection_score = (
            float(rollout_metrics["p_ratio_r2"])
            if checkpoint_metric == "val_rollout_p_ratio_r2" and rollout_metrics is not None
            else terminal_latent_mse
            if checkpoint_metric == "terminal_latent_mse"
            else float("nan")
        )
        improved = checkpoint_evaluated and math.isfinite(selection_score) and (
            best is None
            or (
                selection_score > best["selection_score"] + float(early_stop_min_delta)
                if checkpoint_metric == "val_rollout_p_ratio_r2"
                else selection_score < best["selection_score"] - float(early_stop_min_delta)
            )
        )
        if improved:
            best = {
                "loss": terminal_latent_mse,
                "field_mse": field_mse,
                "selection_score": selection_score,
                "checkpoint_metric": checkpoint_metric,
                "epoch": epoch,
                "state": deepcopy(
                    {key: value.detach().cpu() for key, value in model.state_dict().items()}
                ),
            }
            stale = 0
        elif checkpoint_evaluated:
            stale += 1
        row = {
            "epoch": epoch,
            "train_multistep_loss": train_loss,
            "epoch_seconds": epoch_seconds,
            "val_decoded_field_mse": field_mse,
            "val_terminal_latent_mse": terminal_latent_mse,
            "best_checkpoint_score": (
                best["selection_score"] if best is not None else float("nan")
            ),
            "checkpoint_metric": checkpoint_metric,
            "stale": stale,
        }
        message = (
            f"propagator {epoch:03d} train_loss={train_loss:.6g} "
            f"time={epoch_seconds:.2f}s val_terminal_latent_mse={terminal_latent_mse:.7g} "
            f"val_decoded_field_mse={field_mse:.7g} "
            f"best_{checkpoint_metric}="
            f"{best['selection_score'] if best is not None else float('nan'):.7g} "
            f"stale={stale}"
        )
        if rollout_metrics is not None:
            row.update(
                {
                    "val_rollout_p_ratio_r2": rollout_metrics["p_ratio_r2"],
                    "val_rollout_p_ratio_pearson": rollout_metrics["p_ratio_pearson"],
                }
            )
            message += (
                f" | val_rollout_p_ratio_r2={rollout_metrics['p_ratio_r2']:.4f}"
            )
            for horizon, metrics in rollout_metrics_by_horizon.items():
                row[f"val_rollout_p_ratio_r2_step{horizon}"] = metrics["p_ratio_r2"]
        history.append(row)
        print(message, flush=True)
        if stale >= early_stop_patience:
            print(
                f"propagator early stop at epoch {epoch:03d}; "
                f"best_epoch={best['epoch']:03d}",
                flush=True,
            )
            break
    if best is None:
        raise RuntimeError(
            f"No finite validation checkpoint metric was produced: {checkpoint_metric}."
        )
    model.load_state_dict(best["state"])
    best["history"] = history
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
    selected = None
    for repeat in range(5):
        model, stats, best = fit(
            ae,
            normalizers,
            train_z,
            val_z,
            val_sims,
            SEED + 900 + repeat,
            device,
        )
        val_prediction = rollout(model, stats, val_z, device).cpu()
        test_prediction = rollout(model, stats, test_z, device).cpu()
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
        if selected is None or best["loss"] < selected["val_decoded_field_mse"]:
            selected = {
                "val_decoded_field_mse": best["loss"],
                "repeat": repeat,
                "model_state_dict": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "normalization": {
                    key: value.detach().cpu() for key, value in stats.items()
                },
            }
    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "seed_results.csv", index=False)
    torch.save(
        {
            **selected,
            "configuration": {
                "input": ["z_current", "z3", "initial_latent_velocity", "progress"],
                "target": "delta_z",
                "unroll_steps": UNROLL,
                "hidden_size": 128,
                "depth": 3,
                "learning_rate": 1e-4,
                "weight_decay": 1e-5,
                "target_step": TARGET_STEP,
            },
            "ae_checkpoint": str(AE_CHECKPOINT),
        },
        OUTPUT / "best_propagator.pt",
    )
    print(
        "\n"
        + frame[
            ["val_p_ratio_r2", "test_p_ratio_r2", "test_pred_to_true_std"]
        ].agg(["mean", "std", "min", "max"]).to_string(),
        flush=True,
    )


if __name__ == "__main__":
    main()
