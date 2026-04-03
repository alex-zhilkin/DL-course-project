from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from graph_utils import calc_p_ratio_box, calc_p_ratio_rollout_sides

from .graph import build_graph, rollout


def fit_r2(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.size < 2 or not np.all(np.isfinite(y_true)) or np.allclose(np.std(y_true), 0.0):
        return float("nan")
    coeff = np.polyfit(y_true, y_pred, deg=1)
    y_hat = coeff[0] * y_true + coeff[1]
    return _r2(y_true, y_hat)


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 0.0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def evaluate_rollout_pratio_sides(model, sims: list, history: int, rollout_steps: int, pos_dim: int, device: str, model_inputs_cls) -> dict:
    preds: list[float] = []
    targets: list[float] = []
    pos_mse_values: list[float] = []
    rows: list[dict] = []

    for sim_idx, sim in enumerate(sims):
        target_index = history + rollout_steps
        if len(sim) <= target_index:
            continue

        input_graphs = [sim[i] for i in range(history + 1)]
        roll = rollout(model=model, input_graphs=input_graphs, num_steps=rollout_steps, history=history, pos_dim=pos_dim, device=device, model_inputs_cls=model_inputs_cls)

        pred_pr = float(calc_p_ratio_rollout_sides(roll, -1))
        target_pr = float(calc_p_ratio_rollout_sides(sim, target_index))
        if not np.isfinite(pred_pr) or not np.isfinite(target_pr):
            continue

        pred_pos = roll[-1].x[:, :pos_dim]
        target_pos = sim[target_index].x[:, :pos_dim]
        pos_mse = float(torch.nn.functional.mse_loss(pred_pos, target_pos).item())

        preds.append(pred_pr)
        targets.append(target_pr)
        pos_mse_values.append(pos_mse)
        rows.append({
            "sim_idx": sim_idx,
            "target_index": target_index,
            "pred_rollout_p_ratio": pred_pr,
            "target_rollout_sides_p_ratio": target_pr,
            "rollout_pos_mse": pos_mse,
        })

    return {
        "rollout_r2": _r2(np.asarray(targets, dtype=float), np.asarray(preds, dtype=float)),
        "rollout_pearson_r": _pearson_r(np.asarray(targets, dtype=float), np.asarray(preds, dtype=float)),
        "rollout_pos_mse": float(np.mean(pos_mse_values)) if pos_mse_values else float("nan"),
        "used": len(preds),
        "total": len(sims),
        "rows": rows,
    }


def evaluate_cv_vs_global_pratio(model, sims: list, history: int, pos_dim: int, device: str, max_steps: int | None = None) -> dict:
    cv_means_by_sim = []
    targets = []
    rows = []

    for sim_idx, sim in enumerate(sims):
        n_local = len(sim) - 1 if max_steps is None else min(int(max_steps), len(sim) - 1)
        if n_local <= history:
            continue

        cv_values = []
        for t in range(history, n_local):
            frames = [sim[i].to(device) for i in range(t - history, t + 1)]
            if t > 0:
                frames[-1].vel_state = frames[-1].x[:, :pos_dim] - sim[t - 1].to(device).x[:, :pos_dim]
            input_graph = build_graph(frames).to(device)
            cv = model.extract_cv(input_graph, is_training=False).squeeze(0).detach().cpu().numpy()
            cv_values.append(np.asarray(cv, dtype=float).reshape(-1))

        if len(cv_values) < 3:
            continue

        target_pr = float(calc_p_ratio_box(sim, -1))
        cv_mean = np.mean(np.stack(cv_values, axis=0), axis=0)
        if not np.isfinite(target_pr) or not np.all(np.isfinite(cv_mean)):
            continue

        cv_means_by_sim.append(cv_mean)
        targets.append(target_pr)
        rows.append({
            "sim_idx": sim_idx,
            "target_global_p_ratio": target_pr,
            **{f"mean_cv{i}": float(value) for i, value in enumerate(cv_mean, start=1)},
        })

    cv_abs_r = float("nan")
    fit_r2 = float("nan")
    best_cv_idx = None
    best_cv_name = None
    if len(cv_means_by_sim) >= 3:
        x_all = np.asarray(cv_means_by_sim, dtype=float)
        y = np.asarray(targets, dtype=float)
        for i in range(x_all.shape[1]):
            x = x_all[:, i]
            corr = float(np.corrcoef(x, y)[0, 1])
            slope, intercept = np.polyfit(x, y, 1)
            yhat = slope * x + intercept
            r2 = _r2(y, yhat)
            if best_cv_idx is None or r2 > fit_r2:
                best_cv_idx = i
                best_cv_name = f"cv_{i + 1}"
                cv_abs_r = abs(corr)
                fit_r2 = r2

    return {
        "cv_abs_pearson_r": cv_abs_r,
        "cv_fit_r2": fit_r2,
        "cv_used": len(cv_means_by_sim),
        "best_cv_idx": best_cv_idx,
        "best_cv_name": best_cv_name,
        "rows": rows,
    }


def write_csv(path: str | Path, rows: list[dict]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        p.write_text("")
        return
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
