from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from graph_utils import calc_p_ratio_box, calc_p_ratio_rollout_sides

from .graph import build_graph, rollout


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2:
        return float("nan")
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def _pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size < 2 or y_pred.size < 2:
        return float("nan")
    if np.std(y_true) <= 1e-12 or np.std(y_pred) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def evaluate_rollout_pratio_sides(
    model,
    sims: list,
    history: int,
    rollout_steps: int,
    pos_dim: int,
    device: str,
    node_features: str,
    model_inputs_cls,
) -> dict:
    preds: list[float] = []
    targets: list[float] = []
    pos_mse_values: list[float] = []
    rows: list[dict] = []

    for sim_idx, sim in enumerate(sims):
        target_index = history + rollout_steps
        if len(sim) <= target_index:
            continue

        input_graphs = [sim[i] for i in range(history + 1)]
        roll = rollout(
            model=model,
            input_graphs=input_graphs,
            num_steps=rollout_steps,
            history=history,
            pos_dim=pos_dim,
            device=device,
            node_features=node_features,
            model_inputs_cls=model_inputs_cls,
        )

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
        rows.append(
            {
                "sim_idx": sim_idx,
                "target_index": target_index,
                "pred_rollout_p_ratio": pred_pr,
                "target_rollout_sides_p_ratio": target_pr,
                "rollout_pos_mse": pos_mse,
            }
        )

    return {
        "rollout_r2": _r2(np.asarray(targets, dtype=float), np.asarray(preds, dtype=float)),
        "rollout_pearson_r": _pearson_r(np.asarray(targets, dtype=float), np.asarray(preds, dtype=float)),
        "rollout_pos_mse": float(np.mean(pos_mse_values)) if pos_mse_values else float("nan"),
        "used": len(preds),
        "total": len(sims),
        "rows": rows,
    }


def evaluate_cv_vs_global_pratio(
    model,
    sims: list,
    history: int,
    pos_dim: int,
    device: str,
    max_steps: int,
    node_features: str,
    target_kind: str = "box",
) -> dict:
    if not hasattr(model, "extract_cv"):
        return {
            "cv_abs_pearson_r": float("nan"),
            "cv_fit_r2": float("nan"),
            "cv_used": 0,
            "rows": [],
        }

    cv2_means = []
    targets = []
    rows = []
    target_kind = str(target_kind).strip().lower()
    if target_kind not in {"box", "rollout_sides"}:
        raise ValueError(f"target_kind must be 'box' or 'rollout_sides', got {target_kind!r}")

    for sim_idx, sim in enumerate(sims):
        n_local = min(max_steps, len(sim) - 1)
        if n_local <= history:
            continue

        cv2_values = []
        for t in range(history, n_local):
            frames = [sim[i].to(device) for i in range(t - history, t + 1)]
            if t > 0:
                frames[-1].vel_state = frames[-1].x[:, :pos_dim] - sim[t - 1].to(device).x[:, :pos_dim]
            input_graph = build_graph(frames, node_features=node_features).to(device)
            cv = model.extract_cv(input_graph, is_training=False).squeeze(0).detach().cpu().numpy()
            cv2_values.append(float(cv[1] if len(cv) > 1 else cv[0]))

        if len(cv2_values) < 3:
            continue

        if target_kind == "rollout_sides":
            target_pr = float(calc_p_ratio_rollout_sides(sim, -1))
        else:
            target_pr = float(calc_p_ratio_box(sim, -1))
        cv2_mean = float(np.mean(cv2_values))
        if not np.isfinite(target_pr) or not np.isfinite(cv2_mean):
            continue

        cv2_means.append(cv2_mean)
        targets.append(target_pr)
        rows.append(
            {
                "sim_idx": sim_idx,
                "mean_cv2": cv2_mean,
                "target_global_p_ratio": target_pr,
                "cv_target_kind": target_kind,
            }
        )

    cv_abs_r = float("nan")
    fit_r2 = float("nan")
    if len(cv2_means) >= 3:
        x = np.asarray(cv2_means, dtype=float)
        y = np.asarray(targets, dtype=float)
        if np.std(x) > 1e-12 and np.std(y) > 1e-12:
            corr = float(np.corrcoef(x, y)[0, 1])
            cv_abs_r = abs(corr)
            slope, intercept = np.polyfit(x, y, 1)
            yhat = slope * x + intercept
            fit_r2 = _r2(y, yhat)

    return {
        "cv_abs_pearson_r": cv_abs_r,
        "cv_fit_r2": fit_r2,
        "cv_used": len(cv2_means),
        "cv_target_kind": target_kind,
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
