from __future__ import annotations

import gc
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats as scipy_stats
from torch.utils.data import DataLoader, TensorDataset

from .metrics import fit_r2, mae, rmse
from .peptide import build_split_tensors, compute_norm_stats, ids_for_mutants


def release_cuda_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except RuntimeError:
            pass


def build_holdout_split(
    run_seed: int,
    uniq_mutants,
    train_mutant_count: int,
    mutants,
    history: int,
    time_lag_steps: int,
    frames_per_traj: int | None,
    take_every_kth_frame: int,
    x_all: torch.Tensor,
    time_all: torch.Tensor,
    offsets: torch.Tensor,
    labels: torch.Tensor,
    n_feat: int,
):
    run_mutants = list(uniq_mutants)
    split_rng = np.random.default_rng(int(run_seed))
    split_rng.shuffle(run_mutants)
    train_mutants = list(run_mutants[:train_mutant_count])
    holdout_mutants = list(run_mutants[train_mutant_count:])
    train_ids = ids_for_mutants(mutants, train_mutants)
    holdout_ids = ids_for_mutants(mutants, holdout_mutants)
    x_mean, x_std, dv_mean, dv_std = compute_norm_stats(
        train_ids,
        history,
        time_lag_steps,
        frames_per_traj,
        x_all,
        time_all,
        offsets,
        n_feat,
        frame_stride=take_every_kth_frame,
    )
    train_x, train_dv, train_dv_tau, train_cls, _, train_samples_df = build_split_tensors(
        train_ids,
        history,
        time_lag_steps,
        frames_per_traj,
        x_all,
        time_all,
        offsets,
        labels,
        mutants,
        x_mean,
        x_std,
        dv_mean,
        dv_std,
        frame_stride=take_every_kth_frame,
    )
    holdout_x, holdout_dv, holdout_dv_tau, holdout_cls, _, holdout_samples_df = build_split_tensors(
        holdout_ids,
        history,
        time_lag_steps,
        frames_per_traj,
        x_all,
        time_all,
        offsets,
        labels,
        mutants,
        x_mean,
        x_std,
        dv_mean,
        dv_std,
        frame_stride=take_every_kth_frame,
    )
    return {
        "train_mutants": train_mutants,
        "holdout_mutants": holdout_mutants,
        "train_x": train_x,
        "train_dv": train_dv,
        "train_dv_tau": train_dv_tau,
        "train_cls": train_cls,
        "train_samples_df": train_samples_df,
        "holdout_x": holdout_x,
        "holdout_dv": holdout_dv,
        "holdout_dv_tau": holdout_dv_tau,
        "holdout_cls": holdout_cls,
        "holdout_samples_df": holdout_samples_df,
    }


def build_train_val_test_split(
    run_seed: int,
    uniq_mutants,
    train_mutant_count: int,
    val_mutant_count: int,
    mutants,
    history: int,
    time_lag_steps: int,
    frames_per_traj: int | None,
    take_every_kth_frame: int,
    x_all: torch.Tensor,
    time_all: torch.Tensor,
    offsets: torch.Tensor,
    labels: torch.Tensor,
    n_feat: int,
):
    run_mutants = list(uniq_mutants)
    split_rng = np.random.default_rng(int(run_seed))
    split_rng.shuffle(run_mutants)
    train_mutants = list(run_mutants[:train_mutant_count])
    val_mutants = list(run_mutants[train_mutant_count : train_mutant_count + val_mutant_count])
    test_mutants = list(run_mutants[train_mutant_count + val_mutant_count :])
    train_ids = ids_for_mutants(mutants, train_mutants)
    val_ids = ids_for_mutants(mutants, val_mutants)
    test_ids = ids_for_mutants(mutants, test_mutants)
    x_mean, x_std, dv_mean, dv_std = compute_norm_stats(
        train_ids,
        history,
        time_lag_steps,
        frames_per_traj,
        x_all,
        time_all,
        offsets,
        n_feat,
        frame_stride=take_every_kth_frame,
    )
    train_x, train_dv, train_dv_tau, train_cls, _, train_samples_df = build_split_tensors(
        train_ids,
        history,
        time_lag_steps,
        frames_per_traj,
        x_all,
        time_all,
        offsets,
        labels,
        mutants,
        x_mean,
        x_std,
        dv_mean,
        dv_std,
        frame_stride=take_every_kth_frame,
    )
    val_x, val_dv, val_dv_tau, val_cls, _, val_samples_df = build_split_tensors(
        val_ids,
        history,
        time_lag_steps,
        frames_per_traj,
        x_all,
        time_all,
        offsets,
        labels,
        mutants,
        x_mean,
        x_std,
        dv_mean,
        dv_std,
        frame_stride=take_every_kth_frame,
    )
    test_x, test_dv, test_dv_tau, test_cls, _, test_samples_df = build_split_tensors(
        test_ids,
        history,
        time_lag_steps,
        frames_per_traj,
        x_all,
        time_all,
        offsets,
        labels,
        mutants,
        x_mean,
        x_std,
        dv_mean,
        dv_std,
        frame_stride=take_every_kth_frame,
    )
    return {
        "train_mutants": train_mutants,
        "val_mutants": val_mutants,
        "test_mutants": test_mutants,
        "train_x": train_x,
        "train_dv": train_dv,
        "train_dv_tau": train_dv_tau,
        "train_cls": train_cls,
        "train_samples_df": train_samples_df,
        "val_x": val_x,
        "val_dv": val_dv,
        "val_dv_tau": val_dv_tau,
        "val_cls": val_cls,
        "val_samples_df": val_samples_df,
        "test_x": test_x,
        "test_dv": test_dv,
        "test_dv_tau": test_dv_tau,
        "test_cls": test_cls,
        "test_samples_df": test_samples_df,
    }


def metric_stats_from_train_mutants(train_mutants, target_df: pd.DataFrame, metric_name: str) -> dict[str, float]:
    sub = target_df[target_df["mutant"].astype(str).isin([str(m) for m in train_mutants])].copy()
    values = sub[metric_name].to_numpy(dtype=float)
    std = float(np.nanstd(values))
    return {"mean": float(np.nanmean(values)), "std": 1.0 if std < 1e-8 else std}


def metric_value(mutant: str, target_df: pd.DataFrame, metric_name: str) -> float:
    row = target_df[target_df["mutant"].astype(str) == str(mutant)].iloc[0]
    return float(row[metric_name])


def scaled_metric(mutant: str, target_df: pd.DataFrame, metric_name: str, metric_stats: dict[str, float]) -> float:
    value = metric_value(mutant, target_df, metric_name)
    return (value - metric_stats["mean"]) / metric_stats["std"]


def run_sim_epoch(
    model,
    loader,
    device,
    time_lag_steps: int = 0,
    time_lag_weight: float = 0.0,
    cls_weight: float = 0.0,
    opt=None,
):
    train_mode = opt is not None
    model.train(train_mode)
    sums = {"sim_loss": 0.0, "dv_mse": 0.0, "cls_bce": 0.0}
    n = 0
    for batch in loader:
        xh, y_dv, *rest = batch
        y_cls = rest[-1]
        xh = xh.to(device)
        y_dv = y_dv.to(device)
        y_cls = y_cls.to(device)
        with torch.set_grad_enabled(train_mode):
            dv_pred, cls_logit, _ = model(xh)
            loss_dv = F.mse_loss(dv_pred, y_dv)
            loss_cls = (
                F.binary_cross_entropy_with_logits(cls_logit, y_cls)
                if cls_weight > 0
                else torch.zeros((), device=loss_dv.device)
            )
            loss = loss_dv + cls_weight * loss_cls
            if train_mode:
                opt.zero_grad()
                loss.backward()
                opt.step()
        batch_n = xh.size(0)
        n += batch_n
        sums["sim_loss"] += float(loss.item()) * batch_n
        sums["dv_mse"] += float(loss_dv.item()) * batch_n
        sums["cls_bce"] += float(loss_cls.item()) * batch_n
    return {k: v / n for k, v in sums.items()}


@torch.no_grad()
def extract_cv_sequences(model, x_tensor: torch.Tensor, sample_df: pd.DataFrame, batch_size: int, device: str):
    model.eval()
    rows = []
    loader = DataLoader(TensorDataset(x_tensor), batch_size=batch_size, shuffle=False, drop_last=False)
    for (xh,) in loader:
        rows.append(model.encode_cv(xh.to(device)).detach().cpu())
    cv_all = torch.cat(rows, dim=0)
    frame_df = sample_df.reset_index(drop=True).copy()
    frame_df["cv_tensor"] = [cv_all[i] for i in range(cv_all.shape[0])]
    seqs = {}
    for mutant, grp in frame_df.groupby("mutant"):
        seqs[str(mutant)] = torch.stack(list(grp["cv_tensor"]), dim=0)
    return seqs


def build_metric_dataset(cv_sequences, metric_stats: dict[str, float], target_df: pd.DataFrame, metric_name: str):
    rows = []
    for mutant, seq in cv_sequences.items():
        rows.append(
            {
                "mutant": str(mutant),
                "cv_seq": seq.float(),
                "target_scaled": float(scaled_metric(mutant, target_df, metric_name, metric_stats)),
                "target_value": float(metric_value(mutant, target_df, metric_name)),
            }
        )
    return rows


def metric_batches(metric_dataset, batch_mutants: int, shuffle: bool):
    order = list(range(len(metric_dataset)))
    if shuffle:
        random.shuffle(order)
    for start in range(0, len(order), batch_mutants):
        yield [metric_dataset[i] for i in order[start : start + batch_mutants]]


def _corr_or_nan(fn, x, y) -> float:
    if len(x) < 2:
        return float("nan")
    return float(fn(x, y).statistic)


def _corr_pvalue_or_nan(fn, x, y) -> float:
    if len(x) < 2:
        return float("nan")
    return float(fn(x, y).pvalue)


def run_metric_head_epoch(
    head,
    metric_dataset,
    metric_stats: dict[str, float],
    metric_batch_mutants: int,
    metric_supervision_weight: float,
    device: str,
    opt=None,
):
    train_mode = opt is not None
    head.train(train_mode)
    total_loss = 0.0
    total_n = 0
    preds_scaled = []
    trues_scaled = []
    preds_value = []
    trues_value = []
    for batch in metric_batches(metric_dataset, metric_batch_mutants, shuffle=train_mode):
        losses = []
        for item in batch:
            target = torch.tensor(item["target_scaled"], dtype=torch.float32, device=device)
            pred = head(item["cv_seq"].to(device))
            losses.append((pred - target).pow(2))
            pred_scaled = float(pred.detach().cpu().item())
            preds_scaled.append(pred_scaled)
            trues_scaled.append(float(item["target_scaled"]))
            preds_value.append(pred_scaled * metric_stats["std"] + metric_stats["mean"])
            trues_value.append(float(item["target_value"]))
        loss = metric_supervision_weight * torch.stack(losses).mean()
        if train_mode:
            opt.zero_grad()
            loss.backward()
            opt.step()
        total_loss += float(loss.item()) * len(batch)
        total_n += len(batch)
    return {
        "metric_loss": total_loss / total_n,
        "scaled_rmse": rmse(trues_scaled, preds_scaled),
        "scaled_mae": mae(trues_scaled, preds_scaled),
        "tm_rmse": rmse(trues_value, preds_value),
        "tm_mae": mae(trues_value, preds_value),
        "tm_r2": fit_r2(trues_value, preds_value),
        "tm_pearson": _corr_or_nan(scipy_stats.pearsonr, trues_value, preds_value),
        "tm_spearman": _corr_or_nan(scipy_stats.spearmanr, trues_value, preds_value),
    }


@torch.no_grad()
def predict_metric_df(head, metric_dataset, metric_stats: dict[str, float], metric_name: str, device: str):
    head.eval()
    rows = []
    for item in metric_dataset:
        pred_scaled = float(head(item["cv_seq"].to(device)).cpu().item())
        pred_value = pred_scaled * metric_stats["std"] + metric_stats["mean"]
        rows.append({"mutant": item["mutant"], metric_name: item["target_value"], f"pred_{metric_name}": pred_value})
    return pd.DataFrame(rows)


def prediction_stats(pred_df: pd.DataFrame, metric_name: str) -> dict[str, float]:
    x = pred_df[metric_name].to_numpy(dtype=float)
    y = pred_df[f"pred_{metric_name}"].to_numpy(dtype=float)
    return {
        "n": int(len(pred_df)),
        "rmse": rmse(x, y),
        "mae": mae(x, y),
        "r2": fit_r2(x, y),
        "pearson": _corr_or_nan(scipy_stats.pearsonr, x, y),
        "pearson_p": _corr_pvalue_or_nan(scipy_stats.pearsonr, x, y),
        "spearman": _corr_or_nan(scipy_stats.spearmanr, x, y),
        "spearman_p": _corr_pvalue_or_nan(scipy_stats.spearmanr, x, y),
    }
