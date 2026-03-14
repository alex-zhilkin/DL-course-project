from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def load_target_table(tm_csv, mfpt_csv, evalue_csv):
    tm_df = pd.read_csv(tm_csv)
    mfpt_df = pd.read_csv(mfpt_csv)
    evalue_df = pd.read_csv(evalue_csv)

    tm_df["Mutant"] = tm_df["Mutant"].astype(str).str.strip()
    mfpt_df["Mutant"] = mfpt_df["Mutant"].astype(str).str.strip()
    evalue_df["Mutant"] = evalue_df["Mutant"].astype(str).str.strip()

    tm_df["Tm"] = pd.to_numeric(tm_df["Tm"].astype(str).str.strip(), errors="coerce")
    mfpt_df["mfpt"] = pd.to_numeric(mfpt_df["mfpt"], errors="coerce")
    evalue_df["hlda_evalue"] = pd.to_numeric(evalue_df["hlda_evalue"], errors="coerce")

    mfpt_wt = float(mfpt_df.loc[mfpt_df["Mutant"].str.upper() == "WT", "mfpt"].dropna().iloc[0])

    out = (
        tm_df[["Mutant", "Tm"]]
        .merge(mfpt_df[["Mutant", "mfpt"]], on="Mutant", how="inner")
        .merge(evalue_df[["Mutant", "hlda_evalue"]], on="Mutant", how="inner")
        .assign(
            mutant=lambda df: df["Mutant"],
            mfpt_ratio=lambda df: df["mfpt"] / mfpt_wt,
            log_mfpt_ratio=lambda df: np.log(df["mfpt_ratio"]),
        )
    )
    return out[["mutant", "Tm", "mfpt", "mfpt_ratio", "log_mfpt_ratio", "hlda_evalue"]]


def ids_for_mutants(mutants, mutant_names):
    wanted = set(mutant_names)
    return [i for i, m in enumerate(mutants) if m in wanted]


def traj_slice(x_all, time_all, offsets, traj_id: int):
    start = int(offsets[traj_id])
    end = int(offsets[traj_id + 1])
    return x_all[start:end], time_all[start:end]


def clip_traj(xt, frames_per_traj):
    if frames_per_traj is None:
        return xt
    return xt[:frames_per_traj]


def valid_time_range(xt, history, lag_steps, frames_per_traj):
    xt = clip_traj(xt, frames_per_traj)
    start = max(history - 1, 1)
    end = xt.shape[0] - (lag_steps if lag_steps > 0 else 1)
    return xt, start, end


def compute_norm_stats(train_ids, history, lag_steps, frames_per_traj, x_all, time_all, offsets, n_feat):
    x_sum = torch.zeros(n_feat)
    x_sq = torch.zeros(n_feat)
    x_cnt = 0
    dv_sum = torch.zeros(n_feat)
    dv_sq = torch.zeros(n_feat)
    dv_cnt = 0

    for tid in train_ids:
        xt, _ = traj_slice(x_all, time_all, offsets, tid)
        xt, start, end = valid_time_range(xt, history, lag_steps, frames_per_traj)
        x_sum += xt.sum(dim=0)
        x_sq += (xt * xt).sum(dim=0)
        x_cnt += xt.shape[0]

        vel = xt[1:] - xt[:-1]
        dv_all = vel[1:] - vel[:-1]
        dv = dv_all[start - 1 : end - 1]
        dv_sum += dv.sum(dim=0)
        dv_sq += (dv * dv).sum(dim=0)
        dv_cnt += dv.shape[0]

    x_mean = x_sum / x_cnt
    x_var = x_sq / x_cnt - x_mean * x_mean
    x_std = torch.sqrt(torch.clamp(x_var, min=1e-8))

    dv_mean = dv_sum / dv_cnt
    dv_var = dv_sq / dv_cnt - dv_mean * dv_mean
    dv_std = torch.sqrt(torch.clamp(dv_var, min=1e-8))
    return x_mean, x_std, dv_mean, dv_std


def build_split_tensors(
    traj_ids,
    history,
    lag_steps,
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
):
    x_hist_rows = []
    dv_rows = []
    dv_tau_rows = []
    cls_rows = []
    traj_rows = []
    sample_rows = []

    for tid in traj_ids:
        xt, _ = traj_slice(x_all, time_all, offsets, tid)
        xt, start, end = valid_time_range(xt, history, lag_steps, frames_per_traj)
        t_idx = torch.arange(start, end, dtype=torch.long)
        if t_idx.numel() == 0:
            continue

        vel = xt[1:] - xt[:-1]
        dv_all = vel[1:] - vel[:-1]
        dv = dv_all[t_idx - 1]

        if lag_steps > 0:
            tau_v = xt[t_idx + lag_steps] - xt[t_idx + lag_steps - 1]
            cur_v = xt[t_idx] - xt[t_idx - 1]
            dv_tau = (tau_v - cur_v) / float(lag_steps)
        else:
            dv_tau = torch.zeros_like(dv)

        hist_list = []
        for t in t_idx.tolist():
            hist_list.append(((xt[t - history + 1 : t + 1] - x_mean) / x_std).unsqueeze(0))
            sample_rows.append({"traj_id": int(tid), "t": int(t), "mutant": str(mutants[tid])})
        x_hist_rows.append(torch.cat(hist_list, dim=0))
        dv_rows.append((dv - dv_mean) / dv_std)
        dv_tau_rows.append((dv_tau - dv_mean) / dv_std)

        n = int(t_idx.numel())
        cls_value = float(labels[tid])
        cls_rows.append(torch.full((n,), cls_value, dtype=torch.float32))
        traj_rows.append(torch.full((n,), int(tid), dtype=torch.long))

    x_hist = torch.cat(x_hist_rows, dim=0)
    dv = torch.cat(dv_rows, dim=0)
    dv_tau = torch.cat(dv_tau_rows, dim=0)
    y_cls = torch.cat(cls_rows, dim=0)
    traj_tensor = torch.cat(traj_rows, dim=0)
    return x_hist, dv, dv_tau, y_cls, traj_tensor, pd.DataFrame(sample_rows)
