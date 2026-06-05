"""Shared helpers for latent-space simulator notebooks."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch


def filtered_frame_ids(
    sim,
    *,
    frame_skip: int = 1,
    include_last: bool = False,
    max_frames: int | None = None,
    start_frame_order: int = 0,
) -> list[int]:
    stop = len(sim) if include_last else len(sim) - 1
    step = max(1, int(frame_skip))
    frame_ids = list(range(0, max(stop, 0), step))
    if start_frame_order:
        frame_ids = frame_ids[int(start_frame_order) :]
    if max_frames is not None:
        frame_ids = frame_ids[: int(max_frames)]
    return [int(t) for t in frame_ids]


def frame_for_filtered_step(sim, filtered_step: int, *, frame_skip: int = 1) -> int:
    frame_ids = filtered_frame_ids(sim, frame_skip=frame_skip, include_last=True)
    if not frame_ids:
        return 0
    idx = min(int(filtered_step), len(frame_ids) - 1)
    return int(frame_ids[idx])


def make_frame_index(
    sims,
    *,
    frame_skip: int = 1,
    max_frames_per_sim: int | None = None,
    include_last: bool = False,
    start_frame_order: int = 0,
) -> list[tuple[int, int]]:
    rows = []
    for sim_idx, sim in enumerate(sims):
        frame_ids = filtered_frame_ids(
            sim,
            frame_skip=frame_skip,
            include_last=include_last,
            max_frames=max_frames_per_sim,
            start_frame_order=start_frame_order,
        )
        rows.extend((sim_idx, int(t)) for t in frame_ids)
    return rows


def make_transition_index(
    sims,
    *,
    frame_skip: int = 1,
    max_frames_per_sim: int | None = None,
) -> list[tuple[int, int, int]]:
    rows = []
    for sim_idx, sim in enumerate(sims):
        frame_ids = filtered_frame_ids(sim, frame_skip=frame_skip, include_last=True)
        pairs = list(zip(frame_ids[:-1], frame_ids[1:]))
        if max_frames_per_sim is not None:
            pairs = pairs[: int(max_frames_per_sim)]
        rows.extend((sim_idx, int(t0), int(t1)) for t0, t1 in pairs)
    return rows


def iter_batches(rows, batch_graphs: int, *, shuffle: bool = True):
    rows = list(rows)
    if shuffle:
        random.shuffle(rows)
    for i in range(0, len(rows), int(batch_graphs)):
        yield rows[i : i + int(batch_graphs)]


def edge_features(ref_graph, cur_graph, *, pos_dim: int, device) -> torch.Tensor:
    ref_e = ref_graph.edge_attr.to(device).float()
    cur_e = cur_graph.edge_attr.to(device).float()
    ref_vec = ref_e[:, :pos_dim]
    cur_vec = cur_e[:, :pos_dim]
    ref_len = ref_e[:, pos_dim : pos_dim + 1]
    cur_len = cur_e[:, pos_dim : pos_dim + 1]
    stiffness = ref_e[:, -1:]
    stretch = cur_len - ref_len
    rel_stretch = stretch / ref_len.clamp_min(1e-6)
    return torch.cat([ref_vec, cur_vec, ref_len, cur_len, stretch, rel_stretch, stiffness, cur_e], dim=-1)


def frame_node_feature(sim, t: int, *, pos_dim: int, mode: str, device) -> torch.Tensor:
    t = int(t)
    cur_pos = sim[t].x[:, :pos_dim].to(device).float()
    if mode in ("position", "positions"):
        return cur_pos
    if mode == "delta":
        ref_pos = sim[0].x[:, :pos_dim].to(device).float()
        return cur_pos - ref_pos
    if mode == "velocity":
        if t <= 0:
            return torch.zeros_like(cur_pos)
        prev_pos = sim[t - 1].x[:, :pos_dim].to(device).float()
        return cur_pos - prev_pos
    raise ValueError(f"Unknown node_feature_mode: {mode}")


def batch_delta_graphs(
    sims,
    rows,
    *,
    pos_dim: int,
    device,
    node_feature_mode: str = "delta",
) -> dict[str, torch.Tensor]:
    xs = []
    node_features = []
    cur_positions = []
    ref_xs = []
    edge_attrs = []
    ref_edge_attrs = []
    edge_indices = []
    batch = []
    node_offset = 0
    for local_idx, (sim_idx, t) in enumerate(rows):
        sim = sims[int(sim_idx)]
        ref_graph = sim[0]
        cur_graph = sim[int(t)]
        ref_pos = ref_graph.x[:, :pos_dim].to(device).float()
        cur_pos = cur_graph.x[:, :pos_dim].to(device).float()
        xs.append(cur_pos - ref_pos)
        cur_positions.append(cur_pos)
        node_features.append(
            frame_node_feature(sim, t, pos_dim=pos_dim, mode=node_feature_mode, device=device)
        )
        ref_xs.append(ref_pos)
        edge_attrs.append(edge_features(ref_graph, cur_graph, pos_dim=pos_dim, device=device))
        ref_edge_attrs.append(edge_features(ref_graph, ref_graph, pos_dim=pos_dim, device=device))
        edge_indices.append(ref_graph.edge_index.to(device).long() + node_offset)
        batch.append(torch.full((ref_pos.size(0),), local_idx, dtype=torch.long, device=device))
        node_offset += ref_pos.size(0)
    return {
        "delta": torch.cat(xs, dim=0),
        "cur_pos": torch.cat(cur_positions, dim=0),
        "node_feature": torch.cat(node_features, dim=0),
        "ref_pos": torch.cat(ref_xs, dim=0),
        "edge_attr": torch.cat(edge_attrs, dim=0),
        "ref_edge_attr": torch.cat(ref_edge_attrs, dim=0),
        "edge_index": torch.cat(edge_indices, dim=1),
        "batch": torch.cat(batch, dim=0),
    }


def ae_target_tensor(batch_data: dict[str, torch.Tensor], target_mode: str) -> torch.Tensor:
    if target_mode in ("position", "positions"):
        return batch_data["cur_pos"]
    if target_mode in ("delta", "displacement"):
        return batch_data["delta"]
    raise ValueError(f"Unknown ae_target_mode: {target_mode}")


def fit_ae_target_stats(
    sims,
    rows,
    *,
    pos_dim: int,
    batch_graphs: int,
    device,
    target_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    chunks = []
    for rows_batch in iter_batches(rows, batch_graphs, shuffle=False):
        batch_data = batch_delta_graphs(sims, rows_batch, pos_dim=pos_dim, device=device)
        chunks.append(ae_target_tensor(batch_data, target_mode).detach())
    all_targets = torch.cat(chunks, dim=0)
    mean = all_targets.mean(dim=0, keepdim=True)
    std = all_targets.std(dim=0, keepdim=True).clamp_min(1e-6)
    return mean, std


def fit_node_feature_stats(
    sims,
    rows,
    *,
    pos_dim: int,
    batch_graphs: int,
    device,
    node_feature_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    chunks = []
    for rows_batch in iter_batches(rows, batch_graphs, shuffle=False):
        batch_data = batch_delta_graphs(
            sims,
            rows_batch,
            pos_dim=pos_dim,
            device=device,
            node_feature_mode=node_feature_mode,
        )
        chunks.append(batch_data["node_feature"].detach())
    all_features = torch.cat(chunks, dim=0)
    mean = all_features.mean(dim=0, keepdim=True)
    std = all_features.std(dim=0, keepdim=True).clamp_min(1e-6)
    return mean, std


def fit_edge_stats(
    sims,
    rows,
    *,
    pos_dim: int,
    batch_graphs: int,
    device,
) -> tuple[torch.Tensor, torch.Tensor]:
    chunks = []
    for rows_batch in iter_batches(rows, batch_graphs, shuffle=False):
        batch_data = batch_delta_graphs(sims, rows_batch, pos_dim=pos_dim, device=device)
        chunks.append(batch_data["edge_attr"].detach())
    all_edges = torch.cat(chunks, dim=0)
    mean = all_edges.mean(dim=0, keepdim=True)
    std = all_edges.std(dim=0, keepdim=True).clamp_min(1e-6)
    return mean, std


def safe_linear_fit_1d(x, y) -> tuple[float, float]:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    x_valid = x[valid]
    y_valid = y[valid]
    if len(x_valid) == 0:
        return 0.0, 0.0
    if len(x_valid) < 2 or np.isclose(np.std(x_valid), 0.0):
        return 0.0, float(np.mean(y_valid))
    slope, intercept = np.polyfit(x_valid, y_valid, deg=1)
    return float(slope), float(intercept)


def r2_score(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return float("nan")
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def pearson_r(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() < 2:
        return float("nan")
    if np.isclose(np.std(y_true[mask]), 0.0) or np.isclose(np.std(y_pred[mask]), 0.0):
        return float("nan")
    return float(np.corrcoef(y_true[mask], y_pred[mask])[0, 1])


def initial_graph_descriptors(sim, sim_idx: int, *, p_ratio_fn) -> dict[str, float | int]:
    graph0 = sim[0]
    edge_attr = graph0.edge_attr.detach().cpu().float()
    pos = graph0.x[:, :2].detach().cpu().float().numpy()
    edge_lengths = edge_attr[:, 2].numpy() if edge_attr.shape[1] > 2 else np.array([])
    stiffness = edge_attr[:, -1].numpy() if edge_attr.numel() else np.array([])
    return {
        "sim_idx": int(sim_idx),
        "n_nodes": int(graph0.num_nodes),
        "n_edges": int(graph0.edge_index.shape[1]),
        "final_p_ratio": float(p_ratio_fn(sim, -1)),
        "x_span": float(np.ptp(pos[:, 0])) if len(pos) else float("nan"),
        "y_span": float(np.ptp(pos[:, 1])) if len(pos) else float("nan"),
        "edge_length_mean": float(np.mean(edge_lengths)) if edge_lengths.size else float("nan"),
        "edge_length_std": float(np.std(edge_lengths)) if edge_lengths.size else float("nan"),
        "stiffness_mean": float(np.mean(stiffness)) if stiffness.size else float("nan"),
        "stiffness_std": float(np.std(stiffness)) if stiffness.size else float("nan"),
    }


def latent_pratio_corr_rows(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    target_col: str = "final_p_ratio",
    latent_dim_col: str = "latent_dim",
) -> list[dict[str, object]]:
    rows = []
    if df.empty:
        return rows
    for key, group in df.groupby(group_cols, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        base = dict(zip(group_cols, key, strict=False))
        latent_dim = int(base.get(latent_dim_col, group[latent_dim_col].iloc[0]))
        for z_idx in range(latent_dim):
            z_col = f"z{z_idx}"
            xy = group[[z_col, target_col]].replace([np.inf, -np.inf], np.nan).dropna()
            corr = pearson_r(xy[z_col], xy[target_col]) if len(xy) >= 2 else float("nan")
            if len(xy) >= 2:
                slope, intercept = safe_linear_fit_1d(xy[z_col], xy[target_col])
                pred = slope * xy[z_col].to_numpy(dtype=float) + intercept
                linear_r2 = r2_score(xy[target_col], pred)
            else:
                linear_r2 = float("nan")
            rows.append(
                {
                    **base,
                    "latent_coordinate": z_col,
                    "corr": corr,
                    "corr_r2": float(np.clip(corr**2, 0.0, 1.0))
                    if np.isfinite(corr)
                    else float("nan"),
                    "linear_r2": float(np.clip(linear_r2, 0.0, 1.0))
                    if np.isfinite(linear_r2)
                    else float("nan"),
                    "n_points": int(len(xy)),
                }
            )
    return rows


__all__ = [
    "ae_target_tensor",
    "batch_delta_graphs",
    "edge_features",
    "filtered_frame_ids",
    "fit_ae_target_stats",
    "fit_edge_stats",
    "fit_node_feature_stats",
    "frame_for_filtered_step",
    "frame_node_feature",
    "initial_graph_descriptors",
    "iter_batches",
    "latent_pratio_corr_rows",
    "make_frame_index",
    "make_transition_index",
    "pearson_r",
    "r2_score",
    "safe_linear_fit_1d",
]
