"""Shared helpers for latent-space simulator notebooks."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd
import torch
from graph_utils import directional_side_indices_from_box

from ..graph import box_tensor



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


def complete_graph_edge_data(
    ref_graph,
    cur_graph,
    *,
    pos_dim: int,
    device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build directed all-pairs edges with periodic geometry and stored-bond weights.

    Every ordered pair ``i -> j`` with ``i != j`` is present. The original
    undirected elastic coefficient is copied to both directions; pairs absent
    from the stored elastic graph receive coefficient zero. Geometry uses the
    minimum-image convention when a periodic box is available.
    """
    ref_pos = ref_graph.x[:, :pos_dim].to(device).float()
    cur_pos = cur_graph.x[:, :pos_dim].to(device).float()
    if ref_pos.shape != cur_pos.shape:
        raise ValueError("Complete-graph edge mode requires a fixed node set across frames.")

    num_nodes = ref_pos.size(0)
    nodes = torch.arange(num_nodes, device=device)
    source = nodes.repeat_interleave(num_nodes)
    target = nodes.repeat(num_nodes)
    keep = source != target
    source, target = source[keep], target[keep]
    edge_index = torch.stack([source, target], dim=0)

    def pair_geometry(pos, graph):
        vector = pos[target] - pos[source]
        box = box_tensor(graph, device=device, dtype=pos.dtype)
        if box is not None:
            vector = vector - torch.round(vector / box.reshape(1, -1)) * box.reshape(1, -1)
        length = torch.linalg.vector_norm(vector, dim=-1, keepdim=True)
        return vector, length

    ref_vec, ref_len = pair_geometry(ref_pos, ref_graph)
    cur_vec, cur_len = pair_geometry(cur_pos, cur_graph)

    stiffness_matrix = torch.zeros(
        (num_nodes, num_nodes), device=device, dtype=ref_pos.dtype
    )
    stored_index = ref_graph.edge_index.to(device).long()
    stored_stiffness = ref_graph.edge_attr[:, -1].to(device).float()
    stored_source, stored_target = stored_index
    stiffness_matrix[stored_source, stored_target] = stored_stiffness
    stiffness_matrix[stored_target, stored_source] = stored_stiffness
    stiffness = stiffness_matrix[source, target].reshape(-1, 1)

    def expanded_features(current_vec, current_len):
        stretch = current_len - ref_len
        relative_stretch = stretch / ref_len.clamp_min(1e-6)
        current_raw = torch.cat([current_vec, current_len, stiffness], dim=-1)
        return torch.cat(
            [
                ref_vec,
                current_vec,
                ref_len,
                current_len,
                stretch,
                relative_stretch,
                stiffness,
                current_raw,
            ],
            dim=-1,
        )

    return edge_index, expanded_features(cur_vec, cur_len), expanded_features(ref_vec, ref_len)


def frame_node_feature(sim, t: int, *, pos_dim: int, mode: str, device) -> torch.Tensor:
    t = int(t)
    cur_pos = sim[t].x[:, :pos_dim].to(device).float()
    if mode in ("position", "positions"):
        return cur_pos
    if mode == "delta":
        ref_pos = sim[0].x[:, :pos_dim].to(device).float()
        return cur_pos - ref_pos
    if mode in {"normalized_delta", "self_normalized_delta", "relative_delta"}:
        ref_pos = sim[0].x[:, :pos_dim].to(device).float()
        scale = (ref_pos.max(dim=0).values - ref_pos.min(dim=0).values).clamp_min(1e-6)
        return (cur_pos - ref_pos) / scale.reshape(1, -1)
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
    edge_mode: str = "stored",
) -> dict[str, torch.Tensor]:
    xs = []
    node_features = []
    cur_positions = []
    ref_xs = []
    edge_attrs = []
    ref_edge_attrs = []
    edge_indices = []
    scales = []
    batch = []
    node_offset = 0
    for local_idx, (sim_idx, t) in enumerate(rows):
        sim = sims[int(sim_idx)]
        ref_graph = sim[0]
        cur_graph = sim[int(t)]
        ref_pos = ref_graph.x[:, :pos_dim].to(device).float()
        cur_pos = cur_graph.x[:, :pos_dim].to(device).float()
        scale = (ref_pos.max(dim=0).values - ref_pos.min(dim=0).values).clamp_min(1e-6)
        xs.append(cur_pos - ref_pos)
        scales.append(scale.reshape(1, -1).expand(ref_pos.size(0), -1))
        cur_positions.append(cur_pos)
        node_features.append(
            frame_node_feature(sim, t, pos_dim=pos_dim, mode=node_feature_mode, device=device)
        )
        ref_xs.append(ref_pos)
        if edge_mode == "complete":
            local_edge_index, current_edges, reference_edges = complete_graph_edge_data(
                ref_graph, cur_graph, pos_dim=pos_dim, device=device
            )
        elif edge_mode == "stored":
            local_edge_index = ref_graph.edge_index.to(device).long()
            current_edges = edge_features(ref_graph, cur_graph, pos_dim=pos_dim, device=device)
            reference_edges = edge_features(ref_graph, ref_graph, pos_dim=pos_dim, device=device)
        else:
            raise ValueError(f"Unknown edge_mode: {edge_mode}")
        edge_attrs.append(current_edges)
        ref_edge_attrs.append(reference_edges)
        edge_indices.append(local_edge_index + node_offset)
        batch.append(torch.full((ref_pos.size(0),), local_idx, dtype=torch.long, device=device))
        node_offset += ref_pos.size(0)
    return {
        "delta": torch.cat(xs, dim=0),
        "normalized_delta": torch.cat(xs, dim=0) / torch.cat(scales, dim=0),
        "position_scale": torch.cat(scales, dim=0),
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
    if target_mode in {"normalized_delta", "self_normalized_delta", "relative_delta"}:
        return batch_data["normalized_delta"]
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
    edge_mode: str = "stored",
) -> tuple[torch.Tensor, torch.Tensor]:
    if edge_mode == "complete":
        count = 0
        total = None
        total_square = None
        for rows_batch in iter_batches(rows, batch_graphs, shuffle=False):
            edges = batch_delta_graphs(
                sims, rows_batch, pos_dim=pos_dim, device=device, edge_mode=edge_mode
            )["edge_attr"].detach().double()
            batch_total = edges.sum(dim=0, keepdim=True)
            batch_total_square = edges.square().sum(dim=0, keepdim=True)
            total = batch_total if total is None else total + batch_total
            total_square = batch_total_square if total_square is None else total_square + batch_total_square
            count += int(edges.size(0))
        if not count:
            raise ValueError("Cannot fit complete-edge statistics from an empty frame index.")
        mean = total / count
        denominator = max(count - 1, 1)
        variance = (total_square - total.square() / count) / denominator
        return mean.float(), variance.clamp_min(0).sqrt().clamp_min(1e-6).float()

    chunks = []
    for rows_batch in iter_batches(rows, batch_graphs, shuffle=False):
        batch_data = batch_delta_graphs(
            sims, rows_batch, pos_dim=pos_dim, device=device, edge_mode=edge_mode
        )
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


def trajectory_p_ratio_sides_strain_gated(
    trajectory: list,
    last_index: int = -1,
    *,
    first_index: int = 0,
    min_fit_frames: int = 8,
    min_driven_strain_range: float = 1e-3,
    side_quantile: float = 0.10,
    min_abs_strain: float = 1e-5,
    eps: float = 1e-12,
) -> float:
    """Estimate trajectory p-ratio only after enough driven strain accumulates.

    Linear trajectory fits are unstable at early thermalized frames because both
    fitted strain rates are small. This variant returns NaN until there are
    enough meaningful frames and the larger strain component has moved by at
    least ``min_driven_strain_range``.
    """

    if len(trajectory) < 2:
        return float("nan")

    stop = int(last_index)
    if stop < 0:
        stop += len(trajectory)
    start = int(first_index)
    if start < 0:
        start += len(trajectory)
    if not (0 <= start < stop < len(trajectory)):
        return float("nan")

    reference = trajectory[start]
    side_idx = directional_side_indices_from_box(
        reference,
        quantile=float(side_quantile),
        eps=float(eps),
    )

    def pos_np(graph) -> np.ndarray:
        return graph.x[:, :2].detach().cpu().numpy()

    def side_dimensions(graph) -> tuple[float, float]:
        pos = pos_np(graph)
        width = float(pos[side_idx["right"], 0].mean() - pos[side_idx["left"], 0].mean())
        height = float(pos[side_idx["top"], 1].mean() - pos[side_idx["bottom"], 1].mean())
        return width, height

    width0, height0 = side_dimensions(reference)
    if abs(width0) <= eps or abs(height0) <= eps:
        return float("nan")

    strains = np.asarray(
        [
            (
                (side_dimensions(graph)[0] - width0) / width0,
                (side_dimensions(graph)[1] - height0) / height0,
            )
            for graph in trajectory[start : stop + 1]
        ],
        dtype=float,
    )
    finite = np.isfinite(strains).all(axis=1)
    strains = strains[finite]
    if len(strains) < int(min_fit_frames):
        return float("nan")

    meaningful = (np.abs(strains[:, 0]) >= float(min_abs_strain)) | (
        np.abs(strains[:, 1]) >= float(min_abs_strain)
    )
    strains = strains[meaningful]
    if len(strains) < int(min_fit_frames):
        return float("nan")

    x_range = float(np.ptp(strains[:, 0]))
    y_range = float(np.ptp(strains[:, 1]))
    if max(x_range, y_range) < float(min_driven_strain_range):
        return float("nan")

    time = np.arange(len(strains), dtype=float)
    if float(np.var(time)) <= eps:
        return float("nan")
    dx_rate = float(np.polyfit(time, strains[:, 0], deg=1)[0])
    dy_rate = float(np.polyfit(time, strains[:, 1], deg=1)[0])
    if (not np.isfinite(dx_rate)) or (not np.isfinite(dy_rate)):
        return float("nan")
    if abs(dx_rate) <= eps or abs(dy_rate) <= eps:
        return float("nan")

    p_x_driven = -(dy_rate / dx_rate)
    p_y_driven = -(dx_rate / dy_rate)
    return float(p_x_driven if abs(p_x_driven) < abs(p_y_driven) else p_y_driven)


def _centered_rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    window = int(window)
    if window <= 1 or len(values) < 3:
        return values
    if window % 2 == 0:
        window += 1
    half = window // 2
    out = np.empty_like(values, dtype=float)
    for idx in range(len(values)):
        lo = max(0, idx - half)
        hi = min(len(values), idx + half + 1)
        out[idx] = float(np.nanmean(values[lo:hi]))
    return out


def _theil_sen_slope(x: np.ndarray, y: np.ndarray, *, eps: float) -> float:
    slopes = []
    for i in range(len(x) - 1):
        dx = x[i + 1 :] - x[i]
        dy = y[i + 1 :] - y[i]
        valid = np.abs(dx) > eps
        if np.any(valid):
            slopes.extend((dy[valid] / dx[valid]).tolist())
    if not slopes:
        return float("nan")
    return float(np.median(np.asarray(slopes, dtype=float)))


def trajectory_p_ratio_sides_robust(
    trajectory: list,
    last_index: int = -1,
    *,
    first_index: int = 0,
    min_fit_frames: int = 8,
    min_driven_strain_range: float = 1e-3,
    smooth_window: int = 5,
    side_quantile: float = 0.10,
    min_abs_strain: float = 1e-5,
    eps: float = 1e-12,
) -> float:
    """Robust p-ratio from transverse strain vs driven strain.

    This avoids fitting strain against time. After light smoothing, the axis
    with larger strain range is treated as driven, and the transverse/driven
    slope is estimated by a Theil-Sen median pairwise slope.
    """

    if len(trajectory) < 2:
        return float("nan")

    stop = int(last_index)
    if stop < 0:
        stop += len(trajectory)
    start = int(first_index)
    if start < 0:
        start += len(trajectory)
    if not (0 <= start < stop < len(trajectory)):
        return float("nan")

    reference = trajectory[start]
    side_idx = directional_side_indices_from_box(
        reference,
        quantile=float(side_quantile),
        eps=float(eps),
    )

    def side_dimensions(graph) -> tuple[float, float]:
        pos = graph.x[:, :2].detach().cpu().numpy()
        width = float(pos[side_idx["right"], 0].mean() - pos[side_idx["left"], 0].mean())
        height = float(pos[side_idx["top"], 1].mean() - pos[side_idx["bottom"], 1].mean())
        return width, height

    width0, height0 = side_dimensions(reference)
    if abs(width0) <= eps or abs(height0) <= eps:
        return float("nan")

    strains = np.asarray(
        [
            (
                (side_dimensions(graph)[0] - width0) / width0,
                (side_dimensions(graph)[1] - height0) / height0,
            )
            for graph in trajectory[start : stop + 1]
        ],
        dtype=float,
    )
    finite = np.isfinite(strains).all(axis=1)
    strains = strains[finite]
    if len(strains) < int(min_fit_frames):
        return float("nan")

    strains[:, 0] = _centered_rolling_mean(strains[:, 0], int(smooth_window))
    strains[:, 1] = _centered_rolling_mean(strains[:, 1], int(smooth_window))
    meaningful = (np.abs(strains[:, 0]) >= float(min_abs_strain)) | (
        np.abs(strains[:, 1]) >= float(min_abs_strain)
    )
    strains = strains[meaningful]
    if len(strains) < int(min_fit_frames):
        return float("nan")

    x_range = float(np.ptp(strains[:, 0]))
    y_range = float(np.ptp(strains[:, 1]))
    if max(x_range, y_range) < float(min_driven_strain_range):
        return float("nan")

    if x_range >= y_range:
        slope = _theil_sen_slope(strains[:, 0], strains[:, 1], eps=eps)
    else:
        slope = _theil_sen_slope(strains[:, 1], strains[:, 0], eps=eps)
    return float(-slope) if np.isfinite(slope) else float("nan")


def trajectory_p_ratio_sides_robust_series(
    trajectory: list,
    *,
    min_fit_frames: int = 8,
    min_driven_strain_range: float = 1e-3,
    smooth_window: int = 5,
    side_quantile: float = 0.10,
    min_abs_strain: float = 1e-5,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute the robust p-ratio for every available trajectory prefix."""

    values = np.full(len(trajectory), np.nan, dtype=float)
    if len(trajectory) < 2:
        return values

    side_idx = directional_side_indices_from_box(
        trajectory[0], quantile=float(side_quantile), eps=float(eps)
    )

    def dimensions(graph) -> tuple[float, float]:
        pos = graph.x[:, :2].detach().cpu().numpy()
        return (
            float(pos[side_idx["right"], 0].mean() - pos[side_idx["left"], 0].mean()),
            float(pos[side_idx["top"], 1].mean() - pos[side_idx["bottom"], 1].mean()),
        )

    dims = np.asarray([dimensions(graph) for graph in trajectory], dtype=float)
    width0, height0 = dims[0]
    if abs(width0) <= eps or abs(height0) <= eps:
        return values
    strains = np.column_stack(
        ((dims[:, 0] - width0) / width0, (dims[:, 1] - height0) / height0)
    )

    for stop in range(1, len(trajectory)):
        prefix = strains[: stop + 1]
        prefix = prefix[np.isfinite(prefix).all(axis=1)]
        if len(prefix) < int(min_fit_frames):
            continue
        prefix = prefix.copy()
        prefix[:, 0] = _centered_rolling_mean(prefix[:, 0], int(smooth_window))
        prefix[:, 1] = _centered_rolling_mean(prefix[:, 1], int(smooth_window))
        meaningful = (np.abs(prefix[:, 0]) >= float(min_abs_strain)) | (
            np.abs(prefix[:, 1]) >= float(min_abs_strain)
        )
        prefix = prefix[meaningful]
        if len(prefix) < int(min_fit_frames):
            continue
        x_range = float(np.ptp(prefix[:, 0]))
        y_range = float(np.ptp(prefix[:, 1]))
        if max(x_range, y_range) < float(min_driven_strain_range):
            continue
        if x_range >= y_range:
            slope = _theil_sen_slope(prefix[:, 0], prefix[:, 1], eps=eps)
        else:
            slope = _theil_sen_slope(prefix[:, 1], prefix[:, 0], eps=eps)
        if np.isfinite(slope):
            values[stop] = -slope
    return values


def shrink_p_ratio_series(
    values,
    trajectory: list,
    *,
    prior: float | None,
    strain_scale: float = 6e-3,
    full_weight_frame: int = 20,
) -> np.ndarray:
    """Stabilize short-prefix estimates with a training-set p-ratio prior."""

    out = np.asarray(values, dtype=float).copy()
    try:
        prior_value = float(prior)
    except (TypeError, ValueError):
        return out
    if not np.isfinite(prior_value) or len(out) != len(trajectory):
        return out

    boxes = [box_tensor(graph) for graph in trajectory]
    if any(box is None for box in boxes):
        return out
    widths = np.asarray([float(box[0].detach().cpu()) for box in boxes], dtype=float)
    if not np.isfinite(widths[0]) or abs(widths[0]) <= 1e-12:
        return out
    driven_strain = widths / widths[0] - 1.0
    scale_sq = max(float(strain_scale), 1e-12) ** 2
    for stop in range(1, len(out)):
        if stop >= int(full_weight_frame):
            continue
        signal = float(np.ptp(driven_strain[: stop + 1]))
        signal_weight = signal**2 / (signal**2 + scale_sq) if np.isfinite(signal) else 0.0
        frame_weight = max(
            0.0,
            min(1.0, (stop - 8.0) / max(float(full_weight_frame) - 8.0, 1.0)),
        )
        weight = max(signal_weight, frame_weight)
        if np.isfinite(out[stop]):
            out[stop] = weight * out[stop] + (1.0 - weight) * prior_value
        else:
            out[stop] = prior_value
    return out


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
    "shrink_p_ratio_series",
    "trajectory_p_ratio_sides_robust_series",
]
