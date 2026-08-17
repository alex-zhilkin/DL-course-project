"""Reusable analysis helpers for latent-space paper notebooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from graph_utils import calc_p_ratio_rollout_sides, directional_side_indices_from_box

from ..data import resolve_dataset_splits
from .models import NodeDeltaAttentionAutoEncoder
from .simulation import edge_features, frame_node_feature, pearson_r, r2_score
from .training import encode_frame_latent


def label_evaluation_sources(
    frame: pd.DataFrame,
    source_labels: dict[str, str],
    *,
    source_column: str = "source",
    output_column: str = "source_family",
) -> pd.DataFrame:
    """Label mixed-source evaluation rows without reconstructing source from indices."""

    if source_column not in frame:
        raise KeyError(
            f"Evaluation rows must contain authoritative {source_column!r} metadata."
        )
    sources = frame[source_column]
    invalid = sources.isna() | sources.astype(str).str.strip().eq("")
    if invalid.any():
        raise ValueError(
            f"Evaluation rows contain {int(invalid.sum())} missing source assignments."
        )
    labeled = frame.copy()
    labeled[output_column] = sources.map(source_labels).fillna(sources.astype(str))
    return labeled


@dataclass
class CVAnalysisContext:
    """Load saved sweep models and extract physical and latent trajectory data."""

    cfg: dict
    device: str
    project_root: Path

    def model_path(self, dataset_name, latent_dim) -> Path:
        return Path(self.cfg["model_dir"]) / (
            f"{dataset_name}_{self.cfg['target_mode']}_cv{int(latent_dim):02d}"
            f"_nets{int(self.cfg['train_networks']):03d}"
            f"_frames{int(self.cfg['train_frames_per_network']):03d}"
            f"_rep{int(self.cfg['repeat_idx']):02d}.pt"
        )

    def load_bundle(self, path):
        return torch.load(path, map_location=self.device, weights_only=False)

    def restore_ae(self, bundle):
        spec = dict(bundle.get("spec", {}))
        params = dict(bundle.get("params", {}))
        stats = bundle["stats"]
        ae = NodeDeltaAttentionAutoEncoder(
            pos_dim=int(self.cfg["pos_dim"]),
            node_feature_dim=int(
                stats.get(
                    "node_feature_mean",
                    torch.zeros(int(self.cfg["pos_dim"])),
                ).numel()
            ),
            edge_dim=int(stats["edge_mean"].numel()),
            hidden_size=int(spec.get("hidden_size", params.get("hidden_size", 92))),
            latent_dim=int(spec.get("latent_dim", params.get("latent_dim"))),
            latent_tokens=int(spec.get("latent_tokens", params.get("latent_tokens", 12))),
        ).to(self.device)
        ae.load_state_dict(bundle["ae_state_dict"])
        ae.eval()
        for parameter in ae.parameters():
            parameter.requires_grad_(False)
        return ae

    def resolve_bundle_splits(self, bundle):
        spec = dict(bundle.get("spec", {}))
        params = dict(bundle.get("params", {}))
        snapshot = dict(bundle.get("cfg_snapshot", {}))
        dataset_path = spec.get("path") or params.get("dataset_path") or snapshot.get("dataset_path")
        if dataset_path is None:
            raise ValueError("Saved bundle does not contain a dataset path.")
        dataset_path = Path(dataset_path)
        if not dataset_path.exists() and str(dataset_path).startswith("../"):
            dataset_path = self.project_root / str(dataset_path)[3:]
        return resolve_dataset_splits(
            str(dataset_path),
            train_count=int(spec.get("train_count", params.get("train_count", self.cfg["train_networks"]))),
            val_count=int(spec.get("val_count", params.get("val_count", snapshot.get("val_count", 30)))),
            split_seed=snapshot.get("split_seed", params.get("split_seed")),
            shuffle_within_source=True,
            edge_multiplicity=int(
                spec.get(
                    "edge_multiplicity",
                    params.get("edge_multiplicity", snapshot.get("edge_multiplicity", 1)),
                )
            ),
            edge_vector_dim=int(
                spec.get(
                    "edge_vector_dim",
                    params.get("edge_vector_dim", snapshot.get("edge_vector_dim", 2)),
                )
            ),
        )

    def encode_frame_z(self, ae, bundle, sim, frame_idx):
        stats = bundle["stats"]
        params = dict(bundle.get("params", {}))
        pos_dim = int(self.cfg["pos_dim"])
        ref_graph = sim[0]
        cur_graph = sim[int(frame_idx)]
        ref_pos = ref_graph.x[:, :pos_dim].to(self.device).float()
        node_feature = frame_node_feature(
            sim,
            int(frame_idx),
            pos_dim=pos_dim,
            mode=params.get("node_feature_mode", "delta"),
            device=self.device,
        )
        node_feature_norm = (
            node_feature - stats["node_feature_mean"].to(self.device)
        ) / stats["node_feature_std"].to(self.device)
        edge_attr_norm = (
            edge_features(ref_graph, cur_graph, pos_dim=pos_dim, device=self.device)
            - stats["edge_mean"].to(self.device)
        ) / stats["edge_std"].to(self.device)
        ref_edge_attr_norm = (
            edge_features(ref_graph, ref_graph, pos_dim=pos_dim, device=self.device)
            - stats["edge_mean"].to(self.device)
        ) / stats["edge_std"].to(self.device)
        batch = torch.zeros(ref_pos.size(0), dtype=torch.long, device=self.device)
        with torch.no_grad():
            z, _ = ae.encode(
                node_feature_norm,
                ref_pos,
                edge_attr_norm,
                ref_edge_attr_norm,
                ref_graph.edge_index.to(self.device).long(),
                batch,
            )
        return z.squeeze(0).detach().cpu().numpy()

    def encode_initial_z(self, ae, bundle, sim):
        return self.encode_frame_z(ae, bundle, sim, 0)

    @staticmethod
    def p_ratio_final(sim):
        return float(calc_p_ratio_rollout_sides(sim, -1))

    @staticmethod
    def unique_undirected_edge_values(graph):
        edge_index = graph.edge_index.detach().cpu().numpy()
        edge_attr = graph.edge_attr.detach().cpu().numpy()
        values = {}
        for edge_idx in range(edge_index.shape[1]):
            i, j = int(edge_index[0, edge_idx]), int(edge_index[1, edge_idx])
            if i != j:
                values.setdefault(tuple(sorted((i, j))), float(edge_attr[edge_idx, -1]))
        return np.asarray(list(values.values()), dtype=float)

    def network_descriptor_row(self, sim):
        stiffness = self.unique_undirected_edge_values(sim[0])
        mean = float(np.mean(stiffness)) if len(stiffness) else np.nan
        std = float(np.std(stiffness)) if len(stiffness) else np.nan
        return {
            "stiffness_mean": mean,
            "stiffness_std": std,
            "stiffness_cv": std / mean if np.isfinite(mean) and abs(mean) > 1e-12 else np.nan,
            "soft_frac_lt_0p2": float(np.mean(stiffness < 0.2)) if len(stiffness) else np.nan,
        }

    def position_bbox_dims(self, graph):
        pos = graph.x[:, : int(self.cfg["pos_dim"])].detach().cpu().numpy()
        if pos.size == 0:
            return np.nan, np.nan
        return float(np.ptp(pos[:, 0])), float(np.ptp(pos[:, 1]))

    @staticmethod
    def _numeric_box_array(box):
        if box is None:
            return None
        if torch.is_tensor(box):
            return box.detach().cpu().numpy()
        if isinstance(box, (list, tuple, np.ndarray, float, int)):
            return np.asarray(box)
        for attr in (
            "bounds",
            "box",
            "array",
            "data",
            "tensor",
            "matrix",
            "vectors",
            "lengths",
            "size",
            "extent",
        ):
            if not hasattr(box, attr):
                continue
            value = getattr(box, attr)
            if callable(value):
                try:
                    value = value()
                except TypeError:
                    continue
            array = CVAnalysisContext._numeric_box_array(value)
            if array is not None:
                return array
        return None

    def graph_box_dims(self, graph):
        bbox_w, bbox_h = self.position_bbox_dims(graph)
        array = self._numeric_box_array(getattr(graph, "box", None))
        if array is None:
            return bbox_w, bbox_h
        try:
            array = np.asarray(array, dtype=float).squeeze()
        except (TypeError, ValueError):
            return bbox_w, bbox_h
        candidates = []
        if array.shape == (2,):
            candidates.append((abs(float(array[0])), abs(float(array[1]))))
        elif array.shape == (2, 2):
            candidates.extend(
                [
                    (abs(float(array[0, 1] - array[0, 0])), abs(float(array[1, 1] - array[1, 0]))),
                    (float(np.linalg.norm(array[0])), float(np.linalg.norm(array[1]))),
                ]
            )
        else:
            flat = array.reshape(-1)
            if flat.size >= 4:
                candidates.extend(
                    [
                        (abs(float(flat[1] - flat[0])), abs(float(flat[3] - flat[2]))),
                        (abs(float(flat[2] - flat[0])), abs(float(flat[3] - flat[1]))),
                    ]
                )
            elif flat.size >= 2:
                candidates.append((abs(float(flat[0])), abs(float(flat[1]))))
        candidates = [
            (width, height)
            for width, height in candidates
            if np.isfinite(width) and np.isfinite(height) and width > 0 and height > 0
        ]
        if not candidates:
            return bbox_w, bbox_h
        if np.isfinite(bbox_w) and np.isfinite(bbox_h):
            return min(
                candidates,
                key=lambda dims: abs(dims[0] - bbox_w) + abs(dims[1] - bbox_h),
            )
        return candidates[0]

    def frame_deformation_row(self, sim, frame_idx):
        ref = sim[0]
        cur = sim[int(frame_idx)]
        pos_dim = int(self.cfg["pos_dim"])
        disp = (
            cur.x[:, :pos_dim].detach().cpu().numpy()
            - ref.x[:, :pos_dim].detach().cpu().numpy()
        )
        box_w0, box_h0 = self.graph_box_dims(ref)
        box_w, box_h = self.graph_box_dims(cur)
        bbox_w0, bbox_h0 = self.position_bbox_dims(ref)
        bbox_w, bbox_h = self.position_bbox_dims(cur)
        strain_x = box_w / box_w0 - 1.0 if np.isfinite(box_w0) and abs(box_w0) > 1e-12 else np.nan
        strain_y = box_h / box_h0 - 1.0 if np.isfinite(box_h0) and abs(box_h0) > 1e-12 else np.nan
        return {
            "frame_idx": int(frame_idx),
            "frame_progress": float(frame_idx / max(len(sim) - 1, 1)),
            "box_width": box_w,
            "box_height": box_h,
            "box_delta_width": box_w - box_w0,
            "box_delta_height": box_h - box_h0,
            "box_strain_x": strain_x,
            "box_strain_y": strain_y,
            "box_poisson_path": (
                -strain_y / strain_x
                if np.isfinite(strain_x) and abs(strain_x) > 1e-12
                else np.nan
            ),
            "bbox_strain_x": bbox_w / bbox_w0 - 1.0,
            "bbox_strain_y": bbox_h / bbox_h0 - 1.0,
            "mean_dx": float(np.mean(disp[:, 0])),
            "mean_dy": float(np.mean(disp[:, 1])),
            "rms_dx": float(np.sqrt(np.mean(disp[:, 0] ** 2))),
            "rms_dy": float(np.sqrt(np.mean(disp[:, 1] ** 2))),
            "rms_disp": float(np.sqrt(np.mean(np.sum(disp**2, axis=1)))),
        }


def fit_linear(X, y, ridge=1e-8):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[mask], y[mask]
    x_mean = X.mean(axis=0, keepdims=True)
    x_std = X.std(axis=0, keepdims=True) + 1e-12
    Xz = (X - x_mean) / x_std
    design = np.column_stack([np.ones(len(Xz)), Xz])
    penalty = np.eye(design.shape[1])
    penalty[0, 0] = 0
    coef = np.linalg.solve(design.T @ design + ridge * penalty, design.T @ y)
    return {
        "coef": coef,
        "x_mean": x_mean,
        "x_std": x_std,
        "raw_coef": coef[1:] / x_std.reshape(-1),
        "raw_intercept": float(
            coef[0] - np.sum(coef[1:] * x_mean.reshape(-1) / x_std.reshape(-1))
        ),
    }


def predict_linear(model, X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    Xz = (X - model["x_mean"]) / model["x_std"]
    return np.column_stack([np.ones(len(Xz)), Xz]) @ model["coef"]


def evaluate_readouts(group):
    val = group[group["split"].eq("val")]
    test = group[group["split"].eq("test")]
    metric_rows = []
    weight_rows = []
    specs = [("z0", ["z0"])]
    if int(group["latent_dim"].iloc[0]) >= 2:
        specs.extend([("z1", ["z1"]), ("linear z0,z1", ["z0", "z1"])])
    for label, columns in specs:
        model = fit_linear(val[columns], val["final_p_ratio"])
        pred = predict_linear(model, test[columns])
        metric_rows.append(
            {
                "readout": label,
                "test_r2": r2_score(test["final_p_ratio"], pred),
                "test_pearson_pred_vs_true": pearson_r(pred, test["final_p_ratio"]),
                "raw_pearson_first_coord": pearson_r(
                    test[columns[0]],
                    test["final_p_ratio"],
                ),
                "n_val": len(val),
                "n_test": len(test),
            }
        )
        weights = {
            "readout": label,
            "intercept_raw": model["raw_intercept"],
            "intercept_standardized": float(model["coef"][0]),
        }
        for idx, column in enumerate(columns):
            weights[f"{column}_weight_raw"] = float(model["raw_coef"][idx])
            weights[f"{column}_weight_standardized"] = float(model["coef"][idx + 1])
            weights[f"{column}_val_mean"] = float(model["x_mean"].reshape(-1)[idx])
            weights[f"{column}_val_std"] = float(model["x_std"].reshape(-1)[idx])
        weight_rows.append(weights)
    return pd.DataFrame(metric_rows), pd.DataFrame(weight_rows)


def path_curvature_metrics(group):
    points = group.sort_values("frame_idx")[["z0", "z1"]].to_numpy(float)
    if len(points) < 3 or not np.all(np.isfinite(points)):
        return pd.Series(
            {
                "path_length": np.nan,
                "chord_length": np.nan,
                "tortuosity": np.nan,
                "linearity_r2": np.nan,
                "curvature_score": np.nan,
                "max_perp_deviation": np.nan,
                "mean_perp_deviation": np.nan,
            }
        )
    path_length = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    chord_length = float(np.linalg.norm(points[-1] - points[0]))
    centered = points - points.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    residual = centered - np.outer(centered @ vh[0], vh[0])
    total_ss = float(np.sum(centered**2))
    residual_ss = float(np.sum(residual**2))
    linearity = 1.0 - residual_ss / total_ss if total_ss > 1e-12 else np.nan
    perpendicular = np.linalg.norm(residual, axis=1)
    return pd.Series(
        {
            "path_length": path_length,
            "chord_length": chord_length,
            "tortuosity": path_length / chord_length if chord_length > 1e-12 else np.nan,
            "linearity_r2": linearity,
            "curvature_score": 1.0 - linearity if np.isfinite(linearity) else np.nan,
            "max_perp_deviation": float(np.max(perpendicular)),
            "mean_perp_deviation": float(np.mean(perpendicular)),
        }
    )


def residualize(y, X):
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    out = np.full_like(y, np.nan, dtype=float)
    if mask.sum() < X.shape[1] + 2:
        return out
    design = np.column_stack([np.ones(mask.sum()), X[mask]])
    out[mask] = y[mask] - design @ np.linalg.lstsq(design, y[mask], rcond=None)[0]
    return out


def _box_dimensions(graph) -> tuple[float, float]:
    box = getattr(graph, "box", None)
    if box is not None and all(hasattr(box, attr) for attr in ("x1", "x2", "y1", "y2")):
        return float(box.x2 - box.x1), float(box.y2 - box.y1)
    pos = graph.x[:, :2].detach().cpu().numpy()
    return float(np.ptp(pos[:, 0])), float(np.ptp(pos[:, 1]))


def _unique_edge_mask(graph) -> np.ndarray:
    edge_index = graph.edge_index.detach().cpu().numpy()
    return edge_index[0] < edge_index[1]


def _frame_descriptor_row(
    sim,
    frame_idx: int,
    *,
    side_indices: dict[str, np.ndarray] | None = None,
    side_reference_dimensions: tuple[float, float] | None = None,
) -> dict[str, float]:
    ref = sim[0]
    cur = sim[int(frame_idx)]
    ref_pos = ref.x[:, :2].detach().cpu().numpy()
    cur_pos = cur.x[:, :2].detach().cpu().numpy()
    displacement = cur_pos - ref_pos
    displacement_norm = np.linalg.norm(displacement, axis=1)

    ref_centered = ref_pos - ref_pos.mean(axis=0, keepdims=True)
    affine_design = np.column_stack([np.ones(len(ref_pos)), ref_centered])
    affine_coef = np.linalg.lstsq(affine_design, displacement, rcond=None)[0]
    nonaffine = displacement - affine_design @ affine_coef

    edge_mask = _unique_edge_mask(ref)
    ref_edge = ref.edge_attr.detach().cpu().numpy()[edge_mask]
    cur_edge = cur.edge_attr.detach().cpu().numpy()[edge_mask]
    ref_length = ref_edge[:, 2]
    cur_length = cur_edge[:, 2]
    stiffness = ref_edge[:, -1]
    stretch = cur_length - ref_length
    relative_stretch = stretch / np.maximum(ref_length, 1e-12)

    width0, height0 = _box_dimensions(ref)
    width, height = _box_dimensions(cur)
    strain_x = width / width0 - 1.0
    strain_y = height / height0 - 1.0
    side_strain_x = np.nan
    side_strain_y = np.nan
    if side_indices is not None and side_reference_dimensions is not None:
        side_width0, side_height0 = side_reference_dimensions
        side_width = float(
            np.mean(cur_pos[side_indices["right"], 0])
            - np.mean(cur_pos[side_indices["left"], 0])
        )
        side_height = float(
            np.mean(cur_pos[side_indices["top"], 1])
            - np.mean(cur_pos[side_indices["bottom"], 1])
        )
        if abs(side_width0) > 1e-12 and abs(side_height0) > 1e-12:
            side_strain_x = side_width / side_width0 - 1.0
            side_strain_y = side_height / side_height0 - 1.0
    return {
        "frame_progress": float(frame_idx / max(len(sim) - 1, 1)),
        "box_strain_x": strain_x,
        "box_strain_y": strain_y,
        "box_area_strain": float(width * height / (width0 * height0) - 1.0),
        "side_strain_x": side_strain_x,
        "side_strain_y": side_strain_y,
        "affine_xx": float(affine_coef[1, 0]),
        "affine_yy": float(affine_coef[2, 1]),
        "rms_dx": float(np.sqrt(np.mean(displacement[:, 0] ** 2))),
        "rms_dy": float(np.sqrt(np.mean(displacement[:, 1] ** 2))),
        "rms_disp": float(np.sqrt(np.mean(displacement_norm**2))),
        "nonaffine_rms": float(
            np.sqrt(np.mean(np.sum(nonaffine**2, axis=1)))
        ),
        "edge_stretch_mean": float(np.mean(stretch)),
        "edge_stretch_std": float(np.std(stretch)),
        "edge_rel_stretch_mean": float(np.mean(relative_stretch)),
        "spring_energy_proxy": float(np.mean(stiffness * stretch**2)),
    }


def _bounded_p_ratio(dx: float, dy: float, eps: float = 1e-12) -> float:
    if not np.isfinite(dx) or not np.isfinite(dy):
        return np.nan
    if abs(dx) <= eps or abs(dy) <= eps:
        return np.nan
    direct = -dy / dx
    reciprocal = -dx / dy
    return float(direct if abs(direct) < abs(reciprocal) else reciprocal)


def _finite_mean_median(values) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return np.nan, np.nan
    return float(np.mean(finite)), float(np.median(finite))


def decompose_latent_correlations(
    frame: pd.DataFrame,
    *,
    metric_columns: dict[str, str],
    latent_columns: list[str] | None = None,
    trajectory_columns: tuple[str, ...] = ("split", "sim_idx"),
    between_latent_summary: str = "mean",
) -> pd.DataFrame:
    """Separate pooled frame correlations into within- and between-trajectory parts.

    ``within`` correlates trajectory-demeaned frame values. ``between`` gives
    one point to every trajectory. Its latent descriptor is either the temporal
    mean (the default) or the initial latent code; the metric is temporally
    averaged. Constant per-trajectory quantities such as final p-ratio therefore
    have a meaningful between correlation and an intentionally undefined within
    correlation.
    """

    if latent_columns is None:
        latent_columns = sorted(
            (
                column
                for column in frame.columns
                if column.startswith("z") and column[1:].isdigit()
            ),
            key=lambda column: int(column[1:]),
        )
    group_keys = list(trajectory_columns)
    rows = []
    for latent_column in latent_columns:
        for metric_label, metric_column in metric_columns.items():
            pair = frame[group_keys + [latent_column, metric_column]].replace(
                [np.inf, -np.inf], np.nan
            )
            pooled = pair[[latent_column, metric_column]].dropna()

            within = pair.copy()
            within[latent_column] = within[latent_column] - within.groupby(
                group_keys
            )[latent_column].transform("mean")
            within[metric_column] = within[metric_column] - within.groupby(
                group_keys
            )[metric_column].transform("mean")
            within = within[[latent_column, metric_column]].dropna()

            if between_latent_summary == "mean":
                between = (
                    pair.groupby(group_keys, as_index=False)[
                        [latent_column, metric_column]
                    ]
                    .mean()
                    .dropna()
                )
                between_level = "between trajectories (mean z)"
            elif between_latent_summary == "initial":
                ordered = frame[group_keys + [latent_column, metric_column] + (
                    ["frame_idx"] if "frame_idx" in frame.columns else []
                )].replace([np.inf, -np.inf], np.nan)
                if "frame_idx" in ordered.columns:
                    ordered = ordered.sort_values(group_keys + ["frame_idx"])
                initial_z = ordered.dropna(subset=[latent_column]).groupby(
                    group_keys, as_index=False
                )[latent_column].first()
                metric_mean = ordered.groupby(group_keys, as_index=False)[
                    metric_column
                ].mean()
                between = initial_z.merge(metric_mean, on=group_keys).dropna()
                between_level = "between trajectories (initial z)"
            else:
                raise ValueError(
                    "between_latent_summary must be either 'mean' or 'initial'."
                )
            for level, values in (
                ("pooled", pooled),
                ("within trajectory", within),
                (between_level, between),
            ):
                x = values[latent_column].to_numpy(float)
                y = values[metric_column].to_numpy(float)
                correlation = (
                    float(np.corrcoef(x, y)[0, 1])
                    if len(values) >= 2
                    and not np.isclose(np.std(x), 0.0)
                    and not np.isclose(np.std(y), 0.0)
                    else np.nan
                )
                rows.append(
                    {
                        "coordinate": latent_column,
                        "metric": metric_label,
                        "metric_column": metric_column,
                        "level": level,
                        "pearson_r": correlation,
                        "n": int(len(values)),
                    }
                )
    return pd.DataFrame(rows)


def plot_latent_correlation_decomposition(
    table: pd.DataFrame,
    *,
    metric_order: list[str],
    title: str,
):
    """Plot pooled, within-trajectory, and between-trajectory heatmaps."""

    import matplotlib.pyplot as plt

    print(title)
    between_levels = [
        level
        for level in table["level"].drop_duplicates().tolist()
        if level.startswith("between trajectories")
    ]
    levels = ["pooled", "within trajectory", *between_levels[:1]]
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(12.2, 3.2),
        constrained_layout=True,
        sharey=True,
    )
    image = None
    for ax, level in zip(axes, levels):
        heat = table[table["level"].eq(level)].pivot(
            index="coordinate",
            columns="metric",
            values="pearson_r",
        )
        heat = heat.reindex(columns=metric_order)
        values = heat.to_numpy(float)
        image = ax.imshow(
            values,
            cmap="RdBu_r",
            vmin=-1.0,
            vmax=1.0,
            aspect="auto",
        )
        ax.grid(False, which="both")
        ax.set_xlabel(level)
        ax.set_xticks(range(heat.shape[1]))
        ax.set_xticklabels(heat.columns, rotation=40, ha="right")
        ax.set_yticks(range(heat.shape[0]))
        ax.set_yticklabels(heat.index)
        for row in range(heat.shape[0]):
            for column in range(heat.shape[1]):
                value = values[row, column]
                ax.text(
                    column,
                    row,
                    "—" if not np.isfinite(value) else f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=(
                        "white"
                        if np.isfinite(value) and abs(value) > 0.55
                        else "0.15"
                    ),
                )
    fig.colorbar(image, ax=axes, pad=0.015, fraction=0.025, label="Pearson r")
    return fig


def network_level_latent_correlations(
    descriptors: pd.DataFrame,
    *,
    metric_columns: dict[str, str],
) -> pd.DataFrame:
    """Correlate one latent descriptor with one target value per network."""

    feature_columns = sorted(
        [
            column
            for column in descriptors.columns
            if column.startswith("z")
            and (column.endswith("_initial") or column.endswith("_slope"))
        ],
        key=lambda column: (
            0 if column.endswith("_initial") else 1,
            int(column[1:].split("_")[0]),
        ),
    )
    rows = []
    for feature in feature_columns:
        for metric_label, metric_column in metric_columns.items():
            values = descriptors[[feature, metric_column]].replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            x = values[feature].to_numpy(float)
            y = values[metric_column].to_numpy(float)
            correlation = (
                float(np.corrcoef(x, y)[0, 1])
                if len(values) >= 2
                and not np.isclose(np.std(x), 0.0)
                and not np.isclose(np.std(y), 0.0)
                else np.nan
            )
            rows.append(
                {
                    "descriptor": feature,
                    "metric": metric_label,
                    "metric_column": metric_column,
                    "pearson_r": correlation,
                    "n_networks": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def per_trajectory_latent_correlations(
    frame: pd.DataFrame,
    *,
    metric_columns: dict[str, str],
    latent_columns: list[str] | None = None,
    trajectory_columns: tuple[str, ...] = ("split", "sim_idx"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute framewise correlations inside each trajectory, then summarize."""

    if latent_columns is None:
        latent_columns = sorted(
            [
                column
                for column in frame.columns
                if column.startswith("z") and column[1:].isdigit()
            ],
            key=lambda column: int(column[1:]),
        )
    group_keys = list(trajectory_columns)
    rows = []
    for trajectory_key, group in frame.groupby(group_keys, sort=False):
        if not isinstance(trajectory_key, tuple):
            trajectory_key = (trajectory_key,)
        identifiers = dict(zip(group_keys, trajectory_key))
        for latent_column in latent_columns:
            for metric_label, metric_column in metric_columns.items():
                values = group[[latent_column, metric_column]].replace(
                    [np.inf, -np.inf], np.nan
                ).dropna()
                x = values[latent_column].to_numpy(float)
                y = values[metric_column].to_numpy(float)
                correlation = (
                    float(np.corrcoef(x, y)[0, 1])
                    if len(values) >= 3
                    and not np.isclose(np.std(x), 0.0)
                    and not np.isclose(np.std(y), 0.0)
                    else np.nan
                )
                rows.append(
                    {
                        **identifiers,
                        "coordinate": latent_column,
                        "metric": metric_label,
                        "metric_column": metric_column,
                        "pearson_r": correlation,
                        "n_frames": int(len(values)),
                    }
                )
    per_trajectory = pd.DataFrame(rows)
    finite = per_trajectory.dropna(subset=["pearson_r"])
    summary = finite.groupby(
        ["coordinate", "metric", "metric_column"], as_index=False
    )["pearson_r"].agg(
        mean_r="mean",
        median_r="median",
        std_r="std",
        n_trajectories="count",
    )
    return per_trajectory, summary


def plot_correlation_heatmap(
    table: pd.DataFrame,
    *,
    row: str,
    column: str,
    value: str,
    row_order: list[str] | None = None,
    column_order: list[str] | None = None,
    title: str | None = None,
):
    """Plot a compact annotated correlation heatmap."""

    import matplotlib.pyplot as plt

    if title:
        print(title)
    heat = table.pivot(index=row, columns=column, values=value)
    if row_order is not None:
        heat = heat.reindex(row_order)
    if column_order is not None:
        heat = heat.reindex(columns=column_order)
    values = np.abs(heat.to_numpy(float))
    width = max(4.2, 1.15 * heat.shape[1] + 1.8)
    height = max(2.6, 0.55 * heat.shape[0] + 1.5)
    fig, ax = plt.subplots(figsize=(width, height), constrained_layout=True)
    image = ax.imshow(values, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.grid(False, which="both")
    ax.set_xticks(range(heat.shape[1]), heat.columns, rotation=35, ha="right")
    ax.set_yticks(range(heat.shape[0]), heat.index)
    for y_index in range(heat.shape[0]):
        for x_index in range(heat.shape[1]):
            current = values[y_index, x_index]
            ax.text(
                x_index, y_index,
                "—" if not np.isfinite(current) else f"{current:.2f}",
                ha="center", va="center", fontsize=8,
                color="white" if np.isfinite(current) and current > 0.65 else "0.15",
            )
    fig.colorbar(image, ax=ax, pad=.02, fraction=.045, label="|Pearson r|")
    return fig


def plot_network_descriptor_scatters(
    descriptors: pd.DataFrame,
    *,
    case_order: list[str] | None = None,
    case_labels: dict[str, str] | None = None,
    target_column: str = "final_p_ratio",
):
    """Plot every initial/slope latent against one target value per network."""

    import matplotlib.pyplot as plt

    print("Network-level latent descriptors versus final p-ratio; rows are datasets and columns are descriptors.")
    test = descriptors[descriptors["split"].eq("test")].copy()
    if case_order is None:
        case_order = test["case"].drop_duplicates().tolist()
    features = sorted(
        [
            column
            for column in test.columns
            if column.startswith("z")
            and (column.endswith("_initial") or column.endswith("_slope"))
        ],
        key=lambda column: (
            0 if column.endswith("_initial") else 1,
            int(column[1:].split("_")[0]),
        ),
    )
    fig, axes = plt.subplots(
        len(case_order), len(features),
        figsize=(3.35 * len(features), 3.0 * len(case_order)),
        squeeze=False, constrained_layout=True,
    )
    rows = []
    for row_index, case in enumerate(case_order):
        case_frame = test[test["case"].eq(case)]
        label = (
            case_labels.get(case, case) if case_labels is not None
            else str(case_frame["case_label"].iloc[0])
        )
        for column_index, feature in enumerate(features):
            ax = axes[row_index, column_index]
            values = case_frame[[feature, target_column]].replace(
                [np.inf, -np.inf], np.nan
            ).dropna()
            x = values[feature].to_numpy(float)
            y = values[target_column].to_numpy(float)
            x_std = np.std(x)
            x_display = (x - np.mean(x)) / x_std if x_std > 1e-12 else x * np.nan
            correlation = (
                float(np.corrcoef(x, y)[0, 1])
                if len(values) >= 2 and x_std > 1e-12 and np.std(y) > 1e-12
                else np.nan
            )
            ax.scatter(x_display, y, s=22, alpha=.7, edgecolor="none")
            if len(values) >= 2 and np.all(np.isfinite(x_display)):
                slope, intercept = np.polyfit(x_display, y, 1)
                xx = np.linspace(float(x_display.min()), float(x_display.max()), 100)
                ax.plot(xx, slope * xx + intercept, color="0.2", ls="--", lw=1)
            ax.text(
                .03, .97, f"r={correlation:.2f}", transform=ax.transAxes,
                ha="left", va="top", fontsize=8,
            )
            ax.set_xlabel(f"standardized {feature}")
            ax.set_ylabel(f"{label}\nfinal p-ratio")
            rows.append(
                {
                    "case": case,
                    "case_label": label,
                    "descriptor": feature,
                    "target": target_column,
                    "pearson_r": correlation,
                    "n_networks": int(len(values)),
                }
            )
    return fig, pd.DataFrame(rows)


def _trajectory_p_ratio_columns(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("frame_idx").copy()
    strain_x = group["side_strain_x"].to_numpy(float)
    strain_y = group["side_strain_y"].to_numpy(float)
    frame_count = len(group)

    endpoint = np.asarray(
        [_bounded_p_ratio(dx, dy) for dx, dy in zip(strain_x, strain_y)],
        dtype=float,
    )
    cumulative = np.full(frame_count, np.nan, dtype=float)
    rolling = {
        window: np.full(frame_count, np.nan, dtype=float)
        for window in (5, 10, 20, 40)
    }
    for stop in range(2, frame_count):
        time = np.arange(stop + 1, dtype=float)
        x_rate = np.polyfit(time, strain_x[: stop + 1], deg=1)[0]
        y_rate = np.polyfit(time, strain_y[: stop + 1], deg=1)[0]
        cumulative[stop] = _bounded_p_ratio(x_rate, y_rate, eps=1e-10)
        for window, values in rolling.items():
            start = max(0, stop - window + 1)
            if stop - start + 1 < 3:
                continue
            local_time = np.arange(stop - start + 1, dtype=float)
            x_rate = np.polyfit(local_time, strain_x[start : stop + 1], deg=1)[0]
            y_rate = np.polyfit(local_time, strain_y[start : stop + 1], deg=1)[0]
            values[stop] = _bounded_p_ratio(x_rate, y_rate, eps=1e-10)

    smooth_x = pd.Series(strain_x).rolling(9, center=True, min_periods=5).mean()
    smooth_y = pd.Series(strain_y).rolling(9, center=True, min_periods=5).mean()
    local_x_rate = np.gradient(smooth_x.to_numpy(float))
    local_y_rate = np.gradient(smooth_y.to_numpy(float))
    local = np.asarray(
        [
            _bounded_p_ratio(dx, dy, eps=1e-8)
            for dx, dy in zip(local_x_rate, local_y_rate)
        ],
        dtype=float,
    )
    full_time = np.arange(frame_count, dtype=float)
    full_x_rate = np.polyfit(full_time, strain_x, deg=1)[0]
    full_y_rate = np.polyfit(full_time, strain_y, deg=1)[0]

    group["side_endpoint_p_ratio"] = endpoint
    group["side_cumulative_p_ratio"] = cumulative
    for window, values in rolling.items():
        group[f"side_rolling_{window}_p_ratio"] = values
    group["side_local_p_ratio"] = local
    group["side_final_trajectory_p_ratio"] = _bounded_p_ratio(
        full_x_rate,
        full_y_rate,
        eps=1e-10,
    )
    return group


def framewise_latent_descriptor_sweep(
    result: dict,
    *,
    device,
    frame_stride: int = 1,
    max_frames_per_sim: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Fit latent readouts using an optional evenly sampled frame budget."""

    params = result["params"]
    ae_model = result["ae"]
    ae_model.eval()
    stride = max(1, int(frame_stride))
    rows = []
    with torch.no_grad():
        for split_name in ("val", "test"):
            for sim_idx, sim in enumerate(result[f"{split_name}_data"]):
                if not sim:
                    continue
                side_indices = directional_side_indices_from_box(sim[0], quantile=0.1)
                ref_pos = sim[0].x[:, :2].detach().cpu().numpy()
                side_reference_dimensions = (
                    float(
                        np.mean(ref_pos[side_indices["right"], 0])
                        - np.mean(ref_pos[side_indices["left"], 0])
                    ),
                    float(
                        np.mean(ref_pos[side_indices["top"], 1])
                        - np.mean(ref_pos[side_indices["bottom"], 1])
                    ),
                )
                frame_ids = list(range(0, len(sim), stride))
                if frame_ids[-1] != len(sim) - 1:
                    frame_ids.append(len(sim) - 1)
                if max_frames_per_sim is not None:
                    frame_budget = max(2, int(max_frames_per_sim))
                    if len(frame_ids) > frame_budget:
                        sample_indices = np.linspace(
                            0,
                            len(frame_ids) - 1,
                            num=frame_budget,
                            dtype=int,
                        )
                        frame_ids = [frame_ids[idx] for idx in np.unique(sample_indices)]
                for frame_idx in frame_ids:
                    z = (
                        encode_frame_latent(
                            ae_model,
                            sim,
                            frame_idx,
                            pos_dim=int(params["pos_dim"]),
                            node_feature_mode=params["node_feature_mode"],
                            normalizers=result["normalizers"],
                            device=device,
                        )
                        .detach()
                        .cpu()
                        .numpy()
                        .reshape(-1)
                    )
                    rows.append(
                        {
                            "split": split_name,
                            "sim_idx": int(sim_idx),
                            "source": str(
                                getattr(
                                    sim[0],
                                    "source_name",
                                    result.get("source_name", result["label"]),
                                )
                            ),
                            "frame_idx": int(frame_idx),
                            "temperature": float(
                                getattr(sim[0], "temperature", np.nan)
                            ),
                            **{f"z{idx}": float(value) for idx, value in enumerate(z)},
                            **_frame_descriptor_row(
                                sim,
                                frame_idx,
                                side_indices=side_indices,
                                side_reference_dimensions=side_reference_dimensions,
                            ),
                        }
                    )

    frame_df = pd.DataFrame(rows).sort_values(["split", "sim_idx", "frame_idx"])
    frame_df = (
        frame_df.groupby(["split", "sim_idx"])
        .apply(_trajectory_p_ratio_columns, include_groups=False)
        .reset_index()
        .drop(columns="level_2", errors="ignore")
    )
    z_cols = sorted(
        [col for col in frame_df if col.startswith("z")],
        key=lambda col: int(col[1:]),
    )
    for z_col in z_cols:
        frame_df[f"d{z_col}"] = frame_df.groupby(["split", "sim_idx"])[z_col].transform(
            lambda values: values - values.iloc[0]
        )
    if len(z_cols) == 2:
        frame_df["z0_sq"] = frame_df[z_cols[0]] ** 2
        frame_df["z0_z1"] = frame_df[z_cols[0]] * frame_df[z_cols[1]]
        frame_df["z1_sq"] = frame_df[z_cols[1]] ** 2
        frame_df["dz0_sq"] = frame_df[f"d{z_cols[0]}"] ** 2
        frame_df["dz0_dz1"] = frame_df[f"d{z_cols[0]}"] * frame_df[f"d{z_cols[1]}"]
        frame_df["dz1_sq"] = frame_df[f"d{z_cols[1]}"] ** 2

    targets = [
        "affine_yy",
        "box_strain_y",
        "box_area_strain",
        "affine_xx",
        "rms_dx",
        "rms_disp",
        "edge_stretch_mean",
        "edge_rel_stretch_mean",
        "nonaffine_rms",
        "edge_stretch_std",
        "spring_energy_proxy",
        "frame_progress",
        "side_strain_x",
        "side_strain_y",
        "side_endpoint_p_ratio",
        "side_cumulative_p_ratio",
        "side_rolling_5_p_ratio",
        "side_rolling_10_p_ratio",
        "side_rolling_20_p_ratio",
        "side_rolling_40_p_ratio",
        "side_local_p_ratio",
        "side_final_trajectory_p_ratio",
    ]
    feature_sets = {
        "latent state": z_cols,
        "latent change": [f"d{col}" for col in z_cols],
    }
    if len(z_cols) == 2:
        feature_sets["quadratic latent state"] = [
            *z_cols,
            "z0_sq",
            "z0_z1",
            "z1_sq",
        ]
        feature_sets["quadratic latent change"] = [
            *(f"d{col}" for col in z_cols),
            "dz0_sq",
            "dz0_dz1",
            "dz1_sq",
        ]
    score_rows = []
    prediction_parts = []
    weight_rows = []
    for feature_label, feature_cols in feature_sets.items():
        for target in targets:
            columns = feature_cols + [target]
            val = (
                frame_df.loc[frame_df["split"].eq("val"), columns]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            test_columns = ["sim_idx", "source", "frame_idx", "temperature", *columns]
            test = (
                frame_df.loc[frame_df["split"].eq("test"), test_columns]
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if len(val) < len(feature_cols) + 3 or len(test) < 3:
                continue
            model = fit_linear(val[feature_cols], val[target])
            predicted = predict_linear(model, test[feature_cols])
            within_network = []
            for _, group in test.assign(predicted=predicted).groupby("sim_idx"):
                within_network.append(pearson_r(group[target], group["predicted"]))
            within_mean, within_median = _finite_mean_median(within_network)
            score_rows.append(
                {
                    "feature_set": feature_label,
                    "target": target,
                    "test_r2": r2_score(test[target], predicted),
                    "test_pearson_r": pearson_r(test[target], predicted),
                    "mean_within_network_r": within_mean,
                    "median_within_network_r": within_median,
                    "n_val": int(len(val)),
                    "n_test": int(len(test)),
                }
            )
            prediction_parts.append(
                test[["sim_idx", "source", "frame_idx", "temperature", target]]
                .rename(columns={target: "observed"})
                .assign(
                    feature_set=feature_label,
                    target=target,
                    predicted=predicted,
                )
            )
            weight_rows.append(
                {
                    "feature_set": feature_label,
                    "target": target,
                    "intercept": model["raw_intercept"],
                    **{
                        f"{column}_weight": float(weight)
                        for column, weight in zip(feature_cols, model["raw_coef"])
                    },
                }
            )

    p_ratio_targets = [
        "side_endpoint_p_ratio",
        "side_cumulative_p_ratio",
        "side_rolling_5_p_ratio",
        "side_rolling_10_p_ratio",
        "side_rolling_20_p_ratio",
        "side_rolling_40_p_ratio",
        "side_local_p_ratio",
        "side_final_trajectory_p_ratio",
    ]
    for feature_label, feature_cols in feature_sets.items():
        val = frame_df[frame_df["split"].eq("val")].replace(
            [np.inf, -np.inf],
            np.nan,
        )
        test = frame_df[frame_df["split"].eq("test")].replace(
            [np.inf, -np.inf],
            np.nan,
        )
        val = val.dropna(subset=[*feature_cols, "side_strain_x", "side_strain_y"])
        test = test.dropna(subset=feature_cols).copy()
        x_model = fit_linear(val[feature_cols], val["side_strain_x"])
        y_model = fit_linear(val[feature_cols], val["side_strain_y"])
        test["side_strain_x"] = predict_linear(x_model, test[feature_cols])
        test["side_strain_y"] = predict_linear(y_model, test[feature_cols])
        predicted_parts = []
        for (split_name, sim_idx), group in test.groupby(["split", "sim_idx"]):
            predicted = _trajectory_p_ratio_columns(group)
            predicted_parts.append(predicted)
        predicted_frame = pd.concat(predicted_parts).sort_index()

        for target in p_ratio_targets:
            observed = frame_df.loc[predicted_frame.index, target]
            predicted = predicted_frame[target]
            valid = np.isfinite(observed) & np.isfinite(predicted)
            if int(valid.sum()) < 3:
                continue
            observed = observed[valid]
            predicted = predicted[valid]
            metadata = frame_df.loc[
                observed.index,
                ["sim_idx", "source", "frame_idx", "temperature"],
            ]
            within_network = [
                pearson_r(group["observed"], group["predicted"])
                for _, group in metadata.assign(
                    observed=observed,
                    predicted=predicted,
                ).groupby("sim_idx")
            ]
            within_mean, within_median = _finite_mean_median(within_network)
            score_rows.append(
                {
                    "feature_set": f"{feature_label} via side strains",
                    "target": target,
                    "test_r2": r2_score(observed, predicted),
                    "test_pearson_r": pearson_r(observed, predicted),
                    "mean_within_network_r": within_mean,
                    "median_within_network_r": within_median,
                    "n_val": int(len(val)),
                    "n_test": int(valid.sum()),
                }
            )
            prediction_parts.append(
                metadata.assign(
                    observed=observed,
                    feature_set=f"{feature_label} via side strains",
                    target=target,
                    predicted=predicted,
                )
            )

    score_df = pd.DataFrame(score_rows).sort_values(
        ["test_r2", "mean_within_network_r"],
        ascending=False,
    )
    prediction_df = (
        pd.concat(prediction_parts, ignore_index=True)
        if prediction_parts
        else pd.DataFrame()
    )
    return frame_df, score_df, prediction_df, pd.DataFrame(weight_rows)


__all__ = [
    "CVAnalysisContext",
    "evaluate_readouts",
    "fit_linear",
    "framewise_latent_descriptor_sweep",
    "path_curvature_metrics",
    "pearson_r",
    "predict_linear",
    "r2_score",
    "residualize",
]
