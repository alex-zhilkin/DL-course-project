"""AE-only latent edit experiment for low-temperature dePablo networks.

This script supports the notebook:
``notebooks/latent_space/06_low_temp_auxetic_latent_edit.ipynb``.

It trains or loads a low-temperature dePablo autoencoder, learns a linear
initial-latent readout for final p-ratio, moves a non-auxetic network in the
latent direction that lowers p-ratio, decodes each edited latent with the
original graph context, and renders the decoded path as a GIF.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
import pandas as pd
import torch
from graph_utils import calc_p_ratio_rollout_sides

from lss.graph import box_tensor, clone_graph
from lss.latent.analysis import framewise_latent_descriptor_sweep
from lss.latent.experiment import (
    find_project_root,
    ground_truth_p_ratio,
    run_latent_experiment,
    seed_everything,
)
from lss.latent.training import decode_latent_to_graph, encode_frame_latent
from lss.utils import resolve_device


BASE_SEED = 20260708
DATASET_NAME = "depablo_low_temp"
DATASET_LABEL = "dePablo low temperature"
DATASET_PATH = "data/depablo-near-zero-temp.pt"


@dataclass(frozen=True)
class AuxeticDirection:
    """Linear readout and edit direction in standardized latent space."""

    z_columns: tuple[str, ...]
    z_mean: np.ndarray
    z_std: np.ndarray
    coef_standardized: np.ndarray
    intercept_standardized: float
    auxetic_direction_standardized: np.ndarray
    validation_r: float
    validation_r2: float


def make_config(
    *,
    project_root: str | Path | None = None,
    latent_dim: int = 2,
    train_count: int = 80,
    val_count: int = 30,
    train_frames: int = 80,
    max_epochs: int = 80,
    force_train: bool = False,
) -> tuple[dict, dict, Path]:
    """Return ``(source_spec, cfg, output_dir)`` for the AE-only experiment."""

    root = find_project_root(project_root)
    dataset_path = root / DATASET_PATH
    output_dir = root / "notebooks" / "results" / "06_low_temp_auxetic_latent_edit"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "dataset_name": DATASET_NAME,
        "split_seed": BASE_SEED,
        "split_stratify_temperature": False,
        "min_train_p_ratio": None,
        "pos_dim": 2,
        "batch_graphs": 4,
        "frame_skip": 1,
        "train_frame_start_order": 0,
        "latent_dim": int(latent_dim),
        "latent_tokens": 32,
        "hidden_size": 96,
        "autoencoder_model": "attention",
        "ae_max_train_frames_per_sim": int(train_frames),
        "dyn_max_train_transitions_per_sim": int(train_frames),
        "ae_target_mode": "normalized_delta",
        "node_feature_mode": "normalized_delta",
        "ae_max_epochs": int(max_epochs),
        "ae_patience": 8,
        "ae_lr": 5e-5,
        "ae_weight_decay": 1e-5,
        "dyn_max_epochs": 1,
        "dyn_patience": 1,
        "dyn_lr": 3e-5,
        "dyn_weight_decay": 1e-4,
        "early_stop_min_delta": 1e-5,
        "should_rollout": False,
        "should_train_propagator": False,
        "force_train": bool(force_train),
        "cache_path": str(
            output_dir
            / (
                f"ae_cv{int(latent_dim)}_tr{int(train_count)}_"
                f"val{int(val_count)}_frames{int(train_frames)}.pt"
            )
        ),
        "model_seed": BASE_SEED + 1009 * int(latent_dim),
        "repeat_idx": 1,
    }
    source_spec = {
        "dataset_name": DATASET_NAME,
        "source_name": DATASET_LABEL,
        "label": f"{DATASET_LABEL} CV{latent_dim} AE",
        "path": str(dataset_path),
        "target_mode": cfg["ae_target_mode"],
        "ae_target_mode": cfg["ae_target_mode"],
        "node_feature_mode": cfg["node_feature_mode"],
        "latent_dim": int(latent_dim),
        "repeat_idx": 1,
        "model_seed": int(cfg["model_seed"]),
        "latent_tokens": int(cfg["latent_tokens"]),
        "hidden_size": int(cfg["hidden_size"]),
        "train_count": int(train_count),
        "val_count": int(val_count),
    }
    return source_spec, cfg, output_dir


def train_or_load_autoencoder(
    *,
    project_root: str | Path | None = None,
    latent_dim: int = 2,
    train_count: int = 80,
    val_count: int = 30,
    train_frames: int = 80,
    max_epochs: int = 80,
    force_train: bool = False,
    device: str | torch.device = "auto",
) -> tuple[dict, Path]:
    """Train or load the low-temperature AE experiment."""

    seed_everything(BASE_SEED)
    resolved_device = resolve_device(str(device)) if str(device) == "auto" else torch.device(device)
    source_spec, cfg, output_dir = make_config(
        project_root=project_root,
        latent_dim=latent_dim,
        train_count=train_count,
        val_count=val_count,
        train_frames=train_frames,
        max_epochs=max_epochs,
        force_train=force_train,
    )
    cfg["device"] = str(resolved_device)
    result = run_latent_experiment(source_spec, cfg, device=resolved_device)
    return result, output_dir


def latent_frame_table(
    result: dict,
    *,
    device: str | torch.device,
    frame_stride: int = 4,
    max_frames_per_sim: int = 80,
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Encode validation/test frames and cache the table if requested."""

    frame_df, scores, predictions, weights = framewise_latent_descriptor_sweep(
        result,
        device=device,
        frame_stride=frame_stride,
        max_frames_per_sim=max_frames_per_sim,
    )
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        frame_df.to_csv(output / "latent_frame_table.csv", index=False)
        scores.to_csv(output / "latent_readout_scores.csv", index=False)
        predictions.to_csv(output / "latent_readout_predictions.csv", index=False)
        weights.to_csv(output / "latent_readout_weights.csv", index=False)
    return frame_df


def initial_latent_table(frame_df: pd.DataFrame) -> pd.DataFrame:
    """One row per held-out network with initial latent and final p-ratio."""

    return (
        frame_df.sort_values(["split", "sim_idx", "frame_idx"])
        .groupby(["split", "sim_idx"], as_index=False)
        .first()
    )


def fit_auxetic_direction(frame_df: pd.DataFrame) -> AuxeticDirection:
    """Fit final p-ratio from initial z, then point opposite the p-ratio gradient."""

    initial = initial_latent_table(frame_df)
    z_columns = tuple(
        sorted(
            [col for col in initial.columns if col.startswith("z") and col[1:].isdigit()],
            key=lambda col: int(col[1:]),
        )
    )
    if not z_columns:
        raise ValueError("No latent columns were found in frame_df.")

    train = (
        initial[initial["split"].eq("val")]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=[*z_columns, "side_final_trajectory_p_ratio"])
    )
    if len(train) < len(z_columns) + 3:
        raise ValueError("Not enough validation networks to fit an auxetic direction.")

    z = train.loc[:, z_columns].to_numpy(float)
    y = train["side_final_trajectory_p_ratio"].to_numpy(float)
    z_mean = z.mean(axis=0)
    z_std = z.std(axis=0)
    z_std[z_std < 1e-8] = 1.0
    z_scaled = (z - z_mean) / z_std
    design = np.column_stack([np.ones(len(z_scaled)), z_scaled])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    intercept = float(coef[0])
    weights = coef[1:].astype(float)
    norm = float(np.linalg.norm(weights))
    if norm < 1e-12:
        raise ValueError("The fitted p-ratio direction has near-zero norm.")
    pred = design @ coef
    residual = y - pred
    total = y - y.mean()
    r2 = 1.0 - float(np.sum(residual**2) / max(np.sum(total**2), 1e-12))
    r = float(np.corrcoef(y, pred)[0, 1]) if len(y) > 1 else float("nan")
    return AuxeticDirection(
        z_columns=z_columns,
        z_mean=z_mean,
        z_std=z_std,
        coef_standardized=weights,
        intercept_standardized=intercept,
        auxetic_direction_standardized=-weights / norm,
        validation_r=r,
        validation_r2=r2,
    )


def predict_p_ratio_from_z(z: np.ndarray, direction: AuxeticDirection) -> float:
    """Predict p-ratio from a raw latent vector using the fitted readout."""

    z_scaled = (np.asarray(z, dtype=float) - direction.z_mean) / direction.z_std
    return float(direction.intercept_standardized + z_scaled @ direction.coef_standardized)


def choose_non_auxetic_case(
    result: dict,
    frame_df: pd.DataFrame,
    *,
    split: str = "test",
) -> tuple[int, object, pd.Series]:
    """Pick the held-out network with the largest final p-ratio."""

    initial = initial_latent_table(frame_df)
    candidates = (
        initial[initial["split"].eq(split)]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["side_final_trajectory_p_ratio"])
        .sort_values("side_final_trajectory_p_ratio", ascending=False)
    )
    if candidates.empty:
        raise ValueError(f"No candidate networks found for split={split!r}.")
    row = candidates.iloc[0]
    sim_idx = int(row["sim_idx"])
    return sim_idx, result[f"{split}_data"][sim_idx], row


def decode_auxetic_edit_path(
    result: dict,
    sim,
    direction: AuxeticDirection,
    *,
    device: str | torch.device,
    edit_strength: float = 4.0,
    frames: int = 25,
) -> tuple[list, pd.DataFrame]:
    """Move the initial latent in the learned auxetic direction and decode frames."""

    params = result["params"]
    ae = result["ae"]
    normalizers = result["normalizers"]
    ae.eval()
    z0_t = encode_frame_latent(
        ae,
        sim,
        0,
        pos_dim=int(params["pos_dim"]),
        node_feature_mode=params["node_feature_mode"],
        normalizers=normalizers,
        device=device,
    ).detach()
    z0 = z0_t.cpu().numpy().reshape(-1)
    z0_scaled = (z0 - direction.z_mean) / direction.z_std
    steps = np.linspace(0.0, float(edit_strength), int(frames))
    decoded = [clone_graph(sim[0]).cpu()]
    rows = [
        {
            "frame": 0,
            "edit_step": 0.0,
            "decoded_p_ratio": float("nan"),
            "predicted_p_ratio": predict_p_ratio_from_z(z0, direction),
            **{col: float(value) for col, value in zip(direction.z_columns, z0)},
        }
    ]
    for frame_idx, step in enumerate(steps[1:], start=1):
        z_scaled = z0_scaled + step * direction.auxetic_direction_standardized
        z = direction.z_mean + z_scaled * direction.z_std
        z_t = torch.as_tensor(z, dtype=z0_t.dtype, device=device)
        graph = decode_latent_to_graph(
            ae,
            sim,
            z_t,
            0,
            pos_dim=int(params["pos_dim"]),
            ae_target_mode=params["ae_target_mode"],
            normalizers=normalizers,
            device=device,
        )
        decoded.append(graph)
        rows.append(
            {
                "frame": int(frame_idx),
                "edit_step": float(step),
                "decoded_p_ratio": float(calc_p_ratio_rollout_sides([decoded[0], graph], -1)),
                "predicted_p_ratio": predict_p_ratio_from_z(z, direction),
                **{col: float(value) for col, value in zip(direction.z_columns, z)},
            }
        )
    return decoded, pd.DataFrame(rows)


def recompute_rest_lengths_from_geometry(graph):
    """Return a graph whose bond vectors/rest lengths match its current positions."""

    out = clone_graph(graph)
    pos = out.x[:, :2].float()
    if hasattr(out, "box") and out.box is not None:
        box = out.box
        if all(hasattr(box, attr) for attr in ("x1", "x2", "y1", "y2")):
            xlo, xhi = float(box.x1), float(box.x2)
            ylo, yhi = float(box.y1), float(box.y2)
            width = max(xhi - xlo, 1e-8)
            height = max(yhi - ylo, 1e-8)
            pos = pos.clone()
            pos[:, 0] = ((pos[:, 0] - xlo) % width) + xlo
            pos[:, 1] = ((pos[:, 1] - ylo) % height) + ylo
            out.x = out.x.clone().float()
            out.x[:, :2] = pos
    row, col = out.edge_index.long()
    vectors = pos[col] - pos[row]
    box = box_tensor(out, device=pos.device, dtype=pos.dtype)
    if box is not None:
        vectors = vectors - torch.round(vectors / box.reshape(1, 2)) * box.reshape(1, 2)
    lengths = torch.linalg.vector_norm(vectors, dim=1, keepdim=True).clamp_min(1e-8)
    stiffness = out.edge_attr[:, -1:].clone().float()
    out.edge_attr = torch.cat([vectors, lengths, stiffness], dim=1)
    out.pos = pos.clone()
    return out.cpu()


def select_metaforge_validation_frames(frame_count: int, selection: str | list[int] | tuple[int, ...]) -> list[int]:
    """Choose decoded path frames to validate with LAMMPS."""

    if frame_count <= 0:
        return []
    if isinstance(selection, str):
        value = selection.lower()
        if value in {"final", "last"}:
            return [frame_count - 1]
        if value in {"first_final", "initial_final", "endpoints"}:
            return sorted({0, frame_count - 1})
        if value in {"all", "*"}:
            return list(range(frame_count))
        raise ValueError(f"Unknown MetaForge validation frame selection: {selection!r}")
    indices = []
    for index in selection:
        idx = int(index)
        if idx < 0:
            idx += frame_count
        if 0 <= idx < frame_count:
            indices.append(idx)
    return sorted(set(indices))


def validate_decoded_path_with_metaforge(
    decoded_path: list,
    output_dir: str | Path,
    *,
    frames: str | list[int] | tuple[int, ...] = "endpoints",
    lammps_cmd: str = "lmp",
    fast_screening: bool = True,
    rest_lengths: str = "decoded_geometry",
    raise_on_error: bool = False,
) -> pd.DataFrame:
    """Run MetaForge/LAMMPS elastic validation for selected decoded designs."""

    metaforge_src = Path.home() / "MetaForge" / "src"
    if str(metaforge_src) not in sys.path:
        sys.path.insert(0, str(metaforge_src))
    loaded_auxetic = sys.modules.get("auxetic")
    loaded_auxetic_file = getattr(loaded_auxetic, "__file__", "") if loaded_auxetic is not None else ""
    if loaded_auxetic is not None and not str(loaded_auxetic_file).startswith(str(metaforge_src)):
        for module_name in list(sys.modules):
            if module_name == "auxetic" or module_name.startswith("auxetic."):
                del sys.modules[module_name]

    from metaforge.optimization.latent import network_from_bidirected_data
    from metaforge.simulations import ElasticRunConfig, FastElasticSettings, run_elastic_network

    output = Path(output_dir)
    parent_dir = output / "metaforge_lammps"
    parent_dir.mkdir(parents=True, exist_ok=True)
    config = ElasticRunConfig(
        lammps_cmd=str(lammps_cmd),
        screen="none",
        mass=1e6,
        angles=0.0,
        fast_screening=bool(fast_screening),
        fast=FastElasticSettings(
            maxiter=1000,
            maxeval=4000,
            ftol=1e-8,
            dmax=5e-2,
            srate=1e-3,
        ),
    )

    rows = []
    for frame_idx in select_metaforge_validation_frames(len(decoded_path), frames):
        source_graph = decoded_path[frame_idx]
        if rest_lengths == "decoded_geometry":
            graph = recompute_rest_lengths_from_geometry(source_graph)
        elif rest_lengths == "original_context":
            graph = clone_graph(source_graph).cpu()
        else:
            raise ValueError(
                "rest_lengths must be 'decoded_geometry' or 'original_context'."
            )
        try:
            network = network_from_bidirected_data(graph)
            result = run_elastic_network(
                network,
                parent_dir,
                f"decoded_frame_{frame_idx:03d}",
                config=config,
            )
            rows.append(
                {
                    "frame": int(frame_idx),
                    "metaforge_status": "ok",
                    "rest_lengths": rest_lengths,
                    **{
                        key: result.get(key)
                        for key in (
                            "metaforge_p_ratio",
                            "bulk_modulus",
                            "shear_modulus",
                            "run_dir",
                            "fast_screening",
                            "reused_topology",
                            "lj_active",
                        )
                    },
                    "error": "",
                }
            )
        except Exception as exc:
            if raise_on_error:
                raise
            rows.append(
                {
                    "frame": int(frame_idx),
                    "metaforge_status": "error",
                    "rest_lengths": rest_lengths,
                    "metaforge_p_ratio": float("nan"),
                    "bulk_modulus": float("nan"),
                    "shear_modulus": float("nan"),
                    "run_dir": str(parent_dir / f"decoded_frame_{frame_idx:03d}"),
                    "fast_screening": bool(fast_screening),
                    "reused_topology": False,
                    "lj_active": False,
                    "error": repr(exc),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "metaforge_validation.csv", index=False)
    return frame


def graph_edges_for_plot(graph) -> list[tuple[int, int]]:
    """Return unique non-periodic-looking edges for a readable network plot."""

    pos = graph.x[:, :2].detach().cpu().numpy()
    span = np.ptp(pos, axis=0)
    edge_index = graph.edge_index.detach().cpu().numpy().T
    edges: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for source, target in edge_index:
        i, j = sorted((int(source), int(target)))
        if i == j or (i, j) in seen:
            continue
        delta = np.abs(pos[j] - pos[i])
        if np.any(delta > 0.5 * np.maximum(span, 1e-8)):
            continue
        seen.add((i, j))
        edges.append((i, j))
    return edges


def save_decoded_gif(
    decoded_path: list,
    p_ratio_table: pd.DataFrame,
    output_path: str | Path,
    *,
    fps: int = 8,
) -> Path:
    """Render the decoded latent edit path as a GIF."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    ref = decoded_path[0]
    ref_pos = ref.x[:, :2].detach().cpu().numpy()
    edges = graph_edges_for_plot(ref)
    all_pos = np.concatenate(
        [graph.x[:, :2].detach().cpu().numpy() for graph in decoded_path],
        axis=0,
    )
    margin = 0.05 * np.maximum(np.ptp(all_pos, axis=0), 1e-6)
    xlim = (float(all_pos[:, 0].min() - margin[0]), float(all_pos[:, 0].max() + margin[0]))
    ylim = (float(all_pos[:, 1].min() - margin[1]), float(all_pos[:, 1].max() + margin[1]))

    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    lines = [ax.plot([], [], color="#9aa0a6", linewidth=0.5, alpha=0.55)[0] for _ in edges]
    points = ax.scatter([], [], s=8, color="#1f77b4", zorder=3)
    title = ax.set_title("")

    def update(frame_idx: int):
        pos = decoded_path[frame_idx].x[:, :2].detach().cpu().numpy()
        for line, (i, j) in zip(lines, edges):
            line.set_data([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]])
        points.set_offsets(pos)
        row = p_ratio_table.iloc[frame_idx]
        pr = row["decoded_p_ratio"]
        pr_text = "initial" if not np.isfinite(pr) else f"p-ratio={pr:.3f}"
        title.set_text(f"latent edit step={row['edit_step']:.2f} | {pr_text}")
        return [*lines, points, title]

    animation = FuncAnimation(fig, update, frames=len(decoded_path), interval=1000 / fps, blit=False)
    animation.save(output, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return output


def run_low_temp_auxetic_edit(
    *,
    project_root: str | Path | None = None,
    latent_dim: int = 2,
    train_count: int = 80,
    val_count: int = 30,
    train_frames: int = 80,
    max_epochs: int = 80,
    force_train: bool = False,
    edit_strength: float = 4.0,
    edit_frames: int = 25,
    run_metaforge_validation: bool = False,
    metaforge_validation_frames: str | list[int] | tuple[int, ...] = "endpoints",
    metaforge_lammps_cmd: str = "lmp",
    metaforge_fast_screening: bool = True,
    metaforge_rest_lengths: str = "decoded_geometry",
    device: str | torch.device = "auto",
) -> dict:
    """End-to-end training, latent edit, p-ratio table, and GIF generation."""

    result, output_dir = train_or_load_autoencoder(
        project_root=project_root,
        latent_dim=latent_dim,
        train_count=train_count,
        val_count=val_count,
        train_frames=train_frames,
        max_epochs=max_epochs,
        force_train=force_train,
        device=device,
    )
    resolved_device = resolve_device(str(device)) if str(device) == "auto" else torch.device(device)
    frame_df = latent_frame_table(
        result,
        device=resolved_device,
        frame_stride=4,
        max_frames_per_sim=80,
        output_dir=output_dir,
    )
    direction = fit_auxetic_direction(frame_df)
    sim_idx, sim, source_row = choose_non_auxetic_case(result, frame_df, split="test")
    decoded_path, p_ratio_table = decode_auxetic_edit_path(
        result,
        sim,
        direction,
        device=resolved_device,
        edit_strength=edit_strength,
        frames=edit_frames,
    )
    true_final_p_ratio = ground_truth_p_ratio(
        sim,
        -1,
        dataset_name=DATASET_NAME,
        cfg=result["params"],
    )
    p_ratio_table["source_sim_idx"] = sim_idx
    p_ratio_table["source_true_final_p_ratio"] = float(true_final_p_ratio)
    p_ratio_table["source_initial_readout_p_ratio"] = float(source_row["side_final_trajectory_p_ratio"])
    p_ratio_table.to_csv(output_dir / "decoded_auxetic_edit_p_ratio.csv", index=False)
    gif_path = save_decoded_gif(
        decoded_path,
        p_ratio_table,
        output_dir / f"decoded_auxetic_edit_sim_{sim_idx:03d}.gif",
    )
    metaforge_validation = pd.DataFrame()
    if run_metaforge_validation:
        metaforge_validation = validate_decoded_path_with_metaforge(
            decoded_path,
            output_dir,
            frames=metaforge_validation_frames,
            lammps_cmd=metaforge_lammps_cmd,
            fast_screening=metaforge_fast_screening,
            rest_lengths=metaforge_rest_lengths,
        )
        if not metaforge_validation.empty:
            final_frame = int(p_ratio_table["frame"].iloc[-1])
            matched = metaforge_validation[
                metaforge_validation["frame"].eq(final_frame)
                & metaforge_validation["metaforge_status"].eq("ok")
            ]
            if not matched.empty:
                p_ratio_table.loc[
                    p_ratio_table["frame"].eq(final_frame),
                    "metaforge_p_ratio",
                ] = float(matched["metaforge_p_ratio"].iloc[0])
                p_ratio_table.to_csv(output_dir / "decoded_auxetic_edit_p_ratio.csv", index=False)
    summary = {
        "output_dir": str(output_dir),
        "gif_path": str(gif_path),
        "selected_test_sim_idx": int(sim_idx),
        "source_true_final_p_ratio": float(true_final_p_ratio),
        "validation_readout_r": float(direction.validation_r),
        "validation_readout_r2": float(direction.validation_r2),
        "initial_decoded_p_ratio": float(p_ratio_table["decoded_p_ratio"].dropna().iloc[0])
        if p_ratio_table["decoded_p_ratio"].notna().any()
        else float("nan"),
        "final_decoded_p_ratio": float(p_ratio_table["decoded_p_ratio"].dropna().iloc[-1]),
        "final_predicted_p_ratio": float(p_ratio_table["predicted_p_ratio"].iloc[-1]),
        "metaforge_validated": bool(run_metaforge_validation),
    }
    if run_metaforge_validation and not metaforge_validation.empty:
        final_ok = metaforge_validation[metaforge_validation["metaforge_status"].eq("ok")].tail(1)
        if not final_ok.empty:
            summary["final_metaforge_p_ratio"] = float(final_ok["metaforge_p_ratio"].iloc[0])
            summary["final_metaforge_bulk_modulus"] = float(final_ok["bulk_modulus"].iloc[0])
            summary["final_metaforge_shear_modulus"] = float(final_ok["shear_modulus"].iloc[0])
    pd.DataFrame([summary]).to_csv(output_dir / "decoded_auxetic_edit_summary.csv", index=False)
    return {
        "result": result,
        "frame_df": frame_df,
        "direction": direction,
        "selected_sim_idx": sim_idx,
        "selected_sim": sim,
        "decoded_path": decoded_path,
        "p_ratio_table": p_ratio_table,
        "metaforge_validation": metaforge_validation,
        "summary": summary,
    }


if __name__ == "__main__":
    outputs = run_low_temp_auxetic_edit(force_train=False)
    print(pd.DataFrame([outputs["summary"]]).to_string(index=False))
