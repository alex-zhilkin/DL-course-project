"""Train/load a one-step latent propagator and create compact rollout diagnostics."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from lss.latent.analysis import framewise_latent_descriptor_sweep
from lss.latent.capacity import evaluate_experiment
from lss.latent.experiment import (
    evaluate_autoencoder_reconstruction_horizons,
    evaluate_rollout_horizons,
    find_project_root,
    resolve_existing_path,
    result_tables,
    run_latent_experiment,
    save_result_tables,
    seed_everything,
)
from lss.latent.simulation import edge_features
from lss.latent.training import encode_frame_latent
from lss.plotting import PAPER_COLORS, apply_editorial_style, style_axes
from lss.utils import resolve_device


BASE_SEED = 20260623
DATASET_NAME = "depablo_mixed_temp"
DATASET_LABEL = "dePablo mixed temperature"
DATASET_PATH = "data/depablo-10k-mix-temp.pt"
MIXED_TEMPERATURE_MIN_P_RATIO_REPORT_STEP = 50


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent-dim", type=int, choices=[1, 2], required=True)
    parser.add_argument("--train-count", type=int, required=True)
    parser.add_argument("--holdout-train-count", type=int, default=None)
    parser.add_argument("--val-count", type=int, default=15)
    parser.add_argument("--train-frames", type=int, default=50)
    parser.add_argument("--model-seed", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force-train", action="store_true")
    return parser.parse_args()


def coordinate_summary(frame):
    z_cols = [col for col in ("z0", "z1") if col in frame]
    rows = []
    for z_col in z_cols:
        rows.append(
            {
                "coordinate": z_col,
                "p_ratio_r": frame[z_col].corr(frame["side_final_trajectory_p_ratio"]),
                "progress_r": frame[z_col].corr(frame["frame_progress"]),
                "side_strain_x_r": frame[z_col].corr(frame["side_strain_x"]),
                "side_strain_y_r": frame[z_col].corr(frame["side_strain_y"]),
                "temperature_r": frame[z_col].corr(frame["temperature"]),
            }
        )
    summary = pd.DataFrame(rows)
    if len(z_cols) == 2:
        summary["z0_z1_r"] = frame["z0"].corr(frame["z1"])
        summary["dz0_dz1_r"] = frame["dz0"].corr(frame["dz1"])
    return summary


def select_initial_pratio_coordinate(frame):
    """Select z0 or z1 using initial-frame validation correlation only."""

    initial = (
        frame.sort_values(["sim_idx", "frame_idx"])
        .groupby("sim_idx", as_index=False)
        .first()
    )
    rows = []
    for coordinate in ("z0", "z1"):
        correlation = initial[coordinate].corr(
            initial["side_final_trajectory_p_ratio"]
        )
        rows.append(
            {
                "coordinate": coordinate,
                "validation_pearson_r": correlation,
                "validation_abs_pearson_r": abs(correlation),
            }
        )
    selection = pd.DataFrame(rows).sort_values(
        ["validation_abs_pearson_r", "coordinate"],
        ascending=[False, True],
    )
    selection["selected"] = False
    selection.loc[selection.index[0], "selected"] = True
    return selection.reset_index(drop=True)


def node_latent_sensitivity(result, sim, *, latent_scale, frame_indices, device):
    """Measure each node output's response to typical z0 and z1 changes."""

    ae = result["ae"]
    params = result["params"]
    normalizers = result["normalizers"]
    pos_dim = int(params["pos_dim"])
    ref = sim[0]
    ref_pos = ref.x[:, :pos_dim].to(device).float()
    edge_index = ref.edge_index.to(device).long()
    ref_edge_attr_norm = (
        edge_features(ref, ref, pos_dim=pos_dim, device=device)
        - normalizers["edge_mean"].to(device)
    ) / normalizers["edge_std"].to(device)
    batch = torch.zeros(ref_pos.size(0), dtype=torch.long, device=device)
    with torch.no_grad():
        h0 = ae.encode_reference_graph(ref_pos, ref_edge_attr_norm, edge_index)

    target_std = normalizers["target_std"].to(device)
    target_mean = normalizers["target_mean"].to(device)
    reference_span = (ref_pos.max(dim=0).values - ref_pos.min(dim=0).values).clamp_min(1e-6)
    target_mode = str(params["ae_target_mode"]).lower()
    latent_scale = torch.as_tensor(latent_scale, dtype=ref_pos.dtype, device=device)

    def decode_physical_displacement(z_value):
        target_norm = ae.decode(z_value.unsqueeze(0), h0, batch)
        target = target_norm * target_std + target_mean
        if target_mode in {"normalized_delta", "self_normalized_delta", "relative_delta"}:
            target = target * reference_span.reshape(1, -1)
        return target

    jacobians = []
    for frame_idx in frame_indices:
        z = encode_frame_latent(
            ae,
            sim,
            int(frame_idx),
            pos_dim=pos_dim,
            node_feature_mode=params["node_feature_mode"],
            normalizers=normalizers,
            device=device,
        ).detach()
        jacobian = torch.func.jacfwd(decode_physical_displacement)(z)
        jacobians.append(jacobian.abs() * latent_scale.reshape(1, 1, -1))

    sensitivity = torch.stack(jacobians).mean(dim=0).detach().cpu().numpy()
    ref_xy = ref_pos.detach().cpu().numpy()
    node_rows = []
    for node_idx, (reference_x, reference_y) in enumerate(ref_xy):
        row = {
            "node_idx": node_idx,
            "reference_x": reference_x,
            "reference_y": reference_y,
            "frames": ",".join(str(int(value)) for value in frame_indices),
        }
        for axis_idx, axis in enumerate(("x", "y")):
            z0_use, z1_use = sensitivity[node_idx, axis_idx, :2]
            total = z0_use + z1_use
            row[f"{axis}_z0_sensitivity"] = z0_use
            row[f"{axis}_z1_sensitivity"] = z1_use
            row[f"{axis}_total_sensitivity"] = total
            row[f"{axis}_z1_fraction"] = z1_use / total if total > 0 else np.nan
            row[f"{axis}_dominant_latent"] = "z1" if z1_use > z0_use else "z0"
        node_rows.append(row)

    edge_rows = []
    seen = set()
    for source, target in ref.edge_index.detach().cpu().numpy().T:
        source, target = sorted((int(source), int(target)))
        if source == target or (source, target) in seen:
            continue
        seen.add((source, target))
        delta = np.abs(ref_xy[target] - ref_xy[source])
        # Wrapped periodic bonds are omitted because a straight line across the
        # box would obscure the network rendering.
        if np.any(delta > 0.5 * np.ptp(ref_xy, axis=0)):
            continue
        edge_rows.append({"source": source, "target": target})
    return pd.DataFrame(node_rows), pd.DataFrame(
        edge_rows, columns=["source", "target"]
    )


def reported_p_ratio_stats(stats):
    """Keep thermally unreliable early horizons out of p-ratio reports."""

    test = stats[stats["split"].eq("test")].sort_values("rollout_steps")
    if DATASET_NAME == "depablo_mixed_temp":
        test = test[
            test["rollout_steps"].ge(MIXED_TEMPERATURE_MIN_P_RATIO_REPORT_STEP)
        ]
    return test


def plot_rollout(stats, output_path):
    test = stats[stats["split"].eq("test")].sort_values("rollout_steps")
    p_ratio_test = reported_p_ratio_stats(stats)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1), constrained_layout=True)
    axes[0].plot(
        test["rollout_steps"],
        test["rollout_position_r2"],
        color=PAPER_COLORS["blue"],
        marker="o",
        lw=2,
    )
    axes[0].axhline(0.0, color=PAPER_COLORS["ink"], lw=1, ls="--")
    style_axes(axes[0], xlabel="rollout horizon", ylabel="position rollout R2")
    axes[1].plot(
        p_ratio_test["rollout_steps"],
        p_ratio_test["p_ratio_r2"],
        color=PAPER_COLORS["red"],
        marker="o",
        lw=2,
    )
    axes[1].axhline(0.0, color=PAPER_COLORS["ink"], lw=1, ls="--")
    style_axes(
        axes[1],
        xlabel="rollout horizon (step 50 onward)",
        ylabel="p-ratio rollout R2",
    )
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_coordinates(frame, stats, output_path):
    if "z1" not in frame:
        return
    rng = np.random.default_rng(BASE_SEED + 17)
    sim_ids = np.asarray(sorted(frame["sim_idx"].unique()), dtype=int)
    selected = sorted(rng.choice(sim_ids, size=min(10, len(sim_ids)), replace=False))
    traces = frame[frame["sim_idx"].isin(selected)]
    test_stats = stats[stats["split"].eq("test")].sort_values("rollout_steps")

    fig, axes = plt.subplots(1, 3, figsize=(14.8, 4.2), constrained_layout=True)
    for sim_idx in selected:
        group = traces[traces["sim_idx"].eq(sim_idx)].sort_values("frame_idx")
        for z_col, color in (("z0", PAPER_COLORS["blue"]), ("z1", PAPER_COLORS["red"])):
            values = group[z_col].to_numpy(float)
            standardized = (values - values.mean()) / max(values.std(), 1e-12)
            axes[0].plot(
                group["frame_progress"],
                standardized,
                color=color,
                alpha=0.28,
                lw=1,
            )
    axes[0].plot([], [], color=PAPER_COLORS["blue"], label="z0")
    axes[0].plot([], [], color=PAPER_COLORS["red"], label="z1")
    axes[0].legend(frameon=False)
    style_axes(axes[0], xlabel="trajectory progress", ylabel="standardized coordinate")

    points = axes[1].scatter(
        frame["z0"],
        frame["z1"],
        c=frame["frame_progress"],
        cmap="viridis",
        s=8,
        alpha=0.3,
        linewidth=0,
    )
    fig.colorbar(points, ax=axes[1], pad=0.02, label="trajectory progress")
    style_axes(axes[1], xlabel="z0", ylabel="z1")

    axes[2].plot(
        test_stats["rollout_steps"],
        test_stats["rollout_position_r2"],
        color=PAPER_COLORS["blue"],
        marker="o",
        lw=2,
    )
    axes[2].axhline(0.0, color=PAPER_COLORS["ink"], lw=1, ls="--")
    style_axes(axes[2], xlabel="rollout horizon", ylabel="position rollout R2")
    for label, ax in zip(("a", "b", "c"), axes):
        ax.text(
            -0.13,
            1.03,
            label,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="bottom",
        )
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_analysis(
    *,
    latent_dim,
    train_count,
    holdout_train_count=None,
    val_count=15,
    train_frames=50,
    model_seed=None,
    device_name="auto",
    force_train=False,
    analysis_max_sims=None,
):
    train_count = int(train_count)
    if train_count < 1:
        raise ValueError("train_count must be a positive integer.")
    if holdout_train_count is not None and int(holdout_train_count) < train_count:
        raise ValueError(
            "holdout_train_count must be at least train_count so matched runs "
            "can share the same held-out test networks."
        )
    apply_editorial_style()
    project_root = find_project_root()
    device = resolve_device(device_name)
    model_seed = (
        int(model_seed)
        if model_seed is not None
        else BASE_SEED + 1009 * int(latent_dim)
    )
    seed_everything(model_seed)
    dataset_path = str(resolve_existing_path(DATASET_PATH, project_root=project_root))
    holdout_train_count = (
        int(train_count)
        if holdout_train_count is None
        else int(holdout_train_count)
    )
    run_name = f"train_{int(train_count)}"
    output_dir = (
        project_root / "notebooks" / "results" / "04a_latent_space_analysis" / run_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = {
        "dataset_name": DATASET_NAME,
        "split_seed": BASE_SEED,
        "split_stratify_temperature": False,
        "min_train_p_ratio": None,
        "device": str(device),
        "pos_dim": 2,
        "batch_graphs": 4,
        "frame_skip": 1,
        "train_frame_start_order": 0,
        "latent_dim": int(latent_dim),
        "latent_tokens": 32,
        "hidden_size": 96,
        "autoencoder_model": "attention",
        "edge_feature_dim": 12,
        "ae_max_train_frames_per_sim": int(train_frames),
        "dyn_max_train_transitions_per_sim": int(train_frames),
        "ae_target_mode": "normalized_delta",
        "node_feature_mode": "normalized_delta",
        "ae_max_epochs": 60,
        "ae_patience": 6,
        "ae_lr": 5e-5,
        "ae_weight_decay": 1e-5,
        "dyn_max_epochs": 60,
        "dyn_patience": 6,
        "dyn_lr": 3e-5,
        "dyn_weight_decay": 1e-4,
        "propagator_use_static_context": True,
        "graph_context_dim": 16,
        "propagator_context_include_temperature": False,
        "propagator_step_stride": 1,
        "initial_velocity": "zero",
        "early_stop_min_delta": 1e-5,
        "rollout_steps_grid": [50, 100, 150, 199],
        "rollout_eval_max_sims_per_split": 20,
        "temperature_pratio_window": "full",
        "temperature_pratio_estimator": "robust",
        "temperature_pratio_min_fit_frames": 8,
        "temperature_pratio_min_driven_strain_range": 1e-3,
        "temperature_pratio_smooth_window": 5,
        "should_rollout": True,
        "should_train_propagator": True,
        "force_train": bool(force_train),
        "cache_path": str(output_dir / "model.pt"),
        "propagator_objective": "one_step",
        "propagator_model": "delta_mlp",
        "propagator_loss": "delta",
        "propagator_standardize_latent": False,
        "model_seed": model_seed,
        "repeat_idx": 1,
    }
    # Temperature metadata may be reported after rollout, but must never be
    # appended to the autoencoder or propagator inputs in this experiment.
    if bool(cfg["propagator_context_include_temperature"]):
        raise ValueError("04a must not provide temperature to the simulator.")
    mixture_entry = {
        "name": DATASET_NAME,
        "label": DATASET_LABEL,
        "path": dataset_path,
        "train_count": int(train_count),
        "val_count": int(val_count),
    }
    if holdout_train_count != int(train_count):
        mixture_entry["holdout_train_count"] = holdout_train_count
    mixture = [mixture_entry]
    source_spec = {
        "dataset_name": DATASET_NAME,
        "source_name": DATASET_LABEL,
        "label": f"{DATASET_LABEL} CV{latent_dim} pyramid attention",
        "path": dataset_path,
        "dataset_mixture": mixture,
        "target_mode": "normalized_delta",
        "ae_target_mode": "normalized_delta",
        "node_feature_mode": "normalized_delta",
        "latent_dim": int(latent_dim),
        "repeat_idx": 1,
        "model_seed": model_seed,
        "latent_tokens": 32,
        "hidden_size": 96,
        "autoencoder_model": "attention",
        "edge_feature_dim": 12,
        "ae_max_train_frames_per_sim": int(train_frames),
        "dyn_max_train_transitions_per_sim": int(train_frames),
        "ae_max_epochs": 60,
        "ae_patience": 6,
        "ae_lr": 5e-5,
        "ae_weight_decay": 1e-5,
        "dyn_max_epochs": 60,
        "dyn_patience": 6,
        "dyn_lr": 3e-5,
        "dyn_weight_decay": 1e-4,
    }

    result = run_latent_experiment(source_spec, cfg, device=device)
    if result["rollout_stats"].empty:
        result = evaluate_experiment(result, cfg, device=device)
    ae_stats_existing = result.get("ae_reconstruction_stats", pd.DataFrame())
    if (
        ae_stats_existing.empty
        or "rollout_steps" not in ae_stats_existing
        or not ae_stats_existing["rollout_steps"].eq(1).any()
    ):
        ae_rows, ae_stats = evaluate_autoencoder_reconstruction_horizons(
            result["ae"],
            result["test_data"][: int(cfg["rollout_eval_max_sims_per_split"])],
            cfg=cfg,
            normalizers=result["normalizers"],
            dataset=result["label"],
            split_name="test",
            rollout_steps=[1, *cfg["rollout_steps_grid"]],
            device=device,
        )
        result["ae_reconstruction_rows"] = ae_rows
        result["ae_reconstruction_stats"] = ae_stats
    tables = result_tables([result])
    save_result_tables(tables, output_dir)
    result["split_info"].to_csv(output_dir / "split_info.csv", index=False)
    plot_rollout(result["rollout_stats"], output_dir / "rollout_position_r2.png")

    if int(latent_dim) == 2:
        all_test_path = output_dir / "all_test_rollout_rows_50.csv"
        all_test_stats_path = output_dir / "all_test_rollout_stats_50.csv"
        refresh_all_test = bool(force_train) or not all_test_path.exists()
        if refresh_all_test:
            print(
                f"evaluating 50-step rollout on all {len(result['test_data'])} "
                "held-out test networks"
            )
            all_test_rows, all_test_stats = evaluate_rollout_horizons(
                result["ae"],
                result["dyn"],
                result["test_data"],
                result["latent_stats"],
                cfg=result["params"],
                normalizers=result["normalizers"],
                dataset=result["label"],
                split_name="test",
                rollout_steps=[50],
                device=device,
            )
            all_test_rows.to_csv(all_test_path, index=False)
            all_test_stats.to_csv(all_test_stats_path, index=False)

    if latent_dim == 1:
        print(f"output_dir: {output_dir}")
        print(f"cache_path: {result.get('cache_path')}")
        test_stats = result["rollout_stats"].query("split == 'test'")
        print("position rollout (all horizons)")
        print(test_stats[["rollout_steps", "rollout_position_r2", "rollout_error_fraction"]].to_string(index=False))
        print("p-ratio rollout (step 50 onward for mixed temperature)")
        print(reported_p_ratio_stats(result["rollout_stats"])[["rollout_steps", "p_ratio_r2"]].to_string(index=False))
        return output_dir

    analysis_result = dict(result)
    if analysis_max_sims is not None:
        analysis_result["val_data"] = result["val_data"][: int(analysis_max_sims)]
        analysis_result["test_data"] = result["test_data"][: int(analysis_max_sims)]
    frame_df, _, _, _ = framewise_latent_descriptor_sweep(
        analysis_result,
        device=device,
        frame_stride=2,
        max_frames_per_sim=100,
    )
    val_frame = frame_df[frame_df["split"].eq("val")].copy()
    test_frame = frame_df[frame_df["split"].eq("test")].copy()
    val_frame.to_csv(output_dir / "val_latent_frames.csv", index=False)
    test_frame.to_csv(output_dir / "test_latent_frames.csv", index=False)
    coordinate_selection = select_initial_pratio_coordinate(val_frame)
    coordinate_selection.to_csv(
        output_dir / "selected_initial_pratio_coordinate.csv", index=False
    )
    latent_scale = np.nan_to_num(
        val_frame[["z0", "z1"]].std(ddof=0).to_numpy(float),
        nan=0.0,
    )
    test_pratio = (
        test_frame.sort_values(["sim_idx", "frame_idx"])
        .groupby("sim_idx", as_index=False)
        .first()[["sim_idx", "side_final_trajectory_p_ratio"]]
    )
    extreme_cases = {
        "very auxetic": test_pratio.loc[
            test_pratio["side_final_trajectory_p_ratio"].idxmin()
        ],
        "least auxetic": test_pratio.loc[
            test_pratio["side_final_trajectory_p_ratio"].idxmax()
        ],
    }
    sensitivity_parts = []
    edge_parts = []
    for case, case_row in extreme_cases.items():
        sim_idx = int(case_row["sim_idx"])
        p_ratio = float(case_row["side_final_trajectory_p_ratio"])
        representative_sim = result["test_data"][sim_idx]
        sensitivity_frames = sorted(
            {
                min(step, len(representative_sim) - 1)
                for step in cfg["rollout_steps_grid"]
            }
        )
        node_sensitivity, node_edges = node_latent_sensitivity(
            result,
            representative_sim,
            latent_scale=latent_scale,
            frame_indices=sensitivity_frames,
            device=device,
        )
        for frame in (node_sensitivity, node_edges):
            frame["case"] = case
            frame["sim_idx"] = sim_idx
            frame["final_p_ratio"] = p_ratio
            frame["autoencoder_model"] = result["params"]["autoencoder_model"]
        sensitivity_parts.append(node_sensitivity)
        edge_parts.append(node_edges)
    pd.concat(sensitivity_parts, ignore_index=True).to_csv(
        output_dir / "node_latent_sensitivity.csv", index=False
    )
    pd.concat(edge_parts, ignore_index=True).to_csv(
        output_dir / "node_latent_edges.csv", index=False
    )
    coordinate_stats = coordinate_summary(test_frame)
    coordinate_stats.to_csv(output_dir / "coordinate_summary.csv", index=False)
    plot_coordinates(
        test_frame,
        result["rollout_stats"],
        output_dir / "latent_coordinates_and_rollout.png",
    )

    print(f"output_dir: {output_dir}")
    print(f"cache_path: {result.get('cache_path')}")
    print("initial p-ratio coordinate selected on validation data")
    print(coordinate_selection.to_string(index=False))
    print(coordinate_stats.to_string(index=False))
    test_stats = result["rollout_stats"].query("split == 'test'")
    print("position rollout (all horizons)")
    print(test_stats[["rollout_steps", "rollout_position_r2", "rollout_error_fraction"]].to_string(index=False))
    print("p-ratio rollout (step 50 onward for mixed temperature)")
    print(reported_p_ratio_stats(result["rollout_stats"])[["rollout_steps", "p_ratio_r2"]].to_string(index=False))
    return output_dir


def main():
    args = parse_args()
    run_analysis(
        latent_dim=args.latent_dim,
        train_count=args.train_count,
        holdout_train_count=args.holdout_train_count,
        val_count=args.val_count,
        train_frames=args.train_frames,
        model_seed=args.model_seed,
        device_name=args.device,
        force_train=args.force_train,
        analysis_max_sims=None,
    )


if __name__ == "__main__":
    main()
