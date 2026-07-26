"""Train/load a 1D latent autoencoder and analyze temperature-dependent z0 jitter."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lss.latent.analysis import framewise_latent_descriptor_sweep
from lss.latent.experiment import (
    find_project_root,
    resolve_existing_path,
    run_latent_experiment,
    seed_everything,
)
from lss.latent.simulation import pearson_r, r2_score
from lss.plotting import PAPER_COLORS, apply_editorial_style, style_axes
from lss.utils import resolve_device


BASE_SEED = 20260623
TRAIN_COUNT = 2
VAL_COUNT = 20
RANDOM_TRACE_COUNT = 20
DATASET_NAME = "depablo_mixed_temp"
DATASET_LABEL = "dePablo mixed temperature"
DATASET_PATH = "data/depablo-10k-mix-temp.pt"


def linear_r2(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or np.isclose(np.std(x[mask]), 0.0):
        return float("nan")
    slope, intercept = np.polyfit(x[mask], y[mask], deg=1)
    return r2_score(y[mask], slope * x[mask] + intercept)


def trajectory_temperature_table(frame_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sim_idx, group in frame_df.sort_values(["sim_idx", "frame_idx"]).groupby("sim_idx"):
        time = group["frame_idx"].to_numpy(float)
        z0 = group["z0"].to_numpy(float)
        z0_delta = np.diff(z0)
        slope, intercept = np.polyfit(time, z0, deg=1)
        residual = z0 - (slope * time + intercept)
        rows.append(
            {
                "sim_idx": int(sim_idx),
                "temperature": float(group["temperature"].iloc[0]),
                "final_p_ratio": float(group["side_final_trajectory_p_ratio"].iloc[0]),
                "n_frames": int(len(group)),
                "z0_std": float(np.std(z0)),
                "z0_step_std": float(np.std(z0_delta)),
                "z0_abs_step_mean": float(np.mean(np.abs(z0_delta))),
                "z0_linear_residual_std": float(np.std(residual)),
                "z0_linear_slope": float(slope),
                "z0_slope": float(slope),
            }
        )
    return pd.DataFrame(rows)


def correlation_table(trajectory_df: pd.DataFrame, train_temperatures: list[float]) -> pd.DataFrame:
    metrics = [
        "z0_std",
        "z0_step_std",
        "z0_abs_step_mean",
        "z0_linear_residual_std",
    ]
    rows = []
    for scope, frame in (
        ("all test temperatures", trajectory_df),
        (
            "temperature levels unseen in training",
            trajectory_df[~trajectory_df["temperature"].isin(train_temperatures)],
        ),
    ):
        for metric in metrics:
            clean = frame[[metric, "temperature"]].dropna()
            rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "n": int(len(clean)),
                    "pearson_r": pearson_r(clean[metric], clean["temperature"]),
                    "spearman_r": float(clean[metric].corr(clean["temperature"], method="spearman")),
                    "r2_linear": linear_r2(clean[metric], clean["temperature"]),
                }
            )
    return pd.DataFrame(rows)


def plot_temperature_generalization(
    trajectory_df: pd.DataFrame,
    train_temperatures: list[float],
    output_path: Path,
) -> None:
    specs = [
        ("z0_std", "std(z0)", "Raw trajectory spread"),
        ("z0_linear_residual_std", "std(detrended z0)", "Spread after removing linear slope"),
        ("z0_step_std", "std(delta z0)", "Frame-to-frame jitter"),
    ]
    summary = (
        trajectory_df.groupby("temperature", as_index=False)
        .agg(
            z0_std_mean=("z0_std", "mean"),
            z0_std_sem=("z0_std", "sem"),
            z0_linear_residual_std_mean=("z0_linear_residual_std", "mean"),
            z0_linear_residual_std_sem=("z0_linear_residual_std", "sem"),
            z0_step_std_mean=("z0_step_std", "mean"),
            z0_step_std_sem=("z0_step_std", "sem"),
        )
    )
    summary["seen"] = summary["temperature"].isin(train_temperatures)
    unseen_trajectories = trajectory_df[
        ~trajectory_df["temperature"].isin(train_temperatures)
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.4), constrained_layout=True)
    for ax, (metric, ylabel, title) in zip(axes, specs):
        ax.scatter(
            trajectory_df["temperature"],
            trajectory_df[metric],
            color=PAPER_COLORS["light"],
            s=17,
            alpha=0.3,
            linewidth=0,
        )
        for seen, marker, color, label in (
            (False, "o", PAPER_COLORS["blue"], "unseen temperature"),
            (True, "s", PAPER_COLORS["red"], "temperature used in training"),
        ):
            group = summary[summary["seen"].eq(seen)]
            ax.errorbar(
                group["temperature"],
                group[f"{metric}_mean"],
                yerr=group[f"{metric}_sem"],
                fmt=marker,
                ms=7,
                color=color,
                capsize=3,
                label=label,
            )
        for temperature in train_temperatures:
            ax.axvline(temperature, color=PAPER_COLORS["red"], lw=0.9, ls=":", alpha=0.55)
        all_r = pearson_r(trajectory_df[metric], trajectory_df["temperature"])
        unseen_r = pearson_r(unseen_trajectories[metric], unseen_trajectories["temperature"])
        ax.set_title(f"{title}\nall r={all_r:.3f}, unseen-T r={unseen_r:.3f}")
        style_axes(ax, xlabel="temperature", ylabel=ylabel)
    axes[-1].legend(frameon=False, fontsize=8)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_random_traces(frame_df: pd.DataFrame, output_path: Path) -> None:
    rng = np.random.default_rng(BASE_SEED + 17)
    sim_ids = np.asarray(sorted(frame_df["sim_idx"].unique()), dtype=int)
    selected = sorted(
        rng.choice(sim_ids, size=min(RANDOM_TRACE_COUNT, len(sim_ids)), replace=False).tolist()
    )
    plot_df = frame_df[frame_df["sim_idx"].isin(selected)].copy()
    p_ratio_by_sim = plot_df[["sim_idx", "side_final_trajectory_p_ratio"]].drop_duplicates()
    norm = plt.Normalize(
        p_ratio_by_sim["side_final_trajectory_p_ratio"].min(),
        p_ratio_by_sim["side_final_trajectory_p_ratio"].max(),
    )
    cmap = plt.cm.viridis

    fig, ax = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    for sim_idx in selected:
        group = plot_df[plot_df["sim_idx"].eq(sim_idx)].sort_values("frame_idx")
        p_ratio = float(group["side_final_trajectory_p_ratio"].iloc[0])
        ax.plot(
            group["frame_idx"],
            group["z0"],
            color=cmap(norm(p_ratio)),
            lw=1.35,
            alpha=0.82,
        )
    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        pad=0.015,
        label="final p-ratio",
    )
    colorbar.ax.tick_params(labelsize=9)
    style_axes(ax, xlabel="frame", ylabel="z0")
    ax.set_title(f"1D latent traces for {len(selected)} random held-out trajectories")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    apply_editorial_style()
    project_root = find_project_root()
    device = resolve_device("auto")
    seed_everything(BASE_SEED)
    output_dir = (
        project_root
        / "notebooks"
        / "results"
        / "04a_latent_space_analysis"
        / f"single_{DATASET_NAME}_cv1_seed{BASE_SEED}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = resolve_existing_path(DATASET_PATH, project_root=project_root)
    model_seed = BASE_SEED + 1009
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
        "latent_dim": 1,
        "latent_tokens": 32,
        "hidden_size": 96,
        "edge_feature_dim": 12,
        "ae_target_mode": "normalized_delta",
        "node_feature_mode": "normalized_delta",
        "ae_max_train_frames_per_sim": 80,
        "ae_max_epochs": 300,
        "ae_patience": 6,
        "ae_lr": 5e-5,
        "ae_weight_decay": 1e-5,
        "early_stop_min_delta": 1e-5,
        "temperature_pratio_window": "full",
        "temperature_pratio_estimator": "robust",
        "temperature_pratio_min_fit_frames": 8,
        "temperature_pratio_min_driven_strain_range": 1e-3,
        "temperature_pratio_smooth_window": 5,
        "model_seed": model_seed,
        "repeat_idx": 1,
        "should_rollout": False,
        "should_train_propagator": False,
        "force_train": False,
        "cache_dir": str(project_root / "notebooks" / "results" / "latent_model_cache"),
    }
    source_spec = {
        "dataset_name": DATASET_NAME,
        "source_name": DATASET_LABEL,
        "label": f"{DATASET_LABEL} CV1 latent analysis",
        "path": dataset_path,
        "dataset_mixture": [
            {
                "name": DATASET_NAME,
                "label": DATASET_LABEL,
                "path": dataset_path,
                "train_count": TRAIN_COUNT,
                "val_count": VAL_COUNT,
            }
        ],
        "target_mode": cfg["ae_target_mode"],
        "ae_target_mode": cfg["ae_target_mode"],
        "node_feature_mode": cfg["node_feature_mode"],
        "latent_dim": 1,
        "repeat_idx": 1,
        "model_seed": model_seed,
        "latent_tokens": cfg["latent_tokens"],
        "hidden_size": cfg["hidden_size"],
        "edge_feature_dim": cfg["edge_feature_dim"],
        "ae_max_train_frames_per_sim": cfg["ae_max_train_frames_per_sim"],
        "ae_max_epochs": cfg["ae_max_epochs"],
        "ae_patience": cfg["ae_patience"],
        "ae_lr": cfg["ae_lr"],
        "ae_weight_decay": cfg["ae_weight_decay"],
    }

    result = run_latent_experiment(source_spec, cfg, device=device)
    frame_df, _, _, _ = framewise_latent_descriptor_sweep(
        result,
        device=device,
        frame_stride=1,
    )
    val_frame_df = frame_df[frame_df["split"].eq("val")].copy()
    test_frame_df = frame_df[frame_df["split"].eq("test")].copy()
    val_trajectory_df = trajectory_temperature_table(val_frame_df)
    trajectory_df = trajectory_temperature_table(test_frame_df)
    train_temperatures = sorted(
        {float(getattr(sim[0], "temperature", np.nan)) for sim in result["train_data"]}
    )
    correlations = correlation_table(trajectory_df, train_temperatures)

    test_frame_df.to_csv(output_dir / "test_framewise_latent_trajectory_table.csv", index=False)
    trajectory_df.to_csv(output_dir / "test_z0_temperature_jitter_summary.csv", index=False)
    val_trajectory_df.to_csv(output_dir / "paper_audit_val_trajectory_metrics.csv", index=False)
    trajectory_df.to_csv(output_dir / "paper_audit_test_trajectory_metrics.csv", index=False)
    correlations.to_csv(output_dir / "test_z0_temperature_jitter_correlations.csv", index=False)
    result["ae_history"].to_csv(output_dir / "ae_history.csv", index=False)
    result["split_info"].to_csv(output_dir / "split_info.csv", index=False)
    plot_temperature_generalization(
        trajectory_df,
        train_temperatures,
        output_dir / "test_z0_temperature_generalization.png",
    )
    plot_random_traces(
        test_frame_df,
        output_dir / "random_test_z0_traces_colored_by_pratio.png",
    )

    print(f"output_dir: {output_dir}")
    print(f"training temperatures: {train_temperatures}")
    print(
        "unseen temperature levels:",
        sorted(
            trajectory_df.loc[
                ~trajectory_df["temperature"].isin(train_temperatures), "temperature"
            ].unique()
        ),
    )
    print(correlations.to_string(index=False))


if __name__ == "__main__":
    main()
