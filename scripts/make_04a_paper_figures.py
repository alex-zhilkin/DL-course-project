"""Create the minimal paper figure set for notebook 04a."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lss.latent.experiment import find_project_root
from lss.latent.simulation import pearson_r
from lss.plotting import PAPER_COLORS, apply_editorial_style, style_axes


SEED = 20260623
TRAIN_TEMPERATURES = (1.0, 5.0)


def bootstrap_r(x, y, *, seed=SEED + 1, samples=5000):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)
    values = np.empty(samples, dtype=float)
    for index in range(samples):
        selected = rng.integers(0, len(x), len(x))
        values[index] = np.corrcoef(x[selected], y[selected])[0, 1]
    return np.quantile(values, [0.025, 0.5, 0.975])


def linear_readout(train, test, features, target):
    train_x = np.column_stack([np.ones(len(train)), train[features].to_numpy(float)])
    test_x = np.column_stack([np.ones(len(test)), test[features].to_numpy(float)])
    coefficients = np.linalg.lstsq(
        train_x,
        train[target].to_numpy(float),
        rcond=None,
    )[0]
    predicted = test_x @ coefficients
    actual = test[target].to_numpy(float)
    r2 = 1.0 - np.sum((actual - predicted) ** 2) / np.sum(
        (actual - actual.mean()) ** 2
    )
    return predicted, float(r2), pearson_r(actual, predicted)


def representative_simulations(metrics, count=12):
    quantiles = np.linspace(0.02, 0.98, count)
    targets = metrics["final_p_ratio"].quantile(quantiles).to_numpy(float)
    available = metrics.copy()
    selected = []
    for target in targets:
        distances = (available["final_p_ratio"] - target).abs()
        sim_idx = int(available.loc[distances.idxmin(), "sim_idx"])
        selected.append(sim_idx)
        available = available[available["sim_idx"].ne(sim_idx)]
    return selected


def grouped_temperature_summary(metrics):
    return (
        metrics.groupby("temperature", as_index=False)
        .agg(
            count=("sim_idx", "size"),
            jitter_mean=("z0_step_std", "mean"),
            jitter_sem=("z0_step_std", "sem"),
        )
        .sort_values("temperature")
    )


def main_figure(frame_df, metrics, output_dir):
    selected = representative_simulations(metrics)
    traces = frame_df[frame_df["sim_idx"].isin(selected)].copy()
    p_ratio_by_sim = metrics.set_index("sim_idx")["final_p_ratio"]
    p_min = float(metrics["final_p_ratio"].quantile(0.02))
    p_max = float(metrics["final_p_ratio"].quantile(0.98))
    p_norm = plt.Normalize(p_min, p_max)
    p_cmap = plt.cm.viridis
    temperature_norm = plt.Normalize(
        metrics["temperature"].min(),
        metrics["temperature"].max(),
    )

    fig, axes = plt.subplots(1, 3, figsize=(15.6, 4.45), constrained_layout=True)

    for sim_idx in selected:
        group = traces[traces["sim_idx"].eq(sim_idx)].sort_values("frame_idx")
        p_ratio = float(p_ratio_by_sim.loc[sim_idx])
        axes[0].plot(
            group["frame_idx"],
            group["z0"],
            color=p_cmap(p_norm(p_ratio)),
            lw=1.25,
            alpha=0.86,
        )
    fig.colorbar(
        plt.cm.ScalarMappable(norm=p_norm, cmap=p_cmap),
        ax=axes[0],
        pad=0.02,
        label="final p-ratio",
    )
    style_axes(axes[0], xlabel="frame", ylabel="z0")

    slope_points = axes[1].scatter(
        metrics["z0_slope"],
        metrics["final_p_ratio"],
        c=metrics["temperature"],
        norm=temperature_norm,
        cmap="plasma",
        s=21,
        alpha=0.62,
        linewidth=0,
    )
    slope_fit = np.polyfit(metrics["z0_slope"], metrics["final_p_ratio"], deg=1)
    slope_x = np.linspace(metrics["z0_slope"].min(), metrics["z0_slope"].max(), 100)
    axes[1].plot(
        slope_x,
        slope_fit[0] * slope_x + slope_fit[1],
        color=PAPER_COLORS["ink"],
        lw=1.6,
    )
    fig.colorbar(slope_points, ax=axes[1], pad=0.02, label="temperature")
    style_axes(axes[1], xlabel="z0 slope per frame", ylabel="final p-ratio")

    rng = np.random.default_rng(SEED + 2)
    x_jitter = rng.normal(0.0, 0.14, size=len(metrics))
    axes[2].scatter(
        metrics["temperature"] + x_jitter,
        metrics["z0_step_std"],
        color=PAPER_COLORS["light"],
        s=14,
        alpha=0.3,
        linewidth=0,
    )
    summary = grouped_temperature_summary(metrics)
    for seen, marker, color, label in (
        (False, "o", PAPER_COLORS["blue"], "unseen temperature"),
        (True, "s", PAPER_COLORS["red"], "used for training"),
    ):
        group = summary[summary["temperature"].isin(TRAIN_TEMPERATURES).eq(seen)]
        axes[2].errorbar(
            group["temperature"],
            group["jitter_mean"],
            yerr=group["jitter_sem"],
            fmt=marker,
            ms=7,
            color=color,
            capsize=3,
            lw=1.3,
            label=label,
        )
    style_axes(axes[2], xlabel="temperature", ylabel="std(delta z0)")
    axes[2].legend(frameon=False, fontsize=8, loc="upper left")

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

    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"paper_main_latent_decomposition.{suffix}", dpi=220)
    plt.close(fig)


def strict_control_figure(metrics, output_dir):
    unseen = metrics[~metrics["temperature"].isin(TRAIN_TEMPERATURES)]
    temperature_norm = plt.Normalize(
        metrics["temperature"].min(),
        metrics["temperature"].max(),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.25), constrained_layout=True)

    points = axes[0].scatter(
        metrics["z0_slope"],
        metrics["final_p_ratio"],
        c=metrics["temperature"],
        norm=temperature_norm,
        cmap="plasma",
        s=20,
        alpha=0.62,
        linewidth=0,
    )
    fit = np.polyfit(metrics["z0_slope"], metrics["final_p_ratio"], deg=1)
    x_line = np.linspace(metrics["z0_slope"].min(), metrics["z0_slope"].max(), 100)
    axes[0].plot(
        x_line,
        fit[0] * x_line + fit[1],
        color=PAPER_COLORS["ink"],
        lw=1.6,
    )
    fig.colorbar(points, ax=axes[0], pad=0.02, label="temperature")
    style_axes(axes[0], xlabel="z0 slope per frame", ylabel="final p-ratio")

    rng = np.random.default_rng(SEED + 3)
    axes[1].scatter(
        metrics["temperature"] + rng.normal(0.0, 0.14, size=len(metrics)),
        metrics["z0_step_std"],
        color=PAPER_COLORS["light"],
        s=14,
        alpha=0.3,
        linewidth=0,
    )
    summary = grouped_temperature_summary(metrics)
    for seen, marker, color, label in (
        (False, "o", PAPER_COLORS["blue"], "unseen temperature"),
        (True, "s", PAPER_COLORS["red"], "train/validation temperature"),
    ):
        group = summary[summary["temperature"].isin(TRAIN_TEMPERATURES).eq(seen)]
        axes[1].errorbar(
            group["temperature"],
            group["jitter_mean"],
            yerr=group["jitter_sem"],
            fmt=marker,
            ms=7,
            color=color,
            capsize=3,
            lw=1.3,
            label=label,
        )
    style_axes(axes[1], xlabel="temperature", ylabel="std(delta z0)")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    for label, ax in zip(("a", "b"), axes):
        ax.text(
            -0.13,
            1.03,
            label,
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="bottom",
        )
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"paper_strict_temperature_control.{suffix}", dpi=220)
    plt.close(fig)

    return {
        "strict_unseen_slope_pratio_r": pearson_r(
            unseen["z0_slope"], unseen["final_p_ratio"]
        ),
        "strict_unseen_jitter_temperature_r": pearson_r(
            unseen["z0_step_std"], unseen["temperature"]
        ),
    }


def main():
    apply_editorial_style()
    project_root = find_project_root()
    run_dir = (
        project_root
        / "notebooks"
        / "results"
        / "04a_latent_space_analysis"
        / f"single_depablo_mixed_temp_cv1_seed{SEED}"
    )
    strict_dir = (
        project_root
        / "notebooks"
        / "results"
        / "04a_latent_space_analysis"
        / f"strict_temperature_holdout_cv1_seed{SEED}"
    )
    output_dir = run_dir / "paper_figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_df = pd.read_csv(run_dir / "test_framewise_latent_trajectory_table.csv")
    test_metrics = pd.read_csv(run_dir / "paper_audit_test_trajectory_metrics.csv")
    val_metrics = pd.read_csv(run_dir / "paper_audit_val_trajectory_metrics.csv")
    strict_metrics = pd.read_csv(strict_dir / "test_trajectory_metrics.csv")

    main_figure(frame_df, test_metrics, output_dir)
    strict_stats = strict_control_figure(strict_metrics, output_dir)

    _, p_ratio_r2, p_ratio_prediction_r = linear_readout(
        val_metrics,
        test_metrics,
        ["z0_slope"],
        "final_p_ratio",
    )
    _, temperature_r2, temperature_prediction_r = linear_readout(
        val_metrics,
        test_metrics,
        ["z0_step_std"],
        "temperature",
    )
    rows = [
        {
            "result": "z0 slope vs final p-ratio",
            "pearson_r": pearson_r(
                test_metrics["z0_slope"], test_metrics["final_p_ratio"]
            ),
            "ci_low": bootstrap_r(
                test_metrics["z0_slope"], test_metrics["final_p_ratio"]
            )[0],
            "ci_high": bootstrap_r(
                test_metrics["z0_slope"], test_metrics["final_p_ratio"]
            )[2],
            "validation_fit_test_r2": p_ratio_r2,
            "validation_fit_test_r": p_ratio_prediction_r,
        },
        {
            "result": "std(delta z0) vs temperature",
            "pearson_r": pearson_r(
                test_metrics["z0_step_std"], test_metrics["temperature"]
            ),
            "ci_low": bootstrap_r(
                test_metrics["z0_step_std"], test_metrics["temperature"]
            )[0],
            "ci_high": bootstrap_r(
                test_metrics["z0_step_std"], test_metrics["temperature"]
            )[2],
            "validation_fit_test_r2": temperature_r2,
            "validation_fit_test_r": temperature_prediction_r,
        },
    ]
    summary = pd.DataFrame(rows)
    for key, value in strict_stats.items():
        summary[key] = value
    summary.to_csv(output_dir / "paper_result_summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"figures: {output_dir}")


if __name__ == "__main__":
    main()
