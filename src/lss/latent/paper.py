"""Paper-summary table and plotting helpers for latent capacity sweeps."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def additive_factor_report(df, metric, factors):
    def linear_r2(X, y):
        X = np.column_stack([np.ones(len(X)), np.asarray(X, dtype=float)])
        y = np.asarray(y, dtype=float)
        prediction = X @ np.linalg.lstsq(X, y, rcond=None)[0]
        total = float(((y - y.mean()) ** 2).sum())
        return 1.0 - float(((y - prediction) ** 2).sum()) / total if total > 0 else np.nan

    rows = []
    for dataset_name, group in df.groupby("dataset_name"):
        clean = group.dropna(subset=[metric])
        if len(clean) < 4:
            continue
        y = clean[metric].to_numpy(float)
        full = pd.get_dummies(clean[factors].astype(str), drop_first=False).astype(float)
        full_r2 = linear_r2(full, y)
        rows.append(
            {
                "dataset_name": dataset_name,
                "factor": "full additive model",
                "unique_r2": np.nan,
                "model_r2": full_r2,
            }
        )
        for factor in factors:
            columns = [column for column in full.columns if not column.startswith(factor + "_")]
            without = linear_r2(full[columns], y)
            rows.append(
                {
                    "dataset_name": dataset_name,
                    "factor": factor,
                    "unique_r2": full_r2 - without,
                    "model_r2_without_factor": without,
                }
            )
    return pd.DataFrame(rows)


def plot_factor_grid(
    df,
    metric,
    *,
    dataset_order,
    dataset_labels,
    plot_specs,
    title,
    ylabel,
    color,
    ylim=None,
    log_y=False,
    clip_errors=False,
    value_floor=1e-8,
    value_cap=1e-4,
):
    fig, axes = plt.subplots(
        len(dataset_order),
        len(plot_specs),
        figsize=(10.8, 2.7 * len(dataset_order)),
        sharey="row",
        squeeze=False,
    )
    for row, dataset_name in enumerate(dataset_order):
        dataset_df = df[df["dataset_name"].eq(dataset_name)]
        for col, (factor, label) in enumerate(plot_specs):
            ax = axes[row, col]
            grouped = (
                dataset_df.groupby(factor)
                .agg(mean=(metric, "mean"), std=(metric, "std"))
                .reset_index()
                .sort_values(factor)
            )
            if grouped.empty:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                continue
            error = grouped["std"].fillna(0)
            if clip_errors:
                lower = np.maximum(grouped["mean"] - error, value_floor)
                upper = np.minimum(grouped["mean"] + error, value_cap)
                error = np.vstack([grouped["mean"] - lower, upper - grouped["mean"]])
            ax.errorbar(
                grouped[factor],
                grouped["mean"],
                yerr=error,
                marker="o",
                capsize=3,
                lw=2,
                color=color,
            )
            if log_y:
                ax.set_yscale("log")
                ax.axhline(value_cap, color="0.35", lw=1, ls=":", alpha=0.7)
            if ylim is not None:
                ax.set_ylim(*ylim)
            if row == 0:
                ax.set_title(label)
            if row == len(dataset_order) - 1:
                ax.set_xlabel(label)
            if col == 0:
                ax.set_ylabel(f"{dataset_labels.get(dataset_name, dataset_name)}\n{ylabel}")
    fig.suptitle(title, y=1.01, fontsize=12, fontweight="bold")
    fig.tight_layout()
    plt.show()


def top_table(df, dataset_name, sort_col, ascending, columns, n=5):
    return (
        df[df["dataset_name"].eq(dataset_name)]
        .sort_values(sort_col, ascending=ascending)
        .head(n)[columns]
        .round(6)
    )


def select_simple_strong_rollouts(
    summary_df,
    dataset_name,
    *,
    factors,
    n=5,
    tolerance=0.05,
):
    subset = summary_df[summary_df["dataset_name"].eq(dataset_name)].copy()
    if subset.empty:
        return subset
    best = float(subset["mean_rollout_r2"].max())
    strong = subset[subset["mean_rollout_r2"].ge(best - tolerance)].sort_values(
        ["latent_dim", "train_networks", "train_frames_per_network", "mean_rollout_r2"],
        ascending=[True, True, True, False],
    )
    selected = strong.head(n).copy()
    if len(selected) < n:
        selected_keys = set(map(tuple, selected[factors].to_numpy()))
        fill = subset[
            ~subset[factors].apply(tuple, axis=1).isin(selected_keys)
        ].sort_values("mean_rollout_r2", ascending=False)
        selected = pd.concat([selected, fill.head(n - len(selected))], ignore_index=True)
    selected["best_mean_rollout_r2"] = best
    selected["delta_from_best"] = selected["mean_rollout_r2"] - best
    selected["config_label"] = selected.apply(
        lambda row: (
            f"CV{int(row['latent_dim'])}, nets={int(row['train_networks'])}, "
            f"frames={int(row['train_frames_per_network'])}"
        ),
        axis=1,
    )
    return selected.sort_values(factors).reset_index(drop=True)


def marginal_table(df, metric, *, dataset_order, factors):
    rows = []
    for dataset_name in dataset_order:
        subset = df[df["dataset_name"].eq(dataset_name)]
        for factor in factors:
            part = subset.groupby(factor)[metric].mean().reset_index()
            part.insert(0, "dataset_name", dataset_name)
            part.insert(1, "factor", factor)
            rows.append(part.rename(columns={factor: "value", metric: "mean_metric"}))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


__all__ = [
    "additive_factor_report",
    "marginal_table",
    "plot_factor_grid",
    "select_simple_strong_rollouts",
    "top_table",
]
