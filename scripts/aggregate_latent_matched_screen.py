"""Aggregate completed matched-representation runs without evaluating reserved test data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "notebooks" / "results" / "latent_matched_study"


def main() -> None:
    frames: list[pd.DataFrame] = []
    pca_rows: list[dict] = []
    completed = 0
    for recipe_path in sorted(RESULTS.glob("*/recipe.json")):
        run = recipe_path.parent
        if not (run / "completed.json").is_file():
            continue
        recipe = json.loads(recipe_path.read_text())
        model = recipe["source"]["source_name"]
        if model == "shared_no_mixed":
            continue
        dimension = int(recipe["config"]["ae_config"]["latent_dim"])
        seed = int(recipe["config"]["model_seed"])
        frame = pd.read_csv(run / "validation_summary.csv")
        frame["model"] = model
        frame["dimension"] = dimension
        frame["seed"] = seed
        frames.append(frame)
        pca = np.load(run / "training_pca.npz")
        variance = np.square(pca["singular_values"])
        pca_rows.append(
            {
                "model": model,
                "dimension": dimension,
                "seed": seed,
                "pc1_variance_fraction": float(variance[:1].sum() / variance.sum()),
                "pc2_variance_fraction": float(variance[:2].sum() / variance.sum()),
            }
        )
        completed += 1

    if completed != 72:
        raise RuntimeError(f"Expected 72 completed runs, found {completed}.")
    scores = pd.concat(frames, ignore_index=True)
    scores.to_csv(RESULTS / "all_completed_validation_summary.csv", index=False)
    pd.DataFrame(pca_rows).to_csv(RESULTS / "training_pca_variance.csv", index=False)

    endpoint = scores.loc[(scores.frame == 100) & (scores.control == "ae")].copy()
    endpoint["training_scope"] = np.where(
        endpoint["model"].isin(["shared3", "shared4"]), endpoint["model"], "individual"
    )
    endpoint.to_csv(RESULTS / "endpoint_seed_results.csv", index=False)
    metrics = [
        "position_mse",
        "strain_x_mae",
        "strain_y_mae",
        "p_ratio_r2",
        "p_ratio_mae",
        "legacy_p_ratio_r2",
        "legacy_p_ratio_mae",
    ]
    aggregate = (
        endpoint.groupby(["source", "training_scope", "dimension"], as_index=False)[metrics]
        .agg(["mean", "std"])
    )
    aggregate.columns = [
        "_".join(str(part) for part in column if part).rstrip("_")
        if isinstance(column, tuple)
        else column
        for column in aggregate.columns
    ]
    aggregate.to_csv(RESULTS / "endpoint_aggregate.csv", index=False)

    shared = endpoint.loc[endpoint.model.isin(["shared3", "shared4"])]
    comparison = shared.pivot(
        index=["source", "dimension", "seed"], columns="model", values=metrics
    )
    comparison.columns = [f"{metric}_{model}" for metric, model in comparison.columns]
    comparison = comparison.reset_index()
    for metric in metrics:
        left, right = f"{metric}_shared4", f"{metric}_shared3"
        if left in comparison and right in comparison:
            comparison[f"{metric}_shared4_minus_shared3"] = comparison[left] - comparison[right]
    comparison.to_csv(RESULTS / "adding_lj_matched_contrasts.csv", index=False)
    print(f"Aggregated {completed} completed runs")
    print(
        aggregate.loc[:, [
            "source", "training_scope", "dimension", "p_ratio_r2_mean", "p_ratio_r2_std",
            "position_mse_mean",
        ]].to_string(index=False)
    )


if __name__ == "__main__":
    main()
