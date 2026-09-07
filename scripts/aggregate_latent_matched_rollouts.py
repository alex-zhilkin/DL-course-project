"""Aggregate completed matched rollout runs and report source-wise validation scores."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "notebooks" / "results" / "latent_matched_rollouts"


def main() -> None:
    rows = []
    status = []
    for recipe_path in sorted(RESULTS.glob("*/recipe.json")):
        run = recipe_path.parent
        recipe = json.loads(recipe_path.read_text())
        scope = recipe["source"]["source_name"]
        seed = int(recipe["config"]["model_seed"])
        variant = run.name.rsplit(f"_{scope}_s{seed}", 1)[0]
        complete = (run / "completed.json").is_file()
        status.append({"variant": variant, "scope": scope, "seed": seed, "completed": complete})
        if not complete:
            continue
        frame = pd.read_csv(run / "validation_rollout_summary.csv")
        frame["variant"] = variant
        frame["scope"] = scope
        frame["seed"] = seed
        rows.append(frame)
    status_frame = pd.DataFrame(status).sort_values(["variant", "scope", "seed"])
    status_frame.to_csv(RESULTS / "status.csv", index=False)
    if not rows:
        print(status_frame.to_string(index=False))
        return
    scores = pd.concat(rows, ignore_index=True)
    scores.to_csv(RESULTS / "all_validation_rollout_results.csv", index=False)
    aggregate = (
        scores.groupby(["variant", "scope", "source", "rollout_steps"], as_index=False)
        .agg(
            seeds=("seed", "nunique"),
            p_ratio_r2_mean=("p_ratio_r2", "mean"),
            p_ratio_r2_std=("p_ratio_r2", "std"),
            p_ratio_mae_mean=("p_ratio_mae", "mean"),
            used_min=("used", "min"),
        )
    )
    aggregate.to_csv(RESULTS / "validation_rollout_aggregate.csv", index=False)
    endpoint = aggregate.loc[aggregate.rollout_steps == 100]
    print(endpoint.to_string(index=False))
    print(f"completed={int(status_frame.completed.sum())}/{len(status_frame)}")


if __name__ == "__main__":
    main()
