"""Compare standardized and raw latent propagation on matched 3-seed rollouts."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from lss.latent.experiment import find_project_root
from lss.utils import resolve_device
from scripts.compare_latent_attention_mlp_quick import SEEDS, run_variant


def main() -> None:
    project_root = find_project_root()
    output_dir = project_root / "notebooks" / "results" / "04a_latent_standardization_3seed"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device("auto")

    existing = pd.read_csv(
        project_root
        / "notebooks"
        / "results"
        / "04a_attention_vs_mlp_3seed"
        / "rollout_comparison_by_seed.csv"
    )
    standardized = existing[existing["autoencoder"].eq("attention")].copy()
    standardized["latent_standardization"] = True

    raw_parts = []
    for seed in SEEDS:
        print(f"\n### seed={seed} raw latent propagation ###", flush=True)
        result, stats = run_variant(
            "attention",
            seed=seed,
            project_root=project_root,
            device=device,
            standardize_latent=False,
        )
        raw_parts.append(stats)
        del result

    comparison = pd.concat([standardized, *raw_parts], ignore_index=True)
    comparison["normalization"] = comparison["latent_standardization"].map(
        {True: "standardized", False: "raw"}
    )
    comparison.to_csv(output_dir / "rollout_comparison_by_seed.csv", index=False)
    summary = (
        comparison.groupby(["normalization", "rollout_steps"], as_index=False)
        .agg(
            position_r2_mean=("rollout_position_r2", "mean"),
            position_r2_std=("rollout_position_r2", "std"),
            final_pos_mse_mean=("final_pos_mse", "mean"),
            final_pos_mse_std=("final_pos_mse", "std"),
            p_ratio_r2_mean=("p_ratio_r2", "mean"),
            p_ratio_r2_std=("p_ratio_r2", "std"),
            p_ratio_used_mean=("p_ratio_used", "mean"),
        )
    )
    summary.to_csv(output_dir / "rollout_comparison_summary.csv", index=False)
    print("\nTHREE-SEED STANDARDIZATION SUMMARY", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"saved: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
