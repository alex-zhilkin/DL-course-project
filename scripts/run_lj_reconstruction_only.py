"""Retrain static-only LJ latent rollouts with a reconstruction-only AE."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

import run_lj_latent_comparison as original
from lss.latent.analysis import framewise_latent_descriptor_sweep
from lss.latent.capacity import evaluate_experiment
from lss.latent.experiment import run_latent_experiment, seed_everything
from lss.utils import resolve_device


DATA = ROOT / "data" / "lj-noisy-eps0.01-sigma1.0-cutoff1.122_1348sims_10frames.pt"
OUT = ROOT / "notebooks" / "results" / "13_lj_reconstruction_only"
SEED = 20260716


def make_spec(train_count: int, *, device, epochs: int, force: bool):
    source, cfg = original.specs(
        train_count, output_dir=OUT, device=device, max_epochs=epochs, force=force
    )
    cache = OUT / "models" / f"lj_reconstruction_train{train_count}_seed{SEED}.pt"
    cfg.update(
        {
            "batch_graphs": 32 if train_count >= 500 else 16,
            "ae_max_train_frames_per_sim": 10,
            "dyn_max_train_transitions_per_sim": 9,
            "ae_patience": 5 if train_count >= 500 else 10,
            "dyn_patience": 5 if train_count >= 500 else 10,
            "propagator_standardize_latent": True,
            "propagator_objective": "multistep",
            "propagator_multistep_horizons": [1, 2, 4, 8],
            "rollout_steps_grid": [1, 2, 4, 6, 9],
            "rollout_eval_max_sims_per_split": 100,
            "cache_path": str(cache),
            "force_train": force,
        }
    )
    source.update(
        {
            "path": str(DATA),
            "label": f"LJ reconstruction-only CV2, train={train_count}",
            "ae_max_train_frames_per_sim": 10,
            "dyn_max_train_transitions_per_sim": 9,
            "ae_max_epochs": epochs,
            "ae_patience": 5 if train_count >= 500 else 10,
        }
    )
    source["dataset_mixture"] = [
        {
            "name": "lj_noisy",
            "label": "LJ noisy",
            "path": str(DATA),
            "train_count": train_count,
            "val_count": 50,
        }
    ]
    return source, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-counts", nargs="+", type=int, default=[20, 200])
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not DATA.exists():
        raise FileNotFoundError(DATA)
    OUT.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    latent_rows, rollout_parts = [], []
    for count in args.train_counts:
        print(f"\n### reconstruction-only LJ: train={count}", flush=True)
        seed_everything(SEED)
        source, cfg = make_spec(count, device=device, epochs=args.max_epochs, force=args.force)
        result = run_latent_experiment(source, cfg, device=device)
        if result["rollout_stats"].empty:
            result = evaluate_experiment(result, cfg, device=device)
        frame, *_ = framewise_latent_descriptor_sweep(
            result, device=device, frame_stride=1, max_frames_per_sim=10
        )
        frame.insert(0, "train_count", count)
        frame.to_csv(OUT / f"framewise_train{count}.csv", index=False)
        best = result["ae_history"].loc[result["ae_history"]["val_objective"].idxmin()]
        latent_rows.append(
            {
                "train_count": count,
                **original.latent_metrics(frame),
                "best_val_reconstruction": float(best["val_mse_norm"]),
            }
        )
        rollout = result["rollout_stats"].copy()
        rollout.insert(0, "train_count", count)
        rollout_parts.append(rollout)
        pd.DataFrame(latent_rows).to_csv(OUT / "latent_summary.csv", index=False)
        pd.concat(rollout_parts, ignore_index=True).to_csv(OUT / "rollout_all.csv", index=False)
    summary = pd.concat(rollout_parts, ignore_index=True)
    summary = summary.loc[summary["split"].eq("test"), [
        "train_count", "rollout_steps", "used", "rollout_position_r2",
        "p_ratio_r2", "p_ratio_pearson", "final_pos_mse",
    ]].sort_values(["train_count", "rollout_steps"])
    summary.to_csv(OUT / "test_rollout_summary.csv", index=False)
    print("\nLatents\n", pd.DataFrame(latent_rows).to_string(index=False))
    print("\nRollouts\n", summary.to_string(index=False))


if __name__ == "__main__":
    main()
