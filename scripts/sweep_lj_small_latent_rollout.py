"""Controlled small-latent sweep for the noisy-LJ AE and four-step propagator.

P-ratio is only reported on validation/test data.  It never enters either loss
or checkpoint selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from lss.latent.experiment import run_latent_experiment, seed_everything
from scripts.quick_lj_frozen_ae_propagator_sweep import (
    encode_latent_table,
    evaluate,
)
from scripts.train_lj_four_step_latent_propagator import fit, rollout


SEED = 657567
TARGET_STEP = 100
TRAIN, VAL, TEST = 250, 50, 150
DATA = ROOT / "data/lj-noisy-eps0.01-sigma1.0-cutoff1.122_500sims_200frames.pt"
OUTPUT = ROOT / "notebooks/results/08_history_aware_latent_rollout/lj_noisy/small_latent_sweep"


def ae_config(latent_dim: int, device: torch.device, cache_path: Path) -> dict:
    return {
        "dataset_name": "lj_noisy",
        "split_seed": SEED,
        "model_seed": SEED + latent_dim,
        "device": str(device),
        "train_count": TRAIN,
        "val_count": VAL,
        "split_stratify_temperature": False,
        "min_train_p_ratio": None,
        "pos_dim": 2,
        "batch_graphs": 50,
        "frame_skip": 1,
        "train_frame_start_order": 3,
        "edge_multiplicity": 1,
        "edge_vector_dim": 8,
        "edge_mode": "stored",
        "latent_dim": latent_dim,
        "latent_tokens": 32,
        "hidden_size": 64,
        "autoencoder_model": "attention",
        "node_feature_mode": "modular_history3",
        "ae_target_mode": "modular_history3",
        "ae_max_train_frames_per_sim": 100,
        "ae_max_epochs": 15,
        "ae_patience": 4,
        "ae_lr": 3e-4,
        "ae_weight_decay": 1e-5,
        "early_stop_min_delta": 1e-5,
        "should_rollout": False,
        "should_train_propagator": False,
        "cache_path": str(cache_path),
        "force_train": True,
    }


def save_manual_ae(result: dict, path: Path) -> None:
    torch.save(
        {
            "params": result["params"],
            "ae_state_dict": {
                key: value.detach().cpu() for key, value in result["ae"].state_dict().items()
            },
            "normalizers": {
                key: value.detach().cpu() for key, value in result["normalizers"].items()
            },
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", default="2,4,6,8")
    args = parser.parse_args()
    dims = [int(value) for value in args.dims.split(",")]
    if any(value < 1 or value > 8 for value in dims):
        raise ValueError("All latent dimensions must be in [1, 8].")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_sims = torch.load(DATA, map_location="cpu", weights_only=False)
    split_generator = torch.Generator().manual_seed(SEED)
    order = torch.randperm(len(all_sims), generator=split_generator).tolist()
    prop_train_sims = [all_sims[index] for index in order[:TRAIN]]
    prop_val_sims = [all_sims[index] for index in order[300 : 300 + VAL]]
    test_sims = [all_sims[index] for index in order[350 : 350 + TEST]]
    sims = prop_train_sims + prop_val_sims + test_sims
    rows = []

    for latent_dim in dims:
        case = OUTPUT / f"latent_{latent_dim:02d}"
        case.mkdir(parents=True, exist_ok=True)
        print(f"\n=== latent_dim={latent_dim} | device={device} ===", flush=True)
        cfg = ae_config(latent_dim, device, case / "ae_bundle.pt")
        source = {
            **cfg,
            "source_name": "Noisy LJ",
            "label": f"Noisy LJ | {latent_dim}D history AE",
            "path": str(DATA),
        }
        seed_everything(SEED + latent_dim)
        result = run_latent_experiment(source, cfg, device=device)
        save_manual_ae(result, case / "ae_checkpoint.pt")

        latent_table = encode_latent_table(
            result["ae"],
            result["normalizers"],
            sims,
            max_frame=TARGET_STEP,
            batch_size=24,
            device=device,
            node_feature_mode="modular_history3",
        )
        torch.save(latent_table, case / "encoded_latents_frame100.pt")
        train_z = latent_table[:TRAIN]
        val_z = latent_table[TRAIN : TRAIN + VAL]
        test_z = latent_table[TRAIN + VAL :]

        model, stats, best = fit(
            result["ae"],
            result["normalizers"],
            train_z,
            val_z,
            prop_val_sims,
            seed=SEED + 100 + latent_dim,
            device=device,
            rollout_eval_every=2,
        )
        predicted_test_z = rollout(model, stats, test_z, device).cpu()
        ae_metrics = evaluate(
            result["ae"], result["normalizers"], test_sims,
            test_z[:, TARGET_STEP], TARGET_STEP, device,
        )
        rollout_metrics = evaluate(
            result["ae"], result["normalizers"], test_sims,
            predicted_test_z, TARGET_STEP, device,
        )
        row = {
            "latent_dim": latent_dim,
            "ae_best_epoch": int(result["ae_history"].loc[
                result["ae_history"]["val_objective"].idxmin(), "epoch"
            ]),
            "ae_best_val_objective": float(result["ae_history"]["val_objective"].min()),
            "propagator_best_epoch": best["epoch"],
            "propagator_val_decoded_field_mse": best["loss"],
            "ae_test_p_ratio_r2": ae_metrics["p_ratio_r2"],
            "ae_test_p_ratio_pearson": ae_metrics["p_ratio_pearson"],
            "rollout_test_p_ratio_r2": rollout_metrics["p_ratio_r2"],
            "rollout_test_p_ratio_pearson": rollout_metrics["p_ratio_pearson"],
            "rollout_pred_to_true_std": rollout_metrics["pred_to_true_std"],
        }
        rows.append(row)
        pd.DataFrame(best["history"]).to_csv(case / "propagator_history.csv", index=False)
        print(json.dumps(row, indent=2), flush=True)
        pd.DataFrame(rows).sort_values("rollout_test_p_ratio_r2", ascending=False).to_csv(
            OUTPUT / "summary_partial.csv", index=False
        )

    summary = pd.DataFrame(rows).sort_values("rollout_test_p_ratio_r2", ascending=False)
    summary.to_csv(OUTPUT / "summary.csv", index=False)
    print("\n" + summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
