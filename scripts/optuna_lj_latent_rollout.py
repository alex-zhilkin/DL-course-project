"""Optuna search for a compact noisy-LJ latent AE plus four-step propagator.

The Optuna objective is validation rollout p-ratio R² at frame 100.  P-ratio
never appears in the AE or propagator losses.  The test split is evaluated once,
only after Optuna has selected its best validation trial.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import optuna
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from lss.data import load_dataset
from lss.latent.experiment import run_latent_experiment, seed_everything
from scripts.quick_lj_frozen_ae_propagator_sweep import (
    encode_latent_table,
    evaluate,
    restore_ae,
)
from scripts.train_lj_four_step_latent_propagator import fit, rollout
from scripts.tune_lj_z3_delta_propagator import DeltaMLP


SEED = 657567
TARGET_STEP = 100
PROPAGATOR_UNROLL_STEPS = 4
VAL_START, VAL_COUNT = 300, 50
TEST_START, TEST_COUNT = 350, 150
DATA = ROOT / "data/lj-noisy-eps0.01-sigma1.0-cutoff1.122_500sims_200frames.pt"
OUTPUT = ROOT / "notebooks/results/08_history_aware_latent_rollout/lj_noisy/optuna_rollout_r2"


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


def split_sims(all_sims, train_count: int):
    generator = torch.Generator().manual_seed(SEED)
    order = torch.randperm(len(all_sims), generator=generator).tolist()
    return (
        [all_sims[index] for index in order[:train_count]],
        [all_sims[index] for index in order[VAL_START : VAL_START + VAL_COUNT]],
        [all_sims[index] for index in order[TEST_START : TEST_START + TEST_COUNT]],
    )


def suggest_config(trial: optuna.Trial) -> dict:
    return {
        "train_count": trial.suggest_categorical("train_count", [100, 150, 200, 250]),
        "latent_dim": trial.suggest_int("latent_dim", 2, 8),
        "hidden_size": trial.suggest_categorical("ae_hidden_size", [32, 64, 96]),
        "batch_graphs": trial.suggest_categorical("ae_batch_graphs", [25, 50]),
        "ae_max_train_frames_per_sim": trial.suggest_categorical(
            "ae_frames_per_network", [50, 100, 150, 200]
        ),
        "ae_lr": trial.suggest_float("ae_lr", 1e-4, 5e-4, log=True),
        "prop_hidden_size": trial.suggest_categorical("prop_hidden_size", [64, 128, 256]),
        "prop_depth": trial.suggest_categorical("prop_depth", [2, 3]),
        "prop_lr": trial.suggest_float("prop_lr", 5e-5, 4e-4, log=True),
    }


def ae_config(config: dict, device: torch.device, case: Path, trial_number: int) -> dict:
    return {
        "dataset_name": "lj_noisy",
        "split_seed": SEED,
        "model_seed": SEED + 1000 + trial_number,
        "device": str(device),
        "train_count": config["train_count"],
        "val_count": VAL_COUNT,
        "split_stratify_temperature": False,
        "min_train_p_ratio": None,
        "pos_dim": 2,
        "batch_graphs": config["batch_graphs"],
        "frame_skip": 1,
        "train_frame_start_order": 3,
        "edge_multiplicity": 1,
        "edge_vector_dim": 8,
        "edge_mode": "stored",
        "latent_dim": config["latent_dim"],
        "latent_tokens": 32,
        "hidden_size": config["hidden_size"],
        "autoencoder_model": "attention",
        "node_feature_mode": "modular_history3",
        "ae_target_mode": "modular_history3",
        "ae_max_train_frames_per_sim": config["ae_max_train_frames_per_sim"],
        "ae_max_epochs": 20,
        "ae_patience": 4,
        "ae_lr": config["ae_lr"],
        "ae_weight_decay": 1e-5,
        "early_stop_min_delta": 1e-5,
        "should_rollout": False,
        "should_train_propagator": False,
        "cache_path": str(case / "ae_bundle.pt"),
        "force_train": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=25)
    parser.add_argument("--study-name", default="noisy_lj_compact_rollout")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("WARNING: CUDA unavailable; this full Optuna study is intended for the GPU environment.")
    # Use the shared loader: the AE and the propagator must see the same
    # one-edge-per-undirected-pair representation.
    all_sims = load_dataset(DATA, edge_multiplicity=1, edge_vector_dim=8)
    storage = f"sqlite:///{OUTPUT/'study.db'}"
    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        config = suggest_config(trial)
        case = OUTPUT / f"trial_{trial.number:03d}"
        case.mkdir(parents=True, exist_ok=True)
        print(f"\n=== trial {trial.number}: {config} ===", flush=True)
        cfg = ae_config(config, device, case, trial.number)
        source = {
            **cfg,
            "source_name": "Noisy LJ",
            "label": f"Noisy LJ | Optuna trial {trial.number}",
            "path": str(DATA),
        }
        seed_everything(SEED + trial.number)
        result = run_latent_experiment(source, cfg, device=device)
        save_manual_ae(result, case / "ae_checkpoint.pt")
        train_sims, val_sims, _ = split_sims(all_sims, config["train_count"])
        z = encode_latent_table(
            result["ae"], result["normalizers"], train_sims + val_sims,
            max_frame=TARGET_STEP, batch_size=24, device=device,
            node_feature_mode="modular_history3",
        )
        train_z = z[: len(train_sims)]
        val_z = z[len(train_sims) :]
        model, stats, best = fit(
            result["ae"], result["normalizers"], train_z, val_z, val_sims,
            seed=SEED + 10000 + trial.number,
            device=device,
            rollout_eval_every=2,
            hidden_size=config["prop_hidden_size"],
            depth=config["prop_depth"],
            learning_rate=config["prop_lr"],
            unroll_steps=PROPAGATOR_UNROLL_STEPS,
        )
        predicted_val_z = rollout(model, stats, val_z, device).cpu()
        val_metrics = evaluate(
            result["ae"], result["normalizers"], val_sims,
            predicted_val_z, TARGET_STEP, device,
        )
        torch.save(
            {
                "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "normalization": {key: value.detach().cpu() for key, value in stats.items()},
                "configuration": config,
                "unroll_steps": PROPAGATOR_UNROLL_STEPS,
                "best_epoch": best["epoch"],
                "val_decoded_field_mse": best["loss"],
                "val_metrics": val_metrics,
            },
            case / "propagator_checkpoint.pt",
        )
        pd.DataFrame(best["history"]).to_csv(case / "propagator_history.csv", index=False)
        trial.set_user_attr("case", str(case))
        trial.set_user_attr("best_epoch", best["epoch"])
        trial.set_user_attr("val_r2", val_metrics["p_ratio_r2"])
        trial.set_user_attr("val_field_mse", best["loss"])
        print(json.dumps({"trial": trial.number, **val_metrics}, indent=2), flush=True)
        return float(val_metrics["p_ratio_r2"])

    study.optimize(objective, n_trials=args.trials, gc_after_trial=True)
    trials = study.trials_dataframe(attrs=("number", "value", "params", "user_attrs", "state"))
    trials.to_csv(OUTPUT / "trials.csv", index=False)

    best_trial = study.best_trial
    case = Path(best_trial.user_attrs["case"])
    ae, normalizers, params = restore_ae(case / "ae_checkpoint.pt", device)
    prop_bundle = torch.load(case / "propagator_checkpoint.pt", map_location="cpu", weights_only=False)
    config = prop_bundle["configuration"]
    model = DeltaMLP(
        3 * int(params["latent_dim"]) + 1,
        int(params["latent_dim"]),
        int(config["prop_hidden_size"]),
        int(config["prop_depth"]),
    ).to(device)
    model.load_state_dict(prop_bundle["model_state_dict"])
    stats = {key: value.to(device) for key, value in prop_bundle["normalization"].items()}
    _, _, test_sims = split_sims(all_sims, int(config["train_count"]))
    test_z = encode_latent_table(
        ae, normalizers, test_sims, max_frame=TARGET_STEP,
        batch_size=24, device=device, node_feature_mode="modular_history3",
    )
    predicted_test_z = rollout(model, stats, test_z, device).cpu()
    ae_test = evaluate(ae, normalizers, test_sims, test_z[:, TARGET_STEP], TARGET_STEP, device)
    rollout_test = evaluate(ae, normalizers, test_sims, predicted_test_z, TARGET_STEP, device)
    final = {
        "best_trial": best_trial.number,
        "best_validation_rollout_p_ratio_r2": best_trial.value,
        "config": config,
        "ae_test": ae_test,
        "rollout_test": rollout_test,
    }
    (OUTPUT / "best_trial_test.json").write_text(json.dumps(final, indent=2))
    print("\nBEST TRIAL TEST RESULT\n" + json.dumps(final, indent=2), flush=True)


if __name__ == "__main__":
    main()
