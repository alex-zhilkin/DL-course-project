"""One-step acceleration-propagator Optuna screen on frozen notebook-08 latents.

Optuna maximizes *validation* noisy-LJ p-ratio R².  The test split remains a
reporting-only scorecard; it is never used to select a trial.  Training and
early stopping still use latent-space loss, not p-ratio.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import optuna
import pandas as pd
from optuna.samplers import TPESampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lss.latent.experiment import run_latent_experiment, seed_everything
from lss.latent.simulation import r2_score
from lss.utils import resolve_device
from sweep_08_frozen_ae_acceleration import (
    AE_CACHE,
    AE_CONFIG,
    FEATURE_CACHE,
    OUTPUT,
    SEED,
    source_spec,
)


TRIAL_CACHE = OUTPUT / "one_step_acceleration_optuna"
STUDY_PATH = TRIAL_CACHE / "noisy_lj_70val_r2_mlp_study.sqlite3"


def config_for_trial(args, trial: optuna.Trial) -> dict:
    hidden_size = trial.suggest_categorical(
        "hidden_size", [64, 96, 128, 160, 192]
    )
    lr = trial.suggest_float("lr", 2e-5, 3e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-7, 3e-4, log=True)
    trial_seed = SEED + int(trial.number)
    propagator_config = {
        "model": "history_mlp",
        "hidden_size": hidden_size,
        "history_depth": trial.suggest_int("history_depth", 2, 7),
        "history_activation": trial.suggest_categorical(
            "history_activation", ["gelu", "silu", "relu", "leaky_relu"]
        ),
        "history_dropout": trial.suggest_categorical(
            "history_dropout", [0.0, 0.025, 0.05, 0.10]
        ),
        "objective": "kinematic_multistep",
        "loss": "delta",
        "step_stride": 1,
        "initial_velocity": "three_frames",
        "rollout_history_frames": 3,
        # Strictly one-step acceleration training.
        "multistep_horizons": [1],
        "max_train_transitions_per_sim": args.max_transitions,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "lr": lr,
        "weight_decay": weight_decay,
        "mix_sources": True,
        "source_loss_reduction": "pooled",
        "train_trajectories_per_source": {
            "reid": 30, "depablo_low_temp": 30, "lj_noisy": 60,
        },
        "val_trajectories_per_source": {
            "reid": 30, "depablo_low_temp": 30, "lj_noisy": 70,
        },
        "use_static_context": True,
        "context_pool": "mean",
        "context_dim": 16,
        "context_include_temperature": False,
        "context_include_source_id": False,
        "frozen_latent_cache_dir": str(FEATURE_CACHE),
        # Epoch training and early stopping use latent loss only.
        "rollout_eval_every_epoch": False,
        "checkpoint_metric": None,
        "checkpoint_mode": "min",
    }
    return {
        "ae_config": AE_CONFIG,
        "propagator_config": propagator_config,
        "dataset_name": "08 frozen-AE one-step acceleration Optuna (70 noisy-LJ val)",
        "split_seed": SEED,
        "model_seed": trial_seed,
        "device": str(resolve_device("auto")),
        "pos_dim": 2,
        "batch_graphs": 32,
        "frame_skip": 1,
        "coordinate_normalization": "position_normalization",
        "edge_mode": "compact_stored",
        "static_context_use_physical_reference": True,
        "rollout_steps_grid": [10, 50, 100],
        "rollout_final_eval_sims_per_source": 70,
        "should_rollout": True,
        "should_train_propagator": True,
        "pretrained_ae_cache_path": str(AE_CACHE),
        "pretrained_ae_config_keys": [
            "split_seed", "coordinate_normalization", "edge_mode", "pos_dim",
            "latent_dim", "latent_tokens", "hidden_size", "autoencoder_model",
            "edge_feature_dim", "node_feature_mode", "ae_target_mode",
            "ae_max_train_frames_per_sim", "ae_max_val_frames_per_sim",
        ],
        "pretrained_ae_require_matching_config": True,
        "pretrained_ae_require_matching_normalizers": True,
        "force_train_autoencoder": False,
        "force_train": args.force,
        "cache_require_matching_config": True,
        "cache_dir": str(TRIAL_CACHE / "trial_cache"),
        "early_stop_min_delta": 1e-5,
        # Used only after fitting to select with validation R² and report test R².
        "p_ratio_estimator": "endpoint",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--max-epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--max-transitions", type=int, default=75)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not AE_CACHE.is_file():
        raise FileNotFoundError(f"Frozen AE cache is missing: {AE_CACHE}")
    TRIAL_CACHE.mkdir(parents=True, exist_ok=True)
    study = optuna.create_study(
        study_name="08_frozen_ae_one_step_acceleration_noisy_lj_70val_r2_mlp",
        storage=f"sqlite:///{STUDY_PATH}",
        load_if_exists=True,
        direction="maximize",
        sampler=TPESampler(seed=SEED),
    )
    source = source_spec()
    for item in source["dataset_mixture"]:
        if item["name"] == "lj_noisy":
            item["val_count"] = 70

    def objective(trial: optuna.Trial) -> float:
        config = config_for_trial(args, trial)
        seed_everything(SEED + int(trial.number))
        result = run_latent_experiment(source, config, device=resolve_device("auto"))
        rollout_rows = result["rollout_rows"]
        validation = rollout_rows.query(
            "split == 'val' and source == 'lj_noisy' and rollout_steps == 100"
        )
        if len(validation) < 2:
            raise RuntimeError("Need at least two noisy-LJ validation rollouts for R².")
        validation_r2 = float(
            r2_score(validation.true_p_ratio, validation.pred_p_ratio)
        )
        trial.set_user_attr("val_step100_lj_noisy_p_ratio_r2", validation_r2)
        heldout = rollout_rows.query("split == 'test' and rollout_steps == 100")
        for name, group in heldout.groupby("source"):
            trial.set_user_attr(
                f"test_step100_{name}_p_ratio_r2",
                float(r2_score(group.true_p_ratio, group.pred_p_ratio)),
            )
        return validation_r2

    study.optimize(objective, n_trials=args.trials)
    rows = []
    for trial in study.trials:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
        row = {"trial": trial.number, "val_step100_lj_noisy_p_ratio_r2": trial.value, **trial.params}
        row.update(trial.user_attrs)
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(
        "val_step100_lj_noisy_p_ratio_r2", ascending=False
    )
    summary.to_csv(TRIAL_CACHE / "summary.csv", index=False)
    print("\nRanked by validation noisy-LJ step-100 p-ratio R²:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
