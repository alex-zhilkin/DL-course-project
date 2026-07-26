from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import optuna
import pandas as pd
import torch

from lss.latent.capacity import save_experiment_bundle
from lss.latent.experiment import (
    find_project_root,
    prepare_source_spec,
    seed_everything,
    train_latent_experiment,
)
from lss.utils import resolve_device


PROPAGATOR_PRESETS = {
    "residual_mlp": {
        "propagator_objective": "one_step",
        "propagator_model": "residual_mlp",
        "propagator_loss": "delta",
    },
    "delta_mlp": {
        "propagator_objective": "one_step",
        "propagator_model": "delta_mlp",
        "propagator_loss": "delta",
    },
    "direct_mlp": {
        "propagator_objective": "one_step",
        "propagator_model": "direct_mlp",
        "propagator_loss": "next_z",
    },
    "linear": {
        "propagator_objective": "one_step",
        "propagator_model": "linear",
        "propagator_loss": "delta",
    },
    "velocity_mlp": {
        "propagator_objective": "velocity",
        "propagator_model": "velocity_mlp",
        "propagator_loss": "velocity_delta",
    },
    "polar_rho": {
        "propagator_objective": "one_step",
        "propagator_model": "polar_rho",
        "propagator_loss": "next_z",
        "polar_rho_scale_mode": "box_width",
    },
}


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in str(value).split(",") if item.strip()]


def parse_csv_strings(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def sample_params(trial: optuna.Trial, args: argparse.Namespace) -> dict:
    propagators = parse_csv_strings(args.propagators)
    propagator = trial.suggest_categorical("propagator", propagators)
    if propagator not in PROPAGATOR_PRESETS:
        raise ValueError(
            f"Unknown propagator {propagator!r}. "
            f"Known: {sorted(PROPAGATOR_PRESETS)}"
        )

    latent_dims = parse_csv_ints(args.latent_dims)
    if propagator == "polar_rho":
        if 2 not in latent_dims:
            raise ValueError("polar_rho requires latent dim 2; include --latent-dims 2.")
        latent_dim = 2
    else:
        latent_dim = trial.suggest_categorical("latent_dim", latent_dims)

    cfg = {
        "latent_dim": int(latent_dim),
        "latent_tokens": trial.suggest_categorical(
            "latent_tokens", parse_csv_ints(args.latent_tokens)
        ),
        "hidden_size": trial.suggest_categorical(
            "hidden_size", parse_csv_ints(args.hidden_sizes)
        ),
        "graph_context_dim": trial.suggest_categorical(
            "graph_context_dim", parse_csv_ints(args.graph_context_dims)
        ),
        "ae_lr": trial.suggest_float("ae_lr", args.ae_lr_min, args.ae_lr_max, log=True),
        "dyn_lr": trial.suggest_float("dyn_lr", args.dyn_lr_min, args.dyn_lr_max, log=True),
        "dyn_weight_decay": trial.suggest_float(
            "dyn_weight_decay", args.dyn_weight_decay_min, args.dyn_weight_decay_max, log=True
        ),
        "propagator_step_stride": trial.suggest_categorical(
            "propagator_step_stride", parse_csv_ints(args.step_strides)
        ),
        "propagator_name": propagator,
        **PROPAGATOR_PRESETS[propagator],
    }
    return cfg


def base_config(args: argparse.Namespace, *, device) -> dict:
    return {
        "dataset_name": args.dataset_name,
        "split_seed": int(args.seed),
        "split_stratify_temperature": bool(args.stratify_temperature),
        "min_train_p_ratio": None,
        "device": str(device),
        "pos_dim": 2,
        "batch_graphs": int(args.batch_graphs),
        "frame_skip": int(args.frame_skip),
        "train_frame_start_order": 0,
        "edge_feature_dim": 12,
        "ae_target_mode": args.ae_target_mode,
        "node_feature_mode": args.ae_target_mode,
        "ae_max_train_frames_per_sim": int(args.train_frames_per_sim),
        "dyn_max_train_transitions_per_sim": int(args.train_transitions_per_sim),
        "ae_max_epochs": int(args.ae_epochs),
        "ae_patience": int(args.ae_patience),
        "ae_weight_decay": float(args.ae_weight_decay),
        "dyn_max_epochs": int(args.dyn_epochs),
        "dyn_patience": int(args.dyn_patience),
        "propagator_use_static_context": not args.no_static_context,
        "propagator_context_include_temperature": bool(args.context_temperature),
        "initial_velocity": args.initial_velocity,
        "early_stop_min_delta": float(args.early_stop_min_delta),
        "rollout_steps_grid": parse_csv_ints(args.rollout_steps),
        "temperature_pratio_window": args.temperature_pratio_window,
    }


def score_rollout(stats: pd.DataFrame, args: argparse.Namespace) -> tuple[float, dict]:
    split = stats[stats["split"].eq("val")].copy()
    if split.empty:
        return -1e9, {}
    if args.score_horizon == "max":
        horizon = int(split["rollout_steps"].max())
    else:
        horizon = int(args.score_horizon)
    row = split[split["rollout_steps"].eq(horizon)].tail(1)
    if row.empty:
        row = split.sort_values("rollout_steps").tail(1)
        horizon = int(row["rollout_steps"].iloc[0])

    values = row.iloc[0].to_dict()
    p_ratio_r2 = float(values.get("p_ratio_r2", float("nan")))
    p_ratio_mse = float(values.get("p_ratio_mse", float("nan")))
    final_pos_mse = float(values.get("final_pos_mse", float("nan")))
    error_fraction = float(values.get("rollout_error_fraction", float("nan")))

    if args.objective_metric == "p_ratio_r2":
        score = p_ratio_r2
    elif args.objective_metric == "neg_p_ratio_mse":
        score = -p_ratio_mse
    elif args.objective_metric == "neg_final_pos_mse":
        score = -final_pos_mse
    elif args.objective_metric == "combined":
        penalty = error_fraction if math.isfinite(error_fraction) else 1e6
        score = p_ratio_r2 - float(args.position_weight) * penalty
    else:
        raise ValueError(f"Unknown objective metric: {args.objective_metric}")

    if not math.isfinite(score):
        score = -1e9
    return float(score), {
        "score_horizon": horizon,
        "val_p_ratio_r2": p_ratio_r2,
        "val_p_ratio_mse": p_ratio_mse,
        "val_final_pos_mse": final_pos_mse,
        "val_rollout_error_fraction": error_fraction,
    }


def objective_factory(args: argparse.Namespace, output_dir: Path, device):
    dataset_specs = {
        args.dataset_name: {
            "label": args.dataset_label,
            "path": str(args.dataset_path),
            "train_count": int(args.train_count),
            "val_count": int(args.val_count),
        }
    }
    common_cfg = base_config(args, device=device)

    def objective(trial: optuna.Trial) -> float:
        started = time.perf_counter()
        trial_dir = output_dir / "trials" / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        sampled = sample_params(trial, args)
        trial_seed = int(args.seed + 1009 * trial.number)
        cfg = {
            **common_cfg,
            **sampled,
            "model_seed": trial_seed,
            "repeat_idx": int(trial.number + 1),
        }
        seed_everything(trial_seed)
        source_spec = prepare_source_spec(
            args.dataset_name,
            dataset_specs,
            cfg,
            seed=int(args.seed),
        )

        print(
            f"trial={trial.number:04d} "
            f"propagator={sampled['propagator_name']} "
            f"latent_dim={cfg['latent_dim']} "
            f"hidden={cfg['hidden_size']} "
            f"seed={trial_seed}",
            flush=True,
        )
        result = train_latent_experiment(source_spec, cfg, device=device)
        result["rollout_stats"].to_csv(trial_dir / "rollout_stats.csv", index=False)
        result["rollout_rows"].to_csv(trial_dir / "rollout_rows.csv", index=False)
        result["ae_history"].to_csv(trial_dir / "ae_history.csv", index=False)
        result["dyn_history"].to_csv(trial_dir / "dyn_history.csv", index=False)
        pd.DataFrame([vars(args)]).to_csv(trial_dir / "args.csv", index=False)
        (trial_dir / "config.json").write_text(json.dumps(cfg, indent=2))
        save_experiment_bundle(result, source_spec, trial_dir / "experiment_bundle.pt")

        score, score_info = score_rollout(result["rollout_stats"], args)
        for key, value in score_info.items():
            trial.set_user_attr(key, value)
        trial.set_user_attr("seconds", time.perf_counter() - started)
        trial.set_user_attr("propagator", sampled["propagator_name"])
        trial.set_user_attr("latent_dim", int(cfg["latent_dim"]))
        trial.report(score, step=1)
        if trial.should_prune():
            raise optuna.TrialPruned()

        payload = {
            "trial": trial.number,
            "score": score,
            "score_info": score_info,
            "config": cfg,
        }
        (trial_dir / "score.json").write_text(json.dumps(payload, indent=2))
        print(
            f"trial={trial.number:04d} score={score:.6g} "
            f"val_r2={score_info.get('val_p_ratio_r2', float('nan')):.6g} "
            f"val_pos_mse={score_info.get('val_final_pos_mse', float('nan')):.6g} "
            f"seconds={time.perf_counter() - started:.1f}",
            flush=True,
        )
        return score

    return objective


def write_best_summary(output_dir: Path, study: optuna.Study) -> None:
    trials_df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
    trials_df.to_csv(output_dir / "trials.csv", index=False)
    completed = [
        trial
        for trial in study.trials
        if trial.state is optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    if not completed:
        (output_dir / "best_params.json").write_text(
            json.dumps(
                {
                    "best_trial": None,
                    "best_value": None,
                    "status": "no completed trials",
                },
                indent=2,
            )
        )
        return
    best = study.best_trial
    best_dir = output_dir / "trials" / f"trial_{best.number:04d}"
    best_payload = {
        "best_trial": best.number,
        "best_value": float(best.value),
        "best_params": best.params,
        "best_user_attrs": best.user_attrs,
        "best_trial_dir": str(best_dir),
    }
    config_path = best_dir / "config.json"
    if config_path.exists():
        best_payload["best_config"] = json.loads(config_path.read_text())
    (output_dir / "best_params.json").write_text(json.dumps(best_payload, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna search for latent-space simulator p-ratio rollout metrics."
    )
    parser.add_argument("--dataset-name", default="noizy")
    parser.add_argument("--dataset-label", default="Noizy")
    parser.add_argument("--dataset-path", default="data/200_rand_pruned_OOL_bidirect_val1057.pt")
    parser.add_argument("--output-dir", default="results/optuna_latent_pratio_noizy")
    parser.add_argument("--study-name", default="latent_pratio_noizy")
    parser.add_argument("--storage", default=None)
    parser.add_argument("--n-trials", type=int, default=24)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--train-count", type=int, default=80)
    parser.add_argument("--val-count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-jobs", type=int, default=1)

    parser.add_argument("--latent-dims", default="2,4,6,8")
    parser.add_argument("--latent-tokens", default="16,32")
    parser.add_argument("--hidden-sizes", default="64,96,128")
    parser.add_argument("--graph-context-dims", default="8,16,32")
    parser.add_argument(
        "--propagators",
        default="residual_mlp,delta_mlp,direct_mlp,linear,velocity_mlp",
    )
    parser.add_argument("--step-strides", default="1,2,5")

    parser.add_argument("--batch-graphs", type=int, default=4)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--train-frames-per-sim", type=int, default=60)
    parser.add_argument("--train-transitions-per-sim", type=int, default=60)
    parser.add_argument("--ae-target-mode", default="normalized_delta")
    parser.add_argument("--ae-epochs", type=int, default=200)
    parser.add_argument("--dyn-epochs", type=int, default=200)
    parser.add_argument("--ae-patience", type=int, default=6)
    parser.add_argument("--dyn-patience", type=int, default=6)
    parser.add_argument("--ae-lr-min", type=float, default=1e-5)
    parser.add_argument("--ae-lr-max", type=float, default=2e-4)
    parser.add_argument("--dyn-lr-min", type=float, default=1e-5)
    parser.add_argument("--dyn-lr-max", type=float, default=3e-4)
    parser.add_argument("--ae-weight-decay", type=float, default=1e-5)
    parser.add_argument("--dyn-weight-decay-min", type=float, default=1e-6)
    parser.add_argument("--dyn-weight-decay-max", type=float, default=1e-3)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-5)

    parser.add_argument("--rollout-steps", default="10,20,50,100,150")
    parser.add_argument("--score-horizon", default="max")
    parser.add_argument(
        "--objective-metric",
        default="p_ratio_r2",
        choices=["p_ratio_r2", "neg_p_ratio_mse", "neg_final_pos_mse", "combined"],
    )
    parser.add_argument("--position-weight", type=float, default=0.1)
    parser.add_argument("--temperature-pratio-window", default="full")
    parser.add_argument("--initial-velocity", default="zero", choices=["zero", "mean", "first_step", "gt_first", "observed"])
    parser.add_argument("--context-temperature", action="store_true")
    parser.add_argument("--stratify-temperature", action="store_true")
    parser.add_argument("--no-static-context", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_propagators = parse_csv_strings(args.propagators)
    unknown_propagators = sorted(set(requested_propagators) - set(PROPAGATOR_PRESETS))
    if unknown_propagators:
        raise ValueError(
            f"Unsupported propagators: {unknown_propagators}. "
            f"Choose only from: {sorted(PROPAGATOR_PRESETS)}"
        )
    project_root = find_project_root()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    args.dataset_path = str((project_root / args.dataset_path).resolve())
    storage = args.storage or f"sqlite:///{output_dir / 'study.db'}"
    seed_everything(args.seed)
    device = resolve_device(args.device)

    config_payload = vars(args) | {
        "resolved_dataset_path": args.dataset_path,
        "device": str(device),
        "storage": storage,
    }
    (output_dir / "run_config.json").write_text(json.dumps(config_payload, indent=2))
    print(json.dumps(config_payload, indent=2), flush=True)

    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=1)
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )
    study.optimize(
        objective_factory(args, output_dir, device),
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=args.n_jobs,
        catch=(Exception,),
    )
    write_best_summary(output_dir, study)
    completed = [
        trial
        for trial in study.trials
        if trial.state is optuna.trial.TrialState.COMPLETE and trial.value is not None
    ]
    if completed:
        print("best trial:", study.best_trial.number, flush=True)
        print("best value:", study.best_value, flush=True)
        print("best params:", study.best_params, flush=True)
    else:
        print("study finished without a completed trial", flush=True)


if __name__ == "__main__":
    main()
