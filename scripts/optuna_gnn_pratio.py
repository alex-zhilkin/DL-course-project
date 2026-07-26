from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F

from lss.data import resolve_dataset_splits, simulation_temperature
from lss.graph import box_tensor, inverse_design_velocity_graph
from lss.inverse_design_barostat import STIFF_OPTIMIZED_BAROSTAT
from lss.inverse_design_evaluation import (
    box_p_ratio,
    compute_total_stress,
    inverse_design_rollout,
    prepare_inverse_design_graph,
)
from lss.latent.experiment import find_project_root, seed_everything
from lss.latent.simulation import (
    pearson_r,
    r2_score,
    trajectory_p_ratio_sides_robust_series,
)
from lss.models import create_model, resolve_model_inputs
from lss.utils import resolve_device


def sync_pos_to_x(trajectory):
    synced = []
    for graph in trajectory:
        if hasattr(graph, "pos") and hasattr(graph, "x") and graph.x is not None:
            graph = graph.clone()
            graph.x = graph.x.clone()
            graph.x[:, :2] = graph.pos.to(graph.x.device, graph.x.dtype)
        synced.append(graph)
    return synced


def p_ratio_series(trajectory, *, method: str = "trajectory_linear", min_abs_strain: float = 1e-5):
    values = np.full(len(trajectory), np.nan, dtype=float)
    if method == "endpoint":
        for stop in range(1, len(trajectory)):
            values[stop] = float(box_p_ratio(trajectory, last_index=stop).cpu())
        return values

    return trajectory_p_ratio_sides_robust_series(trajectory)


def acceleration_huber_loss(model, output, inputs, *, is_training: bool):
    target_velocity = inputs.target_position - inputs.cur_position
    current_velocity = inputs.cur_position - inputs.prev_position
    target_acceleration = target_velocity - current_velocity
    target_normalized = model.output_normalizer(
        target_acceleration,
        accumulate=bool(is_training) and not model.freeze_normalizers,
        is_training=is_training,
    )
    return F.huber_loss(output, target_normalized, delta=1.0)


def build_model(train_data, cfg: dict, *, seed: int, device):
    history = int(cfg["history"])
    init_window = [prepare_inverse_design_graph(train_data[0][i]) for i in range(history + 1)]
    init_graph = inverse_design_velocity_graph(init_window).to(device)
    seed_everything(seed)
    model = create_model(
        "inverse_design_simulator",
        init_graph,
        pos_dim=2,
        hidden_size=int(cfg["hidden_size"]),
        n_layers=int(cfg["message_passing_layers"]),
        extras={"num_mlp": int(cfg["mlp_layers"])},
    ).to(device)
    return model, resolve_model_inputs("inverse_design_simulator")


def train_one_epoch(model, model_inputs_cls, train_data, cfg: dict, *, device, optimizer):
    history = int(cfg["history"])
    model.train()
    total = 0.0
    samples = 0
    for sim in train_data:
        max_windows = min(int(cfg["train_windows_per_traj"]), len(sim) - history - 1)
        for start_idx in range(max_windows):
            target_idx = history + 1 + start_idx
            window = [
                prepare_inverse_design_graph(sim[start_idx + step])
                for step in range(history + 1)
            ]
            input_graph = inverse_design_velocity_graph(window).to(device)
            inputs = model_inputs_cls(
                window[-2].to(device),
                window[-1].to(device),
                prepare_inverse_design_graph(sim[target_idx]).to(device),
                2,
            )
            output = model(input_graph, is_training=True)
            loss = acceleration_huber_loss(model, output, inputs, is_training=True)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += float(loss.detach().cpu())
            samples += 1
    return total / max(samples, 1)


@torch.no_grad()
def validation_one_step(model, model_inputs_cls, val_data, cfg: dict, *, device):
    history = int(cfg["history"])
    model.eval()
    total = 0.0
    pos_total = 0.0
    samples = 0
    for sim in val_data:
        max_windows = min(int(cfg["train_windows_per_traj"]), len(sim) - history - 1)
        for start_idx in range(max_windows):
            target_idx = history + 1 + start_idx
            window = [
                prepare_inverse_design_graph(sim[start_idx + step])
                for step in range(history + 1)
            ]
            input_graph = inverse_design_velocity_graph(window).to(device)
            target = prepare_inverse_design_graph(sim[target_idx]).to(device)
            inputs = model_inputs_cls(window[-2].to(device), window[-1].to(device), target, 2)
            output = model(input_graph, is_training=False)
            loss = acceleration_huber_loss(model, output, inputs, is_training=False)
            prediction = model.update(inputs, output)
            total += float(loss.cpu())
            pos_total += float(F.mse_loss(prediction.pos, target.pos).cpu())
            samples += 1
    return {
        "val_huber": total / max(samples, 1),
        "val_position_mse": pos_total / max(samples, 1),
        "val_windows": samples,
    }


@torch.no_grad()
def evaluate_rollout(
    model,
    sims,
    *,
    cfg: dict,
    device,
    split_name: str,
    max_sims: int | None = None,
    p_ratio_method: str = "trajectory_linear",
):
    history = int(cfg["history"])
    rollout_steps = int(cfg["rollout_steps"])
    barostat_base = dict(STIFF_OPTIMIZED_BAROSTAT)
    model.eval()
    model.freeze_normalizers = True
    rows = []
    step_rows = []
    selected = sims[: int(max_sims)] if max_sims is not None else sims
    for sim_idx, sim in enumerate(selected):
        initial = [prepare_inverse_design_graph(sim[i]) for i in range(history + 1)]
        barostat = dict(barostat_base)
        if cfg.get("use_sim_temperature_in_barostat", False):
            barostat["temperature"] = simulation_temperature(sim)
        rollout_steps_for_sim = min(rollout_steps, len(sim) - history - 2)
        rollout = inverse_design_rollout(
            initial,
            model,
            num_steps=rollout_steps_for_sim,
            history=history,
            barostat_config=barostat,
            device=str(device),
        )
        eval_frames = min(len(rollout), len(sim))
        rollout = sync_pos_to_x(rollout[:eval_frames])
        truth = sync_pos_to_x(
            [prepare_inverse_design_graph(sim[i]) for i in range(eval_frames)]
        )
        pred_linear = p_ratio_series(rollout, method="trajectory_linear")
        true_linear = p_ratio_series(truth, method="trajectory_linear")
        pred_endpoint = p_ratio_series(rollout, method="endpoint")
        true_endpoint = p_ratio_series(truth, method="endpoint")
        pred_series = pred_linear if p_ratio_method != "endpoint" else pred_endpoint
        true_series = true_linear if p_ratio_method != "endpoint" else true_endpoint
        r0 = initial[0].edge_attr[:, -2]
        pred_pressure = torch.stack(
            [compute_total_stress(g, r0=r0, temperature=barostat["temperature"]).cpu() for g in rollout]
        )
        true_pressure = torch.stack(
            [compute_total_stress(g, r0=r0, temperature=barostat["temperature"]).cpu() for g in truth]
        )
        rows.append(
            {
                "split": split_name,
                "sim_idx": sim_idx,
                "temperature": simulation_temperature(sim),
                "requested_rollout_steps": rollout_steps,
                "evaluated_rollout_steps": rollout_steps_for_sim,
                "evaluated_frames": eval_frames,
                "pressure_rmse": float(torch.sqrt(torch.mean((pred_pressure - true_pressure).square()))),
                "predicted_linear_p_ratio": pred_linear[-1],
                "true_linear_p_ratio": true_linear[-1],
                "predicted_endpoint_p_ratio": pred_endpoint[-1],
                "true_endpoint_p_ratio": true_endpoint[-1],
                "predicted_p_ratio": pred_series[-1],
                "true_p_ratio": true_series[-1],
            }
        )
        for step in range(eval_frames):
            step_rows.append(
                {
                    "split": split_name,
                    "sim_idx": sim_idx,
                    "step": step,
                    "position_mse": float(
                        F.mse_loss(
                            rollout[step].pos.detach().cpu(),
                            truth[step].pos.detach().cpu(),
                        )
                    ),
                    "pred_p_ratio_linear": pred_linear[step],
                    "true_p_ratio_linear": true_linear[step],
                    "pred_p_ratio_endpoint": pred_endpoint[step],
                    "true_p_ratio_endpoint": true_endpoint[step],
                    "pred_p_ratio": pred_series[step],
                    "true_p_ratio": true_series[step],
                }
            )
    frame = pd.DataFrame(rows)
    step_frame = pd.DataFrame(step_rows)
    metrics = {
        "split": split_name,
        "used": int(len(frame)),
        "p_ratio_r2": r2_score(frame["true_p_ratio"], frame["predicted_p_ratio"]) if len(frame) else float("nan"),
        "p_ratio_pearson": pearson_r(frame["true_p_ratio"], frame["predicted_p_ratio"]) if len(frame) else float("nan"),
        "p_ratio_mse": float(np.nanmean((frame["true_p_ratio"] - frame["predicted_p_ratio"]) ** 2)) if len(frame) else float("nan"),
        "position_mse": float(step_frame["position_mse"].mean()) if len(step_frame) else float("nan"),
    }
    return metrics, frame, step_frame


def sample_params(trial: optuna.Trial, args: argparse.Namespace) -> dict:
    hidden_size = trial.suggest_categorical("hidden_size", [64, 96, 128, 192, 256])
    layers = trial.suggest_int("message_passing_layers", 1, 4)
    mlp_layers = trial.suggest_int("mlp_layers", 2, 5)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 3e-3, log=True)
    lr_decay = trial.suggest_float("learning_rate_decay", 0.985, 0.9995)
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-4, log=True)
    freeze_epoch = trial.suggest_int("freeze_normalizers_epoch", 2, max(2, min(10, args.epochs - 1)))
    return {
        "history": args.history,
        "train_windows_per_traj": args.train_windows_per_traj,
        "epochs": args.epochs,
        "hidden_size": hidden_size,
        "message_passing_layers": layers,
        "mlp_layers": mlp_layers,
        "freeze_normalizers_epoch": freeze_epoch,
        "learning_rate": learning_rate,
        "learning_rate_decay": lr_decay,
        "weight_decay": weight_decay,
        "rollout_steps": args.rollout_steps,
        "use_sim_temperature_in_barostat": args.use_sim_temperature_in_barostat,
    }


def objective_factory(args, output_dir: Path, train_data, val_data, test_data, device):
    model_inputs_cls = resolve_model_inputs("inverse_design_simulator")

    def objective(trial: optuna.Trial) -> float:
        trial_dir = output_dir / "trials" / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        cfg = sample_params(trial, args)
        trial_seed = int(args.seed + 1009 * trial.number)
        seed_everything(trial_seed)
        model, _ = build_model(train_data, cfg, seed=trial_seed, device=device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(cfg["learning_rate"]),
            weight_decay=float(cfg["weight_decay"]),
        )
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=float(cfg["learning_rate_decay"])
        )

        history_rows = []
        best_value = -float("inf")
        best_state = None
        started = time.perf_counter()
        for epoch in range(1, int(cfg["epochs"]) + 1):
            epoch_started = time.perf_counter()
            if epoch == int(cfg["freeze_normalizers_epoch"]):
                model.freeze_normalizers = True
            train_huber = train_one_epoch(
                model, model_inputs_cls, train_data, cfg, device=device, optimizer=optimizer
            )
            val_info = validation_one_step(model, model_inputs_cls, val_data, cfg, device=device)
            rollout_info = {
                "p_ratio_r2": float("nan"),
                "p_ratio_pearson": float("nan"),
                "p_ratio_mse": float("nan"),
                "position_mse": float("nan"),
            }
            if epoch == int(cfg["epochs"]) or epoch % int(args.rollout_every) == 0:
                rollout_info, rollout_rows, rollout_step_rows = evaluate_rollout(
                    model,
                    val_data,
                    cfg=cfg,
                    device=device,
                    split_name="val",
                    max_sims=args.rollout_eval_sims,
                    p_ratio_method=args.p_ratio_method,
                )
                rollout_rows.to_csv(trial_dir / f"val_rollout_epoch_{epoch:03d}.csv", index=False)
                rollout_step_rows.to_csv(
                    trial_dir / f"val_rollout_steps_epoch_{epoch:03d}.csv", index=False
                )
                value = float(rollout_info["p_ratio_r2"])
                if math.isfinite(value) and value > best_value:
                    best_value = value
                    best_state = {
                        key: val.detach().cpu().clone()
                        for key, val in model.state_dict().items()
                    }
                trial.report(value if math.isfinite(value) else -1e9, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            scheduler.step()
            row = {
                "trial": trial.number,
                "epoch": epoch,
                "train_huber": train_huber,
                **val_info,
                "val_rollout_p_ratio_r2": rollout_info["p_ratio_r2"],
                "val_rollout_p_ratio_pearson": rollout_info["p_ratio_pearson"],
                "val_rollout_p_ratio_mse": rollout_info["p_ratio_mse"],
                "val_rollout_position_mse": rollout_info["position_mse"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "epoch_seconds": time.perf_counter() - epoch_started,
            }
            history_rows.append(row)
            pd.DataFrame(history_rows).to_csv(trial_dir / "training_history.csv", index=False)
            print(
                f"trial={trial.number:04d} ep={epoch:03d} "
                f"train={train_huber:.3e} val={val_info['val_huber']:.3e} "
                f"r2={rollout_info['p_ratio_r2']:.4g} best={best_value:.4g} "
                f"t={row['epoch_seconds']:.1f}s",
                flush=True,
            )

        if best_state is None:
            best_state = {
                key: val.detach().cpu().clone()
                for key, val in model.state_dict().items()
            }
        torch.save(
            {
                "state_dict": best_state,
                "config": cfg,
                "trial": trial.number,
                "seed": trial_seed,
                "best_val_p_ratio_r2": best_value,
            },
            trial_dir / "best_checkpoint.pt",
        )
        (trial_dir / "config.json").write_text(json.dumps(cfg, indent=2))
        trial.set_user_attr("seconds", time.perf_counter() - started)
        trial.set_user_attr("best_val_p_ratio_r2", best_value)
        return float(best_value)

    return objective


def evaluate_best(args, output_dir: Path, study, train_data, val_data, test_data, device):
    if study.best_trial is None:
        return
    best_trial = study.best_trial
    checkpoint = torch.load(
        output_dir / "trials" / f"trial_{best_trial.number:04d}" / "best_checkpoint.pt",
        map_location=device,
        weights_only=False,
    )
    cfg = checkpoint["config"]
    model, _ = build_model(train_data, cfg, seed=int(checkpoint["seed"]), device=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    model.freeze_normalizers = True
    metrics = []
    for split_name, sims, max_sims in (
        ("val", val_data, args.rollout_eval_sims),
        ("test", test_data, args.test_eval_sims),
    ):
        split_metrics, rows, step_rows = evaluate_rollout(
            model,
            sims,
            cfg=cfg,
            device=device,
            split_name=split_name,
            max_sims=max_sims,
            p_ratio_method=args.p_ratio_method,
        )
        rows.to_csv(output_dir / f"best_{split_name}_rollout_rows.csv", index=False)
        step_rows.to_csv(output_dir / f"best_{split_name}_rollout_step_rows.csv", index=False)
        metrics.append(split_metrics)
    metrics_frame = pd.DataFrame(metrics)
    metrics_frame.to_csv(output_dir / "best_rollout_metrics.csv", index=False)
    best_payload = {
        "best_trial": best_trial.number,
        "best_value": float(best_trial.value),
        "best_params": best_trial.params,
        "best_config": cfg,
        "rollout_metrics": metrics_frame.to_dict("records"),
    }
    (output_dir / "best_params.json").write_text(json.dumps(best_payload, indent=2))
    torch.save(
        {
            "state_dict": checkpoint["state_dict"],
            "config": cfg,
            "trial": best_trial.number,
            "seed": checkpoint["seed"],
            "study_best_value": float(best_trial.value),
        },
        output_dir / "best_checkpoint.pt",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna search for inverse-design GNN p-ratio rollout R2."
    )
    parser.add_argument("--dataset-path", default="data/depablo-10k-mix-temp.pt")
    parser.add_argument("--output-dir", default="notebooks/results/optuna_gnn_pratio_depablo_mixed_temp")
    parser.add_argument("--study-name", default="gnn_pratio_depablo_mixed_temp")
    parser.add_argument("--storage", default=None)
    parser.add_argument("--n-trials", type=int, default=24)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--train-count", type=int, default=15)
    parser.add_argument("--val-count", type=int, default=15)
    parser.add_argument("--test-count", type=int, default=100)
    parser.add_argument("--train-windows-per-traj", type=int, default=100)
    parser.add_argument("--history", type=int, default=3)
    parser.add_argument("--rollout-steps", type=int, default=199)
    parser.add_argument("--rollout-every", type=int, default=5)
    parser.add_argument("--rollout-eval-sims", type=int, default=15)
    parser.add_argument("--test-eval-sims", type=int, default=100)
    parser.add_argument("--p-ratio-method", default="trajectory_linear", choices=["trajectory_linear", "endpoint"])
    parser.add_argument("--seed", type=int, default=20260622)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-stratify-temperature", action="store_true")
    parser.add_argument("--use-sim-temperature-in-barostat", action="store_true")
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = find_project_root()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    storage = args.storage or f"sqlite:///{output_dir / 'study.db'}"
    seed_everything(args.seed)
    device = resolve_device(args.device)
    train_data, val_data, test_data, split_info = resolve_dataset_splits(
        project_root / args.dataset_path,
        train_count=args.train_count,
        val_count=args.val_count,
        split_seed=args.seed,
        shuffle_within_source=True,
        stratify_temperature=not args.no_stratify_temperature,
    )
    test_data = test_data[: int(args.test_count)]
    pd.DataFrame(split_info).to_csv(output_dir / "split_info.csv", index=False)
    config_payload = vars(args) | {
        "resolved_dataset_path": str((project_root / args.dataset_path).resolve()),
        "device": str(device),
        "storage": storage,
    }
    (output_dir / "run_config.json").write_text(json.dumps(config_payload, indent=2))
    print(json.dumps(config_payload, indent=2), flush=True)

    sampler = optuna.samplers.TPESampler(seed=args.seed, multivariate=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=max(1, args.rollout_every))
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )
    objective = objective_factory(args, output_dir, train_data, val_data, test_data, device)
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, n_jobs=args.n_jobs)
    trials_df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
    trials_df.to_csv(output_dir / "trials.csv", index=False)
    evaluate_best(args, output_dir, study, train_data, val_data, test_data, device)
    print("best trial:", study.best_trial.number, flush=True)
    print("best value:", study.best_value, flush=True)
    print("best params:", study.best_params, flush=True)


if __name__ == "__main__":
    main()
