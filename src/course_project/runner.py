from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig
from .data import split_dataset
from .graph import build_graph
from .metrics import evaluate_cv_vs_global_pratio, evaluate_rollout_pratio_sides, write_csv
from .models import create_model, resolve_model_inputs
from .training import train_graph_cv_model, train_graph_model
from .utils import resolve_device


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _3g(value) -> str:
    return f"{float(value):.3g}"


def _select_best_rollout_checkpoint(entries: list[dict]) -> dict:
    return max(entries, key=lambda e: (float(e["rollout_r2"]), int(e["epoch"])))


def _select_best_cv_checkpoint(entries: list[dict], stats: dict) -> dict:
    epoch_to_path = {int(e["epoch"]): str(e["path"]) for e in entries}
    candidates = []
    for epoch, score in zip(stats["epoch"], stats["cv_fit_r2"]):
        value = float(score)
        ep = int(epoch)
        if np.isfinite(value) and ep in epoch_to_path:
            candidates.append({"epoch": ep, "path": epoch_to_path[ep], "cv_fit_r2": value})
    return max(candidates, key=lambda e: (float(e["cv_fit_r2"]), int(e["epoch"])))


def _write_rollout_scatter(rows: list[dict], out_path: Path, title: str) -> None:
    x = np.asarray([row["target_rollout_sides_p_ratio"] for row in rows], dtype=float)
    y = np.asarray([row["pred_rollout_p_ratio"] for row in rows], dtype=float)
    lo = float(np.min(np.concatenate([x, y])))
    hi = float(np.max(np.concatenate([x, y])))
    pad = max((hi - lo) * 0.05, 1e-6)

    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.scatter(x, y, s=22, alpha=0.75)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=1.0)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("Ground truth p-ratio")
    ax.set_ylabel("Predicted p-ratio")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _build_model_and_data(cfg: ExperimentConfig):
    _set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    cfg_dict = cfg.to_dict()
    train_data, val_data, _test_data = split_dataset(cfg.dataset_path, train_count=cfg.train_count, val_count=cfg.val_count)
    init_frames = [train_data[0][i].to(device) for i in range(cfg.history + 1)]
    init_graph = build_graph(input_graphs=init_frames).to(device)
    model_inputs_cls = resolve_model_inputs(cfg.model_type)
    model = create_model(
        model_type=cfg.model_type,
        init_graph=init_graph,
        pos_dim=cfg.pos_dim,
        hidden_size=cfg.hidden_size,
        n_layers=cfg.n_layers,
        extras=cfg.model_extras,
    ).to(device)
    model.cfg = cfg_dict
    return device, cfg_dict, train_data, val_data, model_inputs_cls, model


def _make_checkpoint_callback(run_dir: Path, cfg_dict: dict):
    rollout_checkpoints_dir = run_dir / "rollout_checkpoints"
    rollout_checkpoints_dir.mkdir(parents=True, exist_ok=True)
    rollout_checkpoints: list[dict] = []

    def _save_rollout_checkpoint(epoch, current_model, optimizer, scheduler, rollout_metrics, train_loss, val_loss):
        path = rollout_checkpoints_dir / f"epoch_{epoch:04d}.pt"
        rollout_summary = {k: v for k, v in rollout_metrics.items() if k != "rows"}
        training_state = {
            "epoch": int(epoch),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "rollout_metrics": rollout_summary,
            "train_loss": float(train_loss),
            "val_loss": float("nan") if val_loss is None else float(val_loss),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        current_model.cfg = cfg_dict
        current_model.save_checkpoint(str(path), training_state=training_state)
        rollout_checkpoints.append(
            {
                "epoch": int(epoch),
                "path": str(path),
                "rollout_r2": float(rollout_metrics.get("rollout_r2", float("nan"))),
                "rollout_pos_mse": float(rollout_metrics.get("rollout_pos_mse", float("nan"))),
                "used": int(rollout_metrics.get("used", 0)),
                "total": int(rollout_metrics.get("total", 0)),
            }
        )

    return rollout_checkpoints, _save_rollout_checkpoint


def run_graph_experiment(cfg: ExperimentConfig) -> dict:
    device, cfg_dict, train_data, val_data, model_inputs_cls, model = _build_model_and_data(cfg)
    run_dir = Path(cfg.output_root) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] {cfg.run_name} model={cfg.model_type} device={device} output={run_dir}", flush=True)

    rollout_checkpoints, save_checkpoint = _make_checkpoint_callback(run_dir, cfg_dict)

    def _rollout_eval(current_model):
        return evaluate_rollout_pratio_sides(model=current_model, sims=val_data, history=cfg.history, rollout_steps=cfg.rollout_steps, pos_dim=cfg.pos_dim, device=device, model_inputs_cls=model_inputs_cls)

    print("[run] training...", flush=True)
    stats = train_graph_model(
        model,
        model_inputs_cls,
        train_data,
        val_data,
        cfg,
        device,
        rollout_eval_fn=_rollout_eval,
        rollout_checkpoint_fn=save_checkpoint,
    )
    torch.save({"model_state_dict": model.state_dict(), "cfg": cfg_dict}, run_dir / "last_checkpoint.pt")

    selected_checkpoint = _select_best_rollout_checkpoint(rollout_checkpoints)
    selection_score = float(selected_checkpoint["rollout_r2"])
    selected_checkpoint_path = Path(selected_checkpoint["path"])
    model.load_checkpoint(str(selected_checkpoint_path))
    print(f"[run] selected checkpoint epoch={selected_checkpoint['epoch']} rollout_r2={_3g(selection_score)}", flush=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "cfg": cfg_dict,
        "selected_checkpoint": selected_checkpoint,
    }, run_dir / "final_checkpoint.pt")

    print("[run] evaluating...", flush=True)
    with torch.no_grad():
        was_training = model.training
        prev_freeze = model.freeze_normalizers
        model.eval()
        model.freeze_normalizers = True
        rollout_metrics = _rollout_eval(model)
        model.freeze_normalizers = prev_freeze
        model.train(was_training)

    rollout_rows = rollout_metrics.pop("rows")
    write_csv(run_dir / "rollout_predictions.csv", rollout_rows)
    _write_rollout_scatter(rollout_rows, run_dir / "rollout_pratio_scatter.png", title=f"{cfg.run_name} rollout p-ratio (step={cfg.rollout_steps})")
    torch.save(stats, run_dir / "train_stats.pt")
    (run_dir / "rollout_checkpoints.json").write_text(json.dumps(rollout_checkpoints, indent=2))

    metrics = {
        "run_name": cfg.run_name,
        "model_type": cfg.model_type,
        "device": device,
        "selected_checkpoint": str(selected_checkpoint_path),
        "best_epoch": int(selected_checkpoint["epoch"]),
        "best_score": float(selection_score),
        **rollout_metrics,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2))
    print(f"[run] done rollout_r2={_3g(metrics['rollout_r2'])} rollout_pos_mse={_3g(metrics['rollout_pos_mse'])}", flush=True)
    return metrics


def run_graph_cv_experiment(cfg: ExperimentConfig) -> dict:
    device, cfg_dict, train_data, val_data, model_inputs_cls, model = _build_model_and_data(cfg)
    run_dir = Path(cfg.output_root) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] {cfg.run_name} model={cfg.model_type} device={device} output={run_dir}", flush=True)

    rollout_checkpoints, save_checkpoint = _make_checkpoint_callback(run_dir, cfg_dict)

    def _cv_eval(current_model):
        return evaluate_cv_vs_global_pratio(model=current_model, sims=val_data, history=cfg.history, pos_dim=cfg.pos_dim, device=device, max_steps=None)

    print("[run] training...", flush=True)
    stats = train_graph_cv_model(
        model,
        model_inputs_cls,
        train_data,
        val_data,
        cfg,
        device,
        cv_eval_fn=_cv_eval,
        checkpoint_fn=save_checkpoint,
    )
    torch.save({"model_state_dict": model.state_dict(), "cfg": cfg_dict}, run_dir / "last_checkpoint.pt")

    selected_checkpoint = _select_best_cv_checkpoint(rollout_checkpoints, stats)
    selection_score = float(selected_checkpoint["cv_fit_r2"])
    selected_checkpoint_path = Path(selected_checkpoint["path"])
    model.load_checkpoint(str(selected_checkpoint_path))
    print(f"[run] selected checkpoint epoch={selected_checkpoint['epoch']} cv_fit_r2={_3g(selection_score)}", flush=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "cfg": cfg_dict,
        "selected_checkpoint": selected_checkpoint,
    }, run_dir / "final_checkpoint.pt")

    print("[run] evaluating...", flush=True)
    with torch.no_grad():
        was_training = model.training
        prev_freeze = model.freeze_normalizers
        model.eval()
        model.freeze_normalizers = True
        cv_metrics = _cv_eval(model)
        model.freeze_normalizers = prev_freeze
        model.train(was_training)

    torch.save(stats, run_dir / "train_stats.pt")
    (run_dir / "rollout_checkpoints.json").write_text(json.dumps(rollout_checkpoints, indent=2))

    metrics = {
        "run_name": cfg.run_name,
        "model_type": cfg.model_type,
        "device": device,
        "selected_checkpoint": str(selected_checkpoint_path),
        "best_epoch": int(selected_checkpoint["epoch"]),
        "best_score": float(selection_score),
        "rollout_r2": float("nan"),
        "rollout_pearson_r": float("nan"),
        "rollout_pos_mse": float("nan"),
        **cv_metrics,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2))
    print(f"[run] done rollout_r2={_3g(metrics['rollout_r2'])} rollout_pos_mse={_3g(metrics['rollout_pos_mse'])}", flush=True)
    return metrics


def run_experiment(cfg: ExperimentConfig) -> dict:
    return run_graph_cv_experiment(cfg) if cfg.model_type == "cv_transformer" else run_graph_experiment(cfg)
