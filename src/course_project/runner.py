from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from .config import ExperimentConfig
from .data import load_dataset
from .graph import build_graph
from .metrics import (
    evaluate_cv_vs_global_pratio,
    evaluate_rollout_pratio_sides,
    write_csv,
)
from .models import create_model, resolve_model_inputs
from .training import train_model
from .utils import resolve_device


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _fmt3g(value) -> str:
    return f"{float(value):.3g}"


def _select_best_rollout_checkpoint(entries: list[dict]) -> dict:
    return max(entries, key=lambda e: (float(e["rollout_r2"]), int(e["epoch"])))


def _select_best_cv_checkpoint(entries: list[dict], stats: dict) -> dict:
    epoch_to_path = {int(e["epoch"]): str(e["path"]) for e in entries}
    candidates = []
    for epoch, score in zip(stats["epoch"], stats["cv_fit_r2"]):
        value = float(score)
        if not np.isfinite(value):
            continue
        ep = int(epoch)
        if ep in epoch_to_path:
            candidates.append(
                {
                    "epoch": ep,
                    "path": epoch_to_path[ep],
                    "cv_fit_r2": value,
                }
            )
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


def run_experiment(cfg: ExperimentConfig) -> dict:
    _set_seed(cfg.seed)
    device = resolve_device(cfg.device)
    verbose = bool(cfg.verbose)
    cfg_dict = cfg.to_dict()

    train_data = load_dataset(cfg.train_dataset)
    val_data = load_dataset(cfg.val_dataset)

    run_dir = Path(cfg.output_root) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if verbose:
        print(
            f"[run] {cfg.run_name} model={cfg.model_type} device={device} "
            f"output={run_dir}",
            flush=True,
        )

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

    def _rollout_eval(current_model):
        return evaluate_rollout_pratio_sides(
            model=current_model,
            sims=val_data,
            history=cfg.history,
            rollout_steps=cfg.rollout_steps,
            pos_dim=cfg.pos_dim,
            device=device,
            model_inputs_cls=model_inputs_cls,
        )

    def _cv_eval(current_model):
        return evaluate_cv_vs_global_pratio(
            model=current_model,
            sims=val_data,
            history=cfg.history,
            pos_dim=cfg.pos_dim,
            device=device,
            max_steps=cfg.rollout_steps,
            target_kind=cfg.cv_pratio_target,
        )

    rollout_checkpoints_dir = run_dir / "rollout_checkpoints"
    rollout_checkpoints_dir.mkdir(parents=True, exist_ok=True)
    rollout_checkpoints: list[dict] = []

    def _save_rollout_checkpoint(
        epoch: int,
        current_model,
        optimizer,
        scheduler,
        rollout_metrics: dict,
        train_loss: float,
        val_loss: float | None,
    ) -> None:
        path = rollout_checkpoints_dir / f"epoch_{epoch:04d}.pt"
        rollout_summary = {k: v for k, v in rollout_metrics.items() if k != "rows"}
        training_state = {
            "epoch": int(epoch),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "rollout_metrics": rollout_summary,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        current_model.cfg = cfg_dict
        current_model.save_checkpoint(str(path), training_state=training_state)
        entry = {
            "epoch": int(epoch),
            "path": str(path),
            "rollout_r2": float(rollout_metrics.get("rollout_r2", float("nan"))),
            "rollout_pos_mse": float(rollout_metrics.get("rollout_pos_mse", float("nan"))),
            "used": int(rollout_metrics.get("used", 0)),
            "total": int(rollout_metrics.get("total", 0)),
        }
        rollout_checkpoints.append(entry)

    if verbose:
        print("[run] training...", flush=True)
    stats = train_model(
        model,
        model_inputs_cls,
        train_data,
        val_data,
        cfg,
        device,
        rollout_eval_fn=_rollout_eval,
        cv_eval_fn=_cv_eval,
        rollout_checkpoint_fn=_save_rollout_checkpoint,
    )

    last_ckpt_path = run_dir / "last_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "cfg": cfg_dict,
        },
        last_ckpt_path,
    )

    if cfg.model_type == "cv_transformer":
        best_rollout = _select_best_cv_checkpoint(rollout_checkpoints, stats)
        selection_metric = "cv_fit_r2"
        selection_score = float(best_rollout["cv_fit_r2"])
    else:
        best_rollout = _select_best_rollout_checkpoint(rollout_checkpoints)
        selection_metric = "rollout_r2"
        selection_score = float(best_rollout["rollout_r2"])

    selected_checkpoint = Path(best_rollout["path"])
    model.load_checkpoint(str(selected_checkpoint))
    if verbose:
        print(
            f"[run] selected best rollout checkpoint epoch={best_rollout['epoch']} "
            f"{selection_metric}={_fmt3g(selection_score)}",
            flush=True,
        )

    final_ckpt_path = run_dir / "final_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "cfg": cfg_dict,
            "selected_best_rollout": best_rollout,
            "selected_checkpoint_metric": selection_metric,
            "selected_checkpoint_score": selection_score,
        },
        final_ckpt_path,
    )

    if verbose:
        print("[run] evaluating...", flush=True)
    with torch.no_grad():
        was_training = model.training
        had_freeze = hasattr(model, "freeze_normalizers")
        prev_freeze = getattr(model, "freeze_normalizers", None) if had_freeze else None
        model.eval()
        if had_freeze:
            model.freeze_normalizers = True
        rollout_metrics = evaluate_rollout_pratio_sides(
            model=model,
            sims=val_data,
            history=cfg.history,
            rollout_steps=cfg.rollout_steps,
            pos_dim=cfg.pos_dim,
            device=device,
            model_inputs_cls=model_inputs_cls,
        )
        cv_metrics = evaluate_cv_vs_global_pratio(
            model=model,
            sims=val_data,
            history=cfg.history,
            pos_dim=cfg.pos_dim,
            device=device,
            max_steps=cfg.rollout_steps,
            target_kind=cfg.cv_pratio_target,
        )
        if had_freeze:
            model.freeze_normalizers = prev_freeze
        model.train(was_training)

    rollout_rows = rollout_metrics.pop("rows")
    cv_rows = cv_metrics.pop("rows")
    write_csv(run_dir / "rollout_predictions.csv", rollout_rows)
    write_csv(run_dir / "cv_vs_pratio.csv", cv_rows)
    _write_rollout_scatter(
        rollout_rows,
        run_dir / "rollout_pratio_scatter.png",
        title=f"{cfg.run_name} rollout p-ratio (step={cfg.rollout_steps})",
    )

    torch.save(stats, run_dir / "train_stats.pt")
    (run_dir / "rollout_checkpoints.json").write_text(json.dumps(rollout_checkpoints, indent=2))

    metrics = {
        "run_name": cfg.run_name,
        "model_type": cfg.model_type,
        "device": device,
        "selected_checkpoint": str(selected_checkpoint),
        "selected_checkpoint_metric": selection_metric,
        "selected_checkpoint_score": float(selection_score),
        "best_rollout_epoch": int(best_rollout["epoch"]),
        "best_rollout_r2": float(best_rollout.get("rollout_r2", float("nan"))),
        **rollout_metrics,
        **cv_metrics,
    }

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2))
    if verbose:
        print(
            f"[run] done rollout_r2={_fmt3g(metrics['rollout_r2'])} "
            f"rollout_pos_mse={_fmt3g(metrics['rollout_pos_mse'])}",
            flush=True,
        )
    return metrics
