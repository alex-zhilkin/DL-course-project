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


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _resolve_device(device: str) -> str:
    return "cpu" if device == "cuda" and not torch.cuda.is_available() else device


def _fmt3g(value) -> str:
    try:
        return f"{float(value):.3g}"
    except Exception:
        return str(value)


def _select_best_rollout_checkpoint(entries: list[dict]) -> dict | None:
    best = None
    best_r2 = float("-inf")
    best_epoch = -1
    for entry in entries:
        r2 = float(entry.get("rollout_r2", float("nan")))
        if not np.isfinite(r2):
            continue
        epoch = int(entry.get("epoch", -1))
        if (r2 > best_r2) or (r2 == best_r2 and epoch > best_epoch):
            best = entry
            best_r2 = r2
            best_epoch = epoch
    return best


def _write_rollout_scatter(rows: list[dict], out_path: Path, title: str, *, verbose: bool = False) -> None:
    if not rows:
        return
    x = np.asarray([row["target_rollout_sides_p_ratio"] for row in rows], dtype=float)
    y = np.asarray([row["pred_rollout_p_ratio"] for row in rows], dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < 1:
        return
    x = x[finite]
    y = y[finite]
    lo = float(np.min(np.concatenate([x, y])))
    hi = float(np.max(np.concatenate([x, y])))
    pad = max((hi - lo) * 0.05, 1e-6)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        if verbose:
            print(f"[run] scatter skipped: matplotlib unavailable ({exc})", flush=True)
        return

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
    device = _resolve_device(cfg.device)
    verbose = bool(getattr(cfg, "verbose", True))
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
    init_graph = build_graph(input_graphs=init_frames, node_features=cfg.node_features).to(device)
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
            node_features=cfg.node_features,
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
            node_features=cfg.node_features,
            target_kind=getattr(cfg, "cv_pratio_target", "box"),
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
            "val_loss": None if val_loss is None else float(val_loss),
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
        rollout_eval_fn=_rollout_eval if getattr(cfg, "rollout_every", 0) > 0 else None,
        cv_eval_fn=_cv_eval if getattr(cfg, "cv_eval_every", 0) > 0 else None,
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

    best_rollout = _select_best_rollout_checkpoint(rollout_checkpoints)
    selected_checkpoint = None
    if best_rollout is not None:
        selected_checkpoint = Path(best_rollout["path"])
        model.load_checkpoint(str(selected_checkpoint))
        if verbose:
            print(
                f"[run] selected best rollout checkpoint epoch={best_rollout['epoch']} "
                f"r2={_fmt3g(best_rollout['rollout_r2'])}",
                flush=True,
            )
    elif verbose:
        print("[run] no rollout checkpoints available; using last training state.", flush=True)

    final_ckpt_path = run_dir / "final_checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "cfg": cfg_dict,
            "selected_best_rollout": best_rollout,
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
            node_features=cfg.node_features,
            model_inputs_cls=model_inputs_cls,
        )
        cv_metrics = evaluate_cv_vs_global_pratio(
            model=model,
            sims=val_data,
            history=cfg.history,
            pos_dim=cfg.pos_dim,
            device=device,
            max_steps=cfg.rollout_steps,
            node_features=cfg.node_features,
            target_kind=getattr(cfg, "cv_pratio_target", "box"),
        )
        if had_freeze and prev_freeze is not None:
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
        verbose=verbose,
    )

    torch.save(stats, run_dir / "train_stats.pt")
    (run_dir / "rollout_checkpoints.json").write_text(json.dumps(rollout_checkpoints, indent=2))

    metrics = {
        "run_name": cfg.run_name,
        "model_type": cfg.model_type,
        "device": device,
        "selected_checkpoint": None if selected_checkpoint is None else str(selected_checkpoint),
        "best_rollout_epoch": None if best_rollout is None else int(best_rollout["epoch"]),
        "best_rollout_r2": None if best_rollout is None else float(best_rollout["rollout_r2"]),
        **rollout_metrics,
        **cv_metrics,
    }

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2))
    if verbose:
        print(
            f"[run] done rollout_r2={_fmt3g(metrics['rollout_r2'])} "
            f"rollout_pos_mse={_fmt3g(metrics.get('rollout_pos_mse'))}",
            flush=True,
        )
    return metrics
