from __future__ import annotations

import math
import time
from typing import Callable

import torch

from .graph import build_graph, clone_graph


def _fmt3g(value) -> str:
    if value is None:
        return "skipped"
    try:
        return f"{float(value):.3g}"
    except Exception:
        return str(value)


def _weighted_rollout_loss(losses: list[torch.Tensor], decay: float) -> torch.Tensor:
    if len(losses) == 1:
        return losses[0]
    if not math.isfinite(decay) or decay <= 0.0:
        raise ValueError(f"train_rollout_loss_decay must be > 0, got {decay}")
    stacked = torch.stack(losses)
    if abs(decay - 1.0) <= 1e-12:
        return stacked.mean()
    weights = stacked.new_tensor([decay**i for i in range(len(losses))])
    weights = weights / weights.sum()
    return (stacked * weights).sum()


def _sample_autoregressive_loss(
    model,
    sim,
    index: int,
    rollout_steps: int,
    cfg,
    device: str,
    model_inputs_cls,
    *,
    is_train: bool,
):
    frames = [clone_graph(sim[i]).to(device) for i in range(index - cfg.history, index + 1)]
    if len(frames) > 1:
        frames[-1].vel_state = frames[-1].x[:, : cfg.pos_dim] - frames[-2].x[:, : cfg.pos_dim]

    losses = []
    loss_decay = float(getattr(cfg, "train_rollout_loss_decay", 1.0))
    for step in range(1, rollout_steps + 1):
        allow_norm_accum = bool(is_train and step == 1)
        input_graph = build_graph(
            input_graphs=frames[-(cfg.history + 1) :],
            node_features=cfg.node_features,
        ).to(device)

        cur_graph = clone_graph(frames[-1]).to(device)
        prev_graph = clone_graph(frames[-2] if len(frames) > 1 else frames[-1]).to(device)
        if len(frames) > 1:
            cur_graph.vel_state = cur_graph.x[:, : cfg.pos_dim] - prev_graph.x[:, : cfg.pos_dim]
        target_graph = clone_graph(sim[index + step]).to(device)
        model_inputs = model_inputs_cls(prev_graph, cur_graph, target_graph, cfg.pos_dim)

        pred = model(input_graph, is_training=allow_norm_accum)
        loss = model.loss(pred, model_inputs, accumulate_norm_stats=allow_norm_accum)
        losses.append(loss)

        if step < rollout_steps:
            predicted_graph = model.update(model_inputs, pred)
            frames.append(predicted_graph)

    return _weighted_rollout_loss(losses, decay=loss_decay)


def _epoch_loss(model, sims, cfg, device: str, model_inputs_cls, optimizer=None):
    is_train = optimizer is not None
    prev_freeze = getattr(model, "freeze_normalizers", None) if hasattr(model, "freeze_normalizers") else None
    model.train(is_train)
    if not is_train and hasattr(model, "freeze_normalizers"):
        model.freeze_normalizers = True

    loss_sum = 0.0
    steps = 0
    train_rollout_steps = max(int(getattr(cfg, "train_rollout_steps", 1)), 1)

    for sim in sims:
        sim_len = min(len(sim), cfg.limit)
        for index in range(sim_len):
            # Match the legacy modular-network-simulator indexing used in the
            # CV-discovery notebook (skip the earliest nominally-valid index).
            if index < (cfg.history + 1) or index + train_rollout_steps >= sim_len:
                continue
            loss = _sample_autoregressive_loss(
                model,
                sim,
                index,
                train_rollout_steps,
                cfg,
                device,
                model_inputs_cls,
                is_train=is_train,
            )

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_sum += float(loss.detach().cpu().item())
            steps += 1

    out = loss_sum / max(steps, 1), steps
    if not is_train and hasattr(model, "freeze_normalizers") and prev_freeze is not None:
        model.freeze_normalizers = prev_freeze
    return out


def train_model(
    model,
    model_inputs_cls,
    train_data,
    val_data,
    cfg,
    device: str,
    rollout_eval_fn: Callable | None = None,
    cv_eval_fn: Callable | None = None,
    rollout_checkpoint_fn: Callable | None = None,
):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer,
        gamma=cfg.learning_rate_decay,
    )

    stats = {
        "epoch": [],
        "loss": [],
        "val_loss": [],
        "lr": [],
        "epoch_seconds": [],
        "rollout_r2": [],
        "rollout_pearson_r": [],
        "rollout_pos_mse": [],
        "rollout_used": [],
        "rollout_total": [],
        "cv_abs_pearson_r": [],
        "cv_fit_r2": [],
        "cv_used": [],
    }
    rollout_every = max(int(getattr(cfg, "rollout_every", 0)), 0)
    cv_eval_every = max(int(getattr(cfg, "cv_eval_every", 0)), 0)
    val_every = max(int(getattr(cfg, "val_every", 0)), 0)
    train_rollout_steps = max(int(getattr(cfg, "train_rollout_steps", 1)), 1)
    train_rollout_loss_decay = float(getattr(cfg, "train_rollout_loss_decay", 1.0))
    freeze_norm_after = max(int(getattr(cfg, "freeze_normalizers_after_epoch", 0)), 0)
    verbose = bool(getattr(cfg, "verbose", True))
    train_start = time.perf_counter()
    if verbose:
        print(
            f"[train] autoregressive loss steps={train_rollout_steps} "
            f"decay={train_rollout_loss_decay:.3g}",
            flush=True,
        )

    for epoch in range(cfg.epochs):
        epoch_start = time.perf_counter()
        if hasattr(model, "freeze_normalizers"):
            model.freeze_normalizers = bool(freeze_norm_after > 0 and (epoch + 1) > freeze_norm_after)
        if hasattr(model, "set_epoch"):
            model.set_epoch(epoch)
        train_loss, train_steps = _epoch_loss(
            model,
            train_data,
            cfg,
            device,
            model_inputs_cls,
            optimizer=optimizer,
        )
        val_loss = None
        val_steps = 0
        if val_every > 0 and (epoch + 1) % val_every == 0:
            with torch.no_grad():
                val_loss, val_steps = _epoch_loss(
                    model,
                    val_data,
                    cfg,
                    device,
                    model_inputs_cls,
                    optimizer=None,
                )

        rollout_metrics = None
        if rollout_eval_fn is not None and rollout_every > 0 and (epoch + 1) % rollout_every == 0:
            with torch.no_grad():
                was_training = model.training
                had_freeze = hasattr(model, "freeze_normalizers")
                prev_freeze = getattr(model, "freeze_normalizers", None) if had_freeze else None
                model.eval()
                if had_freeze:
                    model.freeze_normalizers = True
                rollout_metrics = rollout_eval_fn(model)
                if had_freeze and prev_freeze is not None:
                    model.freeze_normalizers = prev_freeze
                model.train(was_training)
            if rollout_checkpoint_fn is not None:
                rollout_checkpoint_fn(
                    epoch + 1,
                    model,
                    optimizer,
                    scheduler,
                    rollout_metrics,
                    train_loss,
                    val_loss,
                )

        cv_metrics = None
        if cv_eval_fn is not None and cv_eval_every > 0 and (epoch + 1) % cv_eval_every == 0:
            with torch.no_grad():
                was_training = model.training
                had_freeze = hasattr(model, "freeze_normalizers")
                prev_freeze = getattr(model, "freeze_normalizers", None) if had_freeze else None
                model.eval()
                if had_freeze:
                    model.freeze_normalizers = True
                cv_metrics = cv_eval_fn(model)
                if had_freeze and prev_freeze is not None:
                    model.freeze_normalizers = prev_freeze
                model.train(was_training)

        scheduler.step()
        epoch_seconds = time.perf_counter() - epoch_start
        lr = float(optimizer.param_groups[0]["lr"])
        rollout_r2 = None if rollout_metrics is None else rollout_metrics["rollout_r2"]
        rollout_pearson_r = None if rollout_metrics is None else rollout_metrics.get("rollout_pearson_r")
        rollout_pos_mse = None if rollout_metrics is None else rollout_metrics["rollout_pos_mse"]
        rollout_used = None if rollout_metrics is None else rollout_metrics["used"]
        rollout_total = None if rollout_metrics is None else rollout_metrics["total"]
        cv_abs_pearson_r = None if cv_metrics is None else cv_metrics.get("cv_abs_pearson_r")
        cv_fit_r2 = None if cv_metrics is None else cv_metrics.get("cv_fit_r2")
        cv_used = None if cv_metrics is None else cv_metrics.get("cv_used")
        stats["epoch"].append(epoch + 1)
        stats["loss"].append(train_loss)
        stats["val_loss"].append(val_loss)
        stats["lr"].append(lr)
        stats["epoch_seconds"].append(epoch_seconds)
        stats["rollout_r2"].append(rollout_r2)
        stats["rollout_pearson_r"].append(rollout_pearson_r)
        stats["rollout_pos_mse"].append(rollout_pos_mse)
        stats["rollout_used"].append(rollout_used)
        stats["rollout_total"].append(rollout_total)
        stats["cv_abs_pearson_r"].append(cv_abs_pearson_r)
        stats["cv_fit_r2"].append(cv_fit_r2)
        stats["cv_used"].append(cv_used)

        should_log = (
            (epoch + 1 == cfg.epochs)
            or (val_every > 0 and (epoch + 1) % val_every == 0)
            or (rollout_metrics is not None)
            or (cv_metrics is not None)
        )
        if verbose and should_log:
            val_text = _fmt3g(val_loss)
            if rollout_r2 is None:
                rollout_text = "skipped"
            else:
                pearson_text = "nan" if rollout_pearson_r is None else _fmt3g(rollout_pearson_r)
                mse_text = "nan" if rollout_pos_mse is None else _fmt3g(rollout_pos_mse)
                rollout_text = (
                    f"r2={_fmt3g(rollout_r2)} p={pearson_text} mse={mse_text} "
                    f"({rollout_used}/{rollout_total})"
                )
            if cv_metrics is None:
                cv_text = "skipped"
            else:
                cv_p = "nan" if cv_abs_pearson_r is None else _fmt3g(cv_abs_pearson_r)
                cv_r2 = "nan" if cv_fit_r2 is None else _fmt3g(cv_fit_r2)
                cv_n = "?" if cv_used is None else str(int(cv_used))
                cv_text = f"|p|={cv_p} r2={cv_r2} (n={cv_n})"
            gate_text = ""
            if hasattr(model, "get_global_gate_stats"):
                g = model.get_global_gate_stats()
                if isinstance(g, dict) and ("abs_mean" in g) and ("max_abs" in g):
                    ratio = g.get("global_ratio", None)
                    ratio_text = "nan" if ratio is None else _fmt3g(ratio)
                    gate_text = (
                        f"gabs={_fmt3g(g['abs_mean'])} "
                        f"gmax={_fmt3g(g['max_abs'])} "
                        f"g_over_l={ratio_text} "
                    )
            print(
                f"[ep {epoch + 1:>3}/{cfg.epochs}] "
                f"tr={_fmt3g(train_loss)} va={val_text} lr={_fmt3g(lr)} "
                f"roll={rollout_text} "
                f"cv={cv_text} "
                f"{gate_text}"
                f"norm={'fz' if getattr(model, 'freeze_normalizers', False) else 'tr'} "
                f"n=({train_steps},{val_steps}) "
                f"t={_fmt3g(epoch_seconds)}s",
                flush=True,
            )

    if verbose:
        total_seconds = time.perf_counter() - train_start
        print(f"[train] complete in {_fmt3g(total_seconds)}s", flush=True)

    return stats
