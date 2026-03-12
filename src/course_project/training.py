from __future__ import annotations

import time
from typing import Callable

import torch
import torch.nn.functional as F

from .graph import build_graph, clone_graph


def _fmt3g(value) -> str:
    return f"{float(value):.3g}"


def _build_graph_at_index(sim, index: int, cfg, device: str):
    frames = [clone_graph(sim[i]).to(device) for i in range(index - cfg.history, index + 1)]
    if len(frames) > 1:
        frames[-1].vel_state = frames[-1].x[:, : cfg.pos_dim] - frames[-2].x[:, : cfg.pos_dim]
    return build_graph(input_graphs=frames).to(device)


def _build_optimizer(model, cfg):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(
        params,
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
    )
    optimizer.param_groups[0]["name"] = "main"
    return optimizer


def _lr_text(optimizer) -> str:
    return _fmt3g(optimizer.param_groups[0]["lr"])


def _with_frozen_normalizers(model, fn: Callable):
    was_training = model.training
    prev_freeze = getattr(model, "freeze_normalizers", None)
    model.eval()
    if hasattr(model, "freeze_normalizers"):
        model.freeze_normalizers = True
    try:
        return fn(model)
    finally:
        if hasattr(model, "freeze_normalizers"):
            model.freeze_normalizers = prev_freeze
        model.train(was_training)


def _sample_autoregressive_loss(model, sim, index: int, cfg, device: str, model_inputs_cls, *, is_train: bool):
    frames = [clone_graph(sim[i]).to(device) for i in range(index - cfg.history, index + 1)]
    if len(frames) > 1:
        frames[-1].vel_state = frames[-1].x[:, : cfg.pos_dim] - frames[-2].x[:, : cfg.pos_dim]

    allow_norm_accum = bool(is_train)
    input_graph = build_graph(input_graphs=frames[-(cfg.history + 1):]).to(device)
    cur_graph = clone_graph(frames[-1]).to(device)
    prev_graph = clone_graph(frames[-2] if len(frames) > 1 else frames[-1]).to(device)
    if len(frames) > 1:
        cur_graph.vel_state = cur_graph.x[:, : cfg.pos_dim] - prev_graph.x[:, : cfg.pos_dim]
    target_graph = clone_graph(sim[index + 1]).to(device)
    model_inputs = model_inputs_cls(prev_graph, cur_graph, target_graph, cfg.pos_dim)

    pred = model(input_graph, is_training=allow_norm_accum)
    total_loss = model.loss(pred, model_inputs, accumulate_norm_stats=allow_norm_accum)
    parts = {
        "cv_consistency_loss": torch.zeros((), device=total_loss.device),
        "lag_loss": torch.zeros((), device=total_loss.device),
    }

    cv_consistency_weight = float(getattr(model, "cv_consistency_weight", 0.0))
    if hasattr(model, "cv_consistency_loss") and cv_consistency_weight > 0.0:
        pred_graph = model.update(model_inputs, pred)
        cv_loss = model.cv_consistency_loss(pred_graph, target_graph)
        parts["cv_consistency_loss"] = cv_loss.detach()
        total_loss = total_loss + cv_consistency_weight * cv_loss

    lag_steps = int(getattr(model, "time_lag_steps", 0))
    lag_weight = float(getattr(model, "time_lag_weight", 0.0))
    if hasattr(model, "predict_time_lag_acc") and lag_steps > 0 and lag_weight > 0.0 and index + lag_steps < len(sim):
        graph_t = _build_graph_at_index(sim, index, cfg, device)
        pred_tau = model.predict_time_lag_acc(graph_t, is_training=is_train)

        prev_t_pos = clone_graph(sim[index - 1]).to(device).x[:, : cfg.pos_dim]
        cur_t_pos = clone_graph(sim[index]).to(device).x[:, : cfg.pos_dim]
        prev_tau_pos = clone_graph(sim[index + lag_steps - 1]).to(device).x[:, : cfg.pos_dim]
        cur_tau_pos = clone_graph(sim[index + lag_steps]).to(device).x[:, : cfg.pos_dim]
        cur_t_vel = cur_t_pos - prev_t_pos
        cur_tau_vel = cur_tau_pos - prev_tau_pos
        target_tau_acc = (cur_tau_vel - cur_t_vel) / float(lag_steps)
        if hasattr(model, "output_normalizer"):
            target_tau_acc = model.output_normalizer(target_tau_acc, is_training=bool(is_train))
        lag_loss = F.mse_loss(pred_tau, target_tau_acc.detach())
        parts["lag_loss"] = lag_loss.detach()
        total_loss = total_loss + lag_weight * lag_loss

    return total_loss, parts


def _epoch_loss(model, sims, cfg, device: str, model_inputs_cls, optimizer=None):
    is_train = optimizer is not None
    prev_freeze = getattr(model, "freeze_normalizers", None) if hasattr(model, "freeze_normalizers") else None
    model.train(is_train)
    if not is_train and hasattr(model, "freeze_normalizers"):
        model.freeze_normalizers = True

    loss_sum = 0.0
    cv_consistency_loss_sum = 0.0
    lag_loss_sum = 0.0
    steps = 0
    for sim in sims:
        sim_len = min(len(sim), cfg.limit)
        for index in range(sim_len):
            if index < (cfg.history + 1) or index + 1 >= sim_len:
                continue
            loss, parts = _sample_autoregressive_loss(model, sim, index, cfg, device, model_inputs_cls, is_train=is_train)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += float(loss.detach().cpu().item())
            cv_consistency_loss_sum += float(parts["cv_consistency_loss"].cpu().item())
            lag_loss_sum += float(parts["lag_loss"].cpu().item())
            steps += 1

    denom = max(steps, 1)
    out = {
        "loss": loss_sum / denom,
        "cv_consistency_loss": cv_consistency_loss_sum / denom,
        "lag_loss": lag_loss_sum / denom,
        "steps": steps,
    }
    if not is_train and hasattr(model, "freeze_normalizers"):
        model.freeze_normalizers = prev_freeze
    return out


def _init_stats():
    return {
        "epoch": [],
        "loss": [],
        "val_loss": [],
        "cv_consistency_loss": [],
        "val_cv_consistency_loss": [],
        "lag_loss": [],
        "val_lag_loss": [],
        "lr": [],
        "epoch_seconds": [],
        "cv_scale": [],
        "cv_film_mean": [],
    }


def _append_common_stats(stats, epoch, train_info, val_info, val_loss, optimizer, epoch_seconds, model):
    stats["epoch"].append(epoch)
    stats["loss"].append(float(train_info["loss"]))
    stats["val_loss"].append(val_loss)
    stats["cv_consistency_loss"].append(float(train_info["cv_consistency_loss"]))
    stats["val_cv_consistency_loss"].append(float("nan") if val_info is None else float(val_info["cv_consistency_loss"]))
    stats["lag_loss"].append(float(train_info["lag_loss"]))
    stats["val_lag_loss"].append(float("nan") if val_info is None else float(val_info["lag_loss"]))
    stats["lr"].append(float(optimizer.param_groups[0]["lr"]))
    stats["epoch_seconds"].append(epoch_seconds)
    cv_scale = float("nan")
    cv_film_mean = float("nan")
    if hasattr(model, "cv_inject_scale"):
        cv_scale = float(model.cv_inject_scale.detach().cpu().item())
    if hasattr(model, "get_global_film_stats"):
        cv_film_mean = float(model.get_global_film_stats()["mean"])
    stats["cv_scale"].append(cv_scale)
    stats["cv_film_mean"].append(cv_film_mean)


def train_graph_model(model, model_inputs_cls, train_data, val_data, cfg, device: str, *, rollout_eval_fn: Callable, rollout_checkpoint_fn: Callable):
    optimizer = _build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.learning_rate_decay)
    stats = _init_stats()
    stats.update({
        "rollout_r2": [],
        "rollout_pearson_r": [],
        "rollout_pos_mse": [],
        "rollout_used": [],
        "rollout_total": [],
    })
    rollout_every = max(int(cfg.rollout_every), 0)
    val_every = max(int(cfg.val_every), 0)
    freeze_norm_after = max(int(cfg.freeze_normalizers_after_epoch), 0)
    train_start = time.perf_counter()
    if hasattr(model, "time_lag_steps") and hasattr(model, "time_lag_weight"):
        print(
            f"[train] time-lag loss tau={int(getattr(model, 'time_lag_steps', 0))} "
            f"lambda={float(getattr(model, 'time_lag_weight', 0.0)):.3g}",
            flush=True,
        )

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.perf_counter()
        if hasattr(model, "freeze_normalizers"):
            model.freeze_normalizers = bool(freeze_norm_after > 0 and epoch > freeze_norm_after)
        train_info = _epoch_loss(model, train_data, cfg, device, model_inputs_cls, optimizer=optimizer)
        val_info = None
        val_loss = None
        if val_every > 0 and epoch % val_every == 0:
            with torch.no_grad():
                val_info = _epoch_loss(model, val_data, cfg, device, model_inputs_cls, optimizer=None)
                val_loss = float(val_info["loss"])

        rollout_metrics = {"rollout_r2": float("nan"), "rollout_pearson_r": float("nan"), "rollout_pos_mse": float("nan"), "used": 0, "total": 0}
        if rollout_every > 0 and epoch % rollout_every == 0:
            with torch.no_grad():
                rollout_metrics = _with_frozen_normalizers(model, rollout_eval_fn)
            rollout_checkpoint_fn(epoch, model, optimizer, scheduler, rollout_metrics, float(train_info["loss"]), val_loss)

        scheduler.step()
        epoch_seconds = time.perf_counter() - epoch_start
        _append_common_stats(stats, epoch, train_info, val_info, val_loss, optimizer, epoch_seconds, model)
        stats["rollout_r2"].append(float(rollout_metrics["rollout_r2"]))
        stats["rollout_pearson_r"].append(float(rollout_metrics["rollout_pearson_r"]))
        stats["rollout_pos_mse"].append(float(rollout_metrics["rollout_pos_mse"]))
        stats["rollout_used"].append(int(rollout_metrics["used"]))
        stats["rollout_total"].append(int(rollout_metrics["total"]))

        if (epoch == cfg.epochs) or (val_every > 0 and epoch % val_every == 0) or (rollout_every > 0 and epoch % rollout_every == 0):
            line = f"[ep {epoch:>3}/{cfg.epochs}] tr={_fmt3g(train_info['loss'])} va={_fmt3g(val_loss)} lr={_lr_text(optimizer)} "
            if hasattr(model, "cv_inject_scale"):
                line += f"cv_scale={_fmt3g(model.cv_inject_scale.detach().cpu().item())} "
            if hasattr(model, "get_global_film_stats"):
                line += f"film={_fmt3g(model.get_global_film_stats()['mean'])} "
            line += (
                f"roll=r2={_fmt3g(rollout_metrics['rollout_r2'])} "
                f"p={_fmt3g(rollout_metrics['rollout_pearson_r'])} "
                f"mse={_fmt3g(rollout_metrics['rollout_pos_mse'])} "
                f"({int(rollout_metrics['used'])}/{int(rollout_metrics['total'])}) "
                f"t={_fmt3g(epoch_seconds)}s"
            )
            print(line, flush=True)

    total_seconds = time.perf_counter() - train_start
    print(f"[train] complete in {_fmt3g(total_seconds)}s", flush=True)
    return stats


def train_graph_cv_model(model, model_inputs_cls, train_data, val_data, cfg, device: str, *, cv_eval_fn: Callable, checkpoint_fn: Callable):
    optimizer = _build_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.learning_rate_decay)
    stats = _init_stats()
    stats.update({
        "cv_abs_pearson_r": [],
        "cv_fit_r2": [],
        "cv_used": [],
    })
    cv_eval_every = max(int(cfg.cv_eval_every), 0)
    val_every = max(int(cfg.val_every), 0)
    freeze_norm_after = max(int(cfg.freeze_normalizers_after_epoch), 0)
    train_start = time.perf_counter()
    if hasattr(model, "time_lag_steps") and hasattr(model, "time_lag_weight"):
        print(
            f"[train] time-lag loss tau={int(getattr(model, 'time_lag_steps', 0))} "
            f"lambda={float(getattr(model, 'time_lag_weight', 0.0)):.3g}",
            flush=True,
        )

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.perf_counter()
        if hasattr(model, "freeze_normalizers"):
            model.freeze_normalizers = bool(freeze_norm_after > 0 and epoch > freeze_norm_after)
        train_info = _epoch_loss(model, train_data, cfg, device, model_inputs_cls, optimizer=optimizer)
        val_info = None
        val_loss = None
        if val_every > 0 and epoch % val_every == 0:
            with torch.no_grad():
                val_info = _epoch_loss(model, val_data, cfg, device, model_inputs_cls, optimizer=None)
                val_loss = float(val_info["loss"])

        cv_metrics = {"cv_abs_pearson_r": float("nan"), "cv_fit_r2": float("nan"), "cv_used": 0, "cv_text": "|p|=nan r2=nan"}
        if cv_eval_every > 0 and epoch % cv_eval_every == 0:
            with torch.no_grad():
                cv_metrics = _with_frozen_normalizers(model, cv_eval_fn)
            checkpoint_fn(epoch, model, optimizer, scheduler, cv_metrics, float(train_info["loss"]), val_loss)

        scheduler.step()
        epoch_seconds = time.perf_counter() - epoch_start
        _append_common_stats(stats, epoch, train_info, val_info, val_loss, optimizer, epoch_seconds, model)
        stats["cv_abs_pearson_r"].append(float(cv_metrics["cv_abs_pearson_r"]))
        stats["cv_fit_r2"].append(float(cv_metrics["cv_fit_r2"]))
        stats["cv_used"].append(int(cv_metrics["cv_used"]))

        if (epoch == cfg.epochs) or (val_every > 0 and epoch % val_every == 0) or (cv_eval_every > 0 and epoch % cv_eval_every == 0):
            line = (
                f"[ep {epoch:>3}/{cfg.epochs}] tr={_fmt3g(train_info['loss'])} va={_fmt3g(val_loss)} "
                f"lr={_lr_text(optimizer)} cv={cv_metrics.get('cv_text', '|p|=nan r2=nan')} (n={int(cv_metrics['cv_used'])}) "
                f"t={_fmt3g(epoch_seconds)}s"
            )
            print(line, flush=True)

    total_seconds = time.perf_counter() - train_start
    print(f"[train] complete in {_fmt3g(total_seconds)}s", flush=True)
    return stats
