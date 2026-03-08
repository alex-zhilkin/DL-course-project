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


def _set_hybrid_stage(model, stage: str):
    if stage == "global_only":
        for p in model.parameters():
            p.requires_grad = False
        for p in model.cv_encoder.parameters():
            p.requires_grad = False
        for p in model.cv_injector.parameters():
            p.requires_grad = True
        for p in model.cv_node_gate.parameters():
            p.requires_grad = True
        model.cv_inject_scale.requires_grad = True
        return

    for p in model.parameters():
        p.requires_grad = True
    for p in model.cv_encoder.parameters():
        p.requires_grad = False


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
    for step in range(1, rollout_steps + 1):
        allow_norm_accum = bool(is_train and step == 1)
        input_graph = build_graph(input_graphs=frames[-(cfg.history + 1) :]).to(device)

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

    total_loss = torch.stack(losses).mean()

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
        # Trajectory-level lag target: average acceleration over [t, t+tau].
        target_tau_acc = (cur_tau_vel - cur_t_vel) / float(lag_steps)

        if hasattr(model, "output_normalizer"):
            target_tau_acc = model.output_normalizer(target_tau_acc, is_training=bool(is_train))
        lag_loss = F.mse_loss(pred_tau, target_tau_acc.detach())
        total_loss = total_loss + lag_weight * lag_loss

    return total_loss


def _epoch_loss(model, sims, cfg, device: str, model_inputs_cls, optimizer=None):
    is_train = optimizer is not None
    prev_freeze = getattr(model, "freeze_normalizers", None) if hasattr(model, "freeze_normalizers") else None
    model.train(is_train)
    if not is_train and hasattr(model, "freeze_normalizers"):
        model.freeze_normalizers = True

    loss_sum = 0.0
    steps = 0
    train_rollout_steps = max(int(cfg.train_rollout_steps), 1)

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
    if not is_train and hasattr(model, "freeze_normalizers"):
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
    hybrid_global_only_epochs = int(cfg.hybrid_global_only_epochs)
    if cfg.model_type == "hybrid" and hybrid_global_only_epochs > 0:
        _set_hybrid_stage(model, "global_only")
        stage_lr = float(cfg.global_learning_rate or cfg.learning_rate)
    else:
        if cfg.model_type == "hybrid":
            _set_hybrid_stage(model, "full")
        stage_lr = float(cfg.learning_rate)

    optimizer = _build_optimizer(model, cfg)
    optimizer.param_groups[0]["lr"] = stage_lr
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
        "cv_gate": [],
        "cv_node_gate_mean": [],
    }
    rollout_every = max(int(cfg.rollout_every), 0)
    cv_eval_every = max(int(cfg.cv_eval_every), 0)
    checkpoint_every = cv_eval_every if cfg.model_type == "cv_transformer" else rollout_every
    val_every = max(int(cfg.val_every), 0)
    train_rollout_steps = max(int(cfg.train_rollout_steps), 1)
    freeze_norm_after = max(int(cfg.freeze_normalizers_after_epoch), 0)
    verbose = bool(cfg.verbose)
    train_start = time.perf_counter()
    if verbose:
        print(f"[train] autoregressive loss steps={train_rollout_steps}", flush=True)
        if hasattr(model, "time_lag_steps") and hasattr(model, "time_lag_weight"):
            print(
                f"[train] time-lag loss tau={int(getattr(model, 'time_lag_steps', 0))} "
                f"lambda={float(getattr(model, 'time_lag_weight', 0.0)):.3g}",
                flush=True,
            )
        if cfg.model_type == "hybrid" and hybrid_global_only_epochs > 0:
            print(
                f"[train] hybrid stages global_only_epochs={hybrid_global_only_epochs} "
                f"global_lr={_fmt3g(float(cfg.global_learning_rate or cfg.learning_rate))} "
                f"full_lr={_fmt3g(float(cfg.learning_rate))}",
                flush=True,
            )

    for epoch in range(cfg.epochs):
        if cfg.model_type == "hybrid" and hybrid_global_only_epochs > 0 and epoch == hybrid_global_only_epochs:
            _set_hybrid_stage(model, "full")
            optimizer = _build_optimizer(model, cfg)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=cfg.learning_rate_decay,
            )
            if verbose:
                print(
                    f"[train] stage switch epoch={epoch + 1} -> full lr={_fmt3g(float(cfg.learning_rate))}",
                    flush=True,
                )
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

        rollout_metrics = {
            "rollout_r2": float("nan"),
            "rollout_pearson_r": float("nan"),
            "rollout_pos_mse": float("nan"),
            "used": 0,
            "total": 0,
        }
        if rollout_every > 0 and (epoch + 1) % rollout_every == 0:
            with torch.no_grad():
                was_training = model.training
                had_freeze = hasattr(model, "freeze_normalizers")
                prev_freeze = getattr(model, "freeze_normalizers", None) if had_freeze else None
                model.eval()
                if had_freeze:
                    model.freeze_normalizers = True
                rollout_metrics = rollout_eval_fn(model)
                if had_freeze:
                    model.freeze_normalizers = prev_freeze
                model.train(was_training)
        cv_metrics = {
            "cv_abs_pearson_r": float("nan"),
            "cv_fit_r2": float("nan"),
            "cv_used": 0,
        }
        if cv_eval_every > 0 and (epoch + 1) % cv_eval_every == 0:
            with torch.no_grad():
                was_training = model.training
                had_freeze = hasattr(model, "freeze_normalizers")
                prev_freeze = getattr(model, "freeze_normalizers", None) if had_freeze else None
                model.eval()
                if had_freeze:
                    model.freeze_normalizers = True
                cv_metrics = cv_eval_fn(model)
                if had_freeze:
                    model.freeze_normalizers = prev_freeze
                model.train(was_training)
        if checkpoint_every > 0 and (epoch + 1) % checkpoint_every == 0:
            rollout_checkpoint_fn(
                epoch + 1,
                model,
                optimizer,
                scheduler,
                rollout_metrics,
                train_loss,
                val_loss,
            )

        scheduler.step()
        epoch_seconds = time.perf_counter() - epoch_start
        lr = float(optimizer.param_groups[0]["lr"])
        lr_text = _lr_text(optimizer)
        rollout_r2 = float(rollout_metrics["rollout_r2"])
        rollout_pearson_r = float(rollout_metrics["rollout_pearson_r"])
        rollout_pos_mse = float(rollout_metrics["rollout_pos_mse"])
        rollout_used = int(rollout_metrics["used"])
        rollout_total = int(rollout_metrics["total"])
        cv_abs_pearson_r = float(cv_metrics["cv_abs_pearson_r"])
        cv_fit_r2 = float(cv_metrics["cv_fit_r2"])
        cv_used = int(cv_metrics["cv_used"])
        cv_text = cv_metrics.get("cv_text")
        cv_gate = float("nan")
        cv_node_gate_mean = float("nan")
        if hasattr(model, "cv_inject_scale"):
            cv_gate = float(model.cv_inject_scale.detach().cpu().item())
        if hasattr(model, "get_global_gate_stats"):
            g = model.get_global_gate_stats()
            cv_node_gate_mean = float(g["mean"])
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
        stats["cv_gate"].append(cv_gate)
        stats["cv_node_gate_mean"].append(cv_node_gate_mean)

        should_log = (
            (epoch + 1 == cfg.epochs)
            or (val_every > 0 and (epoch + 1) % val_every == 0)
            or (rollout_every > 0 and (epoch + 1) % rollout_every == 0)
            or (cv_eval_every > 0 and (epoch + 1) % cv_eval_every == 0)
        )
        if verbose and should_log:
            val_text = _fmt3g(val_loss)
            if not isinstance(cv_text, str) or len(cv_text) == 0:
                cv_text = f"|p|={_fmt3g(cv_abs_pearson_r)} r2={_fmt3g(cv_fit_r2)} (n={cv_used})"
            gate_text = ""
            if hasattr(model, "cv_inject_scale"):
                gate_text += f"cv_gate={_fmt3g(model.cv_inject_scale.detach().cpu().item())} "
            if hasattr(model, "get_global_gate_stats"):
                g = model.get_global_gate_stats()
                gate_text += f"gmean={_fmt3g(g['mean'])} "
            line = (
                f"[ep {epoch + 1:>3}/{cfg.epochs}] "
                f"tr={_fmt3g(train_loss)} va={val_text} lr={lr_text} "
                f"{gate_text}"
            )
            if cfg.model_type != "cv_transformer":
                rollout_text = (
                    f"r2={_fmt3g(rollout_r2)} p={_fmt3g(rollout_pearson_r)} "
                    f"mse={_fmt3g(rollout_pos_mse)} ({rollout_used}/{rollout_total})"
                )
                line += f"roll={rollout_text} "
            line += f"cv={cv_text} "
            line += f"t={_fmt3g(epoch_seconds)}s"
            print(line, flush=True)

    if verbose:
        total_seconds = time.perf_counter() - train_start
        print(f"[train] complete in {_fmt3g(total_seconds)}s", flush=True)

    return stats
