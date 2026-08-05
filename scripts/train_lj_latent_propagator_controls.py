"""Compare direct-path and GRU propagators on a frozen noisy-LJ autoencoder."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lss.latent.experiment import (
    decode_latent_to_graph,
    ground_truth_p_ratio,
    resolve_train_val_test,
)
from lss.latent.models import NodeDeltaAttentionAutoEncoder
from lss.latent.path_propagator import StaticLatentGRU, StaticLatentPathMLP
from lss.latent.simulation import pearson_r, r2_score
from lss.latent.training import encode_frame_latent, encode_reference_context


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ae-bundle", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_frozen_ae(path: Path):
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    params = dict(bundle["params"])
    stats = bundle["stats"]
    normalizers = {
        key: stats[key].float()
        for key in (
            "target_mean",
            "target_std",
            "node_feature_mean",
            "node_feature_std",
            "edge_mean",
            "edge_std",
        )
    }
    model = NodeDeltaAttentionAutoEncoder(
        pos_dim=int(params["pos_dim"]),
        edge_dim=int(normalizers["edge_mean"].numel()),
        hidden_size=int(params["hidden_size"]),
        latent_dim=int(params["latent_dim"]),
        latent_tokens=int(params["latent_tokens"]),
    )
    model.edge_mode = str(params.get("edge_mode", "stored"))
    model.load_state_dict(bundle["ae_state_dict"])
    model.eval()
    spec = dict(bundle["spec"])
    p_ratio_fn = lambda sim, idx=-1: ground_truth_p_ratio(
        sim, idx, dataset_name="lj_noisy", cfg=params
    )
    train, val, test, _ = resolve_train_val_test(
        spec, params, split_seed=params.get("split_seed"), p_ratio_fn=p_ratio_fn
    )
    return model, normalizers, params, (train, val, test)


def encode_split(model, normalizers, params, sims):
    paths, contexts = [], []
    with torch.no_grad():
        for index, sim in enumerate(sims):
            paths.append(
                torch.stack(
                    [
                        encode_frame_latent(
                            model,
                            sim,
                            frame,
                            pos_dim=2,
                            node_feature_mode=params["node_feature_mode"],
                            normalizers=normalizers,
                            device="cpu",
                        )
                        for frame in range(len(sim))
                    ]
                )
            )
            node_context = encode_reference_context(
                model,
                sim,
                pos_dim=2,
                normalizers=normalizers,
                device="cpu",
                pool_mode="learned_attention",
            )
            contexts.append(
                torch.cat(
                    [
                        node_context.mean(0),
                        node_context.std(0, unbiased=False),
                        node_context.amin(0),
                        node_context.amax(0),
                    ]
                )
            )
            if (index + 1) % 20 == 0:
                print(f"encoded {index + 1}/{len(sims)}", flush=True)
    return torch.stack(paths), torch.stack(contexts)


def fit_context_stats(context):
    return context.mean(0), context.std(0, unbiased=False).clamp_min(1e-6)


def trajectory_stats(z):
    z0 = z[:, :1]
    q = z - z0
    global_scale = q.reshape(-1, q.size(-1)).std(0).clamp_min(1e-6)
    q = q / global_scale
    mean_path = q.mean(0)
    variation = q.std(0, unbiased=False).clamp_min(0.05)
    return global_scale, mean_path, variation


def direct_prediction(model, context, mean_path, variation):
    batch, frames = context.size(0), mean_path.size(0)
    progress = torch.linspace(0, 1, frames).repeat(batch)
    repeated_context = context[:, None, :].expand(-1, frames, -1).reshape(
        batch * frames, -1
    )
    residual = model(repeated_context, progress).reshape(batch, frames, -1)
    return mean_path.unsqueeze(0) + variation.unsqueeze(0) * residual


def train_model(
    name,
    model,
    train_context,
    train_q,
    val_context,
    val_q,
    mean_path,
    variation,
    *,
    epochs,
    patience,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    best, best_loss, stale, rows = None, float("inf"), 0, []
    train_target = (train_q - mean_path) / variation
    val_target = (val_q - mean_path) / variation
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(train_context.size(0))
        losses = []
        for indices in order.split(10):
            if name == "direct":
                prediction = direct_prediction(
                    model, train_context[indices], mean_path, variation
                )
            else:
                prediction = model.rollout(
                    train_context[indices], train_q.size(1) - 1
                )
            residual = (prediction - mean_path) / variation
            loss = F.mse_loss(residual, train_target[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            if name == "direct":
                val_prediction = direct_prediction(
                    model, val_context, mean_path, variation
                )
            else:
                val_prediction = model.rollout(val_context, val_q.size(1) - 1)
            val_residual = (val_prediction - mean_path) / variation
            val_loss = float(F.mse_loss(val_residual, val_target))
        rows.append(
            {"epoch": epoch, "train_loss": np.mean(losses), "val_loss": val_loss}
        )
        if val_loss < best_loss - 1e-4:
            best_loss, stale = val_loss, 0
            best = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(
                f"{name} epoch={epoch} train={np.mean(losses):.5f} "
                f"val={val_loss:.5f}",
                flush=True,
            )
        if stale >= patience:
            break
    model.load_state_dict(best)
    return pd.DataFrame(rows)


def evaluate(
    name,
    model,
    ae,
    normalizers,
    params,
    sims,
    z,
    context,
    global_scale,
    mean_path,
    variation,
):
    model.eval()
    with torch.no_grad():
        if name == "direct":
            q_prediction = direct_prediction(model, context, mean_path, variation)
        else:
            q_prediction = model.rollout(context, z.size(1) - 1)
    z_prediction = z[:, :1] + q_prediction * global_scale
    cfg = dict(params) | {"temperature_pratio_estimator": "endpoint"}
    rows = []
    for sim_index, sim in enumerate(sims):
        for frame in (25, 50, 99, 125, 150, 199):
            pred_graph = decode_latent_to_graph(
                ae,
                sim,
                z_prediction[sim_index, frame],
                frame,
                pos_dim=2,
                ae_target_mode=params["ae_target_mode"],
                normalizers=normalizers,
                device="cpu",
            )
            predicted = [sim[0], pred_graph]
            rows.append(
                {
                    "model": name,
                    "sim_index": sim_index,
                    "frame": frame,
                    "true_p_ratio": ground_truth_p_ratio(
                        [sim[0], sim[frame]],
                        dataset_name="lj_noisy",
                        cfg=cfg,
                    ),
                    "pred_p_ratio": ground_truth_p_ratio(
                        predicted, dataset_name="lj_noisy", cfg=cfg
                    ),
                }
            )
    table = pd.DataFrame(rows).dropna()
    metrics = []
    for frame, group in table.groupby("frame"):
        metrics.append(
            {
                "model": name,
                "frame": frame,
                "n": len(group),
                "p_ratio_r2": r2_score(group.true_p_ratio, group.pred_p_ratio),
                "p_ratio_pearson": pearson_r(
                    group.true_p_ratio, group.pred_p_ratio
                ),
            }
        )
    return table, pd.DataFrame(metrics)


def main():
    args = parse_args()
    torch.manual_seed(20260727)
    ae, normalizers, params, splits = load_frozen_ae(args.ae_bundle)
    encoded = [
        encode_split(ae, normalizers, params, split) for split in splits
    ]
    (train_z, train_context), (val_z, val_context), (test_z, test_context) = encoded
    context_mean, context_std = fit_context_stats(train_context)
    train_context = (train_context - context_mean) / context_std
    val_context = (val_context - context_mean) / context_std
    test_context = (test_context - context_mean) / context_std
    global_scale, mean_path, variation = trajectory_stats(train_z)
    train_q = (train_z - train_z[:, :1]) / global_scale
    val_q = (val_z - val_z[:, :1]) / global_scale
    latent_dim = train_z.size(-1)
    models = {
        "direct": StaticLatentPathMLP(
            context_dim=train_context.size(-1),
            latent_dim=latent_dim,
            hidden_size=args.hidden,
        ),
        "gru": StaticLatentGRU(
            context_dim=train_context.size(-1),
            latent_dim=latent_dim,
            hidden_size=args.hidden,
        ),
    }
    histories, prediction_parts, metric_parts = [], [], []
    for name, model in models.items():
        history = train_model(
            name,
            model,
            train_context,
            train_q,
            val_context,
            val_q,
            mean_path,
            variation,
            epochs=args.epochs,
            patience=args.patience,
        )
        histories.append(history.assign(model=name))
        prediction, metric = evaluate(
            name,
            model,
            ae,
            normalizers,
            params,
            splits[2],
            test_z,
            test_context,
            global_scale,
            mean_path,
            variation,
        )
        prediction_parts.append(prediction)
        metric_parts.append(metric)
    args.output.mkdir(parents=True, exist_ok=True)
    pd.concat(histories).to_csv(args.output / "history.csv", index=False)
    pd.concat(prediction_parts).to_csv(
        args.output / "test_predictions.csv", index=False
    )
    metrics = pd.concat(metric_parts)
    metrics.to_csv(args.output / "test_metrics.csv", index=False)
    torch.save(
        {
            "models": {name: model.state_dict() for name, model in models.items()},
            "context_mean": context_mean,
            "context_std": context_std,
            "global_scale": global_scale,
            "mean_path": mean_path,
            "variation": variation,
        },
        args.output / "models.pt",
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
