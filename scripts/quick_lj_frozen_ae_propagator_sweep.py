"""Quick propagator-only ablation using notebook 08's frozen noisy-LJ AE."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from graph_utils import calc_p_ratio_rollout_sides
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lss.graph import clone_graph
from lss.latent.models import (
    NodeDeltaAttentionAutoEncoder,
    NodeDeltaDirectAttentionAutoEncoder,
    NodeDeltaMLPAutoEncoder,
    NodeDeltaPyramidMLPAutoEncoder,
    NodeDeltaSingleStageAttentionAutoEncoder,
)
from lss.latent.simulation import batch_delta_graphs, edge_features
from lss.latent.training import decode_latent_to_graph


class StandardizedMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_size: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, value):
        return self.net(value)


def r2_score(true, pred) -> float:
    true = np.asarray(true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    denominator = np.square(true - true.mean()).sum()
    return float(1.0 - np.square(true - pred).sum() / max(denominator, 1e-12))


def restore_ae(checkpoint_path: Path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    params = checkpoint["params"]
    normalizers = {
        key: value.to(device) for key, value in checkpoint["normalizers"].items()
    }
    model_type = str(params.get("autoencoder_model", "attention")).lower()
    model_cls = {
        "attention": NodeDeltaAttentionAutoEncoder,
        "attention_mlp": NodeDeltaAttentionAutoEncoder,
        "direct_attention": NodeDeltaDirectAttentionAutoEncoder,
        "attention_direct": NodeDeltaDirectAttentionAutoEncoder,
        "direct_attention_decoder": NodeDeltaDirectAttentionAutoEncoder,
        "mlp": NodeDeltaMLPAutoEncoder,
        "mean_mlp": NodeDeltaMLPAutoEncoder,
        "mean_pool": NodeDeltaMLPAutoEncoder,
        "pyramid_mlp": NodeDeltaPyramidMLPAutoEncoder,
        "mean_pyramid_mlp": NodeDeltaPyramidMLPAutoEncoder,
        "single_stage_attention": NodeDeltaSingleStageAttentionAutoEncoder,
        "direct_latent_attention": NodeDeltaSingleStageAttentionAutoEncoder,
        "node_to_latent_attention": NodeDeltaSingleStageAttentionAutoEncoder,
    }.get(model_type)
    if model_cls is None:
        raise ValueError(f"Unknown autoencoder_model in checkpoint: {model_type}")
    model = model_cls(
        pos_dim=2,
        node_feature_dim=int(normalizers["node_feature_mean"].numel()),
        edge_dim=int(normalizers["edge_mean"].numel()),
        hidden_size=int(params["hidden_size"]),
        latent_dim=int(params["latent_dim"]),
        latent_tokens=int(params["latent_tokens"]),
        reconstruction_dim=int(normalizers["target_mean"].numel()),
    ).to(device)
    model.load_state_dict(checkpoint["ae_state_dict"])
    model.edge_mode = "stored"
    model.eval()
    return model, normalizers, params


def encode_latent_table(
    ae,
    normalizers,
    sims,
    *,
    max_frame,
    batch_size,
    device,
    node_feature_mode="modular_history3",
):
    latent_dim = ae.latent_dim
    values = torch.empty(
        (len(sims), max_frame + 1, latent_dim), dtype=torch.float32
    )
    rows = [
        (sim_index, frame)
        for sim_index in range(len(sims))
        for frame in range(max_frame + 1)
    ]
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            batch_rows = rows[start : start + batch_size]
            data = batch_delta_graphs(
                sims,
                batch_rows,
                pos_dim=2,
                device=device,
                node_feature_mode=node_feature_mode,
                edge_mode="stored",
            )
            node = (
                data["node_feature"] - normalizers["node_feature_mean"]
            ) / normalizers["node_feature_std"]
            edge = (
                data["edge_attr"] - normalizers["edge_mean"]
            ) / normalizers["edge_std"]
            ref_edge = (
                data["ref_edge_attr"] - normalizers["edge_mean"]
            ) / normalizers["edge_std"]
            z, _ = ae.encode(
                node,
                data["ref_pos"],
                edge,
                ref_edge,
                data["edge_index"],
                data["batch"],
            )
            for local_index, (sim_index, frame) in enumerate(batch_rows):
                values[sim_index, frame] = z[local_index].detach().cpu()
            completed = min(start + len(batch_rows), len(rows))
            if completed == len(rows) or completed % 1000 < batch_size:
                print(f"encoded {completed}/{len(rows)} frames", flush=True)
    return values


def static_context_table(ae, normalizers, sims, device):
    contexts = []
    with torch.no_grad():
        for sim in sims:
            graph = sim[0]
            ref_pos = graph.x[:, :2].to(device).float()
            raw_edge = edge_features(graph, graph, pos_dim=2, device=device)
            edge = (raw_edge - normalizers["edge_mean"]) / normalizers["edge_std"]
            h0 = ae.encode_reference_graph(
                ref_pos, edge, graph.edge_index.to(device).long()
            )
            contexts.append(
                torch.cat(
                    [
                        h0.mean(0),
                        h0.std(0, unbiased=False),
                        h0.amin(0),
                        h0.amax(0),
                    ]
                ).cpu()
            )
    return torch.stack(contexts)


def state_parts(z, t):
    current = z[:, t]
    velocity = current - z[:, t - 1]
    previous_velocity = z[:, t - 1] - z[:, t - 2]
    acceleration = velocity - previous_velocity
    return current, velocity, acceleration


def make_features(z, static_context, t, *, history, warm_memory, static_memory):
    current, velocity, acceleration = state_parts(z, t)
    pieces = [current]
    if history:
        pieces.extend([velocity, acceleration])
    if warm_memory:
        pieces.append(z[:, 3])
    if static_memory:
        pieces.append(static_context)
    return torch.cat(pieces, dim=-1)


def training_table(z, static_context, variant):
    features, targets = [], []
    for t in range(3, z.size(1) - 1):
        feature = make_features(
            z,
            static_context,
            t,
            history=variant["history"],
            warm_memory=variant["warm_memory"],
            static_memory=variant["static_memory"],
        )
        next_delta = z[:, t + 1] - z[:, t]
        if variant["target"] == "acceleration":
            target = next_delta - (z[:, t] - z[:, t - 1])
        elif variant["target"] == "delta":
            target = next_delta
        elif variant["target"] == "next_z":
            target = z[:, t + 1]
        else:
            raise ValueError(variant["target"])
        features.append(feature)
        targets.append(target)
    return torch.cat(features), torch.cat(targets)


def fit_variant(train_z, val_z, train_context, val_context, variant, device, seed):
    train_x, train_y = training_table(train_z, train_context, variant)
    val_x, val_y = training_table(val_z, val_context, variant)
    x_mean = train_x.mean(0)
    x_std = train_x.std(0, unbiased=False).clamp_min(1e-6)
    y_mean = train_y.mean(0)
    y_std = train_y.std(0, unbiased=False).clamp_min(1e-6)
    train_x = ((train_x - x_mean) / x_std).to(device)
    train_y = ((train_y - y_mean) / y_std).to(device)
    val_x = ((val_x - x_mean) / x_std).to(device)
    val_y = ((val_y - y_mean) / y_std).to(device)

    torch.manual_seed(seed)
    model = StandardizedMLP(train_x.size(1), train_y.size(1)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    best_state, best_loss, stale = None, float("inf"), 0
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(1, 61):
        model.train()
        order = torch.randperm(len(train_x), generator=generator)
        for start in range(0, len(order), 512):
            indices = order[start : start + 512].to(device)
            loss = nn.functional.mse_loss(model(train_x[indices]), train_y[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(nn.functional.mse_loss(model(val_x), val_y).cpu())
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
        if stale >= 7:
            break
    model.load_state_dict(best_state)
    return model, {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "best_val_loss": best_loss,
        "epochs": epoch,
    }


def rollout_variant(model, stats, z, static_context, variant, target_step, device):
    history = [z[:, index].clone().to(device) for index in range(4)]
    static_context = static_context.to(device)
    warm = z[:, 3].to(device)
    x_mean = stats["x_mean"].to(device)
    x_std = stats["x_std"].to(device)
    y_mean = stats["y_mean"].to(device)
    y_std = stats["y_std"].to(device)
    model.eval()
    with torch.no_grad():
        for _step in range(4, target_step + 1):
            current = history[-1]
            velocity = current - history[-2]
            acceleration = velocity - (history[-2] - history[-3])
            pieces = [current]
            if variant["history"]:
                pieces.extend([velocity, acceleration])
            if variant["warm_memory"]:
                pieces.append(warm)
            if variant["static_memory"]:
                pieces.append(static_context)
            feature = (torch.cat(pieces, dim=-1) - x_mean) / x_std
            output = model(feature) * y_std + y_mean
            if variant["target"] == "acceleration":
                next_z = current + velocity + output
            elif variant["target"] == "delta":
                next_z = current + output
            else:
                next_z = output
            history.append(next_z)
    return history[-1].cpu()


def evaluate(
    ae,
    normalizers,
    sims,
    predicted_z,
    target_step,
    device,
    ae_target_mode="modular_history3",
):
    true_pratio, predicted_pratio = [], []
    with torch.no_grad():
        for sim, z in zip(sims, predicted_z):
            predicted_graph = decode_latent_to_graph(
                ae,
                sim,
                z.to(device),
                target_step,
                pos_dim=2,
                ae_target_mode=ae_target_mode,
                normalizers=normalizers,
                device=device,
            )
            reference = clone_graph(sim[0]).cpu()
            true_pratio.append(
                float(calc_p_ratio_rollout_sides(sim, target_step))
            )
            predicted_pratio.append(
                float(calc_p_ratio_rollout_sides([reference, predicted_graph], -1))
            )
    return {
        "p_ratio_r2": r2_score(true_pratio, predicted_pratio),
        "p_ratio_pearson": float(
            np.corrcoef(true_pratio, predicted_pratio)[0, 1]
        ),
        "true_std": float(np.std(true_pratio)),
        "pred_std": float(np.std(predicted_pratio)),
        "pred_to_true_std": float(
            np.std(predicted_pratio) / max(np.std(true_pratio), 1e-12)
        ),
        "true_mean": float(np.mean(true_pratio)),
        "pred_mean": float(np.mean(predicted_pratio)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=60)
    parser.add_argument("--val", type=int, default=15)
    parser.add_argument("--test", type=int, default=30)
    parser.add_argument("--target-step", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    checkpoint_path = (
        ROOT
        / "notebooks/results/08_history_aware_latent_rollout/lj_noisy/history_aware_ae.pt"
    )
    data_path = (
        ROOT / "data/lj-noisy-eps0.01-sigma1.0-cutoff1.122_500sims_200frames.pt"
    )
    output = (
        ROOT
        / "notebooks/results/08_history_aware_latent_rollout/lj_noisy"
        / "quick_propagator_sweep.csv"
    )
    ae, normalizers, params = restore_ae(checkpoint_path, device)
    all_sims = torch.load(data_path, map_location="cpu", weights_only=False)
    generator = torch.Generator().manual_seed(int(params["split_seed"]))
    order = torch.randperm(len(all_sims), generator=generator).tolist()
    train_sims = [all_sims[index] for index in order[: args.train]]
    val_start = 300
    val_sims = [
        all_sims[index] for index in order[val_start : val_start + args.val]
    ]
    test_start = 350
    test_sims = [
        all_sims[index] for index in order[test_start : test_start + args.test]
    ]
    sims = train_sims + val_sims + test_sims
    z = encode_latent_table(
        ae,
        normalizers,
        sims,
        max_frame=args.target_step,
        batch_size=args.batch_size,
        device=device,
    )
    context = static_context_table(ae, normalizers, sims, device)
    train_stop = len(train_sims)
    val_stop = train_stop + len(val_sims)
    train_z, val_z, test_z = z[:train_stop], z[train_stop:val_stop], z[val_stop:]
    train_context = context[:train_stop]
    val_context = context[train_stop:val_stop]
    test_context = context[val_stop:]

    variants = [
        {
            "name": "current_z_predict_delta",
            "history": False,
            "warm_memory": False,
            "static_memory": False,
            "target": "delta",
        },
        {
            "name": "current_z_plus_z3_predict_delta",
            "history": False,
            "warm_memory": True,
            "static_memory": False,
            "target": "delta",
        },
        {
            "name": "current_z_plus_static_predict_delta",
            "history": False,
            "warm_memory": False,
            "static_memory": True,
            "target": "delta",
        },
        {
            "name": "current_z_plus_z3_static_predict_delta",
            "history": False,
            "warm_memory": True,
            "static_memory": True,
            "target": "delta",
        },
        {
            "name": "current_z_predict_next_z",
            "history": False,
            "warm_memory": False,
            "static_memory": False,
            "target": "next_z",
        },
        {
            "name": "history_predict_acceleration",
            "history": True,
            "warm_memory": False,
            "static_memory": False,
            "target": "acceleration",
        },
        {
            "name": "history_predict_delta",
            "history": True,
            "warm_memory": False,
            "static_memory": False,
            "target": "delta",
        },
        {
            "name": "history_predict_next_z",
            "history": True,
            "warm_memory": False,
            "static_memory": False,
            "target": "next_z",
        },
        {
            "name": "history_plus_z3_predict_acceleration",
            "history": True,
            "warm_memory": True,
            "static_memory": False,
            "target": "acceleration",
        },
        {
            "name": "history_plus_z3_predict_delta",
            "history": True,
            "warm_memory": True,
            "static_memory": False,
            "target": "delta",
        },
        {
            "name": "history_plus_static_predict_delta",
            "history": True,
            "warm_memory": False,
            "static_memory": True,
            "target": "delta",
        },
        {
            "name": "history_plus_z3_static_predict_delta",
            "history": True,
            "warm_memory": True,
            "static_memory": True,
            "target": "delta",
        },
    ]
    rows = []
    for index, variant in enumerate(variants):
        print(f"training {variant['name']}", flush=True)
        model, stats = fit_variant(
            train_z,
            val_z,
            train_context,
            val_context,
            variant,
            device,
            seed=int(params["split_seed"]) + index,
        )
        prediction = rollout_variant(
            model,
            stats,
            test_z,
            test_context,
            variant,
            args.target_step,
            device,
        )
        metrics = evaluate(
            ae,
            normalizers,
            test_sims,
            prediction,
            args.target_step,
            device,
        )
        row = {
            **variant,
            **metrics,
            "one_step_val_loss": stats["best_val_loss"],
            "epochs": stats["epochs"],
            "train_networks": len(train_sims),
            "val_networks": len(val_sims),
            "test_networks": len(test_sims),
        }
        rows.append(row)
        print(json.dumps(row, indent=2), flush=True)
    result = pd.DataFrame(rows).sort_values("p_ratio_r2", ascending=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(result.to_string(index=False))
    print(f"saved {output}")


if __name__ == "__main__":
    main()
