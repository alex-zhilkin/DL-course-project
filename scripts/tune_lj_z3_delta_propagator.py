"""Tune the frozen-AE z(t), z(3) -> delta-z propagator without p-ratio loss."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.quick_lj_frozen_ae_propagator_sweep import (
    encode_latent_table,
    evaluate,
    restore_ae,
)


class DeltaMLP(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, hidden_size: int, depth: int):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_size), nn.GELU()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.GELU()])
        output = nn.Linear(hidden_size, latent_dim)
        nn.init.xavier_uniform_(output.weight)
        output.weight.data.mul_(0.01)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.net = nn.Sequential(*layers)

    def forward(self, value):
        return self.net(value)


def feature(z_current, z3, progress, *, include_progress):
    pieces = [z_current, z3]
    if include_progress:
        pieces.append(
            torch.full(
                (len(z_current), 1),
                float(progress),
                dtype=z_current.dtype,
                device=z_current.device,
            )
        )
    return torch.cat(pieces, dim=-1)


def teacher_forced_table(z, *, include_progress):
    features, targets = [], []
    horizon = z.size(1) - 1
    for frame in range(3, horizon):
        features.append(
            feature(
                z[:, frame],
                z[:, 3],
                frame / horizon,
                include_progress=include_progress,
            )
        )
        targets.append(z[:, frame + 1] - z[:, frame])
    return torch.cat(features), torch.cat(targets)


def rollout(
    model,
    z,
    *,
    target_step,
    include_progress,
    x_mean,
    x_std,
    y_mean,
    y_std,
    device,
):
    current = z[:, 3].to(device)
    z3 = current.clone()
    with torch.no_grad():
        for frame in range(3, target_step):
            raw = feature(
                current,
                z3,
                frame / target_step,
                include_progress=include_progress,
            )
            prediction = model((raw - x_mean) / x_std)
            current = current + prediction * y_std + y_mean
    return current


def fit_config(
    train_z,
    val_z,
    *,
    include_progress,
    hidden_size,
    depth,
    learning_rate,
    weight_decay,
    target_step,
    device,
    seed,
):
    train_x, train_y = teacher_forced_table(
        train_z, include_progress=include_progress
    )
    x_mean = train_x.mean(0).to(device)
    x_std = train_x.std(0, unbiased=False).clamp_min(1e-6).to(device)
    y_mean = train_y.mean(0).to(device)
    y_std = train_y.std(0, unbiased=False).clamp_min(1e-6).to(device)
    train_x = ((train_x - x_mean.cpu()) / x_std.cpu()).to(device)
    train_y = ((train_y - y_mean.cpu()) / y_std.cpu()).to(device)
    val_scale = train_z.std((0, 1), unbiased=False).clamp_min(1e-6).to(device)

    torch.manual_seed(seed)
    model = DeltaMLP(
        train_x.size(1),
        train_y.size(1),
        hidden_size=hidden_size,
        depth=depth,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    generator = torch.Generator().manual_seed(seed)
    best = None
    stale = 0
    for epoch in range(1, 81):
        model.train()
        order = torch.randperm(len(train_x), generator=generator)
        for start in range(0, len(order), 512):
            indices = order[start : start + 512].to(device)
            prediction = model(train_x[indices])
            loss = nn.functional.mse_loss(prediction, train_y[indices])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        predicted_terminal = rollout(
            model,
            val_z,
            target_step=target_step,
            include_progress=include_progress,
            x_mean=x_mean,
            x_std=x_std,
            y_mean=y_mean,
            y_std=y_std,
            device=device,
        )
        true_terminal = val_z[:, target_step].to(device)
        terminal_loss = float(
            (((predicted_terminal - true_terminal) / val_scale) ** 2)
            .mean()
            .cpu()
        )
        if best is None or terminal_loss < best["terminal_loss"] - 1e-5:
            best = {
                "terminal_loss": terminal_loss,
                "epoch": epoch,
                "state_dict": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
            }
            stale = 0
        else:
            stale += 1
        if stale >= 10:
            break
    model.load_state_dict(best["state_dict"])
    return model, {
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
        "val_terminal_latent_mse": best["terminal_loss"],
        "best_epoch": best["epoch"],
        "trained_epochs": epoch,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=int, default=200)
    parser.add_argument("--val", type=int, default=40)
    parser.add_argument("--test", type=int, default=80)
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
    output_dir = (
        ROOT
        / "notebooks/results/08_history_aware_latent_rollout/lj_noisy"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = output_dir / "z3_delta_tuning.csv"
    output_model = output_dir / "z3_delta_best.pt"

    ae, normalizers, params = restore_ae(checkpoint_path, device)
    all_sims = torch.load(data_path, map_location="cpu", weights_only=False)
    generator = torch.Generator().manual_seed(int(params["split_seed"]))
    order = torch.randperm(len(all_sims), generator=generator).tolist()
    train_sims = [all_sims[index] for index in order[: args.train]]
    val_sims = [
        all_sims[index] for index in order[300 : 300 + args.val]
    ]
    test_sims = [
        all_sims[index] for index in order[350 : 350 + args.test]
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
    train_stop = len(train_sims)
    val_stop = train_stop + len(val_sims)
    train_z = z[:train_stop]
    val_z = z[train_stop:val_stop]
    test_z = z[val_stop:]

    configurations = [
        {
            "include_progress": include_progress,
            "hidden_size": hidden_size,
            "depth": depth,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
        }
        for include_progress, hidden_size, depth, learning_rate, weight_decay in itertools.product(
            [False, True],
            [64, 128, 256],
            [1, 2, 3],
            [1e-4, 3e-4],
            [0.0, 1e-5],
        )
    ]
    rows = []
    candidates = []
    for index, configuration in enumerate(configurations):
        print(
            f"[{index + 1}/{len(configurations)}] {configuration}",
            flush=True,
        )
        model, stats = fit_config(
            train_z,
            val_z,
            **configuration,
            target_step=args.target_step,
            device=device,
            seed=int(params["split_seed"]) + index,
        )
        row = {
            **configuration,
            "val_terminal_latent_mse": stats["val_terminal_latent_mse"],
            "best_epoch": stats["best_epoch"],
            "trained_epochs": stats["trained_epochs"],
        }
        rows.append(row)
        candidates.append((row, model, stats))
        print(json.dumps(row, indent=2), flush=True)

    candidates.sort(key=lambda item: item[0]["val_terminal_latent_mse"])
    selected = candidates[:5]
    for rank, (row, model, stats) in enumerate(selected, start=1):
        prediction = rollout(
            model,
            test_z,
            target_step=args.target_step,
            include_progress=bool(row["include_progress"]),
            x_mean=stats["x_mean"],
            x_std=stats["x_std"],
            y_mean=stats["y_mean"],
            y_std=stats["y_std"],
            device=device,
        ).cpu()
        metrics = evaluate(
            ae,
            normalizers,
            test_sims,
            prediction,
            args.target_step,
            device,
        )
        row.update(
            {
                "validation_rank": rank,
                **metrics,
                "train_networks": len(train_sims),
                "val_networks": len(val_sims),
                "test_networks": len(test_sims),
            }
        )
        if rank == 1:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "configuration": row,
                    "normalization": {
                        key: stats[key].detach().cpu()
                        for key in ("x_mean", "x_std", "y_mean", "y_std")
                    },
                    "ae_checkpoint": str(checkpoint_path),
                },
                output_model,
            )
    result = pd.DataFrame(rows).sort_values(
        "val_terminal_latent_mse"
    )
    result.to_csv(output_csv, index=False)
    display_columns = [
        "include_progress",
        "hidden_size",
        "depth",
        "learning_rate",
        "weight_decay",
        "val_terminal_latent_mse",
        "validation_rank",
        "p_ratio_r2",
        "p_ratio_pearson",
        "pred_to_true_std",
        "best_epoch",
    ]
    print(result[display_columns].head(12).to_string(index=False))
    print(f"saved {output_csv}")
    print(f"saved {output_model}")


if __name__ == "__main__":
    main()
