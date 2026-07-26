"""Separate cached-latent study for improving LJ Poisson-ratio rollout."""

from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from lss.latent.capacity import load_experiment_bundle
from lss.latent.experiment import evaluate_rollout_horizons, seed_everything
from lss.latent.models import make_latent_propagator
from lss.latent.training import (
    LatentNormalizer,
    encode_frame_latent,
    encode_reference_context,
)
from lss.utils import resolve_device


BASELINE_CACHE = (
    PROJECT_ROOT
    / "notebooks"
    / "results"
    / "08_lj_train1_vs20"
    / "models"
    / "lj_noisy_train20_seed20260716.pt"
)
OUTPUT_DIR = PROJECT_ROOT / "notebooks" / "results" / "09_lj_rollout_optimization"
SEED = 20260716


def encode_split(result: dict, split: str, device) -> tuple[torch.Tensor, torch.Tensor]:
    sims = result[f"{split}_data"]
    ae = result["ae"]
    normalizers = result["normalizers"]
    all_latents = []
    contexts = []
    ae.eval()
    with torch.no_grad():
        for sim_idx, sim in enumerate(sims):
            contexts.append(
                encode_reference_context(
                    ae,
                    sim,
                    pos_dim=2,
                    normalizers=normalizers,
                    device=device,
                ).cpu()
            )
            all_latents.append(
                torch.stack(
                    [
                        encode_frame_latent(
                            ae,
                            sim,
                            frame,
                            pos_dim=2,
                            node_feature_mode="normalized_delta",
                            normalizers=normalizers,
                            device=device,
                        ).cpu()
                        for frame in range(len(sim))
                    ],
                    dim=0,
                )
            )
            if sim_idx == 0 or (sim_idx + 1) % 5 == 0:
                print(f"encoded {split} {sim_idx + 1}/{len(sims)}", flush=True)
    return torch.stack(all_latents), torch.stack(contexts)


def fit_stats(latents: torch.Tensor, contexts: torch.Tensor) -> LatentNormalizer:
    z0 = latents[:, :-1].reshape(-1, latents.size(-1))
    z1 = latents[:, 1:].reshape(-1, latents.size(-1))
    dz = z1 - z0
    return LatentNormalizer(
        z_mean=z0.mean(0, keepdim=True),
        z_std=z0.std(0, keepdim=True, unbiased=False).clamp_min(1e-6),
        dz_mean=dz.mean(0, keepdim=True),
        dz_std=dz.std(0, keepdim=True, unbiased=False).clamp_min(1e-6),
        z_next_mean=z1.mean(0, keepdim=True),
        z_next_std=z1.std(0, keepdim=True, unbiased=False).clamp_min(1e-6),
        context_mean=contexts.mean(0, keepdim=True),
        context_std=contexts.std(0, keepdim=True, unbiased=False).clamp_min(1e-6),
    )


def flattened_steps(latents: torch.Tensor, contexts: torch.Tensor):
    z0 = latents[:, :-1].reshape(-1, latents.size(-1))
    dz = (latents[:, 1:] - latents[:, :-1]).reshape(-1, latents.size(-1))
    context = contexts[:, None, :].expand(-1, latents.size(1) - 1, -1).reshape(
        -1, contexts.size(-1)
    )
    return z0, dz, context


def one_step_loss(model, tensors, stats: LatentNormalizer, indices: torch.Tensor, device):
    z, dz, context = tensors
    z = z[indices].to(device)
    dz = dz[indices].to(device)
    context = context[indices].to(device)
    prediction = model(stats.normalize_z(z), stats.normalize_context(context))
    return F.mse_loss(prediction, stats.normalize_dz(dz))


def train_one_step(
    model,
    train_tensors,
    val_tensors,
    stats: LatentNormalizer,
    *,
    device,
    epochs: int = 120,
    patience: int = 12,
    learning_rate: float = 1e-4,
    batch_size: int = 256,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    best_state, best_val, stale, history = None, float("inf"), 0, []
    train_n, val_n = len(train_tensors[0]), len(val_tensors[0])
    stats = stats.to(device)
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(train_n)
        train_losses = []
        for start in range(0, train_n, batch_size):
            loss = one_step_loss(
                model, train_tensors, stats, order[start : start + batch_size], device
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            val_losses = [
                float(
                    one_step_loss(
                        model,
                        val_tensors,
                        stats,
                        torch.arange(start, min(start + batch_size, val_n)),
                        device,
                    )
                )
                for start in range(0, val_n, batch_size)
            ]
        train_loss, val_loss = float(np.mean(train_losses)), float(np.mean(val_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0 or stale == 0:
            print(
                f"one-step {epoch:03d} train={train_loss:.5g} val={val_loss:.5g} stale={stale}",
                flush=True,
            )
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    return model.eval(), pd.DataFrame(history), best_val


def multistep_batch_loss(
    model,
    latents: torch.Tensor,
    contexts: torch.Tensor,
    stats: LatentNormalizer,
    pairs: torch.Tensor,
    *,
    horizons: tuple[int, ...],
    device,
):
    sim_idx, starts = pairs[:, 0], pairs[:, 1]
    z = latents[sim_idx, starts].to(device)
    context = contexts[sim_idx].to(device)
    losses = []
    horizon_set = set(horizons)
    for step in range(1, max(horizons) + 1):
        pred_dz = model(stats.normalize_z(z), stats.normalize_context(context))
        z = z + stats.unnormalize_dz(pred_dz)
        if step in horizon_set:
            target = latents[sim_idx, starts + step].to(device)
            losses.append(F.mse_loss(stats.normalize_z(z), stats.normalize_z(target)))
    return torch.stack(losses).mean()


def train_multistep(
    model,
    train_latents,
    train_contexts,
    val_latents,
    val_contexts,
    stats,
    *,
    device,
    horizons=(1, 5, 10, 20),
    epochs=60,
    patience=10,
    learning_rate=2e-5,
    batch_size=128,
):
    max_h = max(horizons)
    train_pairs = torch.tensor(
        [(i, t) for i in range(len(train_latents)) for t in range(train_latents.size(1) - max_h)],
        dtype=torch.long,
    )
    val_pairs = torch.tensor(
        [(i, t) for i in range(len(val_latents)) for t in range(val_latents.size(1) - max_h)],
        dtype=torch.long,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    stats = stats.to(device)
    best_state, best_val, stale, history = None, float("inf"), 0, []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train_pairs))
        train_losses = []
        for start in range(0, len(order), batch_size):
            pairs = train_pairs[order[start : start + batch_size]]
            loss = multistep_batch_loss(
                model,
                train_latents,
                train_contexts,
                stats,
                pairs,
                horizons=horizons,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            # A deterministic subset keeps validation cheap and identical across epochs.
            val_subset = val_pairs[::4]
            val_losses = [
                float(
                    multistep_batch_loss(
                        model,
                        val_latents,
                        val_contexts,
                        stats,
                        val_subset[start : start + batch_size],
                        horizons=horizons,
                        device=device,
                    )
                )
                for start in range(0, len(val_subset), batch_size)
            ]
        train_loss, val_loss = float(np.mean(train_losses)), float(np.mean(val_losses))
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0 or stale == 0:
            print(
                f"multistep {epoch:03d} train={train_loss:.5g} val={val_loss:.5g} stale={stale}",
                flush=True,
            )
        if stale >= patience:
            break
    model.load_state_dict(best_state)
    return model.eval(), pd.DataFrame(history), best_val


def save_variant(path: Path, model, history, stats, spec):
    torch.save(
        {
            "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "history": history,
            "stats": {k: v.detach().cpu() for k, v in stats.as_dict().items()},
            "spec": spec,
        },
        path,
    )


def evaluate_variant(name, model, stats, result, cfg, device):
    rows, summary = evaluate_rollout_horizons(
        result["ae"],
        model,
        result["test_data"][:30],
        stats.to(device),
        cfg=cfg,
        normalizers=result["normalizers"],
        dataset=f"LJ noisy | {name}",
        split_name="test",
        rollout_steps=[10, 20, 50, 100, 150],
        device=device,
    )
    rows.insert(0, "variant", name)
    summary.insert(0, "variant", name)
    return rows, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    seed_everything(SEED)

    baseline_bundle = torch.load(BASELINE_CACHE, map_location="cpu", weights_only=False)
    cfg = dict(baseline_bundle["params"])
    result = load_experiment_bundle(BASELINE_CACHE, cfg=cfg, device=device)
    cache_path = OUTPUT_DIR / "encoded_latents.pt"
    if cache_path.exists() and not args.force:
        encoded = torch.load(cache_path, map_location="cpu", weights_only=False)
        train_latents, train_contexts = encoded["train_latents"], encoded["train_contexts"]
        val_latents, val_contexts = encoded["val_latents"], encoded["val_contexts"]
    else:
        train_latents, train_contexts = encode_split(result, "train", device)
        val_latents, val_contexts = encode_split(result, "val", device)
        torch.save(
            {
                "train_latents": train_latents,
                "train_contexts": train_contexts,
                "val_latents": val_latents,
                "val_contexts": val_contexts,
            },
            cache_path,
        )
    stats = fit_stats(train_latents, train_contexts)
    train_tensors = flattened_steps(train_latents, train_contexts)
    val_tensors = flattened_steps(val_latents, val_contexts)

    variants = [
        {"name": "standardized_ctx16", "graph_context_dim": 16, "hidden_size": 96},
        {"name": "standardized_ctx96", "graph_context_dim": 96, "hidden_size": 192},
    ]
    trained = {}
    histories = []
    for spec in variants:
        name = spec["name"]
        path = OUTPUT_DIR / f"{name}.pt"
        model = make_latent_propagator(
            2,
            spec["hidden_size"],
            model_type="delta_mlp",
            context_dim=96,
            graph_context_dim=spec["graph_context_dim"],
        ).to(device)
        if path.exists() and not args.force:
            bundle = torch.load(path, map_location=device, weights_only=False)
            model.load_state_dict(bundle["state_dict"])
            history = pd.DataFrame(bundle["history"])
        else:
            model, history, _ = train_one_step(
                model, train_tensors, val_tensors, stats, device=device
            )
            save_variant(path, model, history, stats, spec)
        histories.append(history.assign(variant=name, stage="one_step"))
        trained[name] = (model.eval(), spec)

    best_name = min(
        trained,
        key=lambda name: float(
            histories[[h["variant"].iloc[0] for h in histories].index(name)]["val_loss"].min()
        ),
    )
    base_model, base_spec = trained[best_name]
    multistep_name = f"{best_name}_multistep20"
    multistep_path = OUTPUT_DIR / f"{multistep_name}.pt"
    multistep_model = deepcopy(base_model).to(device)
    if multistep_path.exists() and not args.force:
        bundle = torch.load(multistep_path, map_location=device, weights_only=False)
        multistep_model.load_state_dict(bundle["state_dict"])
        multistep_history = pd.DataFrame(bundle["history"])
    else:
        multistep_model, multistep_history, _ = train_multistep(
            multistep_model,
            train_latents,
            train_contexts,
            val_latents,
            val_contexts,
            stats,
            device=device,
        )
        save_variant(
            multistep_path,
            multistep_model,
            multistep_history,
            stats,
            {**base_spec, "base": best_name, "horizons": [1, 5, 10, 20]},
        )
    histories.append(multistep_history.assign(variant=multistep_name, stage="multistep"))
    trained[multistep_name] = (multistep_model.eval(), base_spec)

    rollout_rows, rollout_summaries = [], []
    for name, (model, spec) in trained.items():
        eval_cfg = dict(cfg)
        eval_cfg.update(
            {
                "propagator_objective": "one_step",
                "propagator_loss": "delta",
                "propagator_model": "delta_mlp",
                "propagator_use_static_context": True,
                "graph_context_dim": spec["graph_context_dim"],
                "propagator_standardize_latent": True,
            }
        )
        rows, summary = evaluate_variant(name, model, stats, result, eval_cfg, device)
        rollout_rows.append(rows)
        rollout_summaries.append(summary)
    pd.concat(histories, ignore_index=True).to_csv(OUTPUT_DIR / "training_histories.csv", index=False)
    pd.concat(rollout_rows, ignore_index=True).to_csv(OUTPUT_DIR / "rollout_rows.csv", index=False)
    summary = pd.concat(rollout_summaries, ignore_index=True)
    summary.to_csv(OUTPUT_DIR / "test_rollout_summary.csv", index=False)
    print(summary[["variant", "rollout_steps", "rollout_position_r2", "p_ratio_r2", "p_ratio_pearson"]].to_string(index=False))


if __name__ == "__main__":
    main()
