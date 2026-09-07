"""Frozen-AE code-use diagnostic; observed coordinates only, never selection."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lss.latent.experiment import _load_ae_cache
from lss.latent.training import decode_latent_positions, encode_frame_latent

FRAMES = (5, 10, 25, 50, 75, 100)


def decode(model, sim, z, frame, params, norms):
    return decode_latent_positions(model, sim, z, frame, pos_dim=2,
        ae_target_mode=params["ae_target_mode"], normalizers=norms, device="cpu")


def score(model, sim, z, frame, params, norms):
    residual = decode(model, sim, z, frame, params, norms) - sim[frame].x[:, :2].float()
    scale = norms["target_std"].reshape(-1)[:2].to(residual)
    return float(residual.square().mean()), float((residual / scale).square().mean())


def encoded(model, sim, frame, params, norms):
    with torch.no_grad():
        return encode_frame_latent(model, sim, frame, pos_dim=2,
            node_feature_mode=params["node_feature_mode"], normalizers=norms, device="cpu").detach()


def optimize_code(model, sim, start, frame, params, norms, steps):
    code = torch.nn.Parameter(start.detach().clone())
    optimizer = torch.optim.Adam([code], lr=0.03)
    for _ in range(steps):
        optimizer.zero_grad()
        loss = (decode(model, sim, code, frame, params, norms) - sim[frame].x[:, :2].float()).square().mean()
        loss.backward(); optimizer.step()
    return code.detach()


def select_per_source(sims, limit):
    grouped = defaultdict(list)
    for index, sim in enumerate(sims):
        if len(grouped[sim[0].source_name]) < limit:
            grouped[sim[0].source_name].append((index, sim))
    return grouped


def result_row(split, index, source, frame, condition, model, sim, z, params, norms, start=None):
    mse, normalized_mse = score(model, sim, z, frame, params, norms)
    return dict(split=split, trajectory=index, source=source, frame=frame, condition=condition,
        position_mse=mse, normalized_position_mse=normalized_mse, start=start)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--limit-per-source", type=int, default=3)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--random-starts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=917)
    args = parser.parse_args()
    if args.limit_per_source < 2:
        raise ValueError("Need at least two trajectories per source for permutation controls.")
    torch.manual_seed(args.seed)
    loaded = _load_ae_cache(args.checkpoint, {}, device="cpu")
    model, params, norms = loaded["ae"], loaded["params"], loaded["normalizers"]
    model.eval(); rows = []
    for split, simulations in (("train", loaded["train_data"]), ("val", loaded["val_data"])):
        for source, selected in select_per_source(simulations, args.limit_per_source).items():
            for frame in FRAMES:
                codes = [encoded(model, sim, frame, params, norms) for _, sim in selected]
                mean_code = torch.stack(codes).mean(0)
                for local, (index, sim) in enumerate(selected):
                    code = codes[local]
                    rows.append(result_row(split, index, source, frame, "encoder", model, sim, code, params, norms))
                    rows.append(result_row(split, index, source, frame, "same_source_same_time_permuted", model, sim, codes[(local + 1) % len(codes)], params, norms))
                    rows.append(result_row(split, index, source, frame, "training_source_time_mean", model, sim, mean_code, params, norms))
                    wrong = encoded(model, sim, FRAMES[(FRAMES.index(frame) + 3) % len(FRAMES)], params, norms)
                    rows.append(result_row(split, index, source, frame, "wrong_time", model, sim, wrong, params, norms))
                    fitted = optimize_code(model, sim, code, frame, params, norms, args.steps)
                    rows.append(result_row(split, index, source, frame, "optimized_code", model, sim, fitted, params, norms, "encoder"))
                    for random_start in range(args.random_starts):
                        fitted = optimize_code(model, sim, torch.randn_like(code), frame, params, norms, args.steps)
                        rows.append(result_row(split, index, source, frame, "optimized_code", model, sim, fitted, params, norms, f"normal_{random_start}"))
    base = ROOT / "notebooks/results/ae_information_audit"; base.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows); frame.to_csv(base / "frozen_code_diagnostic.csv", index=False)
    frame.groupby(["split", "source", "frame", "condition", "start"], dropna=False).agg(
        total=("position_mse", "size"), position_mse=("position_mse", "mean"),
        normalized_position_mse=("normalized_position_mse", "mean"),
    ).reset_index().to_csv(base / "frozen_code_diagnostic_summary.csv", index=False)
    (base / "frozen_code_diagnostic.json").write_text(json.dumps({
        "checkpoint": str(args.checkpoint), "limit_per_source": args.limit_per_source, "frames": FRAMES,
        "optimization_steps": args.steps, "random_starts": args.random_starts, "seed": args.seed,
        "scope": "Frozen-checkpoint diagnostic only; fitted codes are not forecasts, validation predictions, or model-selection inputs. No expert targets.",
    }, indent=2))


if __name__ == "__main__":
    main()
