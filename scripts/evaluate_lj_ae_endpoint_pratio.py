"""Endpoint-only p-ratio audit of an LJ latent autoencoder.

This deliberately never calls the latent propagator.  It compares the
two-frame p-ratio measured from real frames 0 and T with the same measurement
after AE reconstruction.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import torch
from graph_utils import calc_p_ratio_rollout_sides

from lss.graph import clone_graph
from lss.latent.capacity import load_experiment_bundle
from lss.latent.simulation import pearson_r, r2_score
from lss.latent.training import decode_latent_to_graph, encode_frame_latent
from lss.utils import resolve_device


DEFAULT_MODEL = (
    ROOT
    / "notebooks/results/08_lj_train1_vs20/models"
    / "lj_noisy_train20_seed20260716.pt"
)
DEFAULT_OUTPUT = (
    ROOT / "notebooks/results/08_lj_train1_vs20/ae_endpoint_pratio.csv"
)


def endpoint_p_ratio(first, final) -> float:
    return float(
        calc_p_ratio_rollout_sides(
            [clone_graph(first).cpu(), clone_graph(final).cpu()], -1
        )
    )


def reconstruct_frame(result, sim, frame: int, device):
    cfg = result["params"]
    z = encode_frame_latent(
        result["ae"],
        sim,
        frame,
        pos_dim=int(cfg["pos_dim"]),
        node_feature_mode=str(cfg["node_feature_mode"]),
        normalizers=result["normalizers"],
        device=device,
    )
    return decode_latent_to_graph(
        result["ae"],
        sim,
        z,
        frame,
        pos_dim=int(cfg["pos_dim"]),
        ae_target_mode=str(cfg["ae_target_mode"]),
        normalizers=result["normalizers"],
        device=device,
    )


def summarize(rows: pd.DataFrame, predicted: str) -> dict[str, float]:
    finite = np.isfinite(rows["true_p_ratio"] * rows[predicted])
    true = rows.loc[finite, "true_p_ratio"].to_numpy(float)
    pred = rows.loc[finite, predicted].to_numpy(float)
    return {
        "used": int(finite.sum()),
        "r2": r2_score(true, pred),
        "pearson": pearson_r(true, pred),
        "mae": float(np.mean(np.abs(true - pred))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    bundle = torch.load(args.model, map_location="cpu", weights_only=False)
    result = load_experiment_bundle(
        args.model, cfg=dict(bundle["params"]), device=device
    )
    rows = []
    result["ae"].eval()
    with torch.no_grad():
        for split in ("train", "val", "test"):
            simulations = result[f"{split}_data"]
            for sim_index, sim in enumerate(simulations):
                final_index = len(sim) - 1
                decoded_first = reconstruct_frame(result, sim, 0, device)
                decoded_final = reconstruct_frame(
                    result, sim, final_index, device
                )
                rows.append(
                    {
                        "split": split,
                        "sim_index": sim_index,
                        "final_frame": final_index,
                        "true_p_ratio": endpoint_p_ratio(sim[0], sim[final_index]),
                        # This isolates reconstruction of the deformed frame.
                        "exact_initial_pred_p_ratio": endpoint_p_ratio(
                            sim[0], decoded_final
                        ),
                        # This measures the complete two-frame AE round trip.
                        "both_decoded_pred_p_ratio": endpoint_p_ratio(
                            decoded_first, decoded_final
                        ),
                    }
                )

    frame = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    for split, group in frame.groupby("split", sort=False):
        print(f"\n{split} ({len(group)} networks)")
        for column in (
            "exact_initial_pred_p_ratio",
            "both_decoded_pred_p_ratio",
        ):
            print(column, summarize(group, column))
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
