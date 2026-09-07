"""Time identical four-source AE batches at several CPU thread counts."""
import copy
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "notebooks/results/latent_matched_study"
sys.path.insert(0, str(BASE / "code_v1/src"))
import torch
from lss.latent.experiment import run_latent_experiment, seed_everything
from lss.latent.simulation import make_frame_index
from lss.latent.training import epoch_autoencoder


def main():
    recipe = json.loads((BASE / "shared4_d4_s123/recipe.json").read_text())
    source, cfg = copy.deepcopy(recipe["source"]), copy.deepcopy(recipe["config"])
    cfg.pop("cache_path")
    cfg["ae_config"].update(max_epochs=1, patience=1, max_train_frames_per_sim=8,
                            max_val_frames_per_sim=8, train_rows_per_source=8)
    for spec in source["dataset_mixture"]:
        spec["train_count"] = spec["val_count"] = 1
        spec["split_indices"] = {k: v[:1] for k, v in spec["split_indices"].items()}
    torch.set_num_threads(1)
    seed_everything(123)
    result = run_latent_experiment(source, cfg, device="cpu")
    model = result["ae"]
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    initial = copy.deepcopy(model.state_dict())
    # Eight consecutive frames/source give one representative 32-graph batch.
    rows = make_frame_index(result["train_data"], max_frames_per_sim=8, include_last=True)
    measurements = []
    for threads in (1, 2, 4, 8, 16):
        torch.set_num_threads(threads)
        model.load_state_dict(initial)
        seed_everything(123)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
        times = []
        for iteration in range(6):
            started = time.perf_counter()
            epoch_autoencoder(model, result["train_data"], rows, batch_graphs=32,
                pos_dim=2, node_feature_mode="normalized_delta", ae_target_mode="normalized_delta",
                normalizers=result["normalizers"], device="cpu", edge_mode="compact_stored",
                mix_sources=True, gradient_method="source_mean", optimizer=optimizer)
            elapsed = time.perf_counter() - started
            if iteration:
                times.append(elapsed)
        record = {"threads": threads, "seconds": times, "median_seconds": statistics.median(times)}
        measurements.append(record)
        print(json.dumps(record), flush=True)
    with (BASE / "cpu_thread_benchmark.json").open("x") as stream:
        json.dump({"job_id": os.environ.get("PBS_JOBID"), "measurements": measurements,
                   "scope": "Throughput only; 1 train and 1 validation trajectory/source; no scientific scores or checkpoint saved."}, stream, indent=2)


if __name__ == "__main__":
    main()
