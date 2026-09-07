"""Matched dynamics-only AE architecture comparison; mixed-T is evaluation-only."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(os.environ["LSS_PROJECT_ROOT"])
CODE = Path(os.environ["LSS_CODE_ROOT"])
sys.path.insert(0, str(CODE / "src"))
import pandas as pd
import torch
from lss.latent.experiment import run_latent_experiment, seed_everything, resolve_train_val_test
from lss.latent.training import encode_frame_latent, decode_latent_to_graph
from latent_diagnostic_metrics import metrics, summarize

VARIANTS = ("orientation_corrected", "mp2", "mp4", "single_stage")
SEEDS = (3456456, 123, 456, 786, 2026)


def build_recipe(baseline, variant, latent_dim, output, smoke):
    source = copy.deepcopy(baseline["source"])
    cfg = copy.deepcopy(baseline["config"])
    mixed = copy.deepcopy(baseline["mixed_evaluation"])
    cfg.update(should_train_propagator=False, should_rollout=False, cache_path=str(output / "ae.pt"))
    cfg.pop("propagator_config")
    ae = cfg["ae_config"]
    ae.update(
        max_epochs=2 if smoke else 40,
        patience=1 if smoke else 8,
        latent_dim=int(latent_dim),
        gradient_method="source_mean",
        checkpoint_metric="val_max_source_reconstruction",
        checkpoint_mode="min",
        pratio_eval_every=0,
    )
    if variant in {"mp2", "mp4"}:
        ae.update(model="message_passing", message_passing_steps=int(variant[-1]))
    elif variant == "orientation_corrected":
        ae.update(model="orientation_corrected", message_passing_steps=0)
    elif variant == "single_stage":
        ae.update(model="single_stage_attention", message_passing_steps=0)
    if smoke:
        cfg.update(train_count=2, val_count=2, ae_batch_graphs=2)
    return source, cfg, mixed


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--latent-dim", choices=(2, 8), type=int, required=True)
    parser.add_argument("--seed", choices=SEEDS, type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    baseline_path = ROOT / f"notebooks/results/06b_ae_mixed_ablation/exclude_mixed_d2_s{args.seed}/recipe.json"
    baseline = json.loads(baseline_path.read_text())
    # A frozen code version is part of a scientific run identity.  In
    # particular, a failed smoke attempt must never make a corrected retry
    # overwrite its recipe or incomplete artifacts.
    code_version = os.environ.get("LSS_CODE_VERSION", "working_tree")
    run_name = f"{args.variant}_d{args.latent_dim}_s{args.seed}_{code_version}"
    if args.smoke:
        run_name += "_smoke"
    output = ROOT / "notebooks/results/shared_ae_architecture" / run_name
    source, cfg, mixed_source = build_recipe(baseline, args.variant, args.latent_dim, output, args.smoke)
    assert {item["name"] for item in source["dataset_mixture"]} == {"reid", "depablo_low_temp", "lj_noisy"}
    manifest = json.loads((ROOT / "notebooks/results/latent_matched_study/split_manifest.json").read_text())
    for spec in [*source["dataset_mixture"], *mixed_source["dataset_mixture"]]:
        if file_sha(spec["path"]) != manifest["sources"][spec["name"]]["sha256"]:
            raise ValueError(f"Dataset changed: {spec['name']}")
    output.mkdir(parents=True, exist_ok=False)
    design = {
        "orientation_corrected": "Existing pyramid-attention AE with only exact raw-coordinate-consistent edge reversal after fitted normalization.",
        "mp2": "Two residual nonlinear neighbour-message layers before the unchanged pyramid attention pool.",
        "mp4": "Four residual nonlinear neighbour-message layers before the unchanged pyramid attention pool.",
        "single_stage": "Existing direct node-to-latent attention pooling; encoder and decoder match the attention baseline.",
    }[args.variant]
    (output / "recipe.json").write_text(json.dumps({
        "source": source, "config": cfg, "mixed_evaluation": mixed_source,
        "variant": args.variant, "latent_dim": args.latent_dim, "seed": args.seed,
        "job_id": os.environ.get("PBS_JOBID", "local"), "code_root": str(CODE),
        "code_version": code_version,
        "baseline_recipe": str(baseline_path), "baseline_sha256": file_sha(baseline_path),
        "mixed_evaluation_only": True, "response_selection": False, "smoke": args.smoke,
        "design": design,
        "selection": "Worst retained-source validation coordinate reconstruction; no response, strain, source label, or mixed-T fitting/selection.",
    }, indent=2))
    seed_everything(args.seed)
    started = time.time()
    result = run_latent_experiment(source, cfg, device="cpu")
    assert not result["test_data"]
    assert all(sim[0].source_name != "depablo_mixed_temp" for key in ("train_data", "val_data") for sim in result[key])
    result["ae_history"].to_csv(output / "ae_history.csv", index=False)
    train, mixed, test, _ = resolve_train_val_test(mixed_source, result["params"], split_seed=cfg["split_seed"])
    assert not train and not test and len(mixed) == (2 if args.smoke else 20)
    ae, params, norms = result["ae"], result["params"], result["normalizers"]
    ae.eval(); counts = {}; rows = []
    with torch.no_grad():
        for sim in [*result["val_data"], *mixed]:
            name = sim[0].source_name; ordinal = counts.get(name, 0); counts[name] = ordinal + 1
            identity = ordinal if args.smoke else manifest["sources"][name]["split_indices"]["val"][ordinal]
            for frame in (5, 10, 25, 50, 75, 100):
                z = encode_frame_latent(ae, sim, frame, pos_dim=2, node_feature_mode=params["node_feature_mode"], normalizers=norms, device="cpu")
                pred = decode_latent_to_graph(ae, sim, z, frame, pos_dim=2, ae_target_mode=params["ae_target_mode"], normalizers=norms, device="cpu")
                rows.append(dict(source=name, original_index=identity, frame=frame, **metrics(sim, frame, pred)))
    frame = pd.DataFrame(rows); frame.to_csv(output / "validation_rows.csv", index=False)
    summarize(frame, ["source", "frame"]).to_csv(output / "source_summary.csv", index=False)
    (output / "completed.json").write_text(json.dumps({
        "seconds": time.time() - started, "seed": args.seed, "variant": args.variant,
        "latent_dim": args.latent_dim, "parameters": sum(p.numel() for p in ae.parameters()),
        "job_id": os.environ.get("PBS_JOBID", "local"), "split": "validation",
        "mixed_evaluation_only": True, "smoke": args.smoke,
    }, indent=2))


if __name__ == "__main__":
    main()
