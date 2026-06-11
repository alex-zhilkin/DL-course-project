from __future__ import annotations

import argparse
import json

from lss.config import ExperimentConfig
from lss.runner import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the standard spatial GNN on balanced mixed-temperature "
            "dePablo trajectories and evaluate a 150-step rollout."
        )
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--train-count", type=int, default=12)
    parser.add_argument("--val-count", type=int, default=12)
    parser.add_argument("--frame-limit", type=int, default=50)
    parser.add_argument("--rollout-steps", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260609)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-root", default="results")
    parser.add_argument(
        "--run-name",
        default="spatial_depablo_mixed_temperature_quick_rollout150",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExperimentConfig(
        run_name=args.run_name,
        model_type="spatial",
        dataset_path="data/depablo-10k-mix-temp.pt",
        train_count=args.train_count,
        val_count=args.val_count,
        output_root=args.output_root,
        pos_dim=2,
        history=1,
        limit=args.frame_limit,
        hidden_size=64,
        n_layers=2,
        model_extras={
            "num_mlp": 2,
            "use_skip": False,
            "final_decoder_local_skip": False,
        },
        learning_rate=1e-4,
        learning_rate_decay=0.995,
        weight_decay=1e-6,
        epochs=args.epochs,
        val_every=2,
        rollout_every=2,
        cv_eval_every=0,
        freeze_normalizers_after_epoch=3,
        device=args.device,
        seed=args.seed,
        rollout_steps=args.rollout_steps,
        split_seed=args.seed,
        shuffle_dataset_within_source=True,
        stratify_temperature=True,
        node_features="positions",
    )
    metrics = run_experiment(cfg)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
