from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ExperimentConfig:
    run_name: str
    model_type: str
    dataset_path: str
    train_count: int
    val_count: int
    output_root: str
    pos_dim: int
    history: int
    limit: int
    hidden_size: int
    n_layers: int
    model_extras: dict
    learning_rate: float
    learning_rate_decay: float
    weight_decay: float
    epochs: int
    val_every: int
    rollout_every: int
    cv_eval_every: int
    freeze_normalizers_after_epoch: int
    device: str
    seed: int
    rollout_steps: int | None = None
    dataset_mixture: list[dict] | None = None
    split_seed: int | None = None
    shuffle_dataset_within_source: bool = False
    stratify_temperature: bool = False
    mix_holdout_across_sources: bool = False
    node_features: str = "positions"

    def to_dict(self) -> dict:
        return asdict(self)
