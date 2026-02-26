from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class ExperimentConfig:
    run_name: str
    model_type: str
    train_dataset: str
    val_dataset: str
    output_root: str = "course_project/results"

    pos_dim: int = 2
    history: int = 1
    limit: int = 20
    node_features: str = "velocity"

    hidden_size: int = 64
    n_layers: int = 2
    model_extras: dict = field(default_factory=dict)

    learning_rate: float = 1e-4
    learning_rate_decay: float = 0.995
    weight_decay: float = 0.0
    epochs: int = 100
    val_every: int = 10
    rollout_every: int = 0
    cv_eval_every: int = 0

    rollout_steps: int = 50
    cv_pratio_target: str = "box"
    train_rollout_steps: int = 1
    train_rollout_loss_decay: float = 1.0
    freeze_normalizers_after_epoch: int = 5

    device: str = "cuda"
    seed: int = 42
    verbose: bool = True
    log_every: int = 1

    def to_dict(self) -> dict:
        return asdict(self)
