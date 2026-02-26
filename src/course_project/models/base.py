from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch_geometric.data import Data

class BaseSimulator(torch.nn.Module, ABC):
    """Abstract base class for simulators to keep a consistent interface."""

    def __init__(self, pos_dim: int, *args, **kwargs):
        super().__init__()
        self._validate_pos_dim(pos_dim)
        self.pos_dim = pos_dim
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Update current training epoch for the model."""
        self.epoch = int(epoch)

    @abstractmethod
    def forward(self, data: Data, is_training: bool = True, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def update(self, inputs, model_output, dummy_sep: bool = False) -> Data:
        raise NotImplementedError

    @abstractmethod
    def loss(self, model_output, inputs, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def save_checkpoint(self, savedir: str, *, training_state: dict):
        raise NotImplementedError

    @abstractmethod
    def load_checkpoint(self, ckpdir: str):
        raise NotImplementedError

    # --- shared helpers ---
    @staticmethod
    def _validate_pos_dim(pos_dim: int) -> None:
        if pos_dim not in (2, 3):
            raise ValueError(f"pos_dim must be 2 or 3, got {pos_dim}")

    @staticmethod
    def _validate_input_dims(
        data: Data,
        min_node_features: int = 2,
        min_edge_features: int = 1,
    ) -> None:
        if data.num_node_features < min_node_features:
            raise ValueError(
                f"Expected at least {min_node_features} node features, got {data.num_node_features}"
            )
        if data.num_edge_features < min_edge_features:
            raise ValueError(
                f"Expected at least {min_edge_features} edge features, got {data.num_edge_features}"
            )


class BaseModelInputs:
    """Shared input container that applies the base pos_dim handling."""

    prev_graph: Data
    cur_graph: Data
    target_graph: Data

    prev_position: torch.Tensor
    cur_position: torch.Tensor
    target_position: torch.Tensor | None

    target_edge_attr: torch.Tensor | None

    def __init__(
        self,
        prev_data: Data,
        cur_data: Data,
        target_data: Data | None,
        pos_dim: int,
    ):
        BaseSimulator._validate_pos_dim(pos_dim)

        self.prev_graph = prev_data
        self.cur_graph = cur_data
        self.target_graph = target_data if target_data is not None else None

        self.prev_position = prev_data.x[:, :pos_dim]
        self.cur_position = cur_data.x[:, :pos_dim]
        self.target_position = target_data.x[:, :pos_dim] if target_data is not None else None
        self.target_edge_attr = target_data.edge_attr if target_data is not None else None
