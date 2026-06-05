from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch_geometric.data import Data


class BaseSimulator(torch.nn.Module, ABC):
    def __init__(self, pos_dim: int, *args, **kwargs):
        super().__init__()
        self.pos_dim = pos_dim

    @abstractmethod
    def forward(self, data: Data, is_training: bool = True, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def update(self, inputs, model_output) -> Data:
        raise NotImplementedError

    @abstractmethod
    def loss(self, model_output, inputs, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError

    def save_checkpoint(self, savedir: str, *, training_state: dict):
        torch.save(self._checkpoint_payload(training_state), savedir)

    def load_checkpoint(self, ckpdir: str):
        self._load_checkpoint_payload(torch.load(ckpdir, weights_only=False, map_location="cpu"))

    def _checkpoint_payload(self, training_state: dict) -> dict:
        return {
            "model": self.state_dict(),
            "output_normalizer": self.output_normalizer.get_variable(),
            "node_normalizer": self.node_normalizer.get_variable(),
            "edge_normalizer": self.edge_normalizer.get_variable(),
            "model_config": self.cfg,
            **training_state,
        }

    def _load_checkpoint_payload(self, payload: dict) -> None:
        self.load_state_dict(payload["model"])
        self.cfg = payload["model_config"]
        for normalizer_name in ("output_normalizer", "node_normalizer", "edge_normalizer"):
            normalizer = getattr(self, normalizer_name)
            for key, value in payload[normalizer_name].items():
                cur = getattr(normalizer, key)
                if isinstance(cur, torch.Tensor) and isinstance(value, torch.Tensor):
                    cur.copy_(value)
                else:
                    setattr(normalizer, key, value)


class BaseModelInputs:
    """Shared input container"""

    prev_graph: Data
    cur_graph: Data
    target_graph: Data

    prev_position: torch.Tensor
    cur_position: torch.Tensor
    target_position: torch.Tensor

    target_edge_attr: torch.Tensor

    def __init__(
        self,
        prev_data: Data,
        cur_data: Data,
        target_data: Data,
        pos_dim: int,
    ):
        self.prev_graph = prev_data
        self.cur_graph = cur_data
        self.target_graph = target_data

        self.prev_position = prev_data.x[:, :pos_dim]
        self.cur_position = cur_data.x[:, :pos_dim]
        self.target_position = target_data.x[:, :pos_dim]
        self.target_edge_attr = target_data.edge_attr
