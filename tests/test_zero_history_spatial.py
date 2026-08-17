from types import SimpleNamespace

import torch
from torch_geometric.data import Data

from lss.graph import clone_graph, rollout
from lss.training import _sample_autoregressive_loss


def _frame(offset: float) -> Data:
    return Data(
        x=torch.tensor([[offset, 0.0], [1.0 + offset, 0.0]]),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        edge_attr=torch.zeros(2, 2),
        box=SimpleNamespace(x=1.0, y=1.0),
    )


class _Inputs:
    def __init__(self, prev_graph, cur_graph, target_graph, pos_dim):
        self.prev_graph = prev_graph
        self.cur_graph = cur_graph
        self.target_graph = target_graph
        self.pos_dim = pos_dim


class _TrainingModel:
    cv_consistency_weight = 0.0
    time_lag_steps = 0
    time_lag_weight = 0.0

    def __init__(self):
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def __call__(self, _graph, *, is_training):
        return self.weight.square()

    def loss(self, output, inputs, *, accumulate_norm_stats):
        torch.testing.assert_close(
            inputs.cur_graph.vel_state,
            torch.zeros_like(inputs.cur_graph.x[:, : inputs.pos_dim]),
        )
        return output


class _RolloutModel:
    def eval(self):
        return self

    def __call__(self, _graph, *, is_training):
        return torch.tensor(0.0)

    def update(self, inputs, _output):
        torch.testing.assert_close(
            inputs.cur_graph.vel_state,
            torch.zeros_like(inputs.cur_graph.x[:, : inputs.pos_dim]),
        )
        predicted = clone_graph(inputs.cur_graph)
        predicted.x = predicted.x.clone()
        return predicted


def test_zero_history_training_initializes_zero_velocity():
    cfg = SimpleNamespace(history=0, pos_dim=2, node_features="positions")
    loss, _ = _sample_autoregressive_loss(
        _TrainingModel(),
        [_frame(0.0), _frame(0.1), _frame(0.2)],
        1,
        cfg,
        "cpu",
        _Inputs,
        is_train=True,
    )
    assert torch.isfinite(loss)


def test_zero_history_rollout_initializes_zero_velocity():
    frames = rollout(
        model=_RolloutModel(),
        input_graphs=[_frame(0.0)],
        num_steps=1,
        history=0,
        pos_dim=2,
        device="cpu",
        model_inputs_cls=_Inputs,
        node_features="positions",
    )
    assert len(frames) == 2
