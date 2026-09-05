"""Repair stored noisy-LJ edge vectors to follow the shared edge convention.

For every stored directed edge ``source -> target``, the first two edge
attributes become the minimum-image displacement ``x[target] - x[source]``.
The geometric length column is recomputed from that displacement; other edge
attributes (including stiffness) are retained unchanged.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch


DEFAULT_PATH = Path(
    "data/lj-noisy-eps0.01-sigma1.0-cutoff1.122_200sims_200frames.pt"
)


def box_lengths(graph: object, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    box = getattr(graph, "box", None)
    if box is not None and all(hasattr(box, name) for name in ("x1", "x2", "y1", "y2")):
        return torch.tensor(
            [abs(float(box.x2) - float(box.x1)), abs(float(box.y2) - float(box.y1))],
            dtype=dtype,
            device=device,
        )
    box_tensor = getattr(graph, "box_tensor", None)
    if isinstance(box_tensor, torch.Tensor) and box_tensor.numel() >= 2:
        return box_tensor[:2].to(dtype=dtype, device=device)
    raise ValueError("Frame has no usable periodic box")


def repair(path: Path) -> tuple[int, int]:
    trajectories = torch.load(path, map_location="cpu", weights_only=False)
    frames = 0
    for trajectory in trajectories:
        for graph in trajectory:
            edge_attr = graph.edge_attr
            if edge_attr.ndim != 2 or edge_attr.size(1) < 3:
                raise ValueError("Expected edge_attr with vector and length columns")
            source, target = graph.edge_index.long()
            lengths = box_lengths(graph, dtype=graph.x.dtype, device=graph.x.device)
            vector = graph.x[target, :2] - graph.x[source, :2]
            vector = vector - torch.round(vector / lengths) * lengths
            graph.edge_attr = edge_attr.clone()
            graph.edge_attr[:, :2] = vector.to(edge_attr)
            graph.edge_attr[:, 2] = torch.linalg.vector_norm(vector, dim=-1).to(edge_attr)
            frames += 1

    temporary_path = path.with_suffix(path.suffix + ".repairing")
    torch.save(trajectories, temporary_path)
    os.replace(temporary_path, path)
    return len(trajectories), frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    trajectories, frames = repair(args.path)
    print(f"repaired {args.path}: {trajectories} trajectories, {frames} frames")


if __name__ == "__main__":
    main()
