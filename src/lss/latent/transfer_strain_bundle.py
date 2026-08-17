"""Topology-independent two-strain autoencoder and one-step transfer propagator.

The learned attention autoencoders used for within-family experiments can encode
family-specific node patterns in their reference tokens.  This module provides a
strict-transfer alternative: the autoencoder coordinates are the two global side
strains, and the decoder applies those strains to any normalized reference graph.
Only the small one-step latent propagator is fitted from source trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from graph_utils import directional_side_indices_from_box

from ..graph import clone_graph


def directional_strain_latents(sim, *, side_quantile: float = 0.10) -> np.ndarray:
    """Encode every frame as horizontal and vertical directional-side strain."""

    side = directional_side_indices_from_box(sim[0], quantile=float(side_quantile))

    def dimensions(graph) -> tuple[float, float]:
        pos = graph.x[:, :2].detach().cpu().numpy()
        width = float(pos[side["right"], 0].mean() - pos[side["left"], 0].mean())
        height = float(pos[side["top"], 1].mean() - pos[side["bottom"], 1].mean())
        return width, height

    width0, height0 = dimensions(sim[0])
    if abs(width0) < 1e-12 or abs(height0) < 1e-12:
        raise ValueError("Cannot encode strain from a zero-width reference graph.")
    return np.asarray(
        [[width / width0 - 1.0, height / height0 - 1.0]
         for width, height in (dimensions(graph) for graph in sim)],
        dtype=np.float64,
    )


def decode_directional_strain(sim, z, *, frame_template: int = 0):
    """Decode a two-strain latent on the normalized reference coordinates."""

    z = np.asarray(z, dtype=np.float64).reshape(2)
    graph = clone_graph(sim[int(frame_template)]).cpu()
    ref = sim[0].x[:, :2].detach().cpu().numpy().astype(np.float64, copy=False)
    center = ref.mean(axis=0, keepdims=True)
    position = center + (ref - center) * (1.0 + z.reshape(1, 2))
    graph.x = graph.x.detach().cpu().clone().float()
    graph.x[:, :2] = torch.as_tensor(position, dtype=graph.x.dtype)
    return graph


@dataclass
class DirectionalStrainTransferBundle:
    """Two-dimensional AE plus a source-fitted, progress-free one-step map."""

    weights: np.ndarray
    observed_frames: tuple[int, int] = (1, 5)
    side_quantile: float = 0.10
    ridge: float = 1e-6

    @classmethod
    def fit(
        cls,
        simulations,
        *,
        observed_frames: tuple[int, int] = (1, 5),
        max_transitions_per_sim: int | None = 100,
        side_quantile: float = 0.10,
        ridge: float = 1e-6,
    ) -> "DirectionalStrainTransferBundle":
        """Fit z(t+1) from z(t), z(5), and z(5)-z(1), source-only."""

        first, last = map(int, observed_frames)
        if first < 0 or last <= first:
            raise ValueError("observed_frames must be ordered non-negative indices.")
        features, targets = [], []
        for sim in simulations:
            z = directional_strain_latents(sim, side_quantile=side_quantile)
            stop = len(z) - 1
            if max_transitions_per_sim is not None:
                stop = min(stop, last + int(max_transitions_per_sim))
            anchor, velocity = z[last], z[last] - z[first]
            for index in range(last, stop):
                features.append(np.concatenate([z[index], anchor, velocity, [1.0]]))
                targets.append(z[index + 1])
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(targets, dtype=np.float64)
        if len(x) == 0:
            raise ValueError("No source transitions were available to fit the bundle.")
        penalty = float(ridge) * np.eye(x.shape[1], dtype=np.float64)
        weights = np.linalg.solve(x.T @ x + penalty, x.T @ y)
        return cls(
            weights=weights,
            observed_frames=(first, last),
            side_quantile=float(side_quantile),
            ridge=float(ridge),
        )

    def step(self, z, z_first, z_last) -> np.ndarray:
        feature = np.concatenate(
            [np.asarray(z), np.asarray(z_last), np.asarray(z_last) - np.asarray(z_first), [1.0]]
        )
        return feature @ self.weights

    def rollout(self, sim, *, last_index: int | None = None) -> list:
        """Use only observed frames through z(5), then recurse one step at a time."""

        first, last = self.observed_frames
        stop = len(sim) - 1 if last_index is None else min(int(last_index), len(sim) - 1)
        if stop < last:
            return [clone_graph(graph).cpu() for graph in sim[: stop + 1]]
        encoded = directional_strain_latents(sim, side_quantile=self.side_quantile)
        rollout = [clone_graph(graph).cpu() for graph in sim[: last + 1]]
        z_first, z_last = encoded[first].copy(), encoded[last].copy()
        z = z_last.copy()
        for _ in range(last, stop):
            z = self.step(z, z_first, z_last)
            rollout.append(decode_directional_strain(sim, z))
        return rollout

    def direct_autoencoder_trajectory(self, sim, *, last_index: int | None = None) -> list:
        stop = len(sim) - 1 if last_index is None else min(int(last_index), len(sim) - 1)
        encoded = directional_strain_latents(sim, side_quantile=self.side_quantile)
        return [decode_directional_strain(sim, encoded[index]) for index in range(stop + 1)]

    def state_dict(self) -> dict:
        return {
            "kind": "directional_strain_transfer_bundle_v1",
            "weights": torch.as_tensor(self.weights, dtype=torch.float64),
            "observed_frames": tuple(self.observed_frames),
            "side_quantile": float(self.side_quantile),
            "ridge": float(self.ridge),
        }

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path) -> "DirectionalStrainTransferBundle":
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("kind") != "directional_strain_transfer_bundle_v1":
            raise ValueError("Not a directional-strain transfer bundle.")
        return cls(
            weights=state["weights"].detach().cpu().numpy(),
            observed_frames=tuple(state["observed_frames"]),
            side_quantile=float(state["side_quantile"]),
            ridge=float(state["ridge"]),
        )


__all__ = [
    "DirectionalStrainTransferBundle",
    "decode_directional_strain",
    "directional_strain_latents",
]
