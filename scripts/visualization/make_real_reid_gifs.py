"""Render representative Real Reid trajectories as diagnostic GIFs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np
import torch

from lss.data import _install_legacy_auxetic_box_alias


def fractional_trajectory(simulation) -> np.ndarray:
    positions = np.stack([graph.x[:, :2].cpu().numpy() for graph in simulation])
    boxes = np.asarray(
        [[graph.box.x1, graph.box.x2, graph.box.y1, graph.box.y2] for graph in simulation],
        dtype=float,
    )
    lower = boxes[:, None, :][:, :, [0, 2]]
    size = boxes[:, None, :][:, :, [1, 3]] - lower
    fractional = (positions - lower) / size - 0.5
    increments = np.diff(fractional, axis=0)
    increments -= np.round(increments)
    return np.concatenate(
        [fractional[:1], fractional[:1] + np.cumsum(increments, axis=0)], axis=0
    )


def cubic_residual(unwrapped: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    time = np.linspace(-1.0, 1.0, len(unwrapped))
    design = np.column_stack([np.ones_like(time), time, time**2, time**3])
    weights = np.linalg.lstsq(
        design, unwrapped.reshape(len(unwrapped), -1), rcond=None
    )[0]
    smooth = (design @ weights).reshape(unwrapped.shape)
    return smooth, unwrapped - smooth


def wrap(values: np.ndarray) -> np.ndarray:
    return (values + 0.5) % 1.0 - 0.5


def undirected_edges(graph) -> np.ndarray:
    edge_index = graph.edge_index.cpu().numpy().T
    edges = {tuple(sorted((int(source), int(target)))) for source, target in edge_index}
    return np.asarray(sorted(edge for edge in edges if edge[0] != edge[1]), dtype=int)


def visible_segments(position: np.ndarray, edges: np.ndarray) -> np.ndarray:
    segments = position[edges]
    keep = np.all(np.abs(segments[:, 1] - segments[:, 0]) < 0.5, axis=1)
    return segments[keep]


def noise_ratio(path: Path) -> float:
    simulation = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    trajectory = fractional_trajectory(simulation)
    step = np.diff(trajectory, axis=0)
    second = np.diff(trajectory, n=2, axis=0)
    return float(np.sqrt(np.mean(second**2)) / np.sqrt(np.mean(step**2)))


def representative_rows(rows: list[dict]) -> list[tuple[str, dict, float]]:
    scored = [(row, noise_ratio(Path(row["source_path"]))) for row in rows]
    scores = np.asarray([score for _, score in scored])
    chosen = []
    used = set()
    for label, quantile in (("lower", 0.1), ("median", 0.5), ("higher", 0.9)):
        target = float(np.quantile(scores, quantile))
        candidates = sorted(scored, key=lambda item: abs(item[1] - target))
        row, score = next(item for item in candidates if item[0]["source_sim_id"] not in used)
        used.add(row["source_sim_id"])
        chosen.append((label, row, score))
    return chosen


def render(
    source_path: Path,
    output_path: Path,
    *,
    source_id: int,
    noise_level: str,
    noise_score: float,
    magnification: float,
    frame_stride: int,
    fps: int,
) -> None:
    simulation = torch.load(source_path, map_location="cpu", weights_only=False, mmap=True)
    unwrapped = fractional_trajectory(simulation)
    smooth, residual = cubic_residual(unwrapped)
    physical_view = wrap(unwrapped)
    enhanced_view = wrap(smooth + magnification * residual)
    residual_magnitude = np.linalg.norm(residual, axis=-1)
    color_limit = max(float(np.quantile(residual_magnitude, 0.99)), 1e-8)
    color_norm = Normalize(vmin=0.0, vmax=color_limit)
    cmap = plt.get_cmap("magma")
    edges = undirected_edges(simulation[0])
    frames = list(range(0, len(simulation), max(1, frame_stride)))
    if frames[-1] != len(simulation) - 1:
        frames.append(len(simulation) - 1)

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.15), constrained_layout=True)
    collections = []
    scatters = []
    for axis, position, label in zip(
        axes,
        (physical_view[0], enhanced_view[0]),
        ("Recorded node motion", f"Fluctuation-enhanced (residual ×{magnification:g})"),
    ):
        lines = LineCollection(
            visible_segments(position, edges), color="#839097", linewidth=0.38, alpha=0.38
        )
        axis.add_collection(lines)
        scatter = axis.scatter(
            position[:, 0],
            position[:, 1],
            c=residual_magnitude[0],
            cmap=cmap,
            norm=color_norm,
            s=8,
            linewidth=0,
        )
        axis.set(xlim=(-0.52, 0.52), ylim=(-0.52, 0.52), xlabel="box-relative x", ylabel="box-relative y")
        axis.set_aspect("equal")
        axis.set_title(label, fontsize=10)
        axis.grid(False)
        axis.spines[["top", "right"]].set_visible(False)
        collections.append(lines)
        scatters.append(scatter)

    frame_label = axes[0].text(
        0.02,
        0.98,
        "",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76, "pad": 2},
    )
    fig.colorbar(scatters[0], ax=axes, label="detrended displacement", shrink=0.82, pad=0.02)

    def update(frame_index: int):
        positions = (physical_view[frame_index], enhanced_view[frame_index])
        for lines, scatter, position in zip(collections, scatters, positions):
            lines.set_segments(visible_segments(position, edges))
            scatter.set_offsets(position)
            scatter.set_array(residual_magnitude[frame_index])
        frame_label.set_text(
            f"frame {frame_index + 1}/{len(simulation)}\n"
            f"network {source_id} · {noise_level} fluctuation example\n"
            f"second/step ratio {noise_score:.2f}"
        )
        return [*collections, *scatters, frame_label]

    animation = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)


def main() -> None:
    _install_legacy_auxetic_box_alias()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/real_reid_200_frames.manifest.json")
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("notebooks/results/06_mixed_dataset_shared_latent_space_boxnorm/real_reid_gifs"),
    )
    parser.add_argument("--magnification", type=float, default=35.0)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()

    rows = json.loads(args.manifest.read_text())["selection"]
    for label, row, score in representative_rows(rows):
        source_id = int(row["source_sim_id"])
        destination = args.output_dir / f"real_reid_{label}_fluctuation_network_{source_id:05d}.gif"
        print(f"rendering {destination} ...", flush=True)
        render(
            Path(row["source_path"]),
            destination,
            source_id=source_id,
            noise_level=label,
            noise_score=score,
            magnification=args.magnification,
            frame_stride=args.frame_stride,
            fps=args.fps,
        )
        print(f"saved {destination}", flush=True)


if __name__ == "__main__":
    main()
