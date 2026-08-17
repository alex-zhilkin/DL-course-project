"""Build a deterministic fixed-size subset of the MetaForge Reid trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from graph_utils.box import Box


DEFAULT_SOURCE = Path(
    "/home/alexz/MetaForge/results/compression_10K_datasets/"
    "data_aux_opt_lowT_448sims_noang/trajectories"
)
DEFAULT_OUTPUT = Path(
    "data/real_reid_200_frames.pt"
)


def evenly_spaced_indices(frame_count: int, target_frames: int) -> list[int]:
    """Select the full trajectory span, retaining both endpoints."""

    if frame_count < target_frames:
        raise ValueError(
            f"Trajectory has {frame_count} frames; need at least {target_frames}."
        )
    indices = np.rint(np.linspace(0, frame_count - 1, target_frames)).astype(int)
    if len(np.unique(indices)) != target_frames:
        raise ValueError("Frame downsampling produced duplicate indices.")
    return indices.tolist()


def build_dataset(
    source_dir: Path,
    output: Path,
    *,
    trajectory_count: int,
    target_frames: int,
    seed: int,
    force: bool,
) -> None:
    source_paths = sorted(source_dir.glob("trajectory_*.pt"))
    if len(source_paths) < trajectory_count:
        raise ValueError(
            f"Found {len(source_paths)} trajectories, need {trajectory_count}."
        )
    if output.exists() and not force:
        raise FileExistsError(f"Output already exists: {output}")

    generator = np.random.default_rng(seed)
    selected_positions = np.sort(
        generator.choice(len(source_paths), size=trajectory_count, replace=False)
    )
    selected_paths = [source_paths[int(index)] for index in selected_positions]

    simulations = []
    manifest_rows = []
    for subset_index, source_path in enumerate(selected_paths):
        trajectory = torch.load(
            source_path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        frame_indices = evenly_spaced_indices(len(trajectory), target_frames)
        selected_frames = [trajectory[index] for index in frame_indices]
        source_id = int(source_path.stem.rsplit("_", 1)[-1])

        reference_edge_index = selected_frames[0].edge_index
        reference_stiffness = selected_frames[0].edge_attr[:, -1]
        for output_frame, (source_frame, graph) in enumerate(
            zip(frame_indices, selected_frames)
        ):
            if not torch.equal(graph.edge_index, reference_edge_index):
                raise ValueError(
                    f"Topology changes in {source_path.name} at frame {source_frame}."
                )
            if not torch.equal(graph.edge_attr[:, -1], reference_stiffness):
                raise ValueError(
                    f"Stiffness changes in {source_path.name} at frame {source_frame}."
                )
            source_box = graph.box
            graph.box = Box(
                float(source_box.x1),
                float(source_box.x2),
                float(source_box.y1),
                float(source_box.y2),
                float(getattr(source_box, "z1", -0.1)),
                float(getattr(source_box, "z2", 0.1)),
            )
            graph.source_name = "real_reid"
            graph.source_sim_id = source_id
            graph.source_frame_index = int(source_frame)
            graph.subset_sim_index = int(subset_index)
            graph.subset_frame_index = int(output_frame)
            graph.edge_stiffness_length_exponent = 2
        simulations.append(selected_frames)
        manifest_rows.append(
            {
                "subset_sim_index": subset_index,
                "source_sim_id": source_id,
                "source_path": str(source_path),
                "source_frame_count": len(trajectory),
                "selected_frame_indices": frame_indices,
            }
        )
        print(
            f"[{subset_index + 1:03d}/{trajectory_count}] "
            f"{source_path.name}: {len(trajectory)} -> {target_frames} frames",
            flush=True,
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(simulations, temporary)
    temporary.replace(output)

    manifest = {
        "source_directory": str(source_dir),
        "output": str(output),
        "selection_seed": seed,
        "available_trajectories": len(source_paths),
        "selected_trajectories": trajectory_count,
        "frames_per_trajectory": target_frames,
        "selection": manifest_rows,
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"saved dataset: {output}", flush=True)
    print(f"saved manifest: {manifest_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--trajectory-count", type=int, default=200)
    parser.add_argument("--target-frames", type=int, default=200)
    parser.add_argument("--seed", type=int, default=34234)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    build_dataset(
        args.source_dir.expanduser().resolve(),
        args.output.expanduser().resolve(),
        trajectory_count=args.trajectory_count,
        target_frames=args.target_frames,
        seed=args.seed,
        force=args.force,
    )


if __name__ == "__main__":
    main()
