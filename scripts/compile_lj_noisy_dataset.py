"""Compile chunked LJ trajectories into the project's list-of-trajectories format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from graph_utils.box import Box


def install_legacy_box_alias() -> None:
    """Allow loading source chunks that serialized their box as network.Box."""

    import types

    module = sys.modules.setdefault("network", types.ModuleType("network"))
    module.Box = Box


def evenly_spaced_indices(length: int, count: int) -> list[int]:
    if length < count:
        raise ValueError(f"Cannot select {count} unique frames from a {length}-frame trajectory.")
    indices = np.rint(np.linspace(0, length - 1, int(count))).astype(int)
    if len(np.unique(indices)) != int(count):
        raise RuntimeError("Even frame selection unexpectedly produced duplicate indices.")
    return indices.tolist()


def convert_frame(frame, *, source_frame: int, sim_id: int, registry_p_ratio: float):
    converted = frame.clone()
    old_box = frame.box
    converted.box = Box(
        old_box.x1,
        old_box.x2,
        old_box.y1,
        old_box.y2,
        old_box.z1,
        old_box.z2,
    )
    converted.time = int(source_frame)
    converted.source_frame_index = int(source_frame)
    converted.source_sim_id = int(sim_id)
    converted.source_name = "lj_noisy"
    converted.registry_poisson_ratio = float(registry_p_ratio)
    converted.lj_active = True
    converted.lj_epsilon = 0.01
    converted.lj_sigma = 1.0
    converted.lj_cutoff = 1.122
    return converted


def validate_trajectory(trajectory: list, *, expected_frames: int) -> None:
    if len(trajectory) != int(expected_frames):
        raise ValueError(f"Expected {expected_frames} frames, found {len(trajectory)}.")
    reference = trajectory[0]
    if reference.x.ndim != 2 or reference.x.size(1) < 2:
        raise ValueError(f"Unexpected node tensor shape: {tuple(reference.x.shape)}")
    for frame in trajectory:
        if frame.x.shape != reference.x.shape:
            raise ValueError("Node shape changes within a trajectory.")
        if frame.edge_index.shape != reference.edge_index.shape:
            raise ValueError("Edge-index shape changes within a trajectory.")
        if frame.edge_attr.shape != reference.edge_attr.shape:
            raise ValueError("Edge-attribute shape changes within a trajectory.")
        if not torch.isfinite(frame.x).all() or not torch.isfinite(frame.edge_attr).all():
            raise ValueError("Trajectory contains non-finite node or edge values.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--max-sims", type=int, default=200)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source_dir = args.source_dir.expanduser().resolve()
    output_path = args.output_path.resolve()
    manifest_path = args.manifest or output_path.with_suffix(".csv")
    if output_path.exists() and not args.force:
        raise FileExistsError(f"Refusing to overwrite {output_path}; pass --force intentionally.")

    registry = pd.read_csv(source_dir / "data_registry.csv")
    required = {"sim_id", "poisson_ratio", "file_path"}
    missing = required - set(registry.columns)
    if missing:
        raise ValueError(f"Registry is missing columns: {sorted(missing)}")
    selected = registry.iloc[: int(args.max_sims)].copy()
    if len(selected) < int(args.max_sims):
        raise ValueError(f"Requested {args.max_sims} simulations but registry has {len(selected)}.")

    install_legacy_box_alias()
    simulations = []
    manifest_rows = []
    for compiled_idx, row in enumerate(selected.itertuples(index=False)):
        source_path = source_dir / Path(row.file_path).name
        source_trajectory = torch.load(source_path, map_location="cpu", weights_only=False)
        frame_ids = evenly_spaced_indices(len(source_trajectory), int(args.frames))
        trajectory = [
            convert_frame(
                source_trajectory[frame_idx],
                source_frame=frame_idx,
                sim_id=int(row.sim_id),
                registry_p_ratio=float(row.poisson_ratio),
            )
            for frame_idx in frame_ids
        ]
        validate_trajectory(trajectory, expected_frames=int(args.frames))
        simulations.append(trajectory)
        manifest_rows.append(
            {
                "compiled_index": compiled_idx,
                "source_sim_id": int(row.sim_id),
                "registry_poisson_ratio": float(row.poisson_ratio),
                "source_file": str(source_path),
                "source_frames": len(source_trajectory),
                "compiled_frames": len(trajectory),
                "first_source_frame": frame_ids[0],
                "last_source_frame": frame_ids[-1],
            }
        )
        if compiled_idx == 0 or (compiled_idx + 1) % 10 == 0:
            print(f"compiled {compiled_idx + 1}/{len(selected)}: {source_path.name}", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(simulations, output_path)
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"saved {len(simulations)} simulations to {output_path}")
    print(f"saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
