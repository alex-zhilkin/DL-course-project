"""Materialize the four shared-rollout datasets in reference-box coordinates.

Every trajectory uses the same per-trajectory map: its frame-zero periodic
box becomes ``[-1, 1] x [-1, 1]``.  The map is then held fixed for every later
frame, so compression is represented directly by the evolving normalized box
and is comparable across all sources.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from lss.data import POSITION_NORMALIZATION, load_dataset  # noqa: E402


DATASETS = {
    "reid": "reid_200_frames.pt",
    "depablo_low_temp": "depablo-near-zero-temp.pt",
    "depablo_mixed_temp": "depablo-10k-mix-temp.pt",
    "lj_noisy": "lj-noisy-eps0.01-sigma1.0-cutoff1.122_200sims_200frames.pt",
}


def _validate_normalized_dataset(simulations: list, *, dataset_name: str) -> dict:
    """Check the saved convention without assuming nodes touch box boundaries."""

    if not simulations:
        raise ValueError(f"{dataset_name}: dataset is empty.")
    trajectory_lengths = {len(simulation) for simulation in simulations}
    if 0 in trajectory_lengths:
        raise ValueError(f"{dataset_name}: contains an empty trajectory.")

    minimum, maximum = float("inf"), float("-inf")
    final_x_box_widths = []
    for simulation in simulations:
        reference = simulation[0]
        reference_box = torch.as_tensor(reference.box_tensor, dtype=torch.float64)
        if not torch.allclose(reference_box[:2], torch.tensor([2.0, 2.0], dtype=torch.float64)):
            raise ValueError(
                f"{dataset_name}: frame-zero box is not [2, 2]: {reference_box.tolist()}"
            )
        for graph in simulation:
            if getattr(graph, "coordinate_normalization", None) != POSITION_NORMALIZATION:
                raise ValueError(f"{dataset_name}: missing coordinate-normalization marker.")
            position = graph.x[:, :2]
            if not torch.isfinite(position).all():
                raise ValueError(f"{dataset_name}: non-finite normalized positions.")
            minimum = min(minimum, float(position.min()))
            maximum = max(maximum, float(position.max()))
        final_x_box_widths.append(float(torch.as_tensor(simulation[-1].box_tensor)[0] / 2.0))

    return {
        "trajectories": len(simulations),
        "frames_per_trajectory": sorted(trajectory_lengths),
        "position_range": [minimum, maximum],
        "final_x_box_ratio_range": [min(final_x_box_widths), max(final_x_box_widths)],
    }


def build_dataset(input_path: Path, output_path: Path, *, dataset_name: str, force: bool) -> dict:
    """Load raw data once, normalize it, validate it, and atomically save it."""

    if output_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite {output_path}; pass --force.")
    simulations = load_dataset(
        input_path,
        coordinate_normalization=POSITION_NORMALIZATION,
        edge_multiplicity=1,
        edge_vector_dim=2,
    )
    summary = _validate_normalized_dataset(simulations, dataset_name=dataset_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(simulations, temporary_path)
    temporary_path.replace(output_path)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "normalized_reference_box_minus1_1",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Keep an existing output file and continue building the remaining datasets.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    manifest = {
        "coordinate_normalization": POSITION_NORMALIZATION,
        "description": "Each trajectory's frame-zero periodic box is [-1, 1]^2; later frames retain that reference scale.",
        "datasets": {},
    }
    for name, filename in DATASETS.items():
        input_path = input_dir / filename
        output_path = output_dir / filename
        if not input_path.is_file():
            raise FileNotFoundError(input_path)
        if output_path.exists() and args.skip_existing:
            manifest["datasets"][name] = {
                "input": str(input_path),
                "output": str(output_path),
                "status": "existing output retained",
            }
            print(f"retained {name}: {output_path}", flush=True)
            continue
        summary = build_dataset(
            input_path,
            output_path,
            dataset_name=name,
            force=args.force,
        )
        manifest["datasets"][name] = {
            "input": str(input_path),
            "output": str(output_path),
            **summary,
        }
        print(f"saved {name}: {output_path}", flush=True)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"saved manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
