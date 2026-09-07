"""Check cluster dependencies and data availability; never train or run notebooks."""
from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import platform
import sys


ROOT = Path(__file__).resolve().parents[1]
DATASETS = {
    "reid": "reid_200_frames.pt",
    "depablo_low_temp": "depablo-near-zero-temp.pt",
    "depablo_mixed_temp": "depablo-10k-mix-temp.pt",
    "lj_noisy": "lj-noisy-eps0.01-sigma1.0-cutoff1.122_200sims_200frames.pt",
}


def main() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    report = {
        "python": sys.executable,
        "host": platform.node(),
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "dependencies": {},
        "datasets": {},
        "errors": [],
    }
    for name in ("numpy", "pandas", "scipy", "torch", "torch_geometric",
                 "graph_utils.box", "lss.latent.experiment"):
        try:
            module = importlib.import_module(name)
            report["dependencies"][name] = {
                "version": getattr(module, "__version__", None),
                "file": getattr(module, "__file__", None),
            }
            if name == "torch":
                report["cuda_available"] = module.cuda.is_available()
                report["cuda_version"] = module.version.cuda
        except Exception as exc:
            report["errors"].append(f"{name}: {type(exc).__name__}: {exc}")
    for source, filename in DATASETS.items():
        path = ROOT / "data" / filename
        report["datasets"][source] = {
            "path": str(path),
            "bytes": path.stat().st_size if path.is_file() else None,
        }
        if not path.is_file():
            report["errors"].append(f"Missing dataset: {path}")
    results = ROOT / "notebooks" / "results"
    report["historical_checkpoints"] = sorted(
        str(path.relative_to(ROOT)) for path in results.rglob("*.pt")
    )
    report["scope"] = "Environment and file metadata only; datasets not deserialized."
    # Exclusive creation prevents accidental overwrites, including manual reruns.
    destination = Path(os.environ["LSS_PREFLIGHT_REPORT"])
    with destination.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2), flush=True)
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
