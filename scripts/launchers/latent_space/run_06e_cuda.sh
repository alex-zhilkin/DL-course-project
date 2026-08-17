#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$project_root"

export MPLBACKEND=Agg
export MPLCONFIGDIR=/tmp/matplotlib-06e-cuda
export PYTHONPATH="$project_root/src"

exec "$project_root/.venv/bin/python" -c '
import json

path = "notebooks/latent_space/06e_zero_shot_mixed_temperature_transfer.ipynb"
with open(path) as handle:
    notebook = json.load(handle)
code = "\n\n".join(
    "".join(
        line
        for line in cell.get("source", [])
        if not line.lstrip().startswith(("%", "!"))
    )
    for cell in notebook["cells"]
    if cell.get("cell_type") == "code"
)
exec(compile(code, path, "exec"))
'
