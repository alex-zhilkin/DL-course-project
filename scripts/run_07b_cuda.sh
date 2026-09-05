#!/usr/bin/env bash
# Execute 07b in a fresh kernel from a CUDA-visible host terminal.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
notebook="$project_root/notebooks/latent_space/07b_mixed_reid_depablo_lj_context_ablation.ipynb"

cd "$project_root"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

"$project_root/.venv/bin/python" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is not visible to this launcher. Run this script from the same "
        "GPU-visible terminal where `nvidia-smi` works."
    )
print({
    "python": sys.executable,
    "torch": torch.__version__,
    "gpu": torch.cuda.get_device_name(0),
})
PY

exec "$project_root/.venv/bin/python" \
  "$project_root/scripts/run_07b_experiment.py" --device cuda
