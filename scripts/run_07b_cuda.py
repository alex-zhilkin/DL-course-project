"""Execute the current 07b notebook code cells in a real Python process."""

import json
from pathlib import Path



NOTEBOOK = Path("notebooks/latent_space/07b_mixed_reid_depablo_lj_context_ablation.ipynb")
namespace = {"__name__": "__main__"}


def display(value):
    if value is not None:
        print(value)


namespace["display"] = display
notebook = json.loads(NOTEBOOK.read_text())
for index in (1, 2, 3, 4, 6, 7, 8):
    source = "".join(notebook["cells"][index].get("source", []))
    source = "\n".join(line for line in source.splitlines() if not line.startswith("%"))
    print(f"\n===== notebook cell {index} =====", flush=True)
    if index == 4:
        source = source.replace(
            "[('A_no_context', False), ('B_small_mean_context', True)]",
            "[('A_no_context', False)]",
        )
    exec(compile(source, f"{NOTEBOOK}:cell-{index}", "exec"), namespace)
