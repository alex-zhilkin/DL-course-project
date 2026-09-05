"""Run 07b with multi-horizon closed-loop propagator supervision."""

import json
from pathlib import Path

import torch


NOTEBOOK = Path("notebooks/latent_space/07b_mixed_reid_depablo_lj_context_ablation.ipynb")
AE_CACHE = Path(
    "notebooks/results/07b_mixed_reid_depablo_lj_context_ablation/"
    "mixed4source_A_no_context_latent8_attention_stored13_unbalancedAE_train20_100frames.pt"
)
notebook = json.loads(NOTEBOOK.read_text())
namespace = {"__name__": "__main__"}
namespace["display"] = lambda value: print(value) if value is not None else None
for index in (1, 2, 3):
    source = "".join(notebook["cells"][index].get("source", []))
    source = "\n".join(line for line in source.splitlines() if not line.startswith("%"))
    exec(compile(source, f"{NOTEBOOK}:cell-{index}", "exec"), namespace)

namespace["LATENT_DIM"] = 8
namespace["TRAIN_PER_SOURCE"] = 20
namespace["VAL_PER_SOURCE"] = 10
for entry in namespace["MIXTURE"]:
    entry["train_count"] = namespace["TRAIN_PER_SOURCE"]
    entry["val_count"] = namespace["VAL_PER_SOURCE"]
source = {
    "dataset_name": "mixed_reid_depablo_lj",
    "source_name": "mixed_reid_depablo_lj",
    "label": "Reid + dePablo + noisy LJ",
    "path": namespace["MIXTURE"][0]["path"],
    "dataset_mixture": namespace["MIXTURE"],
}
config, _ = namespace["build_config"]("B_multistep", use_static_context=True)
config["force_train"] = True
config["force_train_autoencoder"] = False
config["pretrained_ae_cache_path"] = str(AE_CACHE)
config["pretrained_ae_require_matching_normalizers"] = True
config["propagator_config"]["context_dim"] = 16
config["propagator_config"]["context_include_temperature"] = False
config["propagator_config"]["context_include_source_id"] = True
config["propagator_config"]["fixed_history_include_progress"] = False
config["propagator_config"]["checkpoint_metric"] = "val_rollout_min_source_endpoint_p_ratio_r2"
config["propagator_config"]["checkpoint_mode"] = "max"
config["propagator_config"]["position_loss_weight"] = 0.0
config["propagator_config"]["lr"] = 1e-4
config["propagator_config"]["max_epochs"] = 30
config["propagator_config"]["patience"] = 8
config["propagator_config"]["objective"] = "fixed_history_one_step"
config["propagator_config"]["model"] = "fixed_velocity_residual_mlp"
config["propagator_config"]["fixed_observed_frames"] = (1, 5)
config["propagator_config"]["multistep_horizons"] = list(range(1, 17))
config["propagator_config"]["multistep_loss"] = "all_steps_plus_optional_decoded_endpoint"
config["batch_graphs"] = 1024
config["propagator_config"]["rollout_eval_sims_per_source"] = 10
config["rollout_final_eval_sims_per_source"] = 10
config["cache_path"] = str(
    AE_CACHE.with_name("mixed4source_B_multistep_stored13_train20_100frames.pt")
)
namespace["seed_everything"](namespace["SEED"])
print("running", config["cache_path"], flush=True)
result = namespace["run_latent_experiment"](source, config, device=namespace["DEVICE"])
print(result["rollout_stats"].to_string(index=False), flush=True)
