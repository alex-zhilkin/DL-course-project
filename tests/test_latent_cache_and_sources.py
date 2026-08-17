import pandas as pd
import pytest

from lss.latent.analysis import label_evaluation_sources
from lss.latent.experiment import _expand_component_configs, latent_experiment_cache_key


def test_nested_component_configs_expand_without_notebook_prefixes():
    expanded = _expand_component_configs(
        {
            "ae_config": {
                "model": "attention",
                "target_mode": "normalized_delta",
                "max_epochs": 12,
                "lr": 1e-4,
            },
            "propagator_config": {
                "model": "delta_mlp",
                "loss": "delta",
                "max_epochs": 8,
                "context_dim": 16,
                "fixed_observed_frames": (1, 5),
            },
        }
    )
    assert expanded["autoencoder_model"] == "attention"
    assert expanded["ae_target_mode"] == "normalized_delta"
    assert expanded["ae_max_epochs"] == 12
    assert expanded["ae_lr"] == 1e-4
    assert expanded["propagator_model"] == "delta_mlp"
    assert expanded["propagator_loss"] == "delta"
    assert expanded["dyn_max_epochs"] == 8
    assert expanded["graph_context_dim"] == 16
    assert expanded["fixed_observed_frames"] == (1, 5)


def test_source_labels_come_from_evaluation_rows_not_simulation_indices():
    rows = pd.DataFrame(
        {
            "sim_idx": [0, 0, 1],
            "source": ["reid", "lj_noisy", "reid"],
        }
    )

    labeled = label_evaluation_sources(
        rows,
        {"reid": "Reid", "lj_noisy": "noisy LJ"},
    )

    assert labeled["source_family"].tolist() == ["Reid", "noisy LJ", "Reid"]
    assert "source_family" not in rows


def test_source_labeling_rejects_missing_authoritative_metadata():
    with pytest.raises(KeyError, match="authoritative 'source' metadata"):
        label_evaluation_sources(pd.DataFrame({"sim_idx": [0]}), {})


def test_cache_fingerprint_changes_with_model_config_but_not_runtime_path():
    source = {"path": "data.pt", "dataset_name": "example"}
    base = latent_experiment_cache_key(
        source,
        {
            "latent_dim": 2,
            "cache_path": "first.pt",
            "force_train": False,
            "force_train_autoencoder": False,
        },
    )
    same_model = latent_experiment_cache_key(
        source,
        {
            "latent_dim": 2,
            "cache_path": "second.pt",
            "force_train": True,
            "force_train_autoencoder": True,
        },
    )
    changed_model = latent_experiment_cache_key(
        source,
        {"latent_dim": 8, "cache_path": "first.pt", "force_train": False},
    )

    assert base == same_model
    assert base != changed_model
