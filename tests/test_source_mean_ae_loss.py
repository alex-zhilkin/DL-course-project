import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from lss.latent.training import epoch_autoencoder


class SourceMeanLoss(unittest.TestCase):
    def test_source_weighting_is_independent_of_node_and_graph_counts(self):
        class FixedModel(torch.nn.Module):
            def forward(self, *args):
                return torch.tensor([[1.], [1.], [3.], [3.], [3.], [3.]]), None

        zero = torch.zeros(6, 1)
        batch = {"node_feature": zero, "ref_pos": zero, "edge_attr": zero,
                 "ref_edge_attr": zero, "edge_index": torch.zeros(2, 6, dtype=torch.long),
                 "batch": torch.tensor([0, 1, 2, 2, 2, 2])}
        norms = {f"{name}_{stat}": torch.tensor(float(stat == "std"))
                 for name in ("target", "node_feature", "edge", "ref_edge")
                 for stat in ("mean", "std")}
        sims = [[SimpleNamespace(source_name=name)] for name in ("a", "a", "b")]
        with patch("lss.latent.training.batch_delta_graphs", return_value=batch), patch(
            "lss.latent.training.ae_target_tensor", return_value=zero
        ):
            result = epoch_autoencoder(FixedModel(), sims, [(0, 0), (1, 0), (2, 0)],
                batch_graphs=3, pos_dim=1, node_feature_mode="normalized_delta",
                ae_target_mode="normalized_delta", normalizers=norms, device="cpu",
                gradient_method="source_mean")
        self.assertAlmostEqual(result["loss"], 5.0)
        self.assertAlmostEqual(result["macro_source_reconstruction"], 5.0)
        self.assertAlmostEqual(result["max_source_reconstruction"], 9.0)
        self.assertAlmostEqual(result["reconstruction"], 38 / 6, places=5)


if __name__ == "__main__":
    unittest.main()
