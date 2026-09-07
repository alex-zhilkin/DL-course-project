import unittest

import torch

from lss.latent.models import NodeDeltaMessagePassingAutoEncoder


class MessagePassingAutoEncoderTests(unittest.TestCase):
    def _model(self):
        return NodeDeltaMessagePassingAutoEncoder(
            pos_dim=2, edge_dim=4, hidden_size=12, latent_dim=2,
            latent_tokens=4, message_passing_steps=2, message_chunk_size=2,
        )

    def test_forward_backward_and_edge_order_permutation(self):
        torch.manual_seed(7)
        model = self._model()
        delta = torch.randn(4, 2, requires_grad=True)
        ref = torch.randn(4, 2)
        # Canonical (one-per-pair) edges with directional x/y components.
        edges = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]])
        attr = torch.randn(4, 4)
        batch = torch.zeros(4, dtype=torch.long)
        recon, z = model(delta, ref, attr, attr, edges, batch)
        loss = recon.square().mean() + z.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(recon).all())
        self.assertTrue(torch.isfinite(z).all())
        self.assertTrue(torch.isfinite(delta.grad).all())
        order = torch.tensor([2, 0, 3, 1])
        with torch.no_grad():
            recon_permuted, z_permuted = model(
                delta.detach(), ref, attr[order], attr[order], edges[:, order], batch
            )
        torch.testing.assert_close(recon.detach(), recon_permuted, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(z.detach(), z_permuted, atol=1e-6, rtol=1e-6)

    def test_empty_graph_edges_is_finite(self):
        model = self._model()
        delta = torch.randn(3, 2)
        ref = torch.randn(3, 2)
        empty_edges = torch.empty((2, 0), dtype=torch.long)
        empty_attr = torch.empty((0, 4))
        recon, z = model(delta, ref, empty_attr, empty_attr, empty_edges, torch.zeros(3, dtype=torch.long))
        self.assertTrue(torch.isfinite(recon).all())
        self.assertTrue(torch.isfinite(z).all())

    def test_normalized_direction_reversal_matches_raw_then_normalize(self):
        model = self._model()
        mean = torch.tensor([2.0, -3.0, 7.0, 11.0])
        std = torch.tensor([4.0, 5.0, 2.0, 3.0])
        model.set_edge_normalization(mean, std)
        raw = torch.tensor([[6.0, -8.0, 9.0, 12.0]])
        normalized = (raw - mean) / std
        _, _, directed = model._bidirectional_edges(
            normalized, torch.tensor([[0], [1]])
        )
        expected_reversed_raw = raw.clone()
        expected_reversed_raw[:, :2] *= -1
        expected = (expected_reversed_raw - mean) / std
        torch.testing.assert_close(directed[1:], expected)


if __name__ == "__main__":
    unittest.main()
