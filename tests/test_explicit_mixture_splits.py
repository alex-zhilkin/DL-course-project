import unittest
from types import SimpleNamespace
from unittest.mock import patch

from lss.data import resolve_dataset_splits


class ExplicitMixtureSplits(unittest.TestCase):
    def split(self, indices, sources=("a",), seed=123):
        mixture = [{"name": source, "path": source, "train_count": 2,
                    "val_count": 1, "split_indices": indices} for source in sources]
        with patch("lss.data.load_dataset", side_effect=lambda *a, **kw: [
            [SimpleNamespace(identity=i)] for i in range(6)
        ]):
            return resolve_dataset_splits("unused", train_count=0, val_count=0,
                                          dataset_mixture=mixture, split_seed=seed,
                                          shuffle_within_source=True)

    def test_source_membership_and_seed_do_not_change_explicit_identities(self):
        indices = {"train": [4, 0], "val": [2], "test": []}
        first = self.split(indices)
        second = self.split(indices, ("b", "a"), seed=987)
        for split, expected in zip(first[:3], ([4, 0], [2], [])):
            self.assertEqual([s[0].identity for s in split], expected)
        for left, right in zip(first[:3], second[:3]):
            self.assertEqual([s[0].identity for s in left],
                             [s[0].identity for s in right if s[0].source_name == "a"])
        self.assertEqual(first[3][0]["excluded"], 3)

    def test_invalid_indices_fail(self):
        for indices in (
            {"train": [0, 1], "val": [1], "test": []},
            {"train": [0, 0], "val": [1], "test": []},
            {"train": [0, 6], "val": [1], "test": []},
            {"train": [0, -1], "val": [1], "test": []},
            {"train": [0, True], "val": [1], "test": []},
            {"train": [0], "val": [1], "test": []},
        ):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                self.split(indices)


if __name__ == "__main__":
    unittest.main()
