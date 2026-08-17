from collections import Counter
from types import SimpleNamespace

import torch

from lss.latent.training import _source_mixed_rows


def test_source_mixing_preserves_rows_and_spreads_small_sources():
    sims = [
        [SimpleNamespace(source_name="large")],
        [SimpleNamespace(source_name="medium")],
        [SimpleNamespace(source_name="small")],
    ]
    rows = (
        [(0, index) for index in range(100)]
        + [(1, index) for index in range(20)]
        + [(2, index) for index in range(10)]
    )
    torch.manual_seed(7)
    mixed = _source_mixed_rows(sims, rows, shuffle=True)

    assert Counter(mixed) == Counter(rows)
    assert len(mixed) == len(rows)
    for start in range(0, 120, 30):
        sources = {sims[int(row[0])][0].source_name for row in mixed[start : start + 30]}
        assert sources == {"large", "medium", "small"}
