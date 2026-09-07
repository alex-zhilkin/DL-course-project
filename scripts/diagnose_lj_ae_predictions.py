"""Describe completed validation predictions without fitting or selecting on mixed-T."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1] / 'notebooks/results/lj_ae_repair'


def main():
    rows, inputs = [], []
    for marker in sorted(BASE.glob('*/completed.json')):
        run = marker.parent
        recipe = json.loads((run / 'recipe.json').read_text())
        path = run / 'validation_rows.csv'
        inputs.append(dict(run=run.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for (source, frame), g in pd.read_csv(path).groupby(['source', 'frame']):
            valid = np.isfinite(g.true_p_ratio) & np.isfinite(g.pred_p_ratio)
            y = g.loc[valid, 'true_p_ratio'].to_numpy()
            p = g.loc[valid, 'pred_p_ratio'].to_numpy()
            variance = np.var(y)
            bias = np.mean(p-y)
            mse = np.mean((p-y)**2)
            rows.append(dict(variant=recipe['variant'], seed=recipe['seed'], source=source,
                frame=frame, valid=int(valid.sum()), total=len(g),
                response_r2=1-mse/variance if variance > 0 else np.nan,
                response_bias=bias, bias_fraction_mse=bias**2/mse if mse > 0 else 0,
                response_correlation=np.corrcoef(y, p)[0, 1] if np.std(p) > 0 else np.nan,
                predicted_to_true_sd=np.std(p)/np.std(y) if variance > 0 else np.nan,
                strain_x_relative_mae=g.strain_x_error.mean()/g.true_strain_x.abs().mean(),
                strain_y_relative_mae=g.strain_y_error.mean()/g.true_strain_y.abs().mean()))
    data = pd.DataFrame(rows)
    data.to_csv(BASE / 'prediction_diagnostics.csv', index=False)
    measures = [c for c in data if c not in ('variant', 'seed', 'source', 'frame', 'valid', 'total')]
    aggregate = data.groupby(['variant', 'source', 'frame'])[measures].mean().reset_index()
    counts = data.groupby(['variant', 'source', 'frame']).agg(seeds=('seed', 'nunique'),
        valid_min=('valid', 'min'), total_min=('total', 'min')).reset_index()
    aggregate = aggregate.merge(counts, on=['variant', 'source', 'frame'])
    aggregate.to_csv(BASE / 'prediction_diagnostics_aggregate.csv', index=False)
    (BASE / 'prediction_diagnostics_provenance.json').write_text(json.dumps(dict(
        collected_utc=datetime.now(timezone.utc).isoformat(), inputs=inputs,
        interpretation='Descriptive validation diagnostics, no fitted calibration. Ratios use mean absolute strain, not per-network division. Seed means may be incomplete and are not a ranking.',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2))
    print(aggregate.query("source == 'lj_noisy' and frame == 100").to_string(index=False))


if __name__ == '__main__':
    main()
