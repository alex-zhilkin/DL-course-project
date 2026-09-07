"""Describe validation initial coordinates; no response-fitted coordinate selection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'notebooks/results/latent_matched_study'


def main():
    manifest = json.loads((BASE / 'split_manifest.json').read_text())
    reports = []
    for run in sorted(BASE.glob('shared*_d*_s*')):
        if not (run / 'completed.json').exists():
            continue
        recipe = json.loads((run / 'recipe.json').read_text())
        coordinates = pd.read_csv(run / 'latent_coordinates.csv')
        initial = coordinates.query("split == 'val' and frame == 0").copy()
        for source, group in initial.groupby('source', sort=False):
            group = group.sort_values('sim_idx')
            indices = manifest['sources'][source]['split_indices']['val']
            assert len(group) == len(indices)
            initial.loc[group.index, 'original_index'] = indices
        response = pd.read_csv(run / 'validation_rows.csv').query("frame == 100 and control == 'ae'")
        initial = initial.merge(response[['source', 'original_index', 'true_p_ratio']],
                                on=['source', 'original_index'], validate='one_to_one')
        zcols = [f'z{i}' for i in range(recipe['config']['ae_config']['latent_dim'])]
        pca = np.load(run / 'training_pca.npz')
        projected = (initial[zcols].to_numpy() - pca['center']) @ pca['components'].T
        for i in range(projected.shape[1]):
            initial[f'pc{i+1}'] = projected[:, i]
        for source, group in initial.groupby('source'):
            for coordinate in zcols + [f'pc{i+1}' for i in range(projected.shape[1])]:
                reports.append(dict(model=recipe['source']['source_name'],
                    dimension=len(zcols), seed=recipe['config']['model_seed'],
                    source=source, coordinate=coordinate, n=len(group),
                    pearson=group[coordinate].corr(group.true_p_ratio),
                    spearman=group[coordinate].rank().corr(group.true_p_ratio.rank()),
                    coordinate_std=group[coordinate].std()))
    report = pd.DataFrame(reports)
    report.to_csv(BASE / 'initial_coordinate_response_correlations.csv', index=False)
    print(report.query('dimension == 2').to_string(index=False))


if __name__ == '__main__':
    main()
