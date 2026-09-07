"""Persist all AE outcomes; rank retained-source response without mixed-T selection."""
from pathlib import Path
import pandas as pd

BASE=Path(__file__).resolve().parents[1]/'notebooks/results/lj_ae_capacity'
VARIANTS=('dimension6','dimension8')


def main():
    status=[];parts=[]
    for variant in VARIANTS:
        for seed in (3456456,123,456,786,2026):
            run=BASE/f'{variant}_s{seed}';complete=(run/'completed.json').exists()
            status.append(dict(variant=variant,seed=seed,completed=complete))
            if complete:parts.append(pd.read_csv(run/'source_summary.csv').assign(variant=variant,seed=seed))
    pd.DataFrame(status).to_csv(BASE/'status.csv',index=False)
    print('Completed',sum(x['completed'] for x in status),'/',len(status))
    if not parts:return
    frame=pd.concat(parts,ignore_index=True);frame.to_csv(BASE/'sourcewise_results.csv',index=False)
    aggregate=frame.groupby(['variant','source','frame']).agg(seeds=('seed','nunique'),
        r2_mean=('p_ratio_r2','mean'),r2_std=('p_ratio_r2','std'),r2_min=('p_ratio_r2','min'),
        position_mse=('position_mse','mean'),strain_x_mae=('strain_x_error','mean'),strain_y_mae=('strain_y_error','mean'),
        p_ratio_mae=('p_ratio_mae','mean'),valid_min=('valid','min'),total_min=('total','min')).reset_index()
    aggregate.to_csv(BASE/'sourcewise_aggregate.csv',index=False)
    retained=aggregate[(aggregate.frame==100)&aggregate.source.isin(['reid','depablo_low_temp','lj_noisy'])]
    retained=retained[~retained.variant.str.contains('response')]
    ranking=retained.groupby('variant').agg(worst_source_position_mse=('position_mse','max'),sources=('source','nunique'),
        seeds_min=('seeds','min'),valid_min=('valid_min','min'),total_min=('total_min','min'))
    ranking=ranking[(ranking.sources==3)&(ranking.seeds_min==5)]
    ranking.sort_values('worst_source_position_mse').to_csv(BASE/'retained_source_ranking.csv')
    print(aggregate.query('frame==100').to_string(index=False))


if __name__=='__main__':main()
