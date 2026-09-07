"""Collect source-wise reconstruction results and paired dimensionality contrasts."""
import json
from pathlib import Path
import pandas as pd

BASE=Path(__file__).resolve().parents[1]/'notebooks/results/06b_4d_reconstruction'


def main():
    status=[];parts=[]
    for dim in (2,4):
        for seed in (3456456,123,456,786,2026):
            run=BASE/f'd{dim}_s{seed}'
            complete=(run/'completed.json').exists()
            status.append(dict(dimension=dim,seed=seed,completed=complete))
            if not complete:continue
            for kind in ('ae','rollout'):
                parts.append(pd.read_csv(run/f'validation_{kind}_summary.csv').assign(kind=kind,dimension=dim,seed=seed))
    pd.DataFrame(status).to_csv(BASE/'status.csv',index=False)
    print(pd.DataFrame(status).to_string(index=False))
    if not parts:return
    frame=pd.concat(parts,ignore_index=True)
    frame.to_csv(BASE/'sourcewise_results.csv',index=False)
    frame.groupby(['kind','dimension','source','rollout_steps']).agg(
        seeds=('seed','nunique'),r2_mean=('p_ratio_r2','mean'),r2_std=('p_ratio_r2','std'),
        mae_mean=('p_ratio_mae','mean'),valid_min=('used','min'),total_min=('total','min')).to_csv(BASE/'sourcewise_aggregate.csv')
    keys=['kind','source','rollout_steps','seed']
    paired=frame[frame.dimension==2].merge(frame[frame.dimension==4],on=keys,suffixes=('_2d','_4d'),validate='one_to_one')
    paired['r2_2d_minus_4d']=paired.p_ratio_r2_2d-paired.p_ratio_r2_4d
    paired.to_csv(BASE/'paired_dimension_results.csv',index=False)


if __name__=='__main__':main()
