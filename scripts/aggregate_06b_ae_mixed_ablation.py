"""Compare mixed-T AE exposure with paired seeds and source-wise metrics."""
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]/'notebooks/results'
OUT=ROOT/'06b_ae_mixed_ablation'


def main():
    parts=[];status=[]
    for seed in (3456456,123,456,786,2026):
        for condition,run in [('include_mixed',ROOT/'06b_4d_reconstruction'/f'd2_s{seed}'),
                              ('exclude_mixed',OUT/f'exclude_mixed_d2_s{seed}')]:
            complete=(run/'completed.json').exists()
            status.append(dict(seed=seed,condition=condition,completed=complete))
            if not complete:continue
            for kind in ('ae','rollout'):
                parts.append(pd.read_csv(run/f'validation_{kind}_summary.csv').assign(seed=seed,condition=condition,kind=kind))
    pd.DataFrame(status).to_csv(OUT/'status.csv',index=False)
    print(pd.DataFrame(status).to_string(index=False))
    if not parts:return
    frame=pd.concat(parts,ignore_index=True)
    frame.to_csv(OUT/'sourcewise_results.csv',index=False)
    frame.groupby(['condition','kind','source','rollout_steps']).agg(seeds=('seed','nunique'),
        r2_mean=('p_ratio_r2','mean'),r2_std=('p_ratio_r2','std'),mae_mean=('p_ratio_mae','mean'),
        valid_min=('used','min'),total_min=('total','min')).to_csv(OUT/'sourcewise_aggregate.csv')
    keys=['seed','kind','source','rollout_steps']
    pairs=frame[frame.condition=='exclude_mixed'].merge(frame[frame.condition=='include_mixed'],on=keys,
        suffixes=('_excluded','_included'),validate='one_to_one')
    pairs['r2_excluded_minus_included']=pairs.p_ratio_r2_excluded-pairs.p_ratio_r2_included
    pairs.to_csv(OUT/'paired_results.csv',index=False)


if __name__=='__main__':main()
