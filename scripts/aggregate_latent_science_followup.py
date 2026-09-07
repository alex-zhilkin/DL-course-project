"""Aggregate only completed, verified science follow-ups by source and seed."""
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]/'notebooks/results'
OUT=ROOT/'latent_science_followup'


def main():
    diagnostics=[]; probes=[]; status=[]
    for seed in (123,456,786):
        for variant in ('velocity_transfer3','delta_transfer3','acceleration_transfer3'):
            run=ROOT/'latent_rollout_diagnostics'/f'{variant}_shared4_s{seed}'
            complete=(run/'completed.json').exists()
            status.append(dict(task='diagnostics',run=run.name,complete=complete))
            if complete:
                diagnostics.append(pd.read_csv(run/'source_summary.csv').assign(variant=variant,seed=seed))
        for scope in ('shared3','shared4'):
            run=ROOT/'latent_low_data'/f'{scope}_d2_s{seed}'
            complete=(run/'completed.json').exists()
            status.append(dict(task='lowdata',run=run.name,complete=complete))
            if complete:
                for budget in ('one_trajectory','thirty_trajectories'):
                    probes.append(pd.read_csv(run/budget/'probe_summary.csv').assign(scope=scope,seed=seed,budget=budget))
    pd.DataFrame(status).to_csv(OUT/'status.csv',index=False)
    for name,parts,keys in [('diagnostics',diagnostics,['variant','source','frame','mode']),
                             ('probes',probes,['scope','budget','source','probe'])]:
        if not parts: continue
        frame=pd.concat(parts,ignore_index=True)
        frame.to_csv(OUT/f'{name}_sourcewise.csv',index=False)
        frame.groupby(keys).agg(seeds=('seed','nunique'),r2_mean=('p_ratio_r2','mean'),
            r2_std=('p_ratio_r2','std'),mae_mean=('p_ratio_mae','mean'),
            valid_min=('valid','min'),total_min=('total','min')).to_csv(OUT/f'{name}_aggregate.csv')
    print(pd.DataFrame(status).to_string(index=False))


if __name__=='__main__': main()
