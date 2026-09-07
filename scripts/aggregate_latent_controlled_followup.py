"""Collect completed controlled runs without mixing interrupted attempts."""
from pathlib import Path
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]/'notebooks/results'
OUT=ROOT/'latent_controlled_followup'


def main():
    status=[]; dynamics=[]; probes=[]; reconstruction=[]
    for scope in ('transfer3','all4'):
        for horizon in (1,8):
            for seed in (123,456,786):
                run=ROOT/'latent_stability'/f'acceleration_{scope}_matched_h{horizon}_shared4_s{seed}'
                complete=(run/'completed.json').exists()
                status.append(dict(task='stability',run=run.name,completed=complete))
                if complete:
                    dynamics.append(pd.read_csv(run/'validation_rollout_summary.csv').assign(scope=scope,horizon=horizon,seed=seed))
    for experiment in ('latent_low_data','latent_low_data_update_matched'):
        for scope in ('shared3','shared4'):
            for seed in (123,456,786):
                run=ROOT/experiment/f'{scope}_d2_s{seed}'
                complete=(run/'completed.json').exists()
                if experiment.endswith('update_matched'):status.append(dict(task='lowdata',run=run.name,completed=complete))
                if not complete:continue
                reconstruction.append(pd.read_csv(run/'validation_summary.csv').query("control=='ae'").assign(experiment=experiment,scope=scope,seed=seed))
                for budget in ('one_trajectory','thirty_trajectories'):
                    probes.append(pd.read_csv(run/budget/'probe_summary.csv').assign(experiment=experiment,scope=scope,seed=seed,budget=budget))
    pd.DataFrame(status).to_csv(OUT/'status.csv',index=False)
    for name,parts,keys,metric in [
        ('stability',dynamics,['scope','horizon','source','rollout_steps'],'p_ratio_r2'),
        ('reconstruction',reconstruction,['experiment','scope','source','frame'],'legacy_p_ratio_r2'),
        ('probes',probes,['experiment','scope','budget','source','probe'],'p_ratio_r2')]:
        frame=pd.concat(parts,ignore_index=True)
        frame.to_csv(OUT/f'{name}_sourcewise.csv',index=False)
        frame.groupby(keys).agg(seeds=('seed','nunique'),r2_mean=(metric,'mean'),r2_std=(metric,'std')).to_csv(OUT/f'{name}_aggregate.csv')
    print(pd.DataFrame(status).to_string(index=False))
    d=pd.read_csv(OUT/'stability_aggregate.csv').query('rollout_steps==100')
    print(d.pivot(index=['scope','horizon'],columns='source',values='r2_mean').round(3).to_string())
    r=pd.read_csv(OUT/'reconstruction_aggregate.csv').query('frame==100')
    print(r.pivot(index=['experiment','scope'],columns='source',values='r2_mean').round(3).to_string())


if __name__=='__main__':main()
