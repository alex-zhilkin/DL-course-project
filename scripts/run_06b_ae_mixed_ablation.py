"""Remove mixed-T AE exposure from the successful 2D reconstruction recipe."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(os.environ['LSS_PROJECT_ROOT'])
CODE=Path(os.environ['LSS_CODE_ROOT'])
sys.path.insert(0,str(CODE/'src'))
import pandas as pd
import torch
from lss.latent.experiment import (run_latent_experiment,seed_everything,resolve_train_val_test,
    evaluate_rollout_horizons,evaluate_autoencoder_reconstruction_horizons)
from run_latent_matched_rollout import source_summary


def make_recipe(baseline, output):
    source=copy.deepcopy(baseline['source']); cfg=copy.deepcopy(baseline['config'])
    mixed=next(s for s in source['dataset_mixture'] if s['name']=='depablo_mixed_temp')
    source['dataset_mixture']=[s for s in source['dataset_mixture'] if s['name']!='depablo_mixed_temp']
    source['source_name']='AE_reid_lowT_LJ_prop_reid_lowT';source['label']='2D AE mixed-T exclusion'
    cfg['cache_path']=str(output/'bundle.pt')
    assert cfg['ae_config']['latent_dim']==2
    assert set(cfg['propagator_config']['train_trajectories_per_source'])=={'reid','depablo_low_temp'}
    mixed=copy.deepcopy(mixed);mixed['train_count']=0;mixed['split_indices']['train']=[];mixed['split_indices']['test']=[]
    mixed_source=dict(dataset_name=source['dataset_name'],source_name='depablo_mixed_temp',
        label='mixed-T evaluation only',path=mixed['path'],dataset_mixture=[mixed])
    return source,cfg,mixed_source


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args();torch.set_num_threads(8)
    baseline_path=ROOT/f'notebooks/results/06b_4d_reconstruction/d2_s{args.seed}/recipe.json'
    baseline=json.loads(baseline_path.read_text())
    output=ROOT/f'notebooks/results/06b_ae_mixed_ablation/exclude_mixed_d2_s{args.seed}'
    source,cfg,mixed_source=make_recipe(baseline,output)
    manifest=json.loads((ROOT/'notebooks/results/latent_matched_study/split_manifest.json').read_text())
    for spec in [*source['dataset_mixture'],*mixed_source['dataset_mixture']]:
        digest=hashlib.sha256()
        with Path(spec['path']).open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
        if digest.hexdigest()!=manifest['sources'][spec['name']]['sha256']:raise ValueError('Dataset changed')
    output.mkdir(parents=True,exist_ok=False)
    (output/'recipe.json').write_text(json.dumps(dict(source=source,config=cfg,mixed_evaluation=mixed_source,
        baseline_recipe=str(baseline_path),baseline_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        code_root=str(CODE),job_id=os.environ['PBS_JOBID'],
        design='Same 2D architecture, hyperparameters, retained-source split IDs and five paired seeds; remove mixed-T from all AE fitting/statistics/selection. Mixed-T loaded only after AE and propagator selection finish.',
        qualification='Source removal reduces examples/updates per epoch and changes AE-fitted normalizers and RNG consumption. Same seed is not a guarantee of identical propagator initialization. This tests the exclusion recipe, not mathematical necessity.'),indent=2))
    seed_everything(args.seed);started=time.time()
    result=run_latent_experiment(source,cfg,device='cpu')
    assert not result['test_data']
    assert all(s[0].source_name!='depablo_mixed_temp' for split in ('train_data','val_data') for s in result[split])
    # The excluded source enters only now: neither weights nor checkpoint choices can see it.
    train,mixed,test,_=resolve_train_val_test(mixed_source,result['params'],split_seed=cfg['split_seed'])
    assert not train and not test and len(mixed)==20
    kwargs=dict(cfg=result['params'],normalizers=result['normalizers'],dataset=result['label'],
        split_name='val',rollout_steps=cfg['rollout_steps_grid'],device='cpu')
    rollout,_=evaluate_rollout_horizons(result['ae'],result['dyn'],mixed,result['latent_stats'],**kwargs)
    ae,_=evaluate_autoencoder_reconstruction_horizons(result['ae'],mixed,**kwargs)
    result['rollout_rows']=pd.concat([result['rollout_rows'],rollout],ignore_index=True)
    result['ae_reconstruction_rows']=pd.concat([result['ae_reconstruction_rows'],ae],ignore_index=True)
    for key,name in [('ae_history','ae_history'),('dyn_history','dynamics_history'),('rollout_rows','validation_rollout_rows'),('ae_reconstruction_rows','validation_ae_rows')]:
        result[key].to_csv(output/f'{name}.csv',index=False)
    for key,name in [('rollout_rows','validation_rollout_summary'),('ae_reconstruction_rows','validation_ae_summary')]:
        summary=source_summary(result[key]);summary.to_csv(output/f'{name}.csv',index=False)
        print(name,summary.query('rollout_steps==100').to_string(index=False),flush=True)
    (output/'completed.json').write_text(json.dumps(dict(seconds=time.time()-started,job_id=os.environ['PBS_JOBID'],
        split='validation',latent_dim=2,mixed_loaded_after_training=True)))


if __name__=='__main__':main()
