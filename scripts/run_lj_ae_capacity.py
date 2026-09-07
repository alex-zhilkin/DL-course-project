"""AE-only dynamics-data latent-capacity study; mixed-T is evaluation-only."""
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
from lss.latent.experiment import run_latent_experiment,seed_everything,resolve_train_val_test
from lss.latent.training import encode_frame_latent,decode_latent_to_graph
from latent_diagnostic_metrics import metrics,summarize

VARIANTS=('longer','equal_source','response_selection','wider','dimension4',
          'dimension4_response','lj_edges','lj_edges_response','dimension6','dimension8')


def build_recipe(baseline,variant,output):
    source=copy.deepcopy(baseline['source']);cfg=copy.deepcopy(baseline['config'])
    mixed=copy.deepcopy(baseline['mixed_evaluation'])
    cfg.update(should_train_propagator=False,should_rollout=False,cache_path=str(output/'ae.pt'))
    cfg.pop('propagator_config')
    ae=cfg['ae_config']
    ae.update(max_epochs=40,patience=8)
    if variant!='longer':
        ae.update(gradient_method='source_mean',checkpoint_metric='val_max_source_reconstruction',checkpoint_mode='min')
    if variant in ('response_selection','dimension4_response','lj_edges_response'):
        ae.update(checkpoint_metric='val_ae_min_source_p_ratio_r2_sum_100',checkpoint_mode='max')
    ae['pratio_eval_every']=0
    if variant in ('dimension6','dimension8'):ae['latent_dim']=int(variant[-1])
    if variant=='wider':ae['hidden_size']=128
    if variant in ('dimension4','dimension4_response'):ae['latent_dim']=4
    if variant in ('lj_edges','lj_edges_response'):
        ae['edge_feature_dim']=5
        for spec in [*source['dataset_mixture'],*mixed['dataset_mixture']]:
            spec.update(append_lj_indicator=True,lj_max_graph_distance=3 if spec['name']=='lj_noisy' else None)
    return source,cfg,mixed


def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=VARIANTS,required=True);p.add_argument('--seed',type=int,required=True)
    args=p.parse_args();torch.set_num_threads(8)
    baseline_path=ROOT/f'notebooks/results/06b_ae_mixed_ablation/exclude_mixed_d2_s{args.seed}/recipe.json'
    baseline=json.loads(baseline_path.read_text())
    output=ROOT/f'notebooks/results/lj_ae_capacity/{args.variant}_s{args.seed}'
    source,cfg,mixed_source=build_recipe(baseline,args.variant,output)
    assert {s['name'] for s in source['dataset_mixture']}=={'reid','depablo_low_temp','lj_noisy'}
    manifest=json.loads((ROOT/'notebooks/results/latent_matched_study/split_manifest.json').read_text())
    for spec in [*source['dataset_mixture'],*mixed_source['dataset_mixture']]:
        digest=hashlib.sha256()
        with Path(spec['path']).open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
        if digest.hexdigest()!=manifest['sources'][spec['name']]['sha256']:raise ValueError('Dataset changed')
    output.mkdir(parents=True,exist_ok=False)
    (output/'recipe.json').write_text(json.dumps(dict(source=source,config=cfg,mixed_evaluation=mixed_source,
        variant=args.variant,seed=args.seed,job_id=os.environ['PBS_JOBID'],code_root=str(CODE),
        baseline_recipe=str(baseline_path),baseline_sha256=hashlib.sha256(baseline_path.read_bytes()).hexdigest(),
        response_selection='response' in args.variant,
        design='AE only; mixed-T unavailable until checkpoint selection finishes. Response-selection variants use retained-source validation p-ratio labels; gradient objective remains displacement reconstruction.',
        contrasts={'dimension6':'Versus equal_source: latent2->6; response callback disabled',
            'dimension8':'Versus equal_source: latent2->8; response callback disabled',
            'longer':'Original node-pooled objective/field selection; increase cap14->40 and patience3->8',
            'equal_source':'Versus longer: equal source/graph loss and worst-source reconstruction selection',
            'response_selection':'Versus equal_source: worst-source endpoint response selection',
            'wider':'Versus equal_source: hidden96->128',
            'dimension4':'Versus equal_source: latent2->4',
            'dimension4_response':'Versus dimension4: response selection',
            'lj_edges':'Versus equal_source: LJ indicator and graph-distance-three LJ edges',
            'lj_edges_response':'Versus lj_edges: response selection'}[args.variant]),indent=2))
    seed_everything(args.seed);started=time.time()
    result=run_latent_experiment(source,cfg,device='cpu')
    assert not result['test_data']
    assert all(s[0].source_name!='depablo_mixed_temp' for k in ('train_data','val_data') for s in result[k])
    result['ae_history'].to_csv(output/'ae_history.csv',index=False)
    train,mixed,test,_=resolve_train_val_test(mixed_source,result['params'],split_seed=cfg['split_seed'])
    assert not train and not test and len(mixed)==20
    ae=result['ae'];ae.eval();params=result['params'];norms=result['normalizers']
    counts={};rows=[]
    with torch.no_grad():
        for sim in [*result['val_data'],*mixed]:
            name=sim[0].source_name;i=counts.get(name,0);counts[name]=i+1
            identity=manifest['sources'][name]['split_indices']['val'][i]
            for frame in (5,10,25,50,75,100):
                z=encode_frame_latent(ae,sim,frame,pos_dim=2,node_feature_mode=params['node_feature_mode'],normalizers=norms,device='cpu')
                pred=decode_latent_to_graph(ae,sim,z,frame,pos_dim=2,ae_target_mode=params['ae_target_mode'],normalizers=norms,device='cpu')
                rows.append(dict(source=name,original_index=identity,frame=frame,**metrics(sim,frame,pred)))
            print('evaluated',name,identity,flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(output/'validation_rows.csv',index=False)
    summary=summarize(frame,['source','frame']);summary.to_csv(output/'source_summary.csv',index=False)
    print(summary.query('frame==100').to_string(index=False),flush=True)
    (output/'completed.json').write_text(json.dumps(dict(seconds=time.time()-started,seed=args.seed,variant=args.variant,
        job_id=os.environ['PBS_JOBID'],split='validation',mixed_evaluation_only=True)))


if __name__=='__main__':main()
