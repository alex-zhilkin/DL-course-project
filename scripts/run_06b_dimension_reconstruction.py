"""Matched 2D/4D comparison of the reconstructed 06b recipe."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(os.environ['LSS_PROJECT_ROOT'])
CODE=Path(os.environ['LSS_CODE_ROOT'])
sys.path.insert(0,str(CODE/'src'))
import torch
from lss.latent.experiment import run_latent_experiment,seed_everything
from run_latent_matched_rollout import source_summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--dim',type=int,choices=(2,4),required=True)
    args=parser.parse_args()
    torch.set_num_threads(8)
    output=ROOT/f'notebooks/results/06b_4d_reconstruction/d{args.dim}_s{args.seed}'
    output.mkdir(parents=True,exist_ok=False)
    manifest_path=ROOT/'notebooks/results/latent_matched_study/split_manifest.json'
    manifest=json.loads(manifest_path.read_text())
    mixture=[]
    for name,spec in manifest['sources'].items():
        digest=hashlib.sha256()
        with Path(spec['path']).open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
        if digest.hexdigest()!=spec['sha256']:raise ValueError(f'Dataset changed: {name}')
        mixture.append(dict(name=name,label=name,path=spec['path'],train_count=20,val_count=20,
            split_indices=dict(train=spec['split_indices']['train'][:20],val=spec['split_indices']['val'],test=[])))
    trained=['reid','depablo_low_temp']
    ae=dict(model='attention',latent_dim=args.dim,latent_tokens=32,hidden_size=96,
        edge_feature_dim=4,target_mode='normalized_delta',node_feature_mode='normalized_delta',
        max_train_frames_per_sim=101,max_val_frames_per_sim=101,val_frame_skip=1,
        max_epochs=14,patience=3,lr=1e-4,weight_decay=1e-5,mix_sources=True,balance_sources=False,
        pratio_eval_every=1,pratio_eval_step=100)
    prop=dict(model='delta_mlp',objective='one_step',loss='delta',hidden_size=64,
        max_train_transitions_per_sim=100,max_epochs=6,patience=3,lr=1e-4,weight_decay=1e-5,
        step_stride=1,mix_sources=True,balance_sources=False,source_loss_reduction='equal',
        train_trajectories_per_source={s:20 for s in trained},val_trajectories_per_source={s:20 for s in trained},
        use_static_context=True,context_pool='mean',context_dim=16,
        rollout_eval_every_epoch=True,rollout_eval_interval=1,rollout_eval_horizons=[100],
        rollout_eval_sims_per_source=20,rollout_eval_sources=trained,
        checkpoint_metric='val_rollout_macro_source_endpoint_p_ratio_r2',checkpoint_mode='max')
    cfg=dict(ae_config=ae,propagator_config=prop,model_seed=args.seed,split_seed=123,
        dataset_name='06b_4d_reconstruction',pos_dim=2,batch_graphs=32,frame_skip=1,
        edge_mode='compact_stored',coordinate_normalization='position_normalization',
        static_context_use_physical_reference=True,train_frame_start_order=0,
        should_rollout=True,should_train_propagator=True,force_train=True,
        cache_path=str(output/'bundle.pt'),early_stop_min_delta=1e-5,
        rollout_steps_grid=[5,10,25,50,75,100],rollout_eval_splits=['val'],
        rollout_final_eval_sims_per_source=20,p_ratio_estimator='endpoint')
    source=dict(dataset_name=cfg['dataset_name'],source_name='AE_all4_prop_reid_lowT',
        label='06b 4D reconstruction',path=mixture[0]['path'],dataset_mixture=mixture)
    recipe=dict(source=source,config=cfg,job_id=os.environ['PBS_JOBID'],code_root=str(CODE),
        reproduction_status='Reconstruction, not exact historical replay',
        dimension_comparison='2D vs 4D; all other settings and split IDs held fixed',
        documented=['original model seed 3456456 (other seeds are replications)','4D latent','AE all four sources',
            '20 train/20 validation per source','frames 0–100','width-64 delta MLP','16D mean physical context',
            'propagator supervision Reid and low-T only','AE maximum 14 epochs; propagator maximum 6'],
        unresolved=['original checkpoint and exact historical code unavailable','original AE width not preserved in log; 96 from older 06b configuration',
            'patience/lr/weight decay/batch size inferred from surviving notebook lineage',
            'original forecast initialization and p-ratio estimator not fully preserved; current origin 0, endpoint estimator',
            'current corrected per-axis/evolving-box geometry and loader; historical geometry version unknown'],
        deliberate_difference='Use first 20 of current matched training IDs and 20 matched validation IDs; preserve reserved final-test partition. Joint fresh AE+propagator call, no intermediate reseed.')
    (output/'recipe.json').write_text(json.dumps(recipe,indent=2))
    seed_everything(args.seed);started=time.time()
    result=run_latent_experiment(source,cfg,device='cpu')
    assert not result['test_data']
    for key,name in [('ae_history','ae_history'),('dyn_history','dynamics_history'),('rollout_rows','validation_rollout_rows'),('ae_reconstruction_rows','validation_ae_rows')]:
        result[key].to_csv(output/f'{name}.csv',index=False)
    for key,name in [('rollout_rows','validation_rollout_summary'),('ae_reconstruction_rows','validation_ae_summary')]:
        summary=source_summary(result[key]);summary.to_csv(output/f'{name}.csv',index=False)
        print(name,summary.query('rollout_steps==100').to_string(index=False),flush=True)
    (output/'completed.json').write_text(json.dumps(dict(seconds=time.time()-started,job_id=os.environ['PBS_JOBID'],split='validation',latent_dim=args.dim)))


if __name__=='__main__':main()
