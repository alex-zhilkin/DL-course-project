"""Matched 1-versus-30-trajectory AE comparison with training-calibrated probes."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(os.environ['LSS_PROJECT_ROOT'])
CODE = ROOT/'notebooks/results/latent_matched_study/code_v1'
sys.path.insert(0, str(CODE/'src'))
import numpy as np
import pandas as pd
import torch
from lss.latent.experiment import run_latent_experiment, seed_everything, _load_ae_cache
from lss.latent.training import encode_frame_latent
from graph_utils import calc_p_ratio_rollout_sides
from run_latent_matched_screen import evaluate
from latent_diagnostic_metrics import summarize


def probes(model, full, output):
    ae, cfg, norms = (model[k] for k in ('ae','params','normalizers'))
    ae.eval()
    data = []
    manifest = json.loads((ROOT/'notebooks/results/latent_matched_study/split_manifest.json').read_text())
    with torch.no_grad():
        for split, sims in [('calibration', full['train_data']), ('validation', full['val_data'])]:
            counts = {}
            for sim in sims:
                source = sim[0].source_name
                index = counts.get(source, 0); counts[source] = index+1
                # Same 29 labeled networks/source for both AE budgets; exclude AE's one training network.
                if split == 'calibration' and index == 0: continue
                zs = [encode_frame_latent(ae, sim, f, pos_dim=2, node_feature_mode=cfg['node_feature_mode'], normalizers=norms, device='cpu').cpu().numpy().reshape(-1) for f in (0,1,5)]
                original_index = manifest['sources'][source]['split_indices']['train' if split == 'calibration' else 'val'][index]
                data.append(dict(source=source, split=split, index=index, original_index=original_index,
                    target=float(calc_p_ratio_rollout_sides(sim,100)),
                    **{f'z{f}_{j}':float(v) for f,z in zip((0,1,5),zs) for j,v in enumerate(z)}))
    frame = pd.DataFrame(data)
    frame.to_csv(output/'probe_coordinates.csv', index=False)
    train_z = frame[frame.split=='calibration'][['z0_0','z0_1']].to_numpy()
    center = train_z.mean(0)
    _, _, axes = np.linalg.svd(train_z-center, full_matrices=False)
    frame['pc1'] = (frame[['z0_0','z0_1']].to_numpy()-center) @ axes[0]
    for j in (0,1): frame[f'v{j}'] = (frame[f'z5_{j}']-frame[f'z1_{j}'])/4
    definitions = {'constant':[], 'initial_pc1':['pc1'], 'initial_z':['z0_0','z0_1'],
                   'observed_prefix':['z0_0','z0_1','z5_0','z5_1','v0','v1']}
    predictions, parameters, correlations = [], [], []
    for source,g in frame.groupby('source'):
        train = g[(g.split=='calibration') & np.isfinite(g.target)]
        val = g[g.split=='validation']
        for name,cols in definitions.items():
            x = train[cols].to_numpy(); y=train.target.to_numpy()
            mean=x.mean(0); scale=x.std(0); scale[scale<1e-8]=1
            x=(x-mean)/scale
            # Fixed ridge strength; no response-label tuning on validation.
            coef=np.linalg.solve(x.T@x+np.eye(len(cols)), x.T@(y-y.mean()))
            pred=y.mean()+((val[cols].to_numpy()-mean)/scale)@coef
            parameters.append(dict(source=source, probe=name, columns=cols, mean=mean.tolist(),
                scale=scale.tolist(), coefficients=coef.tolist(), intercept=float(y.mean()), calibration_n=len(train)))
            for (_,row),value in zip(val.iterrows(),pred):
                predictions.append(dict(source=source,probe=name,index=row['index'],original_index=row['original_index'],true_p_ratio=row.target,pred_p_ratio=value))
        for col in ('z0_0','z0_1','pc1'):
            correlations.append(dict(source=source,coordinate=col,n=len(val),
                pearson=val[col].corr(val.target),spearman=val[col].rank().corr(val.target.rank())))
    pd.DataFrame(correlations).to_csv(output/'initial_correlations.csv',index=False)
    pd.DataFrame(predictions).to_csv(output/'probe_predictions.csv',index=False)
    summarize(pd.DataFrame(predictions), ['source','probe']).to_csv(output/'probe_summary.csv',index=False)
    (output/'probe_fit.json').write_text(json.dumps(dict(ridge_alpha=1,parameters=parameters,
        pca_center=center.tolist(),pca_axes=axes.tolist(),calibration_budget_per_source=29,
        evaluation='20 validation networks/source; development comparison, not final reserved test'),indent=2))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--scope',choices=('shared3','shared4'),required=True)
    parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args()
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS','8')))
    original=ROOT/f'notebooks/results/latent_matched_study/{args.scope}_d2_s{args.seed}'
    recipe=json.loads((original/'recipe.json').read_text())
    output=ROOT/f'notebooks/results/latent_low_data_update_matched/{args.scope}_d2_s{args.seed}'
    output.mkdir(parents=True,exist_ok=False)
    source,cfg=copy.deepcopy(recipe['source']),copy.deepcopy(recipe['config'])
    for spec in source['dataset_mixture']:
        spec['train_count']=1
        spec['split_indices']['train']=spec['split_indices']['train'][:1]
        spec['split_indices']['test']=[]
    cfg['ae_config']['train_rows_per_source']=3030
    cfg['cache_path']=str(output/'ae.pt')
    (output/'recipe.json').write_text(json.dumps(dict(source=source,config=cfg,matched_recipe=str(original/'recipe.json'),
        design='Same 3030 sampled rows/source/epoch as the 30-trajectory AE, repeating 101 unique frames from one trajectory. Same 40-epoch maximum and patience 8; actual stopping epochs may differ. Frames 0–100, not an exact historical reproduction.',
        code_root=str(CODE),job_id=os.environ['PBS_JOBID']),indent=2))
    seed_everything(args.seed)
    started=time.time()
    low=run_latent_experiment(source,cfg,device='cpu')
    assert not low['test_data']
    low['ae_history'].to_csv(output/'ae_history.csv',index=False)
    manifest=json.loads((ROOT/'notebooks/results/latent_matched_study/split_manifest.json').read_text())
    evaluate(low,output,manifest)
    full=_load_ae_cache(original/'ae.pt',recipe['config'],device='cpu')
    assert not full['test_data']
    for name,model in [('one_trajectory',low),('thirty_trajectories',full)]:
        destination=output/name; destination.mkdir()
        probes(model,full,destination)
    (output/'completed.json').write_text(json.dumps(dict(seconds=time.time()-started,job_id=os.environ['PBS_JOBID'],split='validation')))


if __name__=='__main__': main()
