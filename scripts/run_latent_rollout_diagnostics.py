"""Evaluate existing transfer models and kinematic controls on validation only."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(os.environ['LSS_PROJECT_ROOT'])
parser = argparse.ArgumentParser()
parser.add_argument('--variant', required=True, choices=('velocity_transfer3', 'delta_transfer3', 'acceleration_transfer3'))
parser.add_argument('--seed', type=int, required=True)
args = parser.parse_args()
run = ROOT / f'notebooks/results/latent_matched_rollouts/{args.variant}_shared4_s{args.seed}'
recipe = json.loads((run/'recipe.json').read_text())
sys.path.insert(0, str(Path(recipe['code_root'])/'src'))
import pandas as pd
import torch
from lss.latent.capacity import load_experiment_bundle
from lss.latent.training import (encode_frame_latent, decode_latent_to_graph, encode_reference_context,
    latent_step, latent_step_history, latent_step_fixed_history)
from latent_diagnostic_metrics import metrics, summarize


def main():
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '8')))
    output = ROOT / 'notebooks/results/latent_rollout_diagnostics' / run.name
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    result = load_experiment_bundle(run/'bundle.pt', cfg=recipe['config'], device='cpu')
    assert not result['test_data']
    ae, dyn, cfg, stats, norms = (result[k] for k in ('ae','dyn','params','latent_stats','normalizers'))
    ae.eval(); dyn.eval()
    assert cfg['frame_skip'] == 1 and cfg.get('propagator_step_stride', 1) == 1
    manifest = json.loads((ROOT/'notebooks/results/latent_matched_study/split_manifest.json').read_text())
    counts, rows = {}, []
    with torch.no_grad():
        for sim in result['val_data']:
            source = sim[0].source_name
            index = counts.get(source, 0); counts[source] = index+1
            original_index = manifest['sources'][source]['split_indices']['val'][index]
            zs = [encode_frame_latent(ae, sim, f, pos_dim=2, node_feature_mode=cfg['node_feature_mode'], normalizers=norms, device='cpu') for f in range(101)]
            context = encode_reference_context(ae, sim, pos_dim=2, normalizers=norms, device='cpu', pool_mode='mean')
            def step(z, previous, older):
                if args.variant == 'velocity_transfer3':
                    return latent_step_fixed_history(dyn, z, zs[1], zs[5], stats, observed_frame_gap=4, context=context)
                if args.variant == 'acceleration_transfer3':
                    return latent_step_history(dyn, z, previous, older, zs[0], stats, context=context)
                return latent_step(dyn, z, stats, loss_mode=cfg['propagator_loss'], context=context)
            older, previous, z = zs[3], zs[4], zs[5]
            for frame in range(6, 101):
                next_z = step(z, previous, older)
                tf = step(zs[frame-1], zs[frame-2], zs[frame-3])
                older, previous, z = previous, z, next_z
                predictions = {'free_rollout': z, 'teacher_forced': tf}
                if frame in (10,25,50,75,100):
                    n = frame-5
                    predictions.update(oracle_ae=zs[frame], constant_latent=zs[5],
                        constant_velocity=zs[5]+n*(zs[5]-zs[1])/4,
                        constant_velocity_recent=zs[5]+n*(zs[5]-zs[4]),
                        constant_acceleration=zs[5]+n*(zs[5]-zs[4])+n*(n+1)/2*(zs[5]-2*zs[4]+zs[3]))
                for mode, prediction in predictions.items():
                    row = dict(source=source, original_index=original_index, frame=frame, mode=mode,
                        latent_mse=float((prediction-zs[frame]).square().mean()),
                        normalized_latent_mse=float(((prediction-zs[frame])/stats.z_std.to(prediction).clamp_min(1e-6)).square().mean()))
                    if frame in (10,25,50,75,100):
                        pred = decode_latent_to_graph(ae, sim, prediction, frame, pos_dim=2, ae_target_mode=cfg['ae_target_mode'], normalizers=norms, device='cpu')
                        row.update(metrics(sim, frame, pred))
                    rows.append(row)
            pd.DataFrame(rows).to_csv(output/'validation_rows.csv', index=False)
            print(f'Completed {source} trajectory {original_index}', flush=True)
    frame = pd.DataFrame(rows)
    summarize(frame[frame.frame.isin((10,25,50,75,100))], ['source','frame','mode']).to_csv(output/'source_summary.csv', index=False)
    frame.groupby(['source','frame','mode'])[['latent_mse','normalized_latent_mse']].mean().to_csv(output/'latent_drift.csv')
    # Require independent free-rollout implementation to reproduce saved predictions.
    saved = pd.read_csv(run/'validation_rollout_rows.csv')
    actual = frame.query("mode == 'free_rollout' and frame in [10,25,50,75,100]")
    import numpy as np
    for source, group in saved.groupby('source'):
        for horizon, g in group.groupby('rollout_steps'):
            a = actual[(actual.source == source)&(actual.frame == horizon)]
            np.testing.assert_allclose(a.pred_p_ratio, g.pred_p_ratio, rtol=1e-4, atol=1e-5, equal_nan=True)
    (output/'completed.json').write_text(json.dumps({'seconds':time.time()-started, 'job_id':os.environ['PBS_JOBID'],
        'checkpoint':str(run/'bundle.pt'), 'recipe':recipe, 'saved_rollout_agreement':True,
        'teacher_forcing':'Diagnostic uses true history through t-1; not a causal forecast from frame 5.'}, indent=2))


if __name__ == '__main__': main()
