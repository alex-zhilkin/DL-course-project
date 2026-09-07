"""Compare full AE epoch wall time in fresh processes on one reserved node."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'notebooks/results/latent_full_ae_benchmark'
RECIPE = ROOT / 'notebooks/results/latent_matched_study/shared_no_mixed_d8_s123/recipe.json'
COUNTS = tuple(int(value) for value in os.environ.get('LSS_BENCH_COUNTS', '1:2:4:8:16:32:64').split(':'))


def worker(output, threads):
    recipe = json.loads((output.parent / 'source_recipe.json').read_text())
    sys.path.insert(0, str(output.parent / 'code/src'))
    import torch
    import lss.latent.training as training
    from lss.latent.experiment import run_latent_experiment, seed_everything
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    output.mkdir()
    cfg = copy.deepcopy(recipe['config'])
    cfg['cache_path'] = str(output / 'ae.pt')
    cfg['force_train'] = True
    cfg['ae_config'].update(max_epochs=3, patience=3)
    timings = []
    original = training.epoch_autoencoder

    def timed(*args, **kwargs):
        started = time.perf_counter()
        result = original(*args, **kwargs)
        row = {'phase': 'train' if kwargs.get('optimizer') is not None else 'validation',
               'seconds': time.perf_counter() - started}
        timings.append(row)
        with (output / 'epoch_timings.jsonl').open('a') as stream:
            stream.write(json.dumps(row) + '\n')
        return result

    training.epoch_autoencoder = timed
    (output / 'recipe.json').write_text(json.dumps({'source': recipe['source'], 'config': cfg}, indent=2))
    seed_everything(cfg['model_seed'])
    started = time.perf_counter()
    result = run_latent_experiment(recipe['source'], cfg, device='cpu')
    elapsed = time.perf_counter() - started
    result['ae_history'].to_csv(output / 'ae_history.csv', index=False)
    train = [r['seconds'] for r in timings if r['phase'] == 'train']
    val = [r['seconds'] for r in timings if r['phase'] == 'validation']
    if len(train) != 3 or len(val) != 3:
        raise RuntimeError(f'Expected three complete epochs, got {timings}')
    record = {'threads': threads, 'host': platform.node(), 'torch': torch.__version__,
              'affinity': sorted(os.sched_getaffinity(0)), 'job_id': os.environ.get('PBS_JOBID'),
              'total_seconds': elapsed, 'train_seconds': train, 'validation_seconds': val,
              'steady_epoch_seconds': [a+b for a,b in zip(train[1:], val[1:])],
              'cpu_model': next((x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name')), 'unknown')}
    (output / 'completed.json').write_text(json.dumps(record, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--threads', type=int)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.threads)
        return
    output = BASE / os.environ['PBS_JOBID']
    output.mkdir(parents=True, exist_ok=False)
    # Freeze the exact production code and recipe before any measured runs.
    import shutil
    frozen = BASE / '4669053.zeus-master'
    recipe_path = frozen / 'source_recipe.json'
    shutil.copytree(frozen / 'code/src', output / 'code/src')
    shutil.copy2(__file__, output / 'benchmark.py')
    shutil.copy2(recipe_path, output / 'source_recipe.json')
    hashes = {str(p.relative_to(output / 'code')): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted((output / 'code').rglob('*.py'))}
    (output / 'code_sha256.json').write_text(json.dumps(hashes, indent=2))
    schedule = [(repeat, threads) for repeat in (1, 2) for threads in COUNTS]
    random.Random(123).shuffle(schedule)
    (output / 'schedule.json').write_text(json.dumps(schedule))
    records = []
    for repeat, threads in schedule:
        run = output / f't{threads}_r{repeat}'
        env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads),
                   OPENBLAS_NUM_THREADS=str(threads), NUMEXPR_NUM_THREADS=str(threads))
        print(f'Starting {run.name}', flush=True)
        with (output / f'{run.name}.log').open('x') as log:
            subprocess.run([sys.executable, str(output / 'benchmark.py'), '--worker', str(run),
                            '--threads', str(threads)], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        records.append(json.loads((run / 'completed.json').read_text()))
        ranking = []
        for count in COUNTS:
            group = [r for r in records if r['threads'] == count]
            if len(group) == 2:
                ranking.append({'threads': count, 'median_epoch_seconds': statistics.median(
                    [s for r in group for s in r['steady_epoch_seconds']]),
                    'mean_total_seconds': statistics.mean(r['total_seconds'] for r in group)})
        ranking.sort(key=lambda r: r['median_epoch_seconds'])
        (output / 'summary.json').write_text(json.dumps({'completed_runs': len(records),
            'expected_runs': len(schedule), 'ranking': ranking, 'measurements': records}, indent=2))
        print(f'Completed {run.name}; ranking={ranking}', flush=True)


if __name__ == '__main__':
    main()
