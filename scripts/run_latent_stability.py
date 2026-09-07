"""Controlled history-model horizon comparison on the frozen matched AE."""
import copy
import os
from pathlib import Path
import sys

CODE = Path(os.environ['LSS_CODE_ROOT'])
sys.path.insert(0, str(CODE/'src'))
import lss.latent.experiment as experiment
import run_latent_matched_rollout as runner

# Fix both the sampled start population and latent-stat fitting across horizons.
def restrict_starts(original):
    def index(*args, **kwargs):
        rows = original(*args, **kwargs)
        return [row for row in rows if 5 <= int(row[1]) <= 92]
    return index

experiment.make_multistep_transition_index = restrict_starts(experiment.make_multistep_transition_index)
experiment.make_transition_index = restrict_starts(experiment.make_transition_index)

for scope in ('all4', 'transfer3'):
    for horizon in (1, 8):
        name = f'acceleration_{scope}_matched_h{horizon}'
        definition = copy.deepcopy(runner.VARIANTS[f'acceleration_{scope}'])
        definition['multistep_horizons'] = list(range(1, horizon+1))
        runner.VARIANTS[name] = definition

original_build = runner.build_config

def build_config(**kwargs):
    cfg = original_build(**kwargs)
    cfg['propagator_config'].update(train_rows_per_source=30*88, history_noise_std=0.0)
    cfg['controlled_start_frames'] = [5, 92]
    cfg['controlled_history_noise_std'] = 0.0
    return cfg

runner.build_config = build_config

if __name__ == '__main__':
    runner.main()
