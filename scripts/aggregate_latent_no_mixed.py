"""Collect completed AE-unseen mixed-T transfer results without inference."""
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parents[1] / 'notebooks/results/latent_no_mixed'


def main():
    frames = []
    status = []
    for dim in (2, 6, 8):
        for seed in (123, 456, 786):
            for variant in ('acceleration_transfer3', 'velocity_transfer3', 'delta_transfer3'):
                run = BASE / f'd{dim}_ae{seed}' / f'{variant}_shared4_s{seed}'
                complete = (run / 'completed.json').is_file()
                status.append(dict(dimension=dim, seed=seed, variant=variant, completed=complete))
                if not complete:
                    continue
                for kind, filename in [('rollout', 'validation_rollout_summary.csv'),
                                       ('ae', 'validation_ae_summary.csv')]:
                    frame = pd.read_csv(run / filename)
                    frame = frame.assign(dimension=dim, seed=seed, variant=variant, kind=kind)
                    frames.append(frame)
    pd.DataFrame(status).to_csv(BASE / 'status.csv', index=False)
    print(f"Completed rollout runs: {sum(row['completed'] for row in status)}/27")
    if not frames:
        return
    results = pd.concat(frames, ignore_index=True)
    results.to_csv(BASE / 'sourcewise_results.csv', index=False)
    aggregate = results.groupby(['dimension', 'variant', 'kind', 'source', 'rollout_steps']).agg(
        seeds=('seed', 'nunique'), r2_mean=('p_ratio_r2', 'mean'),
        r2_std=('p_ratio_r2', 'std'), mae_mean=('p_ratio_mae', 'mean'),
        valid_min=('used', 'min'), total_min=('total', 'min')).reset_index()
    aggregate.to_csv(BASE / 'sourcewise_aggregate.csv', index=False)
    print(aggregate.query("rollout_steps == 100 and kind == 'rollout'").to_string(index=False))


if __name__ == '__main__':
    main()
