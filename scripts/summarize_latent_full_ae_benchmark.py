"""Report per-node timings and speedups over the local eight-thread baseline."""
import json
from pathlib import Path
import statistics

BASE = Path(__file__).resolve().parents[1] / 'notebooks/results/latent_full_ae_benchmark'
JOBS = ('4669055.zeus-master', '4669056.zeus-master', '4669057.zeus-master')


def main():
    rows = []
    for job in JOBS:
        path = BASE / job / 'summary.json'
        if not path.exists():
            print(f'{job}: no completed runs yet')
            continue
        summary = json.loads(path.read_text())
        ranking = summary['ranking']
        baseline = next((r['median_epoch_seconds'] for r in ranking if r['threads'] == 8), None)
        print(f"{job}: {summary['completed_runs']}/{summary['expected_runs']} runs completed")
        for record in ranking:
            row = dict(record, job_id=job, speedup_over_local_8=baseline / record['median_epoch_seconds'] if baseline else None)
            rows.append(row)
            print(json.dumps(row))
    (BASE / 'multinode_summary.json').write_text(json.dumps({'jobs': JOBS, 'comparisons': rows,
        'interpretation': 'Compare speedups against each node\'s own eight-thread baseline; confirm finalists on the same node before selecting a production default.'}, indent=2))


if __name__ == '__main__':
    main()
