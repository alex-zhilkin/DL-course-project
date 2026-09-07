"""Submit independent AE-plus-dynamics transfer pipelines on ZEUS."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'notebooks/results/latent_no_mixed/jobs.jsonl'


def main():
    previous = [json.loads(x) for x in LEDGER.read_text().splitlines()] if LEDGER.exists() else []
    accepted = {(x['dim'], x['seed']) for x in previous}
    for dim in (2, 6, 8):
        for seed in (123, 456, 786):
            if (dim, seed) in accepted:
                continue
            command = f'cd /rg/mendels_prj/alexander.z/DL-course-project && /opt/pbs/bin/qsub -N lss_nomix_d{dim}_s{seed} -v LSS_DIM={dim},LSS_SEED={seed} scripts/latent_no_mixed_pipeline.pbs'
            result = subprocess.run(['ssh', '-F', '/dev/null', '-o', 'BatchMode=yes',
                '-o', 'UserKnownHostsFile=/tmp/lss_known_hosts',
                'alexander.z@127.0.0.1', command], text=True, capture_output=True, check=True)
            row = dict(dim=dim, seed=seed, id=result.stdout.strip())
            with LEDGER.open('a') as stream:
                stream.write(json.dumps(row) + '\n')
                stream.flush()
            print(row, flush=True)


if __name__ == '__main__':
    main()
