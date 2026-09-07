"""Submit missing matched 2D/4D seeds while retaining existing 4D runs."""
import json
from pathlib import Path
import re
import shlex
import subprocess

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'notebooks/results/06b_4d_reconstruction'


def main():
    ledger=BASE/'jobs.jsonl'
    previous=[json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    seen={(r['latent_dim'],r['seed']) for r in previous}
    for dim in (2,4):
        for seed in (3456456,123,456,786,2026):
            if (dim,seed) in seen or (BASE/f'd{dim}_s{seed}/completed.json').exists():continue
            args=['/opt/pbs/bin/qsub','-N',f'lss_06b_d{dim}_{seed}','-v',f'LSS_DIM={dim},LSS_SEED={seed}','scripts/06b_dimension_reconstruction.pbs']
            cmd=f'cd {shlex.quote(str(ROOT))} && '+shlex.join(args)
            r=subprocess.run(['ssh','-F','/dev/null','-o','BatchMode=yes','-o','UserKnownHostsFile=/tmp/lss_known_hosts','alexander.z@127.0.0.1',cmd],text=True,capture_output=True,check=True)
            job=r.stdout.strip()
            if not re.fullmatch(r'\d+\.[\w.-]+',job):raise RuntimeError(r)
            record=dict(seed=seed,latent_dim=dim,job_id=job,code_version='code_v2')
            with ledger.open('a') as f:f.write(json.dumps(record)+'\n')
            print(record,flush=True)


if __name__=='__main__':main()
