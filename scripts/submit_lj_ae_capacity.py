"""Submit the predeclared AE-only matrix with an append-only ledger."""
import json
from pathlib import Path
import re
import shlex
import subprocess

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'notebooks/results/lj_ae_capacity'
VARIANTS=('dimension6','dimension8')


def main():
    ledger=BASE/'jobs.jsonl'
    previous=[json.loads(x) for x in ledger.read_text().splitlines()] if ledger.exists() else []
    seen={(x['variant'],x['seed']) for x in previous}
    for variant in VARIANTS:
        for seed in (3456456,123,456,786,2026):
            if (variant,seed) in seen:continue
            cmd=['/opt/pbs/bin/qsub','-N',f'lss_ae_{variant[:7]}_{seed}','-v',f'LSS_VARIANT={variant},LSS_SEED={seed}','scripts/lj_ae_capacity.pbs']
            remote=f'cd {shlex.quote(str(ROOT))} && '+shlex.join(cmd)
            r=subprocess.run(['ssh','-F','/dev/null','-o','BatchMode=yes','-o','UserKnownHostsFile=/tmp/lss_known_hosts','alexander.z@127.0.0.1',remote],capture_output=True,text=True,check=True)
            job=r.stdout.strip()
            if not re.fullmatch(r'\d+\.[\w.-]+',job):raise RuntimeError(r)
            row=dict(variant=variant,seed=seed,job_id=job)
            with ledger.open('a') as f:f.write(json.dumps(row)+'\n')
            print(row,flush=True)


if __name__=='__main__':main()
