"""Submit missing science follow-ups with an append-only PBS ledger."""
import json
from pathlib import Path
import re
import shlex
import subprocess

ROOT=Path(__file__).resolve().parents[1]
LEDGER=ROOT/'notebooks/results/latent_science_followup/jobs.jsonl'


def main():
    previous=[json.loads(line) for line in LEDGER.read_text().splitlines()] if LEDGER.exists() else []
    keys={(r['task'],r.get('scope'),r['seed']) for r in previous}
    jobs=[('diagnostics',None,s) for s in (123,456,786)]
    jobs += [('lowdata',scope,s) for scope in ('shared3','shared4') for s in (123,456,786)]
    for task,scope,seed in jobs:
        if (task,scope,seed) in keys: continue
        env=f'LSS_TASK={task},LSS_SEED={seed}' + (f',LSS_SCOPE={scope}' if scope else '')
        args=['/opt/pbs/bin/qsub','-N',f'lss_{task[:4]}_{scope or "all"}_{seed}','-v',env,'scripts/latent_science_followup.pbs']
        command=f'cd {shlex.quote(str(ROOT))} && '+shlex.join(args)
        response=subprocess.run(['ssh','-F','/dev/null','-o','BatchMode=yes','-o','UserKnownHostsFile=/tmp/lss_known_hosts','alexander.z@127.0.0.1',command],text=True,capture_output=True,check=True)
        job=response.stdout.strip()
        if not re.fullmatch(r'\d+\.[\w.-]+',job): raise RuntimeError(response)
        row=dict(task=task,scope=scope,seed=seed,id=job)
        with LEDGER.open('a') as stream: stream.write(json.dumps(row)+'\n')
        print(row,flush=True)


if __name__=='__main__': main()
