"""Submit the predeclared controlled matrix only after normalization passes."""
import json
from pathlib import Path
import re
import shlex
import subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'notebooks/results/latent_controlled_followup'


def main():
    for source in ('reid','depablo_low_temp','depablo_mixed_temp','lj_noisy'):
        report=json.loads((ROOT/f'notebooks/results/in_memory_normalization_audit/{source}.json').read_text())
        if not report['passed']: raise RuntimeError(f'Normalization audit failed: {source}')
    OUT.mkdir(parents=True,exist_ok=True)
    ledger=OUT/'jobs.jsonl'
    previous=[json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    seen={(r['task'],r['condition'],r['seed']) for r in previous}
    matrix=[('stability',f'acceleration_{scope}_matched_h{h}',s) for scope in ('transfer3','all4') for h in (1,8) for s in (123,456,786)]
    matrix += [('lowdata',scope,s) for scope in ('shared3','shared4') for s in (123,456,786)]
    for task,condition,seed in matrix:
        if (task,condition,seed) in seen: continue
        key='LSS_VARIANT' if task=='stability' else 'LSS_SCOPE'
        args=['/opt/pbs/bin/qsub','-N',f'lss_{task[:4]}_{seed}','-v',f'LSS_TASK={task},{key}={condition},LSS_SEED={seed}','scripts/latent_controlled_followup.pbs']
        command=f'cd {shlex.quote(str(ROOT))} && '+shlex.join(args)
        r=subprocess.run(['ssh','-F','/dev/null','-o','BatchMode=yes','-o','UserKnownHostsFile=/tmp/lss_known_hosts','alexander.z@127.0.0.1',command],capture_output=True,text=True,check=True)
        job=r.stdout.strip()
        if not re.fullmatch(r'\d+\.[\w.-]+',job): raise RuntimeError(r)
        row=dict(task=task,condition=condition,seed=seed,job_id=job)
        with ledger.open('a') as f:f.write(json.dumps(row)+'\n')
        print(row,flush=True)


if __name__=='__main__': main()
