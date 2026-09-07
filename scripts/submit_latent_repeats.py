"""Submit the fixed seed repetitions, recording each accepted PBS job immediately."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, choices=(456, 786), required=True)
    args = parser.parse_args()
    ledger = ROOT / "notebooks/results/latent_matched_study/repeat_jobs.jsonl"
    previous = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    completed = {(row["source"], row["dim"], row["seed"]) for row in previous}
    env = {**os.environ, "TMPDIR": "/tmp"}
    for source in ("shared3", "shared4", "reid", "depablo_low_temp", "depablo_mixed_temp", "lj_noisy"):
        for dim in (2, 4, 6, 8):
            if (source, dim, args.seed) in completed:
                continue
            response = subprocess.run([
                "qsub", "-N", f"lss_{source}_d{dim}_s{args.seed}",
                "-v", f"LSS_DIM={dim},LSS_SOURCES={source},LSS_SEED={args.seed}",
                "scripts/latent_matched_screen.pbs",
            ], cwd=ROOT, env=env, text=True, capture_output=True, check=True)
            job_id = response.stdout.strip()
            if not re.fullmatch(r"\d+\.[A-Za-z0-9_.-]+", job_id):
                raise RuntimeError(f"Unexpected PBS response; inspect queue before retrying: {response}")
            record = {"source": source, "dim": dim, "seed": args.seed, "id": job_id}
            with ledger.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
