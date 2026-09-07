"""Submit the matched dynamics matrix and record every accepted PBS job."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SHARED_VARIANTS = (
    "velocity_one",
    "velocity_source_id_one",
    "velocity_temperature_one",
    "velocity_source_temperature_one",
    "velocity_multistep8",
    "window_one",
    "window_multistep8",
)
SOURCES = ("reid", "depablo_low_temp", "depablo_mixed_temp", "lj_noisy")
SEEDS = (123, 456, 786)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Submit only shared velocity_one seed 123.")
    parser.add_argument("--transfer", action="store_true", help="Run the nine controlled transfer comparisons.")
    parser.add_argument("--acceleration", action="store_true", help="Run six rolling-acceleration comparisons.")
    parser.add_argument(
        "--submit-host",
        help="Submit through SSH on the login host (useful when the local sandbox blocks PBS authentication).",
    )
    args = parser.parse_args()
    jobs = [(variant, "shared4", seed) for variant in SHARED_VARIANTS for seed in SEEDS]
    jobs += [("velocity_one", source, seed) for source in SOURCES for seed in SEEDS]
    if args.smoke:
        jobs = [("velocity_one", "shared4", 123)]
    if args.transfer:
        jobs = [(v, "shared4", s) for v in (
            "velocity_transfer3", "delta_transfer3", "delta_all4"
        ) for s in SEEDS]
    if args.acceleration:
        jobs = [(v, "shared4", s) for v in (
            "acceleration_all4", "acceleration_transfer3"
        ) for s in SEEDS]
    code_version = "code_v3" if args.acceleration else "code_v2" if args.transfer else "code_v1"
    ledger = ROOT / "notebooks" / "results" / "latent_matched_rollouts" / "jobs.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    previous = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    accepted = {(row["variant"], row["scope"], row["seed"]) for row in previous}
    env = {**os.environ, "TMPDIR": "/tmp"}
    for variant, scope, seed in jobs:
        output = ledger.parent / f"{variant}_{scope}_s{seed}" / "completed.json"
        if (variant, scope, seed) in accepted or output.is_file():
            continue
        qsub = [
            "/opt/pbs/bin/qsub" if args.submit_host else "qsub",
            "-N", f"lss_{variant[:5]}_{scope[:5]}_{seed}",
            "-v", f"LSS_VARIANT={variant},LSS_SCOPE={scope},LSS_SEED={seed},LSS_CODE_VERSION={code_version}",
            "scripts/latent_matched_rollout.pbs",
        ]
        command = qsub
        if args.submit_host:
            remote = f"cd {shlex.quote(str(ROOT))} && " + " ".join(
                shlex.quote(part) for part in qsub
            )
            command = [
                "ssh", "-F", "/dev/null", "-o", "BatchMode=yes",
                "-o", "StrictHostKeyChecking=accept-new",
                "-o", "UserKnownHostsFile=/tmp/lss_known_hosts",
                args.submit_host, remote,
            ]
        response = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )
        job_id = response.stdout.strip()
        if not re.fullmatch(r"\d+\.[A-Za-z0-9_.-]+", job_id):
            raise RuntimeError(f"Unexpected PBS response; inspect queue before retrying: {response}")
        record = {"variant": variant, "scope": scope, "seed": seed, "id": job_id}
        with ledger.open("a") as stream:
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
