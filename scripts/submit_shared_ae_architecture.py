"""Freeze and submit the shared-AE architecture study after smoke validation."""
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "notebooks/results/shared_ae_architecture"
SEEDS = (3456456, 123, 456, 786, 2026)


def freeze_code(code_version):
    code = BASE / code_version
    if code.exists():
        return code
    code.mkdir(parents=True)
    shutil.copytree(ROOT / "src", code / "src")
    (code / "scripts").mkdir()
    for name in ("run_shared_ae_architecture.py", "latent_diagnostic_metrics.py"):
        shutil.copy2(ROOT / "scripts" / name, code / "scripts" / name)
    manifest = {}
    for path in sorted(code.rglob("*")):
        if path.is_file():
            manifest[str(path.relative_to(code))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (code / "code_sha256.json").write_text(json.dumps(manifest, indent=2))
    return code


def qsub(variant, latent_dim, seed, code_version, smoke=False):
    variables = f"LSS_VARIANT={variant},LSS_LATENT_DIM={latent_dim},LSS_SEED={seed},LSS_SMOKE={int(smoke)},LSS_CODE_VERSION={code_version}"
    command = ["/opt/pbs/bin/qsub", "-N", f"lss_{variant[:8]}_{latent_dim}_{seed}", "-v", variables, "scripts/shared_ae_architecture.pbs"]
    remote = f"cd {shlex.quote(str(ROOT))} && {shlex.join(command)}"
    result = subprocess.run([
        "ssh", "-F", "/dev/null", "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "UserKnownHostsFile=/tmp/lss_known_hosts_shared_ae",
        "alexander.z@127.0.0.1", remote,
    ], capture_output=True, text=True, check=True)
    job = result.stdout.strip()
    if not re.fullmatch(r"\d+\.[\w.-]+", job):
        raise RuntimeError(result.stdout + result.stderr)
    return job


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true"); parser.add_argument("--retry", action="store_true")
    parser.add_argument("--code-version", default="code_v4")
    args = parser.parse_args()
    freeze_code(args.code_version)
    variants = ("orientation_corrected", "mp2") if args.smoke else ("orientation_corrected", "mp2", "mp4", "single_stage")
    dimensions = (2,) if args.smoke else (2, 8)
    seeds = (123,) if args.smoke else SEEDS
    ledger = BASE / "jobs.jsonl"; existing = [json.loads(line) for line in ledger.read_text().splitlines()] if ledger.exists() else []
    seen = {(row["variant"], row["latent_dim"], row["seed"], row["smoke"], row.get("retry", False), row.get("code_version")) for row in existing}
    for variant in variants:
        for dimension in dimensions:
            for seed in seeds:
                key = (variant, dimension, seed, args.smoke, args.retry, args.code_version)
                if key in seen: continue
                row = {"variant": variant, "latent_dim": dimension, "seed": seed, "smoke": args.smoke, "retry": args.retry, "code_version": args.code_version, "job_id": qsub(variant, dimension, seed, args.code_version, args.smoke)}
                with ledger.open("a") as stream: stream.write(json.dumps(row) + "\n")
                print(row, flush=True)


if __name__ == "__main__":
    main()
