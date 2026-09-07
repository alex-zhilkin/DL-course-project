"""Submit the frozen-checkpoint information diagnostic with immutable provenance."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "notebooks/results/lj_ae_repair/equal_source_s456/ae.pt"
OUT = ROOT / "notebooks/results/ae_information_audit"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = OUT / "frozen_code_diagnostic_submission.json"
    if manifest.exists():
        raise FileExistsError(f"Already submitted or recorded: {manifest}")
    command = ["/opt/pbs/bin/qsub", "-v", f"LSS_CHECKPOINT={CHECKPOINT}", "scripts/diagnose_frozen_ae_information.pbs"]
    remote = f"cd {shlex.quote(str(ROOT))} && {shlex.join(command)}"
    result = subprocess.run([
        "ssh", "-F", "/dev/null", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
        "-o", "UserKnownHostsFile=/tmp/lss_known_hosts_shared_ae", "alexander.z@127.0.0.1", remote,
    ], check=True, capture_output=True, text=True)
    manifest.write_text(json.dumps({
        "job_id": result.stdout.strip(), "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "script_sha256": sha256(ROOT / "scripts/diagnose_frozen_ae_information.py"),
        "recipe": {"frames": [5, 10, 25, 50, 75, 100], "limit_per_source": 3,
                   "optimization_steps": 100, "random_starts": 2, "seed": 917},
        "scope": "Diagnostic only; coordinates and graph/state data only. It cannot select a checkpoint or model.",
    }, indent=2))
    print(result.stdout.strip())


if __name__ == "__main__":
    main()
