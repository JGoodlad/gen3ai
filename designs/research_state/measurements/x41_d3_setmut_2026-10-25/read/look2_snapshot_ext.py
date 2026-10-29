"""D3 look 2 (REVIEW m5): snapshot the external state the pin does not pin -- the Metamon checkout and both conda
envs (conda list --explicit --md5 and pip freeze --all) -- as one JSON line appended to <state>/external_snapshots.jsonl. Read-only.

usage: python snapshot_ext.py <d3 state> <label>
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

CONDA = "/home/goodlad/miniconda3/bin/conda"
METAMON = "/home/goodlad/dev/metamon"
ENVS = {"metamon": "/home/goodlad/miniconda3/envs/metamon", "gen3ai_torch28": "/home/goodlad/miniconda3/envs/gen3ai_torch28"}


def run(argv):
    r = subprocess.run(argv, capture_output=True, text=True, timeout=600)
    return r.returncode, r.stdout


def main(argv):
    state, label = Path(argv[0]), argv[1]
    snap = {"t": round(time.time(), 1), "label": label}
    _, head = run(["git", "-C", METAMON, "rev-parse", "HEAD"])
    _, tracked = run(["git", "-C", METAMON, "status", "--porcelain", "--untracked-files=no"])
    _, untracked = run(["git", "-C", METAMON, "status", "--porcelain", "--untracked-files=normal"])
    snap["metamon_checkout"] = {"head": head.strip(), "tracked_changes": tracked.strip().splitlines(),
                                "untracked": [ln for ln in untracked.strip().splitlines() if ln.startswith("??")]}
    for name, prefix in ENVS.items():
        rc, out = run([CONDA, "list", "--explicit", "--md5", "-p", prefix])
        prc, pip = run([f"{prefix}/bin/python", "-m", "pip", "freeze", "--all"])
        snap[f"env_{name}"] = {"rc": rc, "lines": len(out.splitlines()),
                               "sha256": hashlib.sha256(out.encode()).hexdigest(),
                               "pip_rc": prc, "pip_lines": len(pip.splitlines()),
                               "pip_sha256": hashlib.sha256(pip.encode()).hexdigest()}
    with open(state / "external_snapshots.jsonl", "a") as f:
        f.write(json.dumps(snap) + "\n")
    print(json.dumps(snap))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
