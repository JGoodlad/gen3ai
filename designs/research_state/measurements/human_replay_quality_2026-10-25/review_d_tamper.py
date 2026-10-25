"""REVIEW D: does `hrq.py extract --snapshot` really REFUSE a tampered snapshot? Run in a TEMP dir, on COPIES.

    PYTHONPATH=$PWD/src CUDA_VISIBLE_DEVICES="" nice -n 19 taskset -c 5-7,13-15 \
        $PY designs/research_state/measurements/human_replay_quality_2026-10-25/review_d_tamper.py   # < 2 min

Copies 40 real logs of one `hr_spec_v1` date (read-only on the corpus) with their REAL manifest rows into a temp
corpus, writes a temp registry whose digest is that sub-manifest's, and runs the real CLI (`hrq.py extract`) as a
subprocess under `timeout`, through the env overrides `HRQ_REPO` / `HRQ_CORPUS` / `HRQ_SNAPSHOTS`. Cases: clean;
one byte changed; the registry digest changed; an archive-population row with a spectator header; a tamper AFTER a
clean extract with the cache reused. Each case prints its exit code and whether it timed out (a hang is not a refusal).
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = sys.executable
CORPUS = Path("/home/goodlad/dev/gen3ai/replays/showdown/gen3ou")
REAL_MANIFEST = Path.home() / "gen3ai_archive/human_replay_snapshots/hr_spec_v1.tsv.gz"
DATE = "2026-07-01"
N = 40
TIMEOUT_S = 60


def build(root: Path, population: str = "spectator", files=None) -> list:
    with gzip.open(REAL_MANIFEST, "rt") as fh:
        rows = [ln.split("\t") for ln in fh.read().splitlines() if ln.startswith(DATE + "/")][:N]
    (root / "corpus" / DATE).mkdir(parents=True, exist_ok=True)
    for rel, _, _ in rows:
        shutil.copyfile(CORPUS / rel, root / "corpus" / rel)
    body = "".join("\t".join(r) + "\n" for r in rows).rstrip("\n").encode()
    (root / "store").mkdir(exist_ok=True)
    with gzip.open(root / "store" / "hr_t.tsv.gz", "wb") as f:
        f.write(body)
    reg = root / "designs" / "research_state"
    reg.mkdir(parents=True, exist_ok=True)
    (reg / "human_replay_snapshots.json").write_text(json.dumps({"snapshots": {"hr_t": {
        "population": population, "files": len(rows), "manifest_sha256": hashlib.sha256(body).hexdigest()}}}))
    return rows


def run(root: Path, cache: str) -> dict:
    env = dict(os.environ, HRQ_REPO=str(root), HRQ_CORPUS=str(root / "corpus"), HRQ_SNAPSHOTS=str(root / "store"),
               HRQ_DURABLE=str(root / "durable"), CUDA_VISIBLE_DEVICES="")
    try:
        p = subprocess.run([PY, str(HERE / "hrq.py"), "extract", "--snapshot", "hr_t", "--cache", str(root / cache),
                            "--workers", "1"], env=env, capture_output=True, text=True, timeout=TIMEOUT_S, check=False)
        return {"exit": p.returncode, "timed_out": False, "tail": (p.stdout + p.stderr).strip().splitlines()[-1:]}
    except subprocess.TimeoutExpired:
        return {"exit": None, "timed_out": True, "tail": [f"HUNG: no result within {TIMEOUT_S} s"]}


def records(root: Path, cache: str) -> int:
    n = 0
    for f in (root / cache / "hr_t").glob("*.jsonl.gz"):
        with gzip.open(f, "rt") as fh:
            n += sum(1 for _ in fh)
    return n


def main() -> None:
    out = {}
    with tempfile.TemporaryDirectory(prefix="review_d_tamper_") as td:
        root = Path(td)
        rows = build(root)
        out["clean"] = run(root, "c_clean") | {"records": records(root, "c_clean"), "of": len(rows)}

        victim = root / "corpus" / rows[7][0]
        b = bytearray(victim.read_bytes())
        b[len(b) // 2] ^= 0x01
        victim.write_bytes(bytes(b))
        out["one_byte_changed_fresh_cache"] = run(root, "c_tamper") | {"records": records(root, "c_tamper")}
        out["one_byte_changed_cache_reused"] = run(root, "c_clean") | {"records": records(root, "c_clean")}
        shutil.copyfile(CORPUS / rows[7][0], victim)

        reg = root / "designs/research_state/human_replay_snapshots.json"
        r = json.loads(reg.read_text())
        good = r["snapshots"]["hr_t"]["manifest_sha256"]
        r["snapshots"]["hr_t"]["manifest_sha256"] = "0" * 64
        reg.write_text(json.dumps(r))
        out["registry_digest_changed"] = run(root, "c_digest")
        r["snapshots"]["hr_t"]["manifest_sha256"] = good
        r["snapshots"]["hr_t"]["population"] = "archive"
        reg.write_text(json.dumps(r))
        out["spectator_logs_declared_archive"] = run(root, "c_pop")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
