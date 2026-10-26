"""A tampered snapshot is REFUSED, never hung on (REVIEW D M1).

REVIEW D's tamper harness (``designs/research_state/measurements/human_replay_quality_2026-10-25/review_d_tamper.py``)
found that ``hrq.py extract`` HUNG on a changed log or a wrong population header: ``SnapshotError`` subclassed
``SystemExit``, a worker that raised it died, and the pool waited forever for its result. These tests run the same
cases on SYNTHETIC logs in a temp dir (no corpus read): the pool-level propagation (a bounded ``get``), and the real
``hrq.py extract`` CLI under a subprocess timeout. Both fail on a revert of ``SnapshotError`` to ``SystemExit``.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path

import pytest

from main.human_data import snapshots as S
from utils.paths import repo_path

HRQ = repo_path("designs", "research_state", "measurements", "human_replay_quality_2026-10-25", "hrq.py")
DATE = "2026-07-01"
TIMEOUT_S = 90


def _log(i: int) -> bytes:
    return ("\n".join(["|init|battle", f"|title|A{i} vs. B{i}", f"|player|p1|A{i}||1500", f"|player|p2|B{i}||1510",
                       "|gen|3", "|tier|[Gen 3] OU", "|rated|", "|start",
                       "|switch|p1a: Tyranitar|Tyranitar, M|100/100", "|switch|p2a: Zapdos|Zapdos|100/100",
                       "|turn|1", "|move|p1a: Tyranitar|Rock Slide|p2a: Zapdos", f"|win|A{i}"]) + "\n").encode()


def build(root: Path, population: str = "spectator", n: int = 6) -> list:
    """A temp snapshot ``hr_t``: corpus, manifest store and registry, the shapes ``hrq.py`` reads."""
    (root / "corpus" / DATE).mkdir(parents=True)
    rows = []
    for i in range(n):
        rel = f"{DATE}/battle-gen3ou-{i}.log"
        b = _log(i)
        (root / "corpus" / rel).write_bytes(b)
        rows.append((rel, hashlib.sha256(b).hexdigest(), len(b)))
    body = "\n".join("\t".join(map(str, r)) for r in rows).encode()
    (root / "store").mkdir()
    with gzip.open(root / "store" / "hr_t.tsv.gz", "wb") as f:
        f.write(body)
    reg = root / "designs" / "research_state"
    reg.mkdir(parents=True)
    (reg / "human_replay_snapshots.json").write_text(json.dumps({"snapshots": {"hr_t": {
        "population": population, "files": n, "manifest_sha256": hashlib.sha256(body).hexdigest()}}}))
    return rows


def _flip(path: Path) -> None:
    b = bytearray(path.read_bytes())
    b[len(b) // 2] ^= 0x01
    path.write_bytes(bytes(b))


def _job(args):
    root, item = args
    return len(S.read_verified("hr_t", "spectator", Path(root) / "corpus", tuple(item)))


def test_snapshot_error_is_an_exception_not_a_system_exit():
    assert issubclass(S.SnapshotError, Exception) and not issubclass(S.SnapshotError, SystemExit)


def test_a_worker_refusal_reaches_the_parent_in_bounded_time(tmp_path):
    rows = build(tmp_path)
    reg = tmp_path / "designs/research_state/human_replay_snapshots.json"
    items = S.read_manifest("hr_t", tmp_path / "store", reg)
    assert [tuple(r) for r in rows] == items
    _flip(tmp_path / "corpus" / rows[3][0])
    with multiprocessing.get_context("fork").Pool(2) as pool:
        res = pool.map_async(_job, [(str(tmp_path), list(it)) for it in items])
        # a SystemExit in the worker would leave this result pending forever: get() then times out, and FAILS
        with pytest.raises(S.SnapshotError, match=rows[3][0]):
            res.get(timeout=60)


def test_manifest_digest_and_count_are_checked(tmp_path):
    build(tmp_path)
    reg = tmp_path / "designs/research_state/human_replay_snapshots.json"
    r = json.loads(reg.read_text())
    r["snapshots"]["hr_t"]["manifest_sha256"] = "0" * 64
    reg.write_text(json.dumps(r))
    with pytest.raises(S.SnapshotError, match="manifest sha256"):
        S.read_manifest("hr_t", tmp_path / "store", reg)
    with pytest.raises(S.SnapshotError, match="not in the registry"):
        S.read_manifest("hr_nope", tmp_path / "store", reg)


def _hrq(root: Path, cache: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, HRQ_REPO=str(root), HRQ_CORPUS=str(root / "corpus"), HRQ_SNAPSHOTS=str(root / "store"),
               HRQ_DURABLE=str(root / "durable"), CUDA_VISIBLE_DEVICES="",
               PYTHONPATH=os.pathsep.join([str(repo_path("src")), os.environ.get("PYTHONPATH", "")]))
    try:
        return subprocess.run([sys.executable, str(HRQ), "extract", "--snapshot", "hr_t", "--cache",
                               str(root / cache), "--workers", "2"], env=env, capture_output=True, text=True,
                              timeout=TIMEOUT_S, check=False)
    except subprocess.TimeoutExpired:
        pytest.fail(f"hrq.py extract HUNG for {TIMEOUT_S} s instead of refusing (REVIEW D M1)")


@pytest.mark.integration
def test_hrq_extract_cli_refuses_a_tampered_log_a_wrong_header_and_a_stale_cache(tmp_path):
    rows = build(tmp_path)
    clean = _hrq(tmp_path, "c_clean")
    assert clean.returncode == 0, clean.stderr
    victim = tmp_path / "corpus" / rows[2][0]
    good = victim.read_bytes()
    _flip(victim)
    for cache in ("c_tamper", "c_clean"):            # a fresh cache, and the clean cache reused (re-verified)
        p = _hrq(tmp_path, cache)
        assert p.returncode != 0, f"{cache}: a changed log must be refused"
        assert "REFUSED" in p.stderr and rows[2][0] in p.stderr, p.stderr
    victim.write_bytes(good)
    reg = tmp_path / "designs/research_state/human_replay_snapshots.json"
    r = json.loads(reg.read_text())
    r["snapshots"]["hr_t"]["population"] = "archive"
    reg.write_text(json.dumps(r))
    p = _hrq(tmp_path, "c_pop")
    assert p.returncode != 0 and "wrong header signature" in p.stderr, p.stderr
