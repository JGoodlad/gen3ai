"""The test temp root is on the REAL DISK and a passing session leaves NOTHING behind.

Why: on 2026-09-30 `/tmp` (tmpfs — RAM + a fixed 1,048,576-inode table) went 42% -> 61% inodes in 15
minutes, from concurrent `-n 2` routine gates each retaining a `pytest-of-<user>/pytest-NNNN` of ~80k
files. The root `conftest.py` now points every temp dir at `_test_scratch_root()` and `pytest.ini`
sets `tmp_path_retention_policy = failed` / `_count = 1`.

This runs a CHILD pytest session (`-n 2`, so xdist workers and a grandchild subprocess are covered)
over `pytest_tmp_probe_child.py` with `$GEN3AI_SCRATCH` pointed at a fresh on-disk dir, and asserts
(1) every temp location it reports is under that root and not on tmpfs, and (2) after it PASSES, no
`pytest-N` basetemp remains. Reverting the conftest half fails (1); reverting the ini half fails (2).
A third case: a tmpfs scratch root is REFUSED.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from utils.contention import scale_timeout
from utils.paths import repo_root

pytestmark = pytest.mark.integration

_PROBE = "src/utils/pytest_tmp_probe_child.py"


def _fs_type(path: str) -> str | None:
    real, best, kind = os.path.realpath(path), "", None
    with open("/proc/mounts") as f:
        for line in f:
            p = line.split()
            if len(p) >= 3 and (real == p[1] or real.startswith(p[1].rstrip("/") + "/")) \
                    and len(p[1]) >= len(best):
                best, kind = p[1], p[2]
    return kind


def _child(scratch: Path, out: Path, *extra: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, GEN3AI_SCRATCH=str(scratch), GEN3AI_TMP_PROBE_OUT=str(out),
               GEN3AI_SKIP_SLOW_STATUS_RECORD="1")
    env.pop("TMPDIR", None)                  # the child must set it itself, not inherit ours
    env.pop("PYTEST_XDIST_WORKER", None)
    env.pop("PYTEST_ADDOPTS", None)
    return subprocess.run([sys.executable, "-m", "pytest", _PROBE, "-q", "-p", "no:cacheprovider",
                           *extra], cwd=repo_root(), env=env, capture_output=True, text=True,
                          timeout=scale_timeout(180))


def _on_disk_dir(tmp_path_factory, name: str) -> Path:
    d = tmp_path_factory.mktemp(name)
    if _fs_type(str(d)) in ("tmpfs", "ramfs"):
        pytest.fail(f"this session's own temp root {d} is on tmpfs — the conftest half is reverted")
    return d


def test_child_session_temp_is_on_disk_and_a_pass_leaves_nothing(tmp_path_factory):
    scratch = _on_disk_dir(tmp_path_factory, "scratch")
    out = _on_disk_dir(tmp_path_factory, "out")
    r = _child(scratch, out, "-n", "2")
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]

    reports = {p.stem: json.loads(p.read_text()) for p in out.glob("*.json")}
    assert {"gw0", "gw1"} <= set(reports), (reports, r.stdout[-2000:])
    root = os.path.realpath(scratch)
    for worker, rep in reports.items():
        for key in ("tmp_path", "gettempdir", "subprocess_gettempdir"):
            where = os.path.realpath(rep[key])
            assert where == root or where.startswith(root + os.sep), (worker, key, rep)
            assert _fs_type(where) not in ("tmpfs", "ramfs"), (worker, key, rep)

    retained = sorted(str(p) for p in scratch.glob("pytest-of-*/pytest-[0-9]*"))
    assert not retained, f"a PASSING session retained its basetemp: {retained}"


def test_a_tmpfs_scratch_root_is_refused(tmp_path_factory):
    shm = Path("/dev/shm")
    if _fs_type(str(shm)) != "tmpfs" or not os.access(shm, os.W_OK):
        pytest.skip("no writable tmpfs at /dev/shm to point the child at")
    out = _on_disk_dir(tmp_path_factory, "out_refused")
    r = _child(shm / f"gen3ai_tmp_refusal_probe_{os.getpid()}", out)
    try:
        assert r.returncode != 0
        assert "real disk" in (r.stdout + r.stderr), r.stdout[-2000:] + r.stderr[-2000:]
        assert not list(out.glob("*.json")), "the child ran tests on a tmpfs root"
    finally:
        import shutil
        shutil.rmtree(shm / f"gen3ai_tmp_refusal_probe_{os.getpid()}", ignore_errors=True)
