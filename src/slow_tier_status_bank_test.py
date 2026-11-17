"""An ORDINARY pytest session never writes the committed slow-tier status file; banking is a deliberate act.

**The defect (2026-11-16, recurring).** The root `conftest.py` merged every `slow` verdict a session measured into the
TRACKED `designs/ops/slow_tier_status.json`, whatever the session was for: a full `pytest src/`, the targeted re-run of a
touched module's slow tests, a child session another test started. The file came out of an unrelated session modified,
dirtied every worktree it ran in, and rode into unrelated commits (and rows banked at a worktree commit that the ship's
rebase then rewrote are orphaned: not an ancestor of main, so STALE forever).

**The rule now (`utils.slow_tier_status.recording_target`), pinned here two ways.**

* the TABLE, by injecting the environment: no variable -> nowhere; `GEN3AI_SLOW_STATUS_BANK=1` -> the committed file;
  `GEN3AI_SLOW_STATUS_FILE=<p>` -> `<p>` and never the committed file; `record_results` takes no default path;
* the SESSION, which is what fails on a revert of the conftest's use of it: a real child pytest session (the ACTUAL root
  `conftest.py`) runs two `slow` tests (one pass, one red) inside a FAKE checkout built in `tmp_path` (a copy of
  `src/utils`, so `utils.paths.repo_root()` is the fake and the real tracked file is never in reach), and the tracked
  file's bytes are compared before and after in each mode.

`integration`: it needs out-of-process pytest. A few seconds per session, so it rides the routine gate.
"""
from __future__ import annotations

import inspect
import json
import os
import shutil
import sys
import subprocess
from pathlib import Path
from typing import Dict, Optional

import pytest

from utils.child_guard import guarded_run
from utils.contention import scale_timeout
from utils.paths import repo_root, src_root
from utils.slow_tier_status import (
    BANK_ENV, SCHEMA, make_row, recording_target, record_results, tracked_path)

pytestmark = pytest.mark.integration

_FILE_ENV = "GEN3AI_SLOW_STATUS_FILE"
_PASS = "dummy_slow_test.py::test_a_measured_slow_test"
_RED = "dummy_slow_test.py::test_a_red_slow_test"
_SENTINEL = "sentinel_test.py::test_a_row_banked_long_ago"

_DUMMY = '''\
import pytest

@pytest.mark.slow
def test_a_measured_slow_test():
    pass

@pytest.mark.slow
def test_a_red_slow_test():
    assert 1 == 2
'''


# ───────────────────────────────────────────────────────────────── the table, no session ──


def test_the_recording_target_table_with_the_environment_injected(tmp_path):
    scratch = str(tmp_path / "scratch.json")
    assert recording_target({}) is None, "an ordinary session records NOWHERE"
    assert recording_target({BANK_ENV: ""}) is None
    assert recording_target({BANK_ENV: "0"}) is None, "`=0` is off, not 'set'"
    assert recording_target({BANK_ENV: "1"}) == tracked_path()
    assert recording_target({_FILE_ENV: scratch}) == Path(scratch), "an explicit file needs no BANK"
    assert recording_target({_FILE_ENV: scratch, BANK_ENV: "1"}) == Path(scratch), \
        "an explicit file wins: it never reaches the committed one"
    assert tracked_path() == repo_root() / "designs" / "ops" / "slow_tier_status.json"


def test_record_results_has_no_default_path_so_no_caller_can_write_the_tracked_file_by_omission():
    """Read off the signature, never by calling it: a call without a path, on a regression, would write the real file."""
    path = inspect.signature(record_results).parameters["path"]
    assert path.default is inspect.Parameter.empty, "`record_results(rows)` must name its target"


# ───────────────────────────────────────────────────── the session, in a fake checkout ──


def _git(cwd: Path, *argv: str) -> str:
    return subprocess.check_output(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *argv], cwd=str(cwd), text=True,
        stderr=subprocess.STDOUT).strip()


def _fake_checkout(tmp_path: Path) -> Dict[str, Path]:
    """A one-commit git repo whose `src/utils` is a copy of ours, the REAL root conftest, a seeded tracked file and two
    `slow` tests. Nothing in it can reach the real checkout's `designs/ops/slow_tier_status.json`."""
    repo = tmp_path / "repo"
    (repo / "src" / "utils").mkdir(parents=True)
    for f in (src_root() / "utils").iterdir():
        if f.is_dir():
            if f.name != "__pycache__":
                (repo / "src" / "utils" / f.name).symlink_to(f)      # a subpackage, read-only to this session
        elif f.suffix == ".py" and not f.name.endswith("_test.py"):
            shutil.copy(f, repo / "src" / "utils" / f.name)       # `utils.paths` is COPIED: its root is the fake
    shutil.copy(repo_root() / "conftest.py", repo / "conftest.py")
    (repo / "pytest.ini").write_text("[pytest]\nmarkers =\n    slow: slow\n")
    (repo / "dummy_slow_test.py").write_text(_DUMMY)
    (repo / "designs").mkdir()
    for f in (repo_root() / "designs").iterdir():                   # the conftest's autouse fixtures read designs/*.json
        if f.name != "ops":
            (repo / "designs" / f.name).symlink_to(f)
    tracked = repo / "designs" / "ops" / "slow_tier_status.json"
    tracked.parent.mkdir()
    seeded = {"schema": SCHEMA, "last_written": "2026-01-01T00:00:00Z", "tests": {
        _SENTINEL: make_row("pass", commit="a" * 40, duration_s=1, contention=1)}}
    tracked.write_text(json.dumps(seeded, indent=1) + "\n")
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "fake checkout")
    return {"repo": repo, "tracked": tracked}


def _run_session(repo: Path, **extra_env: str) -> str:
    """Run the fake checkout's slow tests under the real conftest with ONLY ``extra_env`` deciding where they bank."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("PYTEST_") and k not in (_FILE_ENV, BANK_ENV, "GEN3AI_SKIP_SLOW_STATUS_RECORD")}
    env.update(GEN3AI_SKIP_DEPS_GUARD="1", GEN3AI_SKIP_TIER_BUDGET="1", PYTHONPATH=os.pathsep.join([str(repo / "src"), str(src_root())]),
               CUDA_VISIBLE_DEVICES="", **extra_env)
    r = guarded_run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "-m", "slow",
                     "dummy_slow_test.py"], cwd=str(repo), env=env, capture_output=True, text=True,
                    timeout=scale_timeout(180.0))
    out = (r.stdout or "") + (r.stderr or "")
    assert "1 failed, 1 passed" in out, f"the fake checkout's session did not run both slow tests:\n{out}"
    return out


def test_an_ordinary_session_leaves_the_tracked_file_byte_identical(tmp_path):
    """THE FIX, at the session. Fails on a revert of the conftest (or of `recording_target`): the pre-fix writer
    merges both rows into the fake checkout's tracked file and stamps `last_written`."""
    fake = _fake_checkout(tmp_path)
    before = fake["tracked"].read_bytes()
    out = _run_session(fake["repo"])
    assert fake["tracked"].read_bytes() == before, "an ordinary session rewrote the tracked slow-tier status file"
    assert sorted(p.name for p in fake["tracked"].parent.iterdir()) == ["slow_tier_status.json"], \
        "an ordinary session left a stray file beside the tracked one"
    assert "2 slow verdict(s) measured and NOT banked" in out and BANK_ENV in out, \
        f"a session that banked nothing must say so in its summary:\n{out}"


def test_the_bank_path_still_updates_the_tracked_file_and_merges(tmp_path):
    """`GEN3AI_SLOW_STATUS_BANK=1`: both verdicts land in the committed file at the checkout's HEAD, the seeded row is
    kept (a merge, never a truncation), and the red is recorded as a red."""
    fake = _fake_checkout(tmp_path)
    head = _git(fake["repo"], "rev-parse", "HEAD")
    out = _run_session(fake["repo"], **{BANK_ENV: "1"})
    doc = json.loads(fake["tracked"].read_text())
    assert set(doc["tests"]) == {_SENTINEL, _PASS, _RED}, doc["tests"].keys()
    assert doc["tests"][_PASS]["status"] == "pass" and doc["tests"][_RED]["status"] == "fail"
    assert doc["tests"][_PASS]["commit"] == head == doc["tests"][_RED]["commit"]
    assert doc["tests"][_SENTINEL]["commit"] == "a" * 40, "the banked row of another test must survive"
    assert doc["last_written"] != "2026-01-01T00:00:00Z"
    assert "2 row(s) recorded" in out and "NOT banked" not in out, out


@pytest.mark.parametrize("bank", [None, "1"], ids=["file-only", "file-and-bank"])
def test_an_explicit_status_file_gets_the_rows_and_the_tracked_file_is_never_touched(tmp_path, bank):
    """`GEN3AI_SLOW_STATUS_FILE` (a GPU window's scratch file) is a deliberate target of its own and wins over BANK."""
    fake = _fake_checkout(tmp_path)
    before = fake["tracked"].read_bytes()
    scratch = tmp_path / "elsewhere" / "rows.json"
    extra: Dict[str, Optional[str]] = {_FILE_ENV: str(scratch)}
    if bank:
        extra[BANK_ENV] = bank
    _run_session(fake["repo"], **{k: v for k, v in extra.items() if v is not None})
    assert set(json.loads(scratch.read_text())["tests"]) == {_PASS, _RED}
    assert fake["tracked"].read_bytes() == before
