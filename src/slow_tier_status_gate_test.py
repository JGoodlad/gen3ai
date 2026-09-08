"""`gen3_slow_tier_status_v1` — THE SEVENTH STATIC GATE: a red `slow` test reaches the routine gate.

The routine gate is `-m "not slow and not e2e"`. A `slow` test is DESELECTED there, and a
deselected test cannot fail — so a red one is invisible until the next 31-minute full-suite run.
That is not a hypothetical: `tb_relevance_test`'s winprob smoke was red on main for a day behind
that marker (2026-09-07), and the obs-golden linchpin rode main red three separate times behind the
old `-m "not integration"` cut. **A gate that reads GREEN while a test is red is the GIGO class.**

This test closes it for one JSON read: the slow tier WRITES its verdict (the root `conftest.py`
merges every `slow` test that actually ran into `designs/ops/slow_tier_status.json`), and this
unmarked test READS it in every tier. The full contract — the four verdict classes, why staleness
is reported and never fatal, why the artifact is committed rather than gitignored, why a missing
file fails, and the honest limitation that a test which broke *since* the last recorded run still
reads green — is in `src/utils/slow_tier_status.py`'s module docstring.

**One thing is fatal here and only one: a RECORDED RED.** Inconclusive (a timeout signature —
*a timeout is never a semantic outcome*), unrecorded (a slow test nobody has run yet) and stale (the
tier has not been run lately) are REPORTED, as warnings that survive `-q`, and pass.

Refresh the artifact:

    export PYTHONPATH=$PYTHONPATH:src && python3 -m pytest src/ -m slow -q -n 2

Opt out with `GEN3AI_SKIP_SLOW_STATUS_GATE=1`. Runs in every tier; measured **0.03 s** plus one
`git rev-list` per distinct recorded commit.
"""
from __future__ import annotations

import json
import os
import subprocess
import warnings

import pytest

from utils.paths import repo_path
from utils.slow_tier_status import (
    SCHEMA,
    STALE_COMMITS,
    Verdict,
    classify,
    evaluate,
    make_row,
    merge_outcome,
    record_results,
    status_path,
)

_SKIP_ENV = "GEN3AI_SKIP_SLOW_STATUS_GATE"

_MISSING = """\
{path} is missing — the `slow`-tier last-known-status artifact.

This is a CHECKOUT problem, not a test failure. The file is COMMITTED precisely so that "missing"
cannot mean "nobody has run the tier yet": every checkout of main has one. Its absence means a bad
rebase, a stray delete, or a hand-edited tree.

FIX — restore it from git, or regenerate it by running the tier:

    git checkout -- designs/ops/slow_tier_status.json
    export PYTHONPATH=$PYTHONPATH:src && python3 -m pytest src/ -m slow -q -n 2

Set {env}=1 to run without this gate (a pure-unit CI that does not carry the artifact)."""


def _skip_if_opted_out():
    if os.environ.get(_SKIP_ENV):
        pytest.skip(f"{_SKIP_ENV} set")


def _load_doc():
    path = status_path()
    if not path.exists():
        pytest.fail(_MISSING.format(path=path, env=_SKIP_ENV))
    try:
        with open(path) as f:
            doc = json.load(f)
    except Exception as exc:
        pytest.fail(f"{path} is not readable JSON ({exc!r}). It is a committed artifact — "
                    f"`git checkout -- {path}` restores it.")
    if not isinstance(doc, dict) or not isinstance(doc.get("tests"), dict):
        pytest.fail(f"{path} is not a {SCHEMA} document: no top-level 'tests' object.")
    return path, doc


_DIST_CACHE: dict = {}


def _commit_distance(sha: str):
    """How many commits HEAD is ahead of ``sha``; ``None`` when git does not know the commit."""
    if not sha or sha == "unknown":
        return None
    if sha not in _DIST_CACHE:
        try:
            out = subprocess.check_output(
                ["git", "rev-list", "--count", f"{sha}..HEAD"],
                cwd=str(repo_path()), text=True, stderr=subprocess.DEVNULL).strip()
            _DIST_CACHE[sha] = int(out)
        except Exception:
            _DIST_CACHE[sha] = None
    return _DIST_CACHE[sha]


# ══════════════════════════════════════════════════════════ THE GATE ══


def test_the_slow_tier_status_artifact_exists_and_is_a_status_document():
    """A missing artifact FAILS and never skips — the house rule for every gate here."""
    _skip_if_opted_out()
    path, doc = _load_doc()
    assert doc["tests"], (
        f"{path} records ZERO slow tests. NO SLOW-TIER STATUS RECORDED — the routine gate is "
        f"reading nothing at all, which is exactly the blind spot this gate exists to remove. "
        f"Run: export PYTHONPATH=$PYTHONPATH:src && python3 -m pytest src/ -m slow -q -n 2")


def test_no_slow_tier_test_is_recorded_red(request):
    """THE GATE. A `slow` test the tier recorded as FAILED fails the routine run, by name."""
    _skip_if_opted_out()
    path, doc = _load_doc()
    collected = getattr(request.config, "_gen3ai_slow_collected", set())
    v = evaluate(doc, collected=collected, commit_distance=_commit_distance)
    _report(v, path)
    assert v.ok, v.red_message()


def _report(v: Verdict, path) -> None:
    """The three NON-fatal classes, as warnings — visible in `-q`'s warnings summary."""
    print(f"[slow-tier status] {path}: {v.n_rows} row(s), {v.n_pass} pass, {v.n_skip} skip, "
          f"{len(v.reds)} FAIL, {len(v.inconclusive)} inconclusive, "
          f"{len(v.unrecorded)} unrecorded, {len(v.stale)} stale")
    if v.inconclusive:
        warnings.warn(
            "slow-tier INCONCLUSIVE (a timeout signature — never a verdict): "
            + ", ".join(n for n, _ in v.inconclusive), stacklevel=2)
    if v.unrecorded:
        warnings.warn(
            f"{len(v.unrecorded)} `slow` test(s) collected with NO recorded status — their green "
            f"is not a measurement, it is an absence: {', '.join(v.unrecorded[:6])}"
            + (" …" if len(v.unrecorded) > 6 else ""), stacklevel=2)
    if v.stale:
        warnings.warn(
            f"{len(v.stale)} slow-tier row(s) older than {STALE_COMMITS} commits — the tier has "
            f"not been run lately, so how much a green here is worth is a question about the "
            f"schedule: {v.stale[0][0]} ({v.stale[0][1]})", stacklevel=2)


# ═════════════════════════════════════════ THE META-TESTS — a planted red MUST fire ══
#
# A gate nobody has watched fail is a gate nobody knows works. Each of these plants one condition
# into a scratch document and asserts the checker's verdict, so the gate's own behaviour is pinned
# rather than described.


def _row(status, commit="deadbeef", detail="boom"):
    return {"status": status, "commit": commit, "at": "2026-09-07T00:00:00Z",
            "duration_s": 1.0, "contention": 1.0, "detail": detail}


def test_a_planted_red_fails_the_gate():
    v = evaluate({"tests": {"src/a_test.py::test_x": _row("pass"),
                            "src/b_test.py::test_y": _row("fail")}})
    assert not v.ok
    assert [n for n, _ in v.reds] == ["src/b_test.py::test_y"]
    msg = v.red_message()
    assert "src/b_test.py::test_y" in msg and "deadbeef" in msg


def test_an_all_green_document_passes():
    v = evaluate({"tests": {"src/a_test.py::test_x": _row("pass")}})
    assert v.ok and v.n_pass == 1 and not v.inconclusive


def test_an_inconclusive_row_is_reported_and_NOT_fatal():
    """A TIMEOUT IS NEVER A SEMANTIC OUTCOME — the tree's standing rule, applied here."""
    v = evaluate({"tests": {"src/a_test.py::test_x": _row("inconclusive")}})
    assert v.ok
    assert [n for n, _ in v.inconclusive] == ["src/a_test.py::test_x"]


def test_a_collected_slow_test_with_no_row_is_reported_and_NOT_fatal():
    v = evaluate({"tests": {"src/a_test.py::test_x": _row("pass")}},
                 collected={"src/a_test.py::test_x", "src/new_test.py::test_new"})
    assert v.ok
    assert v.unrecorded == ["src/new_test.py::test_new"]


def test_staleness_is_reported_and_NOT_fatal():
    v = evaluate({"tests": {"src/a_test.py::test_x": _row("pass")}},
                 commit_distance=lambda sha: STALE_COMMITS + 1)
    assert v.ok and len(v.stale) == 1
    v_fresh = evaluate({"tests": {"src/a_test.py::test_x": _row("pass")}},
                       commit_distance=lambda sha: 0)
    assert not v_fresh.stale


def test_an_unknown_commit_is_stale_not_silently_fresh():
    v = evaluate({"tests": {"src/a_test.py::test_x": _row("pass")}},
                 commit_distance=lambda sha: None)
    assert v.ok and "not an ancestor" in v.stale[0][1]


# ═══════════════════════════════════════════════ the WRITER's two invariants ══


def test_a_timeout_shaped_failure_records_INCONCLUSIVE_and_a_real_one_records_FAIL():
    assert classify(True, False, "E   asyncio.TimeoutError: battle 3 timed out") == "inconclusive"
    assert classify(True, False, "E   AssertionError: 2 != 3") == "fail"
    assert classify(False, True, "") == "skip"
    assert classify(False, False, "") == "pass"


def test_a_teardown_failure_beats_a_passing_call_but_a_pass_never_overwrites_a_fail():
    assert merge_outcome("pass", "fail") == "fail"
    assert merge_outcome("fail", "pass") == "fail"
    assert merge_outcome("inconclusive", "fail") == "fail"
    assert merge_outcome("fail", "inconclusive") == "fail"
    assert merge_outcome(None, "pass") == "pass"


def test_a_SKIPPED_test_never_banks_as_a_PASS():
    """The bug the first real tier run exposed: a skipped test reports `skipped` in SETUP and then
    `passed` in TEARDOWN, so all 44 skips banked GREEN until `skip` outranked `pass`. A test that
    did not run must not read as a test that ran and was fine."""
    assert merge_outcome("skip", "pass") == "skip"
    assert merge_outcome("pass", "skip") == "skip"
    assert merge_outcome("skip", "fail") == "fail"
    v = evaluate({"tests": {"a::t": _row("skip"), "b::t": _row("pass")}})
    assert v.ok and v.n_pass == 1 and v.n_skip == 1


def test_recording_MERGES_and_never_truncates_the_other_rows(tmp_path):
    """Running ONE slow test must not erase the other 74 verdicts — the merge is the whole point."""
    path = tmp_path / "status.json"
    record_results({"a::t1": make_row("pass", commit="c1", duration_s=1, contention=1)}, path)
    record_results({"b::t2": make_row("fail", commit="c2", duration_s=2, contention=1)}, path)
    doc = json.loads(path.read_text())
    assert set(doc["tests"]) == {"a::t1", "b::t2"}
    assert doc["tests"]["a::t1"]["status"] == "pass"
    assert doc["schema"] == SCHEMA
    v = evaluate(doc)
    assert not v.ok and [n for n, _ in v.reds] == ["b::t2"]


def test_the_env_override_points_the_reader_at_a_scratch_file(tmp_path, monkeypatch):
    """`$GEN3AI_SLOW_STATUS_FILE` is what lets a plant be proven without touching the real file."""
    scratch = tmp_path / "elsewhere.json"
    monkeypatch.setenv("GEN3AI_SLOW_STATUS_FILE", str(scratch))
    assert status_path() == scratch
    monkeypatch.delenv("GEN3AI_SLOW_STATUS_FILE")
    assert status_path() == repo_path("designs", "ops", "slow_tier_status.json")
