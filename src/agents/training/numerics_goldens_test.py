"""The routine guard of ``numerics_goldens``: a re-recorded routine golden forces every registered SLOW golden.

Unmarked (routine tier, in the ship gate): it reads source files and one JSON, no torch, well under a second.
"""
from __future__ import annotations

import json

import pytest

from agents.training import numerics_goldens as NG
from utils.paths import repo_path


def test_every_registered_slow_golden_was_run_under_the_current_routine_goldens():
    stale = NG.stale()
    assert not stale, (
        "OBS_SEMANTICS_VERSION or a ROUTINE numerics golden (the K9 learner golden, the played-bits sentinel) moved "
        "since these SLOW goldens were last run — the ship gate does not run them:\n  " + "\n  ".join(stale)
        + "\nFor EACH file: run it (`pytest <file> -m slow`), re-record what moved by the file's own procedure with the "
          f"cause, then set its {NG.STAMP} to the anchor `python -m agents.training.numerics_goldens` prints. "
          "Never re-stamp a file you did not run.")


def test_no_slow_test_pins_a_digest_outside_the_registry():
    loose = NG.unregistered()
    assert not loose, (
        "a slow test reads a sha256 constant that `numerics_goldens.SLOW_GOLDENS` does not list:\n  "
        + "\n  ".join(loose) + "\nRegister it (and stamp the file), or add the file to EXEMPT with the reason "
        "the model's numerics cannot move it.")
    for rel, why in NG.EXEMPT.items():
        assert repo_path(rel).is_file() and why.strip(), rel
        assert rel not in NG.SLOW_GOLDENS, f"{rel} is both registered and exempt"


def _patched_anchor(monkeypatch, tmp_path, edit):
    """The anchor with ``edit`` applied to a COPY of the learner golden (the real file is never written)."""
    g = json.loads(repo_path(NG.LEARNER_GOLDEN).read_text())
    edit(g)
    alt = tmp_path / "learner_golden.json"
    alt.write_text(json.dumps(g))
    real = NG.repo_path
    monkeypatch.setattr(NG, "repo_path", lambda rel: alt if rel == NG.LEARNER_GOLDEN else real(rel))
    return NG.anchor()


def _anchor_with(monkeypatch, rel, name, value):
    """The anchor with ONE source constant read as ``value`` (undone before returning)."""
    real = NG.constants
    with monkeypatch.context() as m:
        m.setattr(NG, "constants", lambda r: {**real(r), name: value} if r == rel else real(r))
        return NG.anchor()


@pytest.mark.parametrize("field", ["post_params_sha256", "fp32_opt_out"])
def test_the_guard_has_teeth_a_re_recorded_learner_golden_turns_every_stamp_stale(monkeypatch, tmp_path, field):
    before = NG.anchor()

    def edit(g):
        e = g["entries"][sorted(g["entries"])[0]]
        if field == "fp32_opt_out":
            e["fp32_opt_out"]["post_params_sha256"] = "0" * 64
        else:
            e[field] = "0" * 64

    assert _patched_anchor(monkeypatch, tmp_path, edit) != before
    stale = NG.stale()
    assert len(stale) == len(NG.SLOW_GOLDENS) and all(NG.STAMP in s for s in stale), stale


@pytest.mark.parametrize("digest", ["PACKED_DIGEST", "PLAYED_DIGEST"])
def test_the_guard_has_teeth_a_re_recorded_sentinel_digest_moves_the_anchor(monkeypatch, digest):
    assert _anchor_with(monkeypatch, NG.SENTINEL, digest, "0" * 64) != NG.anchor()


def test_the_guard_has_teeth_a_raised_obs_semantics_marker_moves_the_anchor(monkeypatch):
    """Config v166 moved the overlap file's goldens and no routine golden: the semantics marker is the trigger. The
    source reader agrees with the imported constant, and a config-version bump alone (a new per-run setting) is NOT
    a trigger."""
    from agents.model.model_version import constants as C

    now, before = NG.semantics_version(), NG.anchor()
    assert now == C.OBS_SEMANTICS_VERSION
    with monkeypatch.context() as m:
        m.setattr(NG, "semantics_version", lambda: now + 1)
        assert NG.anchor() != before
    assert _anchor_with(monkeypatch, NG.VERSION_CONSTANTS, "MODEL_CONFIG_VERSION", C.MODEL_CONFIG_VERSION + 1) == before


def test_a_semantics_history_the_reader_cannot_read_is_refused_not_hashed(monkeypatch, tmp_path):
    real = NG.repo_path
    for body in ("OBS_SEMANTICS_HISTORY = rows()\n", "OBS_SEMANTICS_HISTORY = (('v1', 'x', 'warn'),)\n", "X = 1\n"):
        f = tmp_path / "constants.py"
        f.write_text(body)
        with monkeypatch.context() as m:
            m.setattr(NG, "repo_path", lambda rel, f=f: f if rel == NG.VERSION_CONSTANTS else real(rel))
            with pytest.raises(ValueError, match="OBS_SEMANTICS_HISTORY"):
                NG.anchor()


def test_the_scan_finds_a_slow_digest_in_a_file_marked_per_test_and_skips_a_routine_one(monkeypatch, tmp_path):
    src = (
        'import pytest\n'
        'ROUTINE = "' + "a" * 64 + '"\n'
        'SLOW = "' + "b" * 64 + '"\n'
        'def test_routine():\n    assert ROUTINE\n'
        '@pytest.mark.slow\n'
        'def test_slow():\n    assert SLOW\n')
    f = tmp_path / "x_test.py"
    f.write_text(src)
    real = NG.repo_path
    monkeypatch.setattr(NG, "repo_path", lambda rel: f if rel == "x_test.py" else real(rel))
    assert NG.slow_read_digests("x_test.py") == ["SLOW"]
    f.write_text("import pytest\npytestmark = [pytest.mark.slow]\n" + src)
    assert NG.slow_read_digests("x_test.py") == ["ROUTINE", "SLOW"]
