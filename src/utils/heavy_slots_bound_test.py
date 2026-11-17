"""The 45-minute chunk policy, made TRUE (2026-11-17): a hold past a chunk is legitimate exactly when it is BOUNDED.

``heavy_slots`` said "hold a slot per bounded chunk (<= 45 min), never per run" and flagged every hold past 60 min, so the
indivisible jobs (a 70-min slow-test bundle, a 100-min cross cell, 4.8 % of the long lane's holds, 26.6 % of its slot-seconds)
could not satisfy it and the flag became noise. The policy is now: <= 45 min by chunking, or a DECLARED, ENFORCED bound
(``mem_cap.sh --max-run-s S``, S <= ``HOLD_BOUND_MAX_S``); what is flagged is the hold nobody bounded. Every state here is
a temp slot directory and a controllable timer: nothing is the real box's, nothing sleeps. Unmarked (in-process).
"""
from __future__ import annotations

import json
import time
from types import SimpleNamespace
from typing import List

import pytest

from utils import heavy_slots as HS


class _Timer:
    made: List["_Timer"] = []

    def __init__(self, interval, fn) -> None:
        self.interval, self.fn, self.started, self.cancelled, self.daemon = interval, fn, False, False, False
        _Timer.made.append(self)

    def start(self) -> None:
        self.started = True

    def cancel(self) -> None:
        self.cancelled = True

    def fire(self) -> None:
        if self.started and not self.cancelled:
            self.fn()


@pytest.fixture
def slots(tmp_path, monkeypatch):
    monkeypatch.setenv(HS.DIR_ENV, str(tmp_path / "heavy"))
    monkeypatch.setenv(HS.SLOTS_ENV, "2")
    for k in (HS.HELD_ENV, HS.BLOCKING_ENV):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GEN3AI_GPU_LOCK", str(tmp_path / "gpu.lock"))
    monkeypatch.setattr(HS, "threading", SimpleNamespace(Timer=_Timer))
    _Timer.made.clear()
    return tmp_path / "heavy"


BAR, CEIL = HS.LONG_HOLD_S, HS.HOLD_BOUND_MAX_S


def test_the_policy_numbers_are_the_measured_ones_and_ordered() -> None:
    """45 min is the chunk, 60 the chunk plus its stop and re-queue, 2 h the ceiling of a declared bound: the measured long
    lane (939 holds 2026-11-05..17) held > 2 h in 0.6 %, which is what must be chunked."""
    assert HS.CHUNK_MIN * 60 < BAR < CEIL == 7200.0


@pytest.mark.parametrize("held, bound, long_hold, overdue", [
    (BAR - 1, 0.0, False, False),                      # a chunk and a bit: fine
    (BAR + 1, 0.0, True, False),                       # nobody bounded it
    (BAR + 1, 5400.0, False, False),                   # an indivisible job that declared 90 min, enforced
    (5400.0 + 299, 5400.0, False, False),              # still inside its stop grace
    (5400.0 + 301, 5400.0, False, True),               # the supervisor should have stopped it: enforcement failed
    (CEIL + 60, CEIL, False, False),                   # the ceiling itself is within policy
    (BAR + 1, CEIL + 1, True, False),                  # a bound above the ceiling is no bound: split it
])
def test_a_hold_is_flagged_only_when_nobody_bounded_it(held, bound, long_hold, overdue) -> None:
    assert HS.hold_flags(held, lane=HS.LANE_LONG, max_s=0.0, run_bound_s=bound) == (long_hold, overdue)


def test_the_short_lane_overdue_rule_is_unchanged() -> None:
    worst = 600.0 + HS.SHORT_MARGIN_S + HS.SHORT_GRACE_MAX_S + 30.0
    assert HS.hold_flags(worst, lane=HS.LANE_SHORT, max_s=600.0, run_bound_s=0.0) == (False, False)
    assert HS.hold_flags(worst + 1, lane=HS.LANE_SHORT, max_s=600.0, run_bound_s=0.0) == (False, True)


def test_a_bounded_holder_is_recorded_shown_with_its_end_and_never_warned_or_flagged(slots) -> None:
    """The indivisible job: `mem_cap.sh --max-run-s 5400` reaches the slot as ``run_bound_s``. `--status` prints the bound and
    what is left (the ETA of the jobs queued behind it), the holder gets NO long-hold warning, and the release event carries
    the bound. Fails on a revert of the record field, the tag, the timer skip or the event field."""
    lines: List[str] = []
    with HS.heavy_slot(["python", "-m", "pytest", "a_slow_file.py"], force=True, log=lines.append, run_bound_s=5400.0):
        assert not [t for t in _Timer.made if t.started], "a bounded hold is inside the policy: no timer, no warning"
        rec = json.loads((slots / "slot0.json").read_text())
        assert rec["run_bound_s"] == 5400.0
        since = rec["since"]
        st = HS.collect(now=since + 4200.0)                          # 70 min in: past the 60-min bar
        row = st["slots"][0]
        assert (row["run_bound_s"], row["long_hold"], row["overdue"]) == (5400.0, False, False)
        text = HS.format_human(st)
        assert "BOUND<=1h30m (20m00s left)" in text and HS.LONG_HOLD_TAG not in text and HS.BOUND_OVERDUE_TAG not in text, text
        late = HS.format_human(HS.collect(now=since + 5400.0 + 400.0))
        assert HS.BOUND_OVERDUE_TAG in late and "PAST IT" in late, late
    (ev,) = [json.loads(x) for x in (slots / HS.EVENTS_FILE).read_text().splitlines() if '"kind":"slot"' in x]
    assert ev["bound_s"] == 5400.0


def test_an_unbounded_hold_is_still_flagged_and_the_warning_says_how_to_bound_it(slots) -> None:
    lines: List[str] = []
    with HS.heavy_slot(["a", "job"], force=True, log=lines.append):
        (timer,) = [t for t in _Timer.made if t.started]
        timer.fire()
        (warn,) = [ln for ln in lines if HS.LONG_HOLD_TAG in ln]
        assert "--max-run-s" in warn and f"{HS.CHUNK_MIN} min" in warn, warn
        st = HS.collect(now=time.time() + BAR + 60.0)
        assert st["slots"][0]["long_hold"] is True and HS.LONG_HOLD_TAG in HS.format_human(st)


def test_a_bound_above_the_ceiling_is_warned_as_one_to_split(slots) -> None:
    lines: List[str] = []
    with HS.heavy_slot(["a", "cross"], force=True, log=lines.append, run_bound_s=16600.0):
        (timer,) = [t for t in _Timer.made if t.started]
        timer.fire()
        (warn,) = [ln for ln in lines if HS.LONG_HOLD_TAG in ln]
        assert "split it into resumable chunks" in warn and "4.6 h" in warn, warn
        assert HS.collect(now=time.time() + BAR + 60.0)["slots"][0]["long_hold"] is True
