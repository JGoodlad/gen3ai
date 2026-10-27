"""``main.reads.attempts``: a unit whose retries were SPENT (EXHAUSTED) is told apart from one the registered stop cut off,
in ``status``, in the assembled cells, in the supervisor's stop event and in the ``INCOMPLETE_BY_STOP`` record
(backlog row ``main.reads`` open item 2). Each guard fails on revert. Units are stubbed (no game is played): a stub
that fails writes the same ``units.jsonl`` events ``kakuna._play`` does."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, List

import pytest

from main.reads import api, attempts as AT, lanectl as LC, layout as LY
from main.reads import record as REC
from main.reads import runner as RN
from main.reads.anchor_pool_test import write_played
from main.reads.lanectl_test import lane  # noqa: F401  (the fixture)
from main.reads.partial_test import _planned
from main.reads.record_test import _rec
from main.x35 import kakuna as KA


def _log(state: Path, *events: dict) -> None:
    with open(state / "units.jsonl", "a") as f:
        for ev in events:
            f.write(json.dumps({"t": time.time(), **ev}) + "\n")


def _end(name: str, counted: bool, short: bool = True, rc: int = 0) -> dict:
    return {"event": "end", "unit": name, "rc": rc, "counted": counted, "short": short}


def _start(name: str) -> dict:
    return {"event": "start", "unit": name, "short": True}


def test_failed_attempts_count_the_ends_that_did_not_count_and_net_out_a_short_lane_overrun(tmp_path):
    st = tmp_path / "k"
    st.mkdir()
    (st / "units.jsonl").write_text("this line is damaged\n[1, 2]\n")
    _log(st,
         _start("two"), _end("two", False, rc=1), _start("two"), _end("two", False, rc=1),            # two failed attempts
         _start("overran_then_counted"), _end("overran_then_counted", False, rc=88),
         {"event": "short_lane_overrun", "unit": "overran_then_counted", "rc": 88},
         _start("overran_then_counted"), _end("overran_then_counted", True, short=False),               # ONE attempt, counted
         _start("overran_then_failed"), _end("overran_then_failed", False, rc=88),
         {"event": "short_lane_overrun", "unit": "overran_then_failed", "rc": 88},
         _start("overran_then_failed"), _end("overran_then_failed", False, short=False, rc=1),         # ONE failed attempt
         _start("killed"),                                                                              # no end: interrupted
         _start("late"), _end("late", False, rc=1), _start("late"), _end("late", True),                 # failed once, then counted
         {"event": "supervisor_start", "pid": 1})
    assert AT.failed_attempts(st) == {"two": 2, "overran_then_failed": 1, "late": 1}
    assert AT.failed_attempts(tmp_path / "no_such_state") == {}


def _kakuna_plan(tmp_path: Path, seeds=(0, 1, 2)) -> Path:
    st = tmp_path / "ks"
    st.mkdir()
    units = KA.plan_units([{"label": "a", "arm": "A", "seed": 1, "path": "/m.zip"}], hash_models=False,
                          teamsets=("away",), unit_seeds=seeds)
    (st / "plan.json").write_text(json.dumps({"units": [u.__dict__ for u in units]}))
    return st


def test_a_unit_is_exhausted_only_when_uncounted_with_every_attempt_spent(tmp_path):
    st = _kakuna_plan(tmp_path)
    n0, n1, n2 = (u.name for u in KA.load_plan(st))
    assert KA.MAX_ATTEMPTS == 2, "the test below spends exactly the registered attempts"
    _log(st, _end(n0, False, rc=1), _end(n0, False, rc=1),         # n0: both attempts failed -> EXHAUSTED
         _end(n1, False, rc=1))                                    # n1: one failure of two -> not exhausted
    assert AT.exhausted_units(st) == [n0]
    write_played(st, next(u for u in KA.load_plan(st) if u.name == n0))      # n0 counts after all (a later life)
    assert AT.exhausted_units(st) == [], "a unit that counts is never exhausted, whatever its history"
    _log(st, _end(n1, False, rc=1))
    assert AT.exhausted_units(st) == [n1]


@pytest.fixture
def stopped(tmp_path, run_archive_env, monkeypatch):
    """A lane stopped at its registered stop with 8 planned units: ``old__00`` always fails with NO summary (two attempts,
    then given up on: EXHAUSTED), ``new__01`` fails twice but leaves an uncounted summary (EXHAUSTED, but played), four
    units count, and the stop never reaches the last three."""
    state = _planned(tmp_path, run_archive_env, monkeypatch)
    now = [time.time()]
    stop = now[0] + 1000
    bad_no_summary, bad_summary = "old__heldout72_hpfix__00", "new__heldout72_hpfix__01"

    def unit(st, u):
        _log(st, _start(u.name))
        if u.name == bad_no_summary:
            _log(st, _end(u.name, False, rc=1))
            return 1
        write_played(st, u)
        if u.name == bad_summary:
            s = json.loads((st / "cells" / u.name / "summary.json").read_text())
            s["peer_clean"] = False                       # played, but it does not count
            (st / "cells" / u.name / "summary.json").write_text(json.dumps(s))
            _log(st, _end(u.name, False, rc=0))
            return 0
        _log(st, _end(u.name, True))
        now[0] += 250
        return 0

    assert RN.run(state, "anchor", stop, run_unit=unit, clock=lambda: now[0], sleep=lambda s: None) == 0
    return state, bad_no_summary, bad_summary


def test_status_and_the_cells_tell_an_exhausted_unit_from_one_the_stop_never_reached(stopped):
    state, bad_no_summary, bad_summary = stopped
    st = RN.status(state)["anchor"]["u"]
    assert st["old"] == {"counted": 2, "planned": 4, "exhausted": 1}
    assert st["new"] == {"counted": 2, "planned": 4, "exhausted": 1}, "an exhausted unit that left a summary is exhausted too"
    grp = api.assemble(state)["anchor"]["u"]
    assert grp["exhausted_units"] == sorted([bad_no_summary, bad_summary])
    assert grp["unplayed_exhausted_units"] == [bad_no_summary], "only the one with no summary is part of `unplayed`"
    assert grp["unplayed"] == 3                         # old00 (exhausted), old03 and new03 (cut off by the stop)
    ev = [json.loads(x) for x in LY.supervisor_log(state, "anchor").read_text().splitlines()][-1]
    assert ev["event"] == "stop_at" and ev["left"] == 2 and ev["not_counted"] == sorted([bad_no_summary, bad_summary])


def test_the_assemble_verb_prints_the_unplayed_and_exhausted_units_of_an_anchor_group(stopped, capsys):
    from main.reads import cli

    state, bad_no_summary, bad_summary = stopped
    capsys.readouterr()
    assert cli.main(["assemble", "--state", str(state)]) == 0
    out = json.loads(capsys.readouterr().out)["u"]
    assert out["unplayed"] == 3 and out["unplayed_exhausted_units"] == [bad_no_summary]
    assert out["exhausted_units"] == sorted([bad_no_summary, bad_summary])
    assert cli.main(["status", "--state", str(state)]) == 0
    assert json.loads(capsys.readouterr().out)["anchor"]["u"]["old"]["exhausted"] == 1


def test_the_incomplete_by_stop_record_names_the_exhausted_units_apart_from_the_stop_cut_ones(stopped):
    state, bad_no_summary, _bad_summary = stopped
    rec = api.record(state)
    g = rec["completeness"]["groups"]["anchor.u"]
    assert g == {"planned": 8, "played": 5, "unplayed": 3, "counted": 4, "unplayed_exhausted": 1, "stop_cut": 2,
                 "unplayed_exhausted_units": [bad_no_summary]}
    assert rec["completeness"]["stops"]["anchor"]["not_counted"] == sorted([bad_no_summary, "new__heldout72_hpfix__01"])
    line = api.summary_line(rec)
    assert "anchor.u 5 of 8 units played, 1 EXHAUSTED (retries spent)" in line, line
    assert api.load_record(LY.record_file(state))["completeness"]["groups"]["anchor.u"] == g


def test_a_stop_that_cut_units_off_without_any_exhausted_one_says_so(tmp_path, run_archive_env, monkeypatch):
    from main.reads.partial_test import _stop_after

    state = _planned(tmp_path, run_archive_env, monkeypatch)
    _stop_after(state, 4)
    g = api.record(state)["completeness"]["groups"]["anchor.u"]
    assert (g["unplayed"], g["unplayed_exhausted"], g["stop_cut"], g["unplayed_exhausted_units"]) == (4, 0, 4, [])
    assert "EXHAUSTED" not in api.summary_line(api.load_record(LY.record_file(state)))


def test_the_record_check_refuses_exhausted_and_stop_cut_counts_that_do_not_add_up():
    ok = _rec(completeness={"class": "INCOMPLETE_BY_STOP", "opt_in": "decision.allow_partial_at_stop",
                            "groups": {"anchor.u": {"planned": 8, "played": 5, "unplayed": 3, "counted": 4,
                                                    "unplayed_exhausted": 1, "stop_cut": 2,
                                                    "unplayed_exhausted_units": ["x"]}},
                            "stops": {"anchor": {"event": "stop_at"}}})
    assert REC.check(ok) == []
    for key, val in (("stop_cut", 1), ("unplayed_exhausted", 2), ("unplayed_exhausted_units", [])):
        bad = copy.deepcopy(ok)
        bad["completeness"]["groups"]["anchor.u"][key] = val
        assert any("is not unplayed_exhausted" in p for p in REC.check(bad)), (key, REC.check(bad))
    old = copy.deepcopy(ok)                                  # a record written before the split carries neither key
    for key in ("stop_cut", "unplayed_exhausted", "unplayed_exhausted_units"):
        del old["completeness"]["groups"]["anchor.u"][key]
    assert REC.check(old) == []


def test_a_lane_that_finished_with_exhausted_units_is_refused_as_a_stop_and_says_why(lane):  # noqa: F811
    """`done` is not a registered stop: a lane that ended because every remaining unit was exhausted must not be
    recorded as one cut off by the stop. The refusal names the units."""
    log = LY.supervisor_log(lane, "anchor")
    log.write_text("".join(json.dumps(e) + "\n" for e in (
        {"event": "start", "stop_epoch": time.time() + 99, "t": 1.0},
        {"event": "done", "not_counted": ["old__heldout72_hpfix__00", "new__heldout72_hpfix__02"], "t": 2.0})))
    with pytest.raises(LC.LaneError, match=r"FINISHED with 2 unit\(s\) whose retries were spent \(EXHAUSTED"):
        LC.stop_evidence(lane, "anchor")
    log.write_text(json.dumps({"event": "done", "not_counted": [], "t": 2.0}) + "\n")
    with pytest.raises(LC.LaneError) as exc:
        LC.stop_evidence(lane, "anchor")
    assert "EXHAUSTED" not in str(exc.value)


def test_a_lane_that_gives_up_on_a_unit_logs_it_in_its_done_event(lane):  # noqa: F811
    """The supervisor's own accounting is unchanged: `done` lists the units it gave up on (now through `_gave_up`)."""
    def unit(st: Path, u: Any) -> int:
        if u.name == "old__heldout72_hpfix__00":
            return 1                                       # never counts
        write_played(st, u)
        return 0

    assert RN.run(lane, "anchor", time.time() + 3600, run_unit=unit, sleep=lambda s: None) == 0
    last: List[dict] = [json.loads(x) for x in LY.supervisor_log(lane, "anchor").read_text().splitlines()]
    assert last[-1]["event"] == "done" and last[-1]["not_counted"] == ["old__heldout72_hpfix__00"]
