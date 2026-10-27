"""A reduced read at a REGISTERED stop (belief-prior-mix 71c76dfb FINDING 8: `record` needed every planned unit played even
when the rule's `min_unit_frac` would decide). `decision.allow_partial_at_stop: true` is the spec's own opt-in; the record
is then class INCOMPLETE_BY_STOP with the counted units, and only when the lane ENDED at its stop (`stop_at`). Without the
opt-in, `record` refuses exactly as before. Units are stubbed; no game is played."""
from __future__ import annotations

import copy
import json
import time

import pytest

from main.reads import api, lanectl as LC, layout as LY
from main.reads import record as REC
from main.reads import runner as RN
from main.reads.anchor_pool_test import write_played
from main.reads.runner_test import CODE, _spec, _step, _zip
from main.reads.spec import SpecError, parse
from main.reads.supervisors_test import _fake_supervisor, _stop
from main.reads.spec_test import base as spec_base


def _planned(tmp_path, run_archive_env, monkeypatch, *, allow=True, name="state"):
    """A planned state of 8 anchor units (2 models x 1 team set x 4 unit seeds), `min_unit_frac` 0.5."""
    p = _spec(tmp_path, run_archive_env)
    doc = json.loads(p.read_text())
    doc["anchor"]["u"]["unit_seeds"] = "0-3"
    doc["decision"]["params"]["min_unit_frac"] = 0.5
    if allow:
        doc["decision"]["allow_partial_at_stop"] = True
    p.write_text(json.dumps(doc))
    _zip(run_archive_env / "rb_new", age=3600)
    state = tmp_path / name
    RN.plan(p, state, code=CODE, step_of=_step)
    monkeypatch.setattr(RN, "checkout_code", lambda root=None: CODE)
    return state


def _stop_after(state, n_units):
    """Run the anchor lane on a fake clock that passes its registered stop after ``n_units`` units: it ends with
    ``stop_at`` and the rest of the queue unplayed."""
    now = [time.time()]
    stop = now[0] + 1000

    def unit(st, u):
        write_played(st, u)
        now[0] += 1001 / n_units
        return 0

    assert RN.run(state, "anchor", stop, run_unit=unit, clock=lambda: now[0], sleep=lambda s: None) == 0
    return stop


def test_without_the_opt_in_a_stop_with_units_unplayed_is_still_refused(tmp_path, run_archive_env, monkeypatch):
    state = _planned(tmp_path, run_archive_env, monkeypatch, allow=False)
    _stop_after(state, 4)
    assert RN.status(state)["anchor"]["u"]["old"] == {"counted": 2, "planned": 4, "exhausted": 0}
    with pytest.raises(api.RecordError, match=r"cells not conclusive.*4 unit\(s\) unplayed"):
        api.record(state)
    assert not LY.record_file(state).exists()


def test_a_registered_reduced_read_records_the_counted_units_as_incomplete_by_stop(tmp_path, run_archive_env, monkeypatch):
    state = _planned(tmp_path, run_archive_env, monkeypatch)
    stop = _stop_after(state, 4)
    rec = api.record(state, out=tmp_path / "reads" / "demo_look1.json")
    comp = rec["completeness"]
    assert comp["class"] == REC.INCOMPLETE_BY_STOP == "INCOMPLETE_BY_STOP" and comp["opt_in"] == "decision.allow_partial_at_stop"
    assert comp["groups"] == {"anchor.u": {"planned": 8, "played": 4, "unplayed": 4, "counted": 4,
                                           "unplayed_exhausted": 0, "stop_cut": 4, "unplayed_exhausted_units": []}}
    assert comp["stops"]["anchor"]["event"] == "stop_at" and comp["stops"]["anchor"]["stop_epoch"] == stop
    assert comp["stops"]["anchor"]["left"] == 4
    assert rec["cells_inconclusive"] == [] and rec["verdict"] == rec["decision"]["decision"]["verdict"]
    assert rec["verdict"] != "VOID", "the rule's own min_unit_frac decided (2 of 4 per model clears 0.5)"
    line = api.summary_line(rec)
    assert "[INCOMPLETE_BY_STOP: anchor.u 4 of 8 units played]" in line
    assert api.load_record(tmp_path / "reads" / "demo_look1.json")["completeness"] == comp
    assert api.load_record(LY.record_file(state))["completeness"]["class"] == "INCOMPLETE_BY_STOP"


def test_the_rules_own_floor_still_decides_a_reduced_read(tmp_path, run_archive_env, monkeypatch):
    """The opt-in lets the record be WRITTEN; it never lowers the rule's `min_unit_frac`: 1 unit of 4 per model is VOID."""
    state = _planned(tmp_path, run_archive_env, monkeypatch)
    _stop_after(state, 2)
    rec = api.record(state)
    assert rec["verdict"] == "VOID" and rec["completeness"]["groups"]["anchor.u"]["played"] == 2


@pytest.mark.parametrize("how", ["never ran", "held", "killed", "refused", "live"])
def test_a_lane_that_did_not_end_at_its_registered_stop_is_not_recorded_partial(tmp_path, run_archive_env, monkeypatch,
                                                                                 how):
    state = _planned(tmp_path, run_archive_env, monkeypatch)
    fake = None
    if how == "never ran":
        for lab in ("old", "new"):                       # units on disk (as if played elsewhere), no supervisor log
            from main.x35 import kakuna as KA
            st = LY.anchor_state(state, "u", lab)
            for u in KA.load_plan(st)[:2]:
                write_played(st, u)
        needle = "never ran"
    else:
        _stop_after(state, 4)
        needle = "last supervisor event is"
        log = LY.supervisor_log(state, "anchor")
        if how == "held":
            log.write_text(log.read_text().replace('"stop_at"', '"held_stopped"'))
        elif how == "killed":
            lines = log.read_text().splitlines()
            log.write_text("\n".join(lines[:-1]) + "\n")  # the supervisor died: no closing event
        elif how == "refused":
            log.write_text(log.read_text() + json.dumps({"event": "refused", "why": "x", "t": time.time()}) + "\n")
        else:
            fake = _fake_supervisor(tmp_path, ["run", "--state", str(state), "--lane", "anchor", "--hours", "1"])
            needle = "live supervisor"
    try:
        with pytest.raises(LC.LaneError, match=needle):
            api.record(state)
    finally:
        if fake is not None:
            _stop(fake)
    assert not LY.record_file(state).exists()


def test_a_read_with_every_unit_played_is_complete_even_when_the_spec_allows_a_reduced_one(tmp_path, run_archive_env,
                                                                                            monkeypatch):
    state = _planned(tmp_path, run_archive_env, monkeypatch)
    assert RN.run(state, "anchor", time.time() + 3600, run_unit=lambda st, u: write_played(st, u) or 0,
                  sleep=lambda s: None) == 0
    rec = api.record(state)
    assert rec["completeness"] == {"class": "COMPLETE"} and "INCOMPLETE" not in api.summary_line(rec)


def test_other_reasons_a_read_is_not_conclusive_still_refuse_a_reduced_read(tmp_path, run_archive_env, monkeypatch):
    """Only UNPLAYED units are exempt: a unit whose summary stamps another regime is still not conclusive."""
    state = _planned(tmp_path, run_archive_env, monkeypatch)
    _stop_after(state, 4)
    s = LY.anchor_state(state, "u", "new") / "cells" / "new__heldout72_hpfix__01" / "summary.json"
    doc = json.loads(s.read_text())
    doc["cell"]["their_regime"] = "another"
    s.write_text(json.dumps(doc))
    with pytest.raises(api.RecordError, match="span 2 regimes"):
        api.record(state)


def test_the_record_check_refuses_a_partial_record_the_spec_did_not_register():
    from main.reads.record_test import _rec

    r = _rec(completeness={"class": "INCOMPLETE_BY_STOP", "groups": {"anchor.u": {"planned": 8, "played": 4,
                                                                                    "unplayed": 4, "counted": 4}},
                           "stops": {"anchor": {"event": "stop_at"}}})
    assert any("without decision.allow_partial_at_stop" in p for p in REC.check(r)), REC.check(r)
    ok = copy.deepcopy(r)
    ok["completeness"]["opt_in"] = "decision.allow_partial_at_stop"
    assert REC.check(ok) == []
    for mutate, needle in [(lambda c: c["groups"].clear(), "names no group"),
                           (lambda c: c["stops"].clear(), "no recorded stop"),
                           (lambda c: c["stops"]["anchor"].update(event="held_stopped"), "not a registered stop"),
                           (lambda c: c["groups"]["anchor.u"].update(unplayed=0), "no unplayed unit")]:
        bad = copy.deepcopy(ok)
        mutate(bad["completeness"])
        assert any(needle in p for p in REC.check(bad)), (needle, REC.check(bad))
    assert any("neither" in p for p in REC.check(_rec(completeness={"class": "MOSTLY"})))
    assert REC.check(_rec()) == [], "a record written before `completeness` existed reads as complete"


def test_build_refuses_a_partial_record_for_a_spec_without_the_opt_in(tmp_path, run_archive_env, monkeypatch):
    state = _planned(tmp_path, run_archive_env, monkeypatch, allow=False)
    _stop_after(state, 4)
    spec = RN.load_state(state)
    cells = api.assemble(state)
    dec = api.decide(state)
    with pytest.raises(REC.RecordError, match="needs the spec's decision.allow_partial_at_stop"):
        REC.build(spec, cells, json.loads(LY.inputs_file(state).read_text()), dec, {"commit": "c" * 40},
                  partial={"groups": {"anchor.u": {"planned": 8, "played": 4, "unplayed": 4, "counted": 4}},
                           "stops": {"anchor": {"event": "stop_at"}}})


def test_the_opt_in_is_refused_where_it_cannot_be_honest():
    d = spec_base()
    assert parse(d).allow_partial_at_stop is False
    d["decision"]["allow_partial_at_stop"] = True
    assert parse(d).allow_partial_at_stop is True                  # an anchor group, and a record this module owns
    for bad in ("yes", 1):
        e = copy.deepcopy(d)
        e["decision"]["allow_partial_at_stop"] = bad
        with pytest.raises(SpecError, match="true / false"):
            parse(e)
    from main.reads.rules import RULES

    assert RULES["d1"].x35_record and RULES["x35_r1"].x35_record and not RULES["anchor_trajectory"].x35_record
    x35 = copy.deepcopy(d)
    x35["decision"]["rule"] = "d1"
    with pytest.raises(SpecError, match="X35-stage record"):
        parse(x35, rule_params=False)
    s_only = copy.deepcopy(d)
    s_only["anchor"] = {}
    with pytest.raises(SpecError, match="no anchor group"):
        parse(s_only, rule_params=False)
