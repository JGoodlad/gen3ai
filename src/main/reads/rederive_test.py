"""The anchor guards RE-DERIVED at assemble time (review H1, ``main.reads.rederive``): the spec's mode, the HOLD a
``report``-mode read's hit puts on ``decide``, the orchestrator's release of exactly those hits, and the record.

The per-unit re-derivation itself (a capture's FORCELOSE, a peer log's long-tail line) is pinned on a kakuna state in
``main/anchors/forced_end_guard_test.py``."""
from __future__ import annotations

import json

import pytest

from main.reads import api, rederive as RD
from main.reads import spec as SP
from main.reads.rules_test import GAMES, _banked, _d3_spec

HIT = {"check": "forced_end", "tag": "battle-gen3ou-1", "half": "peer_challenge", "index": 1, "slot": "p2",
       "booked": "win", "turns": 3, "hit_forfeit_limit": False, "forfeit_turn_limit": 250}


def _cells(mode: str, *, hits=(HIT,), counted=True, unavailable=False) -> dict:
    chk = {"schema": RD.SCHEMA, "forced_end": "hit" if hits else ("unavailable" if unavailable else "clean"),
           "peer_fault": "clean", "hits": list(hits), "why": ["no capture"] if unavailable else []}
    unit = {"unit": "c0_s3501__forum72__00", "counted": counted, "rederived": chk}
    return {"guards": {"rederived": mode, "choices": "require"}, "anchor": {"h": {"units": [unit]}}}


def test_a_spec_registered_before_the_rederivation_defaults_to_report_every_later_one_to_count(tmp_path):
    sp = _d3_spec(tmp_path, _banked(GAMES))
    assert sp.rederived_guards == "count"                      # a spec this change has never seen
    for sha in RD.REGISTERED_BEFORE:
        assert SP.parse(sp.raw, sha256=sha).rederived_guards == "report"
    assert SP.parse(dict(sp.raw, rederived_guards="report"), sha256="").rederived_guards == "report"
    with pytest.raises(SP.SpecError, match="rederived_guards"):
        SP.parse(dict(sp.raw, rederived_guards="maybe"), sha256="")


#: Every read spec committed when the re-derivation landed (the in-flight reads among them).
SPECS_AT_THE_CHANGE = (
    "belief_prior_mix_2026-10-26/read_spec_l025.json", "belief_prior_mix_2026-10-26/read_spec_l100.json",
    "hctx_rc72_2026-10-26/read_spec.json", "hctx_rc72_2026-10-26/read_spec_am1.json",
    "hctx_rc72_2026-10-26/read_spec_am2.json", "su_headroom_2026-10-27/read_spec.json", "su_gate_2026-10-27/read_spec.json",
    "x41_d3_setmut_2026-10-25/read/read_spec_look1.json", "x41_d3_setmut_2026-10-25/read/read_spec_look2.json",
    "x59s_matched_students_2026-10-25/read_spec_full.json", "x59s_matched_students_2026-10-25/read_spec_gate.json",
    "x59s_matched_students_2026-10-25/read_spec_pilot.json",
    "x59s_matched_students_2026-10-25/read_spec_restriction.json",
    "x61p_n1_production_2026-10-26/read_spec_look1.json", "x61p_n1_production_2026-10-26/read_spec_look2.json")


def test_every_spec_on_file_at_the_change_is_registered_before_it():
    """The frozen set holds the bytes of every read spec committed at the change, so an in-flight read's counting
    never changes (``report``); the one dropped from it would flip to ``count`` silently."""
    from utils.paths import repo_path

    for rel in SPECS_AT_THE_CHANGE:
        p = repo_path("designs/research_state/measurements") / rel
        assert SP.load(p, rule_params=False).rederived_guards == "report", rel


def test_a_report_mode_hit_on_a_counted_unit_holds_decide_until_the_orchestrator_releases_it(tmp_path):
    cells = _cells("report")
    with pytest.raises(RD.GuardHoldError, match="HELD FOR THE ORCHESTRATOR.*forced_end battle-gen3ou-1"):
        RD.check_hold(tmp_path, cells)
    with pytest.raises(RD.GuardHoldError, match="--by"):
        RD.release(tmp_path, cells, by="", why="x")
    rel = RD.release(tmp_path, cells, by="orchestrator", why="the drop is in an uncounted half; keep the counting")
    got = RD.check_hold(tmp_path, cells)
    assert got["held"] and got["released"]["by"] == "orchestrator"
    assert got["released"]["hits_sha256"] == rel["hits_sha256"]
    other = _cells("report", hits=(dict(HIT, tag="battle-gen3ou-2"),))
    with pytest.raises(RD.GuardHoldError, match="names other hits"):     # a release never covers a later hit
        RD.check_hold(tmp_path, other)
    (tmp_path / RD.RELEASE_NAME).write_text("{not json")
    with pytest.raises(RD.GuardHoldError, match="damaged release"):
        RD.check_hold(tmp_path, cells)


def test_nothing_is_held_in_count_mode_on_an_uncounted_unit_or_without_a_hit(tmp_path):
    assert RD.check_hold(tmp_path, _cells("count"))["held"] is False         # count mode voided the unit instead
    assert RD.check_hold(tmp_path, _cells("report", counted=False))["hits"] == []
    summ = RD.check_hold(tmp_path, _cells("report", hits=(), unavailable=True))
    assert summ["held"] is False and summ["units_checked"] == 1 and summ["unavailable"][0]["why"]
    with pytest.raises(RD.GuardHoldError, match="nothing is held"):
        RD.release(tmp_path, _cells("report", hits=()), by="o", why="w")


def test_count_mode_records_the_unit_it_voided(tmp_path):
    cells = _cells("count", counted=False)
    cells["anchor"]["h"]["units"][0]["rederived_uncounted"] = True
    summ = RD.summarize(cells)
    assert summ["hits"][0]["counted"] is False and summ["held"] is False


def test_decide_refuses_while_held(tmp_path, monkeypatch):
    """``api.decide`` runs the HOLD before the rule (a typed refusal: the CLI's exit 2)."""
    state = tmp_path / "state"
    state.mkdir()
    (state / "cells.json").write_text(json.dumps(_cells("report")))
    monkeypatch.setattr(api, "void_gate", lambda _s: None)
    monkeypatch.setattr(api, "load_state", lambda _s: _d3_spec(tmp_path, _banked(GAMES)))
    with pytest.raises(api.REFUSALS, match="HELD FOR THE ORCHESTRATOR"):
        api.decide(state)
    assert not (state / "decision.json").exists()


def test_the_record_carries_the_rederivation_and_refuses_a_held_one_never_released():
    from main.reads.record import check

    base = {"schema": "gen3_read_record_v1"}
    held = {"held": True, "hits": [HIT]}
    assert any("never released" in p for p in check(dict(base, **_record_keys(), rederived_guards=held)))
    released = dict(held, released={"by": "o", "why": "w", "hits_sha256": "x"})
    assert not any("rederived" in p for p in check(dict(base, **_record_keys(), rederived_guards=released)))


def _record_keys() -> dict:
    """A minimal record body that otherwise checks, read from the golden x26c record's shape."""
    from main.reads.record import REQUIRED

    rec = {k: None for k in REQUIRED}
    rec.update(read="r", rule="anchor_trajectory", spec_sha256="0" * 64, created="now", models={},
               regimes={"anchor": {}, "s": {}}, ledgers={}, code={"commit": "a" * 40}, cells_inconclusive=[],
               inputs={}, decision={"verdict": "VOID"}, verdict="VOID")
    rec.pop("schema")
    return rec


def test_the_cells_carry_the_spec_guard_policy(tmp_path, monkeypatch):
    """``assemble`` records the policy the rules' guards read (H2's ``choices`` and H1's ``rederived``)."""
    from main.reads import assemble as AS

    monkeypatch.setattr(AS, "members_of", lambda *_a, **_k: {})
    monkeypatch.setattr(AS, "s_group", lambda *_a, **_k: {"nodes": {}, "inconclusive": []})
    sp = _d3_spec(tmp_path, _banked(GAMES))
    state = tmp_path / "state"
    state.mkdir()
    assert AS.assemble(sp, state)["guards"] == {"rederived": "count", "choices": "require"}
    sp = SP.parse(dict(sp.raw, choices_guard="five_tuple"), sha256=sorted(RD.REGISTERED_BEFORE)[0])
    assert AS.assemble(sp, state)["guards"] == {"rederived": "report", "choices": "five_tuple"}


def test_record_carries_the_rederivation(tmp_path, monkeypatch):
    """``api.record`` hands ``record.build`` the re-derivation as checked (its released hold included)."""
    state = tmp_path / "state"
    state.mkdir()
    cells = _cells("report")
    RD.release(state, cells, by="orchestrator", why="ruled")
    (state / "inputs.json").write_text("{}")
    (state / "code.json").write_text(json.dumps({"commit": "a" * 40}))
    got = {}
    monkeypatch.setattr(api, "load_state", lambda _s: SP.parse(_d3_spec(tmp_path, _banked(GAMES)).raw, sha256=""))
    monkeypatch.setattr(api, "assemble", lambda _s: cells)
    monkeypatch.setattr(api, "decide", lambda _s: {})
    monkeypatch.setattr(api, "load_repins", lambda _s: [])
    monkeypatch.setattr(api, "partial_at_stop", lambda *a: None)
    monkeypatch.setattr(api, "build_record", lambda *a, **kw: got.update(kw) or {})
    monkeypatch.setattr(api, "write", lambda *a: None)
    api.record(state)
    assert got["rederived"]["held"] and got["rederived"]["released"]["by"] == "orchestrator"
