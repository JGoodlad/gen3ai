"""A SEARCHED read in ``main.reads`` (S-U, ``designs/endstate/design_search_u_headroom.md`` §6 B3): the spec's
``search`` load option and ``per_game`` groups, the per-game queue (deal by deal, raw before searched), G-PREFIX (a
pair whose protocol diverges before the search's first turn HOLDS the lane and keeps the group inconclusive),
``--max-units`` (stage 0 is the queue's first units) and the search-cost report. No game is played: a stub unit runner
writes each unit's summary and front-end capture."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from main.reads import prefix as PX
from main.reads import runner as RN
from main.reads import search_cost as SC
from main.reads.spec import SpecError, load_stamp, parse

CODE = {"commit": "e05c45eeb" + "0" * 31, "dirty": False, "root": "/tmp/pin"}
START = {"formatid": "gen3ou", "p1": {"name": "Gen3AIAnchor1", "team": "A"}, "p2": {"name": "MetaKakuna1", "team": "B"},
         "seed": [1, 2, 3, 4], "core_obs": {"sides": ["p1"]}}


def _doc(path: str, sha: str, **anchor):
    g = {"opponent": "metamon:Kakuna", "teamsets": ["home72", "forum72"], "unit_seeds": "0", "per_game": True,
         "models": ["pstar", "pstar_s"]}
    g.update(anchor)
    ref = {"kind": "pinned", "path": path, "sha256": sha}
    return {"schema": "gen3_read_spec_v1", "read": "su_demo_2026-10-27",
            "models": {"pstar": {"arm": "pstar", "seed": 2601, "ref": ref},
                       "pstar_s": {"arm": "pstar_s", "seed": 2601, "ref": ref, "load": {"search": "r0s4"}}},
            "anchor": {"su": g},
            "decision": {"rule": "anchor_trajectory", "params": {"on": "pstar_s", "off": "pstar", "anchor_group": "su",
                                                                 "goal_bar": 0.5, "min_unit_frac": 0.9}}}


@pytest.fixture
def model(tmp_path):
    from utils.hashing import sha256_file

    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "model_config.json").write_text("{}")
    z = d / "checkpoint_280000170_steps.zip"
    z.write_bytes(b"pstar")
    return str(z), sha256_file(z)


def test_the_spec_takes_a_search_load_in_a_per_game_group_with_its_raw_twin(model):
    spec = parse(_doc(*model))
    assert spec.models["pstar_s"].load == {"search": "r0s4@10"} and spec.anchor["su"].per_game
    assert load_stamp(spec.models["pstar_s"].load) == "+search=r0s4@10" and load_stamp({}) == ""
    with pytest.raises(SpecError, match="per_game"):
        parse(_doc(*model, per_game=False))
    with pytest.raises(SpecError, match="raw twin"):
        parse(_doc(*model, models=["pstar_s"]))
    doc = _doc(*model)
    doc["models"]["pstar_s"]["load"] = {"search": "r0s4@12x"}
    with pytest.raises(SpecError, match="search"):
        parse(doc)


def _capture(cell: Path, *, diverge_at_turn=None):
    lines = ["|player|p1|Gen3AIAnchor1", "|t:|1700000000", "|turn|1"]
    for t in range(2, 14):
        if t == diverge_at_turn:
            lines.append("|move|p1a: Tyranitar|Earthquake|p2a: Swampert")
        lines += [f"|move|p1a: Tyranitar|Rock Slide|p2a: Swampert|t{t}", f"|t:|{1700000000 + t}", f"|turn|{t}"]
    chunks = [["p1", "\n".join(lines)], ["p2", "\n".join(ln.replace("p1a", "P1A") for ln in lines)]]
    (cell / "capture").mkdir(parents=True, exist_ok=True)
    (cell / "capture" / "battle-gen3ou-1.json").write_text(json.dumps({
        "tag": "battle-gen3ou-1", "seed": START["seed"], "commands": ["START " + json.dumps(START)],
        "chunks": chunks}))


def _fake_unit(played, diverge=None):
    def run(st, u):
        played.append(u.name)
        d = st / "cells" / u.name
        d.mkdir(parents=True)
        (d / "summary.json").write_text(json.dumps({
            "status": "OK", "n": 1, "wins": 1, "ties": 0, "regime_verified_decisions": True, "peer_clean": True,
            "team_source_asymmetry": False}))
        bad = diverge is not None and u.search is not None and (u.teamset, u.game) == diverge[0]
        _capture(d, diverge_at_turn=diverge[1] if bad else None)
        if u.search is not None:
            (d / "search_decisions.jsonl").write_text("\n".join(json.dumps(r) for r in (
                {"turn": 9, "n_legal": 3, "searched": False, "why": "before_turn", "search": "r0s4@10"},
                {"turn": 10, "n_legal": 1, "searched": False, "why": "one_legal", "search": "r0s4@10"},
                {"turn": 10, "n_legal": 4, "searched": True, "deviation": True, "cpu_s": 30.0, "wall_s": 40.0,
                 "world_cpu_s": 24.0, "playout_cpu_s": 7.5, "search": "r0s4@10"},
                {"turn": 11, "n_legal": 4, "searched": True, "deviation": False, "cpu_s": 50.0, "wall_s": 70.0,
                 "world_cpu_s": 36.0, "playout_cpu_s": 16.5, "search": "r0s4@10"})) + "\n")
        return 0
    return run


def _plan(tmp_path, model, monkeypatch, **anchor):
    spec_p = tmp_path / "read_spec.json"
    spec_p.write_text(json.dumps(_doc(*model, **anchor)))
    state = tmp_path / "state"
    RN.plan(spec_p, state, code=CODE, step_of=lambda p: 280000170)
    monkeypatch.setattr(RN, "checkout_code", lambda root=None: CODE)
    return state


def test_the_per_game_queue_plays_deal_by_deal_raw_before_searched_and_max_units_stops_it(
        tmp_path, model, monkeypatch):
    state = _plan(tmp_path, model, monkeypatch)
    played = []
    assert RN.run(state, "anchor", time.time() + 3600, run_unit=_fake_unit(played), sleep=lambda s: None,
                  max_units=6) == 0
    assert played == ["pstar__home72__00__g00", "pstar_s__home72__00__g00", "pstar__forum72__00__g00",
                      "pstar_s__forum72__00__g00", "pstar__home72__00__g01", "pstar_s__home72__00__g01"]
    ev = [json.loads(x) for x in (state / "anchor_supervisor.jsonl").read_text().splitlines()]
    assert ev[-1]["event"] == "max_units" and ev[-1]["started"] == 6
    oks = [e for e in ev if e["event"] == "g_prefix"]
    assert len(oks) == 3 and all(e["ok"] for e in oks)
    st = RN.status(state)["anchor"]["su"]
    assert st == {"pstar": {"counted": 3, "planned": 40, "exhausted": 0}, "pstar_s": {"counted": 3, "planned": 40, "exhausted": 0}}


def test_a_pair_that_diverges_before_the_first_searched_turn_holds_the_lane_and_voids_the_group(
        tmp_path, model, monkeypatch):
    from main.reads import api
    from main.reads import lanectl as LC

    state = _plan(tmp_path, model, monkeypatch)
    played = []
    assert RN.run(state, "anchor", time.time() + 3600, run_unit=_fake_unit(played, diverge=(("forum72", 0), 9)),
                  sleep=lambda s: None) == 0
    assert len(played) == 4, "the lane stopped at the hold: no unit after the diverged pair"
    hold = LC.read_hold(state, "anchor")
    assert hold["by"] == "G-PREFIX" and "forum72:0:0" in hold["why"]
    cells = api.assemble(state)
    grp = cells["anchor"]["su"]
    assert grp["g_prefix"]["mismatches"][0]["deal"] == "forum72:0:0"
    assert any(w.startswith("G-PREFIX: deal forum72:0:0") for w in grp["inconclusive"])
    assert grp["cells"]["pstar"]["home72"]["planned"] == 20 and grp["unplayed"] == 80 - 4


def test_a_divergence_after_the_first_searched_turn_is_the_treatment_not_a_mismatch(tmp_path):
    raw, srch = tmp_path / "r", tmp_path / "s"
    _capture(raw)
    _capture(srch, diverge_at_turn=11)
    res = PX.compare(PX.capture_of(raw), PX.capture_of(srch), 10)
    assert res["ok"] and res["reached"] == {"p1": True, "p2": True}
    _capture(srch, diverge_at_turn=10)          # the line lands BEFORE |turn|10: inside the prefix
    res = PX.compare(PX.capture_of(raw), PX.capture_of(srch), 10)
    assert not res["ok"] and "before turn 10" in res["why"]


def test_the_wall_clock_line_is_not_compared_and_a_start_difference_is(tmp_path):
    a = {"commands": ["START " + json.dumps(START)], "chunks": [["p1", "|t:|1\n|turn|1"], ["p1", "|turn|2"]]}
    b = {"commands": ["START " + json.dumps(START)], "chunks": [["p1", "|t:|99\n|turn|1"], ["p1", "|turn|2"]]}
    assert PX.compare(a, b, 2)["ok"]
    c = {"commands": ["START " + json.dumps(dict(START, seed=[1, 2, 3, 5]))], "chunks": b["chunks"]}
    res = PX.compare(a, c, 2)
    assert not res["ok"] and "seed" in res["why"]


def test_the_search_cost_report_reads_the_decision_logs(tmp_path, model, monkeypatch):
    state = _plan(tmp_path, model, monkeypatch)
    played = []
    RN.run(state, "anchor", time.time() + 3600, run_unit=_fake_unit(played), sleep=lambda s: None, max_units=4)
    rep = SC.state_report(state)
    assert rep["games"] == 2 and rep["searched"] == 4 and rep["excluded"] == 0
    assert rep["searched_decision_cpu_s"]["median"] == 40.0 and rep["searched_decision_wall_s"]["max"] == 70.0
    assert rep["ge2_legal_share"] == round(4 / 6, 4)          # turn >= 10: two of three decisions have >= 2 legal
    assert rep["deviation_rate"] == 0.5
    assert rep["game_search_cpu_min"]["median"] == round(80.0 / 60.0, 3)
    # the split: world building vs playouts, per game, summed over its searched and excluded decisions
    assert rep["game_world_cpu_min"]["median"] == round(60.0 / 60.0, 3)
    assert rep["game_playout_cpu_min"]["median"] == round(24.0 / 60.0, 3)
    assert rep["resize"]["above"] is False
    assert rep["g_prefix"]["su"]["checked"] == 2 and not rep["g_prefix"]["su"]["mismatches"]
