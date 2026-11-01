"""``main.reads`` rule ``deviation_live`` (the deviation-only live read): the spec-load check, the raw-arm and quota VOID
reasons, and GO / VOID through ``live_branch.read`` on the synthetic raw and branch states of
``main.value_search.live_branch_test``. No game."""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from main.reads import rules as RU
from main.reads import rules_deviation_live as DL
from main.reads.spec import SpecError, parse
from main.value_search import live_branch as LB
from main.value_search.live_branch_test import ENGINE, PANELS, _play, _raw_state, _scored

SHA = "ab"


def _params(tmp_path, **over: Any) -> Dict[str, Any]:
    p = {"label": "pstar", "u_group": "u", "home_group": "home", "u_panels": ["heldout72_hpfix"],
         "home_panel": "home72", "quota": {"heldout72_hpfix": 6, "home72": 6}, "bar": 0.0,
         "bootstrap": {"B": 1000, "seed": 20261028}, "rule8_eps": 0.0005, "input_log": "10-30", "per_game": 2,
         "search": "r0s4_rolls@10", "max_drop_frac": 0.1, "selected": str(tmp_path / "selected.jsonl"),
         "scored": str(tmp_path / "scored.jsonl"), "branch_state": str(tmp_path / "br"), "engine": ENGINE}
    p.update(over)
    return p


def _spec(tmp_path, **over: Any) -> Dict[str, Any]:
    ref = {"kind": "pinned", "path": "/x/checkpoint_280000170_steps.zip", "sha256": "a" * 64}
    load = over.pop("load", {"input_log": "10-30"})
    gu = {"opponent": "metamon:Kakuna", "teamsets": ["heldout72_hpfix"], "unit_seeds": "200-241", "models": ["pstar"]}
    gh = {"opponent": "metamon:Kakuna", "teamsets": ["home72"], "unit_seeds": "200-222", "models": ["pstar"]}
    gu.update(over.pop("u_over", {}))
    return {"schema": "gen3_read_spec_v1", "read": "dlive_demo_2026-10-28",
            "models": {"pstar": {"arm": "pstar", "seed": 2601, "ref": ref, "load": load}},
            "anchor": {"u": gu, "home": gh}, "decision": {"rule": "deviation_live", "params": _params(tmp_path, **over)}}


def test_the_spec_check_accepts_the_registered_params(tmp_path) -> None:
    spec = parse(_spec(tmp_path))
    assert spec.rule == "deviation_live"


@pytest.mark.parametrize("over, needle", [
    ({"bogus": 1}, "unknown 'bogus'"),
    ({"u_group": "nope"}, "is not an anchor group"),
    ({"input_log": "10-20"}, "keeps input logs"),
    ({"quota": {"heldout72_hpfix": 6}}, "quota"),
    ({"search": "r0s4_rolls"}, "canonically"),
    ({"search": "nope@10"}, "search"),
    ({"max_drop_frac": 0.9}, "max_drop_frac"),
    ({"u_over": {"teamsets": ["heldout72_hpfix", "home72"]}}, "must play exactly"),
    ({"u_over": {"per_game": True}}, "not per_game"),
])
def test_the_spec_check_refuses(tmp_path, over, needle) -> None:
    with pytest.raises(SpecError) as e:
        parse(_spec(tmp_path, **over))
    assert needle in str(e.value)


def _world(tmp_path, *, branch: str = "win") -> List[Dict[str, Any]]:
    raw, sel = _raw_state(tmp_path / "raw")
    sc = _scored(sel)
    (tmp_path / "selected.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sel))
    (tmp_path / "scored.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sc))
    LB.plan_branches(sel, sc, {"heldout72_hpfix": 6, "home72": 6}, [raw], tmp_path / "br")
    _play(tmp_path / "br", y={"branch": branch})
    return [{"unit": f"pstar__{ts}__200", "label": "pstar", "panel": ts, "unit_seed": 200, "counted": True,
             "our_regime": "greedy", "search": None, "model_sha256": SHA, "engine": ENGINE} for ts in PANELS]


def _read(tmp_path, units, **over) -> Dict[str, Any]:
    p = _params(tmp_path, **over)
    g = {"engine": ENGINE, "regime_id": "r" * 16, "unit_seeds": [200], "unplayed": 0}
    cells = {"anchor": {"u": dict(g, units=[u for u in units if u["panel"] != "home72"], teamsets=["heldout72_hpfix"]),
                        "home": dict(g, units=[u for u in units if u["panel"] == "home72"], teamsets=["home72"])}}
    spec = SimpleNamespace(models={"pstar": SimpleNamespace(ref=SimpleNamespace(sha256=SHA))})
    doc = json.loads(json.dumps(RU.inputs("deviation_live", cells, p, spec)))
    return RU.decide("deviation_live", doc, p)


def test_go_through_the_branch_reader_with_its_artefacts_recorded(tmp_path) -> None:
    res = _read(tmp_path, _world(tmp_path))
    assert res["verdict"] == "GO" and res["decision"]["class"] == "GO"
    assert RU.RULES["deviation_live"].verdict_of(res).startswith("GO: pooled-U mean D")
    assert res["artefacts"]["branch_plan"]["sha256"] and res["quota"]["heldout72_hpfix"]["deviations"] == 6


@pytest.mark.parametrize("case, needle", [
    ("other_engine", "not the registered"),
    ("search_stamp", "raw arm"),
    ("uncounted", "not a counted cell"),
    ("quota", "quota"),
])
def test_each_raw_void_reason(tmp_path, case, needle) -> None:
    units = _world(tmp_path)
    over: Dict[str, Any] = {}
    if case == "other_engine":
        for u in units:
            u["engine"] = "port=c/1;env=d/2"
    elif case == "search_stamp":
        units[0]["search"] = "r0s4@10"
    elif case == "uncounted":
        units[0]["counted"] = False
    elif case == "quota":
        over["quota"] = {"heldout72_hpfix": 2, "home72": 6}
    res = _read(tmp_path, units, **over)
    assert res["verdict"] == DL.VOID and any(needle in w for w in res["void"]), res["void"]
    assert RU.RULES["deviation_live"].verdict_of(res).startswith("VOID: ")


def test_a_branch_tooth_voids_through_the_rule(tmp_path) -> None:
    units = _world(tmp_path)
    br = tmp_path / "br" / "cells"
    one = sorted(p for p in br.iterdir() if "__branch__" in p.name)[0]
    (one / "branch.jsonl").write_text(json.dumps({"g_branch": "fail", "forced": 3, "mismatch": ["log_sha256"]}) + "\n")
    res = _read(tmp_path, units)
    assert res["verdict"] == DL.VOID and any("G-BRANCH" in w for w in res["void"])


def test_a_missing_artefact_refuses_the_inputs(tmp_path) -> None:
    with pytest.raises(RU.RuleInputError, match="missing"):
        _read(tmp_path, [])


# ------------------------------------------------------------------- AM-DL-PEER (the PREREG's blind amendment)
def _classify(tmp_path, extra: List[Dict[str, Any]]) -> None:
    """The classifier's output for the branch state: every null's live attempt ``identical``, plus ``extra``."""
    rows = [{"unit": m["unit"], "attempt": m["unit"], "kind": m["kind"], "class": "identical", "peer_drop": False}
            for m in (json.loads(x) for x in (tmp_path / "br" / "branch_plan.jsonl").read_text().splitlines())
            if m["kind"] == "null"]
    rows = [r for r in rows if r["unit"] not in {e["unit"] for e in extra if e["attempt"] == e["unit"]}] + extra
    (tmp_path / "br" / DL.PEER_DROPS_NAME).write_text("".join(json.dumps(r) + "\n" for r in rows))


def _peer_failed_first_attempt(tmp_path) -> str:
    """One branch whose FIRST attempt failed G-BRANCH (a moved-aside partial) and whose second counted."""
    br = tmp_path / "br" / "cells"
    one = sorted(p for p in br.iterdir() if "__branch__" in p.name)[0]
    part = br / f"{one.name}.partial.1"
    part.mkdir()
    (part / "branch.jsonl").write_text(json.dumps({"g_branch": "fail", "forced": 3, "mismatch": ["log_sha256"]}) + "\n")
    return one.name


@pytest.mark.parametrize("cls, peer, verdict", [("peer_first", True, "GO"), ("ours_first", False, DL.VOID),
                                                ("unattributable", False, DL.VOID)])
def test_a_peer_first_g_branch_failure_is_a_peer_drop_and_any_other_still_voids(tmp_path, cls, peer, verdict) -> None:
    units = _world(tmp_path)
    unit = _peer_failed_first_attempt(tmp_path)
    assert _read(tmp_path, units)["verdict"] == DL.VOID          # no classification: as registered
    _classify(tmp_path, [{"unit": unit, "attempt": f"{unit}.partial.1", "kind": "branch", "class": cls,
                          "peer_drop": peer}])
    res = _read(tmp_path, units)
    assert res["verdict"] == verdict, res.get("void")
    assert res["artefacts"]["peer_drops"]["sha256"] if verdict == "GO" else True
    panel = next(r["panel"] for r in RU.inputs("deviation_live", *_inputs_args(tmp_path, units))["branches"]
                 if r["unit"] == unit)
    rep = res["peer_drops"]["per_panel"][panel]
    assert rep["g_branch_peer_units"] == ([unit] if peer else [])
    assert rep["branch_peer_drops"] == 0                          # its second attempt counted: not a drop


def _inputs_args(tmp_path, units):
    p = _params(tmp_path)
    g = {"engine": ENGINE, "regime_id": "r" * 16, "unit_seeds": [200], "unplayed": 0}
    cells = {"anchor": {"u": dict(g, units=[u for u in units if u["panel"] != "home72"], teamsets=["heldout72_hpfix"]),
                        "home": dict(g, units=[u for u in units if u["panel"] == "home72"], teamsets=["home72"])}}
    spec = SimpleNamespace(models={"pstar": SimpleNamespace(ref=SimpleNamespace(sha256=SHA))})
    return cells, p, spec


def test_a_null_that_changed_its_game_peer_first_is_dropped_not_void(tmp_path) -> None:
    units = _world(tmp_path)
    nulls = [json.loads(x) for x in (tmp_path / "br" / "branch_plan.jsonl").read_text().splitlines()]
    null = next(m for m in nulls if m["kind"] == "null")
    cap = tmp_path / "br" / "cells" / null["unit"] / "capture" / "battle-gen3ou-1.json"
    doc = json.loads(cap.read_text())
    doc["chunks"][-1][1] += "\n|move|late"                      # the null's game changed after its branch turn
    cap.write_text(json.dumps(doc))
    assert _read(tmp_path, units)["verdict"] == DL.VOID          # as registered
    _classify(tmp_path, [{"unit": null["unit"], "attempt": null["unit"], "kind": "null", "class": "peer_first_late",
                          "peer_drop": True}])
    res = _read(tmp_path, units)
    assert res["verdict"] == "GO", res.get("void")
    assert res["peer_drops"]["per_panel"][null["panel"]]["null_peer_drops"] == 1
    assert res["peer_drops"]["pooled"]["null_peer_drops"] == 1


def test_an_unclassified_null_refuses_the_inputs(tmp_path) -> None:
    units = _world(tmp_path)
    (tmp_path / "br" / DL.PEER_DROPS_NAME).write_text("")
    with pytest.raises(RU.RuleInputError, match="not classified"):
        _read(tmp_path, units)
