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
