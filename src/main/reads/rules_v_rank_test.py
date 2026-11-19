"""``main.reads`` rule ``v_rank`` (V-RANK): the spec-load check, the raw-arm VOID reasons, and the classes / VOID
through ``v_rank.read`` on the synthetic raw and branch states of ``main.value_search.v_rank_test``. No game."""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from main.reads import rules as RU
from main.reads import rules_v_rank as VR
from main.reads.spec import SpecError, parse
from main.value_search import v_rank as V
from main.value_search.v_rank_test import ENGINE, GAMES, MODEL_SHA as SHA, PANELS, _play, _raw_state, _scored


def _params(tmp_path, **over: Any) -> Dict[str, Any]:
    p = {"label": "pstar", "u_groups": ["u"], "home_groups": ["home"], "u_panels": ["heldout72_hpfix"],
         "home_panel": "home72", "look": 2, "null_bar": 0.25, "margin": 0.25,
         "bootstrap": {"B": 1000, "seed": 20261027}, "rule8_eps": 0.0005, "input_log": "10-30",
         "search": "r0s4_honest@10", "referee": {"preset": "oracle_leaf8", "seeds": 8}, "max_drop_frac": 0.1,
         "selected": [str(tmp_path / "selected.jsonl")], "scored": [str(tmp_path / "scored")],
         "branch_states": [str(tmp_path / "br")], "engine": ENGINE}
    p.update(over)
    return p


def _spec(tmp_path, **over: Any) -> Dict[str, Any]:
    ref = {"kind": "pinned", "path": "/x/checkpoint_280000170_steps.zip", "sha256": "a" * 64}
    load = over.pop("load", {"input_log": "10-30"})
    gu = {"opponent": "metamon:Kakuna", "teamsets": ["heldout72_hpfix"], "unit_seeds": "2000-2004",
          "models": ["pstar"]}
    gh = {"opponent": "metamon:Kakuna", "teamsets": ["home72"], "unit_seeds": "2100-2114", "models": ["pstar"]}
    gu.update(over.pop("u_over", {}))
    return {"schema": "gen3_read_spec_v1", "read": "v_rank_demo_2026-11-18",
            "models": {"pstar": {"arm": "pstar", "seed": 2601, "ref": ref, "load": load}},
            "anchor": {"u": gu, "home": gh}, "decision": {"rule": "v_rank", "params": _params(tmp_path, **over)}}


def test_the_spec_check_accepts_the_registered_params(tmp_path) -> None:
    assert parse(_spec(tmp_path)).rule == "v_rank"


@pytest.mark.parametrize("over, needle", [
    ({"bogus": 1}, "unknown 'bogus'"),
    ({"u_groups": ["nope"]}, "is not an anchor group"),
    ({"home_groups": ["u"]}, "no group in both"),
    ({"input_log": "10-20"}, "keeps input logs"),
    ({"look": 3}, "look: 1 or 2"),
    ({"null_bar": 0}, "null_bar"),
    ({"margin": 1.5}, "margin"),
    ({"bootstrap": {"B": 10, "seed": 1}}, "bootstrap"),
    ({"search": "r0s4_honest"}, "canonically"),
    ({"search": "nope@10"}, "search"),
    ({"referee": {"preset": "oracle_leaf8", "seeds": 4}}, "referee"),
    ({"max_drop_frac": 0.9}, "max_drop_frac"),
    ({"scored": "one/path"}, "scored"),
    ({"u_over": {"teamsets": ["heldout72_hpfix", "home72"]}}, "must play exactly"),
    ({"u_over": {"per_game": True}}, "not per_game"),
    ({"load": {"input_log": "10-30", "search": "r0s4_honest@10"}}, "RAW"),
])
def test_the_spec_check_refuses(tmp_path, over, needle) -> None:
    with pytest.raises(SpecError) as e:
        parse(_spec(tmp_path, **over))
    assert needle in str(e.value)


def _world(tmp_path, results: Dict[int, str], **play: Any) -> List[Dict[str, Any]]:
    raw, sel = _raw_state(tmp_path / "raw")
    sc = _scored(sel)
    (tmp_path / "selected.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sel))
    (tmp_path / "scored").mkdir()
    for i in range(2):                            # two scorers' part files
        part = [r for r in sc if V.in_part(r, (i, 2))]
        (tmp_path / "scored" / f"part_{i}of2.jsonl").write_text("".join(json.dumps(r) + "\n" for r in part))
    V.plan_branches(sel, sc, [raw], tmp_path / "br", 2)
    for s in V.shard_states(tmp_path / "br"):
        _play(s, results, **play)
    assert V.main(["classify", *[x for s in V.shard_states(tmp_path / "br") for x in ("--branch-state", str(s))]]) == 0
    return [{"unit": f"pstar__{ts}__200", "label": "pstar", "panel": ts, "unit_seed": 200, "counted": True,
             "our_regime": "greedy", "search": None, "model_sha256": SHA, "engine": ENGINE} for ts in PANELS]


def _read(tmp_path, units, **over) -> Dict[str, Any]:
    p = _params(tmp_path, **over)
    g = {"engine": ENGINE, "regime_id": "r" * 16, "unit_seeds": [200], "unplayed": 0}
    cells = {"anchor": {"u": dict(g, units=[u for u in units if u["panel"] != "home72"], teamsets=["heldout72_hpfix"]),
                        "home": dict(g, units=[u for u in units if u["panel"] == "home72"], teamsets=["home72"])}}
    spec = SimpleNamespace(models={"pstar": SimpleNamespace(ref=SimpleNamespace(sha256=SHA))})
    doc = json.loads(json.dumps(RU.inputs("v_rank", cells, p, spec)))       # the record stores it as JSON
    return RU.decide("v_rank", doc, p)


def test_holds_through_the_reader_with_its_artefacts_recorded(tmp_path) -> None:
    res = _read(tmp_path, _world(tmp_path, {3: "win", 6: "loss"}))
    assert res["verdict"] == "RANK-HOLDS-ON-U" == res["decision"]["class"]
    assert RU.RULES["v_rank"].verdict_of(res).startswith("RANK-HOLDS-ON-U (look 2): β_U +2.500 [")
    assert res["positions"] == {p: 2 * GAMES for p in PANELS} and res["coverage"]["home72"]["raw_units_counted"] == 1
    arts = res["artefacts"]
    assert set(arts["scored"][str(tmp_path / "scored")]) == {"part_0of2.jsonl", "part_1of2.jsonl"}
    assert all(set(v) == {"plan.json", "branch_plan.jsonl", "peer_drops.jsonl"} for v in arts["branch_states"].values())
    assert len(arts["branch_states"]) == 2 and res["peer_drops"]["pooled"]["nulls"] == 2
    look1 = _read(tmp_path, [dict(u) for u in _units()], look=1)
    assert look1["verdict"] == "CONTINUE" and look1["delta_class"] is None


def _units() -> List[Dict[str, Any]]:
    return [{"unit": f"pstar__{ts}__200", "label": "pstar", "panel": ts, "unit_seed": 200, "counted": True,
             "our_regime": "greedy", "search": None, "model_sha256": SHA, "engine": ENGINE} for ts in PANELS]


def test_a_value_with_no_ranking_reads_null_at_look_1_with_the_delta_class_beside_it(tmp_path) -> None:
    res = _read(tmp_path, _world(tmp_path, {3: "loss", 6: "loss"}), look=1)
    assert res["verdict"] == "RANK-NULL-ON-U" and res["delta_class"] == "UNDECIDED"
    assert "Δβ's own class UNDECIDED" in res["decision"]["verdict"]


@pytest.mark.parametrize("case, needle", [
    ("other_engine", "not the registered"),
    ("search_stamp", "raw arm"),
    ("other_bytes", "raw arm"),
    ("scored_other_net", "scored on other bytes"),
    ("uncounted", "not a counted cell"),
    ("bare_unit", "with no selected state"),
])
def test_each_raw_void_reason(tmp_path, case, needle) -> None:
    units = _world(tmp_path, {3: "win", 6: "loss"})
    if case == "other_engine":
        for u in units:
            u["engine"] = "port=c/1;env=d/2"
    elif case == "search_stamp":
        units[0]["search"] = "r0s4@10"
    elif case == "other_bytes":
        units[0]["model_sha256"] = "cd"
    elif case == "scored_other_net":                       # the scorer was pointed at another zip than the raw arm's
        f = tmp_path / "scored" / "part_0of2.jsonl"
        rows = [dict(json.loads(x), model_sha256="cd") for x in f.read_text().splitlines()]
        f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    elif case == "uncounted":
        units[0]["counted"] = False
    elif case == "bare_unit":
        units.append(dict(units[0], unit="pstar__heldout72_hpfix__201", unit_seed=201))
    res = _read(tmp_path, units)
    assert res["verdict"] == VR.VOID and any(needle in w for w in res["void"]), res["void"]
    assert RU.RULES["v_rank"].verdict_of(res).startswith("VOID: ")


def test_a_branch_tooth_voids_through_the_rule_and_a_peer_first_failure_is_a_drop(tmp_path) -> None:
    units = _world(tmp_path, {3: "win", 6: "loss"})
    s0 = V.shard_states(tmp_path / "br")[0]
    one = sorted(p for p in (s0 / "cells").iterdir() if "__branch__" in p.name)[0]
    part = s0 / "cells" / f"{one.name}.partial.1"                 # its FIRST attempt failed G-BRANCH; the second counted
    part.mkdir()
    (part / "branch.jsonl").write_text(json.dumps({"g_branch": "fail", "forced": 3, "mismatch": ["log_sha256"]}) + "\n")
    res = _read(tmp_path, units)
    assert res["verdict"] == VR.VOID and any(w.startswith("G-BRANCH") for w in res["void"])
    with open(s0 / V.PEER_DROPS_NAME, "a") as f:
        f.write(json.dumps({"unit": one.name, "attempt": part.name, "kind": "branch", "class": "peer_first",
                            "peer_drop": True}) + "\n")
    res = _read(tmp_path, units)
    assert res["verdict"] == "RANK-HOLDS-ON-U", res.get("void")
    panel = "home72" if "home72" in one.name else "heldout72_hpfix"
    assert res["peer_drops"]["per_panel"][panel]["g_branch_peer_units"] == [one.name]


def test_missing_artefacts_and_an_unclassified_shard_refuse_the_inputs(tmp_path) -> None:
    with pytest.raises(RU.RuleInputError, match="missing"):
        _read(tmp_path, [])
    units = _world(tmp_path, {3: "win", 6: "loss"})
    s0 = V.shard_states(tmp_path / "br")[0]
    (s0 / V.PEER_DROPS_NAME).write_text("")
    with pytest.raises(RU.RuleInputError, match="not classified"):
        _read(tmp_path, units)
    (s0 / V.PEER_DROPS_NAME).unlink()
    with pytest.raises(RU.RuleInputError, match="peer_drops.jsonl: missing"):
        _read(tmp_path, units)
    (tmp_path / "scored" / "part_1of2.jsonl").unlink()
    (tmp_path / "scored" / "part_0of2.jsonl").unlink()
    with pytest.raises(RU.RuleInputError, match="no scored part file"):
        _read(tmp_path, units)
