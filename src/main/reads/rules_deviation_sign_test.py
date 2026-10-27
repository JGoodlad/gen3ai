"""``main.reads`` rule ``deviation_sign`` (the S-U gate test): the spec-load check, every VOID reason, and GO / STOP
through ``sign_gate.read`` on synthetic artefacts. No game."""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from main.reads import rules as RU
from main.reads import rules_deviation_sign as DS
from main.reads.spec import SpecError, parse
from main.value_search import sign_gate as SG

SHA = "a" * 64
ENGINE = "port=aaaaaaaaaaaaaaaa/10;env=bbbbbbbbbbbbbbbb/20"
HOME, U = "home72", ["heldout72_hpfix", "forum72", "p72"]
PANELS = [HOME, *U]
SEEDS = range(100, 103)


def _params(tmp_path, **over: Any) -> Dict[str, Any]:
    p = {"group": "gate", "label": "pstar", "home_panel": HOME, "u_panels": list(U), "bar": 0.5,
         "bootstrap": {"B": 2000, "seed": 20261027}, "rule8_eps": 0.0005, "input_log": "10-30", "per_game": 2,
         "selected": str(tmp_path / "selected.jsonl"), "scored": str(tmp_path / "scored.jsonl"), "engine": ENGINE}
    p.update(over)
    return p


def _spec(tmp_path, **over: Any) -> Dict[str, Any]:
    ref = {"kind": "pinned", "path": "/x/checkpoint_280000170_steps.zip", "sha256": SHA}
    load = over.pop("load", {"input_log": "10-30"})
    g = {"opponent": "metamon:Kakuna", "teamsets": PANELS, "unit_seeds": "100-109", "models": ["pstar"]}
    g.update(over.pop("group_over", {}))
    return {"schema": "gen3_read_spec_v1", "read": "su_gate_demo_2026-10-27",
            "models": {"pstar": {"arm": "pstar", "seed": 2601, "ref": ref, "load": load}},
            "anchor": {"gate": g}, "decision": {"rule": "deviation_sign", "params": _params(tmp_path, **over)}}


def test_the_spec_check_accepts_the_registered_params(tmp_path):
    spec = parse(_spec(tmp_path))
    assert spec.rule == "deviation_sign" and spec.models["pstar"].load == {"input_log": "10-30"}


@pytest.mark.parametrize("over, needle", [
    ({"bogus": 1}, "unknown 'bogus'"),
    ({"group": "nope"}, "is not an anchor group"),
    ({"input_log": "10-20"}, "keeps input logs"),
    ({"load": {}}, "keeps input logs"),
    ({"u_panels": ["heldout72_hpfix", "forum72"]}, "must be exactly group"),
    ({"home_panel": "forum72"}, "home_panel not among them"),
    ({"bar": 1.5}, "bar"),
    ({"bootstrap": {"B": 100, "seed": 1}}, "bootstrap"),
    ({"rule8_eps": 0.5}, "rule8_eps"),
    ({"per_game": 0}, "per_game"),
    ({"group_over": {"per_game": True}}, "not per_game"),
])
def test_the_spec_check_refuses(tmp_path, over, needle):
    with pytest.raises(SpecError) as e:
        parse(_spec(tmp_path, **over))
    assert needle in str(e.value)


def test_a_searched_label_cannot_keep_input_logs(tmp_path):
    with pytest.raises(SpecError, match="RAW label"):
        parse(_spec(tmp_path, load={"input_log": "10-30", "search": "r0s4@10"}))


def _units(mutate=None) -> List[Dict[str, Any]]:
    us = [{"unit": f"pstar__{ts}__{s:02d}", "label": "pstar", "panel": ts, "unit_seed": s, "counted": True,
           "our_regime": "greedy", "search": None, "model_sha256": SHA, "engine": ENGINE}
          for s in SEEDS for ts in PANELS]
    if mutate:
        mutate(us)
    return us


def _artefacts(tmp_path, acc: float, *, drop_score: bool = False, extra_sel: bool = False) -> None:
    sel, sc = [], []
    for s in SEEDS:
        for ts in PANELS:
            for g in range(10):
                deal = f"{ts}{s}g{g}"
                for k, (turn, n) in enumerate(((12, 20), (25, 44))):
                    key = f"su_live_v1|{deal}|t{turn}|n{n}"
                    sel.append({"cell": f"pstar__{ts}__{s:02d}", "panel": ts, "deal": deal, "turn": turn, "n": n,
                                "key": key, "select_hash": SG.select_hash(deal, turn, n)})
                    right = (g * 2 + k) % 10 < acc * 10
                    tp, tt = (0.7, 0.3) if right else (0.3, 0.7)
                    sc.append({"panel": ts, "deal": deal, "key": key,
                               "search": {"searched": True, "deviation": True, "played": 1, "cpu_s": 9.0},
                               "referee": {"T_pick": tp, "T_top": tt, "sign": SG.sign_value(tp, tt), "cpu_s": 20.0}})
    if extra_sel:
        sel.append(dict(sel[0], turn=13, n=21, key="su_live_v1|x|t13|n21", select_hash=SG.select_hash(sel[0]["deal"], 13, 21)))
    if drop_score:
        sc = sc[1:]
    (tmp_path / "selected.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sel))
    (tmp_path / "scored.jsonl").write_text("".join(json.dumps(r) + "\n" for r in sc))


def _read(tmp_path, units=None, **over) -> Dict[str, Any]:
    p = _params(tmp_path, **over)
    cells = {"anchor": {"gate": {"units": units or _units(), "engine": ENGINE, "regime_id": "r" * 16,
                                 "teamsets": PANELS, "unit_seeds": list(SEEDS), "unplayed": 0}}}
    spec = SimpleNamespace(models={"pstar": SimpleNamespace(ref=SimpleNamespace(sha256=SHA))})
    doc = json.loads(json.dumps(RU.inputs("deviation_sign", cells, p, spec)))
    return RU.decide("deviation_sign", doc, p)


def test_go_and_stop_through_the_gates_reader(tmp_path):
    _artefacts(tmp_path, 0.9)
    go = _read(tmp_path)
    assert go["verdict"] == "GO" and go["n_states"] == len(SEEDS) * 4 * 20
    assert go["artefacts"]["scored"]["sha256"] and go["engine"] == ENGINE
    _artefacts(tmp_path, 0.5)
    assert _read(tmp_path)["verdict"] == "STOP"


@pytest.mark.parametrize("case, needle", [
    ("two_engines", "span 2 engines"),
    ("other_engine", "not the registered"),
    ("search_stamp", "raw arm"),
    ("uncounted", "not a counted cell"),
    ("unscored", "not scored"),
    ("three_per_game", "more than 2 states"),
])
def test_each_void_reason(tmp_path, case, needle):
    _artefacts(tmp_path, 0.9, drop_score=case == "unscored", extra_sel=case == "three_per_game")
    units = _units()
    if case == "two_engines":
        units[0]["engine"] = "port=c/1;env=d/2"
    elif case == "other_engine":
        for u in units:
            u["engine"] = "port=c/1;env=d/2"
    elif case == "search_stamp":
        units[3]["search"] = "r0s4@10"
    elif case == "uncounted":
        units[0]["counted"] = False
    res = _read(tmp_path, units=units)
    assert res["verdict"] == DS.VOID and any(needle in w for w in res["void"]), res["void"]


def test_a_missing_artefact_refuses_the_inputs(tmp_path):
    with pytest.raises(RU.RuleInputError, match="missing"):
        _read(tmp_path)
