"""``main.reads.rules_x59s`` (X59-S's floor pilot and full read, PREREG Amendment 4) on constructed cells: the floor's
exact boundary, VOID 6's game-for-game repeat, the pilot releasing only its boolean, VOID 3 / 5 / 7, FR-2's gate,
the pilot's floor overriding the full read's, and the spec-level parameter checks."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from main.reads import rules as RU
from main.reads import rules_x59s as X5
from main.reads.spec import SpecError, parse

PANELS = ["heldout72_hpfix", "forum72", "p72"]
ARMS = {"S-H": ["s_h_1", "s_h_2"], "S-P": ["s_p_1", "s_p_2"], "S-S": ["s_s_1", "s_s_2"]}
G = 80                       # games a cell: 12 / 80 = 0.15 exactly (the floor bar)


def _games(w: int, n: int = 20) -> List[List[Any]]:
    return [["ours_challenge", i, f"t{i}.txt", "win" if i <= w else "loss", 30] for i in range(1, n + 1)]


def _group(rates: Dict[str, int], seeds: List[int], *, short: Dict[str, int] = None,
           games: Dict[str, Any] = None) -> Dict[str, Any]:
    """``rates``: label -> wins per cell of G games; ``short``: label/panel -> units lost."""
    short = short or {}
    cells, units = {}, []
    for lab, w in rates.items():
        cells[lab] = {}
        for ts in PANELS:
            lost = short.get(f"{lab}/{ts}", 0)
            cells[lab][ts] = {"wins": w, "ties": 0, "n": G, "units": len(seeds) - lost, "planned": len(seeds),
                              "skipped": []}
            for s in seeds:
                units.append({"unit": f"{lab}__{ts}__{s:02d}", "label": lab, "panel": ts, "unit_seed": s,
                              "counted": True, "wins": w // len(seeds), "n": G // len(seeds),
                              "games": (games or {}).get(f"{lab}/{ts}/{s}", _games(w // len(seeds)))})
    return {"cells": cells, "units": units, "teamsets": PANELS, "unit_seeds": seeds, "regime_id": "r" * 16, "engine": "port=e/1;env=e/2"}


def _floor_params(**over: Any) -> Dict[str, Any]:
    p = {"group": "pilot", "panels": PANELS, "arms": ARMS, "floor_bar": 0.15,
         "repeat": {"group": "repeat", "label": "s_h_1_rep", "of": "s_h_1"}}
    p.update(over)
    return p


def _floor(wins: Dict[str, int], *, rep_games: Any = None) -> Dict[str, Any]:
    labs = [lab for v in ARMS.values() for lab in v]
    pilot = _group({lab: wins.get(lab, 12) for lab in labs}, [0, 1, 2, 3])
    same = next(u["games"] for u in pilot["units"] if u["unit"] == "s_h_1__forum72__00")
    rep = _group({"s_h_1_rep": wins.get("s_h_1", 12)}, [0], games=rep_games or {"s_h_1_rep/forum72/0": same})
    rep["teamsets"] = ["forum72"]
    rep["units"] = [u for u in rep["units"] if u["panel"] == "forum72"]
    cells = {"anchor": {"pilot": pilot, "repeat": rep}}
    p = _floor_params()
    doc = RU.inputs("x59s_floor", cells, p, None)
    return RU.decide("x59s_floor", json.loads(json.dumps(doc)), p)


def test_the_floor_fires_strictly_below_the_bar():
    labs = [lab for v in ARMS.values() for lab in v]
    assert _floor({lab: 11 for lab in labs})["floor"] is True                    # 0.1375 everywhere
    assert _floor({lab: 12 for lab in labs})["floor"] is False                   # 0.15 exactly: does not fire
    one_arm_up = {lab: 11 for lab in labs} | {"s_s_1": 14, "s_s_2": 13}          # S-S mean 0.169
    assert _floor(one_arm_up)["floor"] is False


def test_the_pilot_releases_only_its_boolean():
    labs = [lab for v in ARMS.values() for lab in v]
    dec = _floor({lab: 11 for lab in labs})
    assert set(dec) == {"floor", "short_cells", "repeat_pass", "decision"}
    assert dec["decision"] == {"verdict": X5.FLOOR_FIRES}


def test_a_repeat_that_differs_game_for_game_is_void_6():
    labs = [lab for v in ARMS.values() for lab in v]
    bad = {"s_h_1_rep/forum72/0": _games(2)[:-1] + [["ours_challenge", 20, "t20.txt", "win", 31]]}
    dec = _floor({lab: 11 for lab in labs}, rep_games=bad)
    assert dec["decision"]["verdict"] == X5.VOID and "VOID 6" in dec["decision"]["reasons"][0]
    assert dec["floor"] is None


# ------------------------------------------------------------------------------------------------ full read
def _write(tmp: Path, name: str, doc: Any) -> str:
    p = tmp / f"{name}.json"
    p.write_text(json.dumps(doc))
    return str(p)


def _x51(tmp: Path) -> Dict[str, Any]:
    p = tmp / "x51.jsonl"
    with open(p, "w") as f:
        for ts in PANELS:
            for s in range(4):
                for g in _games(10):
                    f.write(json.dumps({"read_cell": f"pstar_{ts}", "unit_seed": s, "half": g[0], "index": g[1],
                                        "our_team": g[2], "result": g[3], "turns": g[4]}) + "\n")
    return {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "cells": {ts: f"pstar_{ts}" for ts in PANELS}}


def _artefacts(tmp: Path, *, fr2: Any = "NO-ARTEFACT", identity: bool = True, uninf: Any = None,
               void3: List[str] = (), fit_last: float = 1.2995, share: float = 0.0, ih_env: str = "env=e/2"
               ) -> Dict[str, str]:
    labs = [lab for v in ARMS.values() for lab in v]
    yd = {"s_h": 1.4, "s_p": 1.3, "s_s": 1.25}
    stud = {"void3": list(void3), "students": {lab: {
        "yardsticks": {"selfplay_val_pstar_labels": yd[lab[:3]], "human_val_human_labels": 1.2},
        "val_hist": [[100, 1.5], [200, 1.3], [300, fit_last]], "val_rose_last_epoch": False} for lab in labs}}
    gate = {"uninformative": uninf or {"delta_data": False, "delta_labels": False, "delta_states": False},
            "continue": True}
    return {"students": _write(tmp, "students", stud),
            "floor_refs": _write(tmp, "floor", {"rates": {"random": 0.02, "aggressive": 0.05}, "uncounted": [],
                                                 "engines": ["port=e/1;env=e/2"]}),
            "inhouse": _write(tmp, "inhouse", {"short": [], "core_stamps": [ih_env],
                                               "rates": {"delta_data": [0.55, 0.56], "delta_labels": [0.52, 0.53]}}),
            "fr2": _write(tmp, "fr2", {"class": fr2}),
            "identity": _write(tmp, "identity", {"pass": identity, "unit": "pstar__home72__00"}),
            "gate_record": _write(tmp, "gate", {"decision": {"gate": gate, "engine": "port=e/1;env=e/2",
                                                             "decision": {"verdict": "x"}}}),
            "m3_reencode": _write(tmp, "m3", {"share_differing": share})}


def _full(tmp: Path, wins: Dict[str, int], *, short: Dict[str, int] = None, **art: Any) -> Dict[str, Any]:
    labs = [lab for v in ARMS.values() for lab in v]
    w = {lab: wins.get(lab, 20) for lab in labs}
    cells = {"anchor": {"pilot": _group(w, [0, 1, 2, 3], short=short), "rest": _group(w, [4, 5, 6, 7])}}
    p = {"groups": ["pilot", "rest"], "panels": PANELS, "arms": ARMS, "artefacts": _artefacts(tmp, **art),
         "x51": _x51(tmp)}
    doc = RU.inputs("x59s", cells, p, None)
    return RU.decide("x59s", json.loads(json.dumps(doc)), p)


def test_the_full_read_wires_the_registered_verdict(tmp_path):
    dec = _full(tmp_path, {"s_h_1": 40, "s_h_2": 41})
    assert dec["decision"]["verdict"] != X5.VOID
    assert set(dec["decision"]["classes"]) == {"delta_data", "delta_labels"}
    assert dec["fr2"] == "NO-ARTEFACT" and dec["verdict_detail"]["sp_learned_its_teacher"]
    # the student-to-P* gap: every student unit at 5 / 20 = 0.25 (wins 20 of 80 over 4 units) against P*'s 10 / 20
    gap = dec["descriptive"]["student_to_pstar_gap_secondary_cross_engine"]
    assert gap["S-S"]["delta"] == pytest.approx(-0.25)
    assert set(dec["descriptive"]["per_panel"]) == set(PANELS)


def test_the_gates_booleans_make_a_contrast_uninformative(tmp_path):
    dec = _full(tmp_path, {}, uninf={"delta_data": True, "delta_labels": False, "delta_states": False})
    assert dec["decision"]["classes"]["delta_data"] == "UNINFORMATIVE"
    assert "UNINFORMATIVE (both arms at the floor)" in dec["decision"]["verdict"]


def test_the_closure_needs_s_h_fitted(tmp_path):
    w = {"s_s_1": 60, "s_s_2": 61, "s_p_1": 60, "s_p_2": 61}                   # S-S far above S-H
    fitted = _full(tmp_path, w, fit_last=1.2995)                                # last-quarter drop 0.0005
    unfit = _full(tmp_path, w, fit_last=1.25)                                   # still falling 0.05
    assert fitted["sh_fit_drop"] < 0.01 < unfit["sh_fit_drop"]


@pytest.mark.parametrize("kw, why", [
    ({"fr2": None}, "FR-2"),
    ({"identity": False}, "VOID 5"),
    ({"void3": ["s_h_1: 7000 steps"]}, "VOID 3"),
    ({"short": {"s_h_1/forum72": 2, "s_s_2/p72": 2}}, "VOID 7"),
    ({"ih_env": "env=other/9"}, "in-house"),
])
def test_each_void_refuses_a_verdict(tmp_path, kw, why):
    dec = _full(tmp_path, {}, **kw)
    assert dec["decision"]["verdict"] == X5.VOID and any(why in r for r in dec["decision"]["reasons"])


def test_one_short_cell_is_reported_not_void(tmp_path):
    dec = _full(tmp_path, {}, short={"s_h_1/forum72": 2})
    assert dec["decision"]["verdict"] != X5.VOID and dec["short_cells"] == ["s_h_1/forum72"]


# ------------------------------------------------------------------------------------------- spec checks
def _spec(tmp: Path, rep_sha: str, rest_seeds: str = "4-19") -> Dict[str, Any]:
    sha = "a" * 64
    models = {lab: {"arm": a, "seed": i + 1, "ref": {"kind": "pinned", "path": f"/x/{lab}.zip", "sha256": sha}}
              for a, labs in ARMS.items() for i, lab in enumerate(labs)}
    models["s_h_1_rep"] = {"arm": "S-H", "seed": 1, "ref": {"kind": "pinned", "path": "/x/s_h_1.zip",
                                                             "sha256": rep_sha}}
    labs = [lab for v in ARMS.values() for lab in v]
    return {"schema": "gen3_read_spec_v1", "read": "x59s_test", "models": models,
            "anchor": {"repeat": {"opponent": "metamon:Kakuna", "teamsets": ["forum72"], "unit_seeds": "0",
                                  "models": ["s_h_1_rep"]},
                       "pilot": {"opponent": "metamon:Kakuna", "teamsets": PANELS, "unit_seeds": "0-3",
                                 "models": labs}},
            "decision": {"rule": "x59s_floor", "params": _floor_params()}}


def test_the_floor_spec_checks_the_repeat_pins_the_same_bytes(tmp_path):
    parse(_spec(tmp_path, "a" * 64))
    with pytest.raises(SpecError, match="SAME bytes"):
        parse(_spec(tmp_path, "b" * 64))


def test_the_full_spec_refuses_overlapping_groups(tmp_path):
    d = _spec(tmp_path, "a" * 64)
    labs = [lab for v in ARMS.values() for lab in v]
    d["anchor"]["rest"] = {"opponent": "metamon:Kakuna", "teamsets": PANELS, "unit_seeds": "3-19", "models": labs}
    d["decision"] = {"rule": "x59s", "params": {"groups": ["pilot", "rest"], "panels": PANELS, "arms": ARMS,
                                                "artefacts": {k: "x.json" for k in X5.ARTEFACTS},
                                                "x51": {"path": "r", "sha256": "s",
                                                        "cells": {ts: ts for ts in PANELS}}}}
    with pytest.raises(SpecError, match="disjoint"):
        parse(d)
    d["anchor"]["rest"]["unit_seeds"] = "4-19"
    parse(d)


# ------------------------------------------------------------------------------------------------ the gate
def _gate(wins: Dict[str, int], *, rep_games: Any = None) -> Dict[str, Any]:
    labs = [lab for v in ARMS.values() for lab in v]
    pilot = _group({lab: wins.get(lab, 12) for lab in labs}, [0, 1, 2, 3])
    same = next(u["games"] for u in pilot["units"] if u["unit"] == "s_h_1__forum72__00")
    rep = _group({"s_h_1_rep": wins.get("s_h_1", 12)}, [0], games=rep_games or {"s_h_1_rep/forum72/0": same})
    rep["teamsets"] = ["forum72"]
    rep["units"] = [u for u in rep["units"] if u["panel"] == "forum72"]
    cells = {"anchor": {"pilot": pilot, "repeat": rep}}
    return cells


def test_the_gate_releases_only_per_contrast_booleans(tmp_path):
    fl = _write(tmp_path, "floor", {"rates": {"random": 0.0, "aggressive": 0.0458}, "uncounted": [],
                                    "engines": ["port=e/1;env=e/2"]})
    p = {"group": "pilot", "panels": PANELS, "arms": ARMS, "floor_refs": fl, "band": 0.05,
         "repeat": {"group": "repeat", "label": "s_h_1_rep", "of": "s_h_1"}}
    # S-H and S-P at 6 / 80 = 0.075 (<= 0.0958), S-S at 16 / 80 = 0.20
    cells = _gate({"s_h_1": 6, "s_h_2": 6, "s_p_1": 6, "s_p_2": 6, "s_s_1": 16, "s_s_2": 16})
    dec = RU.decide("x59s_gate", json.loads(json.dumps(RU.inputs("x59s_gate", cells, p, None))), p)
    assert dec["gate"] == {"uninformative": {"delta_data": False, "delta_labels": True, "delta_states": False},
                           "continue": True}
    assert set(dec) == {"gate", "short_cells", "repeat_pass", "engine", "decision"}
    assert dec["decision"]["verdict"].startswith("CONTINUE")
    cells = _gate({lab: 6 for v in ARMS.values() for lab in v})
    dec = RU.decide("x59s_gate", json.loads(json.dumps(RU.inputs("x59s_gate", cells, p, None))), p)
    assert dec["gate"]["continue"] is False and dec["decision"]["verdict"].startswith("STOP")
