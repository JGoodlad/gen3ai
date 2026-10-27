"""``main.reads.rules`` ``premium_recovery`` (the H-CTX read, amended): the exact KILL / SUPPORT boundaries on the
recovered share of the premium, INCONCLUSIVE on an unresolved premium, the VOID branches (engine, regime, counts),
the deterministic unit bootstrap, the cross-engine secondary line and the parameter check."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from main.reads import rules as RU

GAMES = 20
SEEDS = list(range(10))
CELLS = {"heldout72_hpfix": "pstar_hpfix", "forum72": "pstar_forum72", "p72": "pstar_p72", "home72": "pstar_home72"}
ENGINE = "port=aaaaaaaaaaaaaaaa/10;env=bbbbbbbbbbbbbbbb/20"


def _bank(tmp_path: Path) -> "tuple[str, str]":
    p = tmp_path / "rows.jsonl"
    with open(p, "w") as f:
        for ts, cell in CELLS.items():
            for s in SEEDS:
                for i in range(1, GAMES + 1):
                    f.write(json.dumps({"read_cell": cell, "unit_seed": s, "half": "ours_challenge", "index": i,
                                        "our_team": f"t{s}_{i}.txt", "result": "win" if i <= 10 else "loss",
                                        "turns": 30 + i}) + "\n")
    return str(p), hashlib.sha256(p.read_bytes()).hexdigest()


def _params(path: str, sha: str, **over: Any) -> Dict[str, Any]:
    p = {"label": "pstar",
         "arms": {"rc": {"panel": "rc72", "groups": ["rc_a", "rc_b"]},
                  "familiar": {"panel": "home72", "groups": ["home_a"]},
                  "unfamiliar": {"panel": "forum72", "groups": ["f_a"]}},
         "kill_frac": 0.75, "support_frac": 0.25, "min_unit_frac": 0.75, "bootstrap": {"B": 2000, "seed": 7},
         "secondary": {"kill_at": 0.57, "support_at": 0.52, "familiar": "home72",
                       "unfamiliar": ["heldout72_hpfix", "forum72", "p72"],
                       "banked": {"path": path, "sha256": sha, "cells": CELLS}},
         "verdict_text": {"KILL": "H-CTX KILLED", "SUPPORT": "H-CTX SUPPORTED", "UNDECIDED": "H-CTX UNDECIDED",
                          "INCONCLUSIVE": "H-CTX INCONCLUSIVE"}}
    p.update(over)
    return p


def _group(ts: str, wins: List[int], seeds: List[int], *, engine: Optional[str] = ENGINE, counted: bool = True,
           regime: str = "greedy") -> Dict[str, Any]:
    units = [{"unit": f"pstar__{ts}__{s:02d}", "label": "pstar", "panel": ts, "unit_seed": s, "counted": counted,
              "wins": w, "ties": 0, "n": GAMES, "our_regime": regime, "games": None} for s, w in zip(seeds, wins)]
    return {"units": units, "teamsets": [ts], "unit_seeds": seeds, "regime_id": "r" * 16, "engine": engine}


def _jitter(mean_wins: int, k: int) -> List[int]:
    """``k`` unit scores whose mean is exactly ``mean_wins`` (± 2 alternating), so the bootstrap has spread."""
    return [mean_wins + (2 if i % 2 else -2) for i in range(k)]


def _read(tmp_path: Path, rc: int, fam: int, unf: int, *, engines: Optional[Dict[str, Optional[str]]] = None,
          **kw: Any) -> Dict[str, Any]:
    path, sha = _bank(tmp_path)
    p = _params(path, sha)
    e = engines or {}
    cells = {"anchor": {
        "rc_a": _group("rc72", _jitter(rc, 10), SEEDS, engine=e.get("rc_a", ENGINE), **kw),
        "rc_b": _group("rc72", _jitter(rc, 10), [s + 10 for s in SEEDS], engine=e.get("rc_b", ENGINE)),
        "home_a": _group("home72", _jitter(fam, 10), SEEDS),
        "f_a": _group("forum72", _jitter(unf, 10), SEEDS)}}
    doc = RU.inputs("premium_recovery", cells, p, object())
    return RU.decide("premium_recovery", json.loads(json.dumps(doc)), p)


@pytest.mark.parametrize("rc, band", [
    (11, RU.PR_UNDECIDED),   # premium 16 - 8 = 8 wins / unit; recovery 3 = 0.375 of it
    (14, RU.PR_KILL),        # recovery 6 = 0.75 exactly: >= kill_frac
    (13, RU.PR_UNDECIDED),   # 0.625
    (10, RU.PR_SUPPORT),     # 2 = 0.25 exactly: <= support_frac
    (6, RU.PR_SUPPORT),      # below the unfamiliar panel
])
def test_the_recovered_share_is_banded_exactly(tmp_path, rc, band):
    dec = _read(tmp_path, rc, 16, 8)
    assert dec["decision"]["band"] == band, dec["decision"]
    assert dec["engine"] == ENGINE
    assert dec["estimates"]["premium"] == pytest.approx(0.4)
    assert dec["decision"]["verdict"].startswith({RU.PR_KILL: "H-CTX KILLED", RU.PR_SUPPORT: "H-CTX SUPPORTED",
                                                  RU.PR_UNDECIDED: "H-CTX UNDECIDED"}[band])


def test_an_unresolved_premium_is_inconclusive(tmp_path):
    dec = _read(tmp_path, 10, 10, 10)
    assert dec["decision"]["band"] == RU.PR_INCONCLUSIVE and "not resolved" in dec["decision"]["verdict"]


def test_the_bootstrap_is_deterministic_and_brackets_the_point(tmp_path):
    a, b = _read(tmp_path, 12, 16, 8), _read(tmp_path, 12, 16, 8)
    assert a["estimates"]["boot95"] == b["estimates"]["boot95"]
    lo, hi = a["estimates"]["boot95"]["premium"]
    assert lo < a["estimates"]["premium"] < hi


@pytest.mark.parametrize("engines, why", [
    ({"rc_b": "port=cccccccccccccccc/10;env=bbbbbbbbbbbbbbbb/20"}, "ONE known engine"),
    ({"rc_a": None}, "ONE known engine"),
])
def test_two_engines_or_an_unknown_one_void_the_read(tmp_path, engines, why):
    dec = _read(tmp_path, 14, 16, 8, engines=engines)
    assert dec["decision"]["verdict"] == RU.VOID and any(why in r for r in dec["decision"]["reasons"])


def test_a_wrong_regime_or_too_few_units_void_the_read(tmp_path):
    dec = _read(tmp_path, 14, 16, 8, regime="greedy+belief_prior_mix=1")
    assert dec["decision"]["verdict"] == RU.VOID and any("not 'greedy'" in r for r in dec["decision"]["reasons"])
    dec = _read(tmp_path, 14, 16, 8, counted=False)        # rc_a's 10 of 20 planned: 50 % < 75 %
    assert dec["decision"]["verdict"] == RU.VOID and any("units counted" in r for r in dec["decision"]["reasons"])


def test_the_secondary_line_is_cross_engine_and_read_by_no_verdict(tmp_path):
    dec = _read(tmp_path, 12, 16, 8)                      # rc72 0.60 >= 0.57 by the absolute bar
    sec = dec["secondary_cross_engine"]
    assert sec["band"] == RU.PR_KILL and sec["x51_familiar"] == pytest.approx(0.5)
    assert dec["decision"]["band"] == RU.PR_UNDECIDED and "cross-engine" in dec["decision"]["verdict"]


class _A:
    def __init__(self, models: List[str], teamsets: List[str]) -> None:
        self.models, self.teamsets = models, teamsets


def test_the_params_check_names_each_defect():
    anchor = {"rc_a": _A(["pstar"], ["rc72"]), "rc_b": _A(["pstar"], ["rc72"]), "home_a": _A(["pstar"], ["home72"]),
              "f_a": _A(["pstar"], ["forum72"])}
    models = {"pstar": object()}
    ok = _params("x", "0" * 64)
    chk = RU.RULES["premium_recovery"].check_params
    assert chk(ok, models=models, anchor=anchor, s={}) == []
    wrong = dict(anchor, home_a=_A(["pstar"], ["forum72"]))
    assert any("must play exactly" in m for m in chk(ok, models=models, anchor=wrong, s={}))
    assert any("support_frac < kill_frac" in m for m in chk(_params("x", "0" * 64, kill_frac=0.2), models=models,
                                                            anchor=anchor, s={}))
    assert any("bootstrap" in m for m in chk(_params("x", "0" * 64, bootstrap={"B": 10, "seed": 1}), models=models,
                                             anchor=anchor, s={}))
