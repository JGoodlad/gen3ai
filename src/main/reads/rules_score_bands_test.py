"""``main.reads.rules`` ``score_bands`` (the H-CTX rc72 read) on constructed cases: the exact band boundaries, the
game-for-game guard and the other VOID branches, the parameter check, and the descriptive comparisons."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from main.reads import rules as RU

UNF = ["heldout72_hpfix", "forum72", "p72"]
FAM = "home72"
CELLS = {"heldout72_hpfix": "pstar_hpfix", "forum72": "pstar_forum72", "p72": "pstar_p72", "home72": "pstar_home72"}
SEEDS = range(4)
GAMES = 25          # 4 units x 25 = 100 games: a band boundary of 0.57 / 0.52 is an integer count of wins
BANK_WINS = {"heldout72_hpfix": 12, "forum72": 13, "p72": 11, "home72": 15}


def _params(path: str, sha: str, **over: Any) -> Dict[str, Any]:
    p = {"group": "rc", "label": "pstar", "panel": "rc72", "kill_at": 0.57, "support_at": 0.52,
         "min_unit_frac": 0.9, "banked": {"path": path, "sha256": sha, "cells": CELLS},
         "guard": {"group": "guard", "label": "pstar"}, "familiar": FAM, "unfamiliar": UNF,
         "verdict_text": {"KILL": "H-CTX KILLED", "SUPPORT": "H-CTX SUPPORTED", "UNDECIDED": "H-CTX UNDECIDED"}}
    p.update(over)
    return p


def _games(seed: int, wins: int) -> List[List[Any]]:
    return sorted(([h, i, f"t{seed}_{i}.txt", "win" if i <= wins else "loss", 30 + i]
                   for h in ("ours_challenge",) for i in range(1, GAMES + 1)), key=lambda g: (g[0], g[1]))


def _bank(tmp_path: Path) -> "tuple[str, str]":
    p = tmp_path / "rows.jsonl"
    with open(p, "w") as f:
        for ts, cell in CELLS.items():
            for s in SEEDS:
                for g in _games(s, BANK_WINS[ts]):
                    f.write(json.dumps({"read_cell": cell, "unit_seed": s, "half": g[0], "index": g[1],
                                        "our_team": g[2], "result": g[3], "turns": g[4]}) + "\n")
    return str(p), hashlib.sha256(p.read_bytes()).hexdigest()


def _unit(ts: str, s: int, wins: int, *, counted: bool = True, regime: str = "greedy", games: Any = None
          ) -> Dict[str, Any]:
    return {"unit": f"pstar__{ts}__{s:02d}", "label": "pstar", "panel": ts, "unit_seed": s, "counted": counted,
            "wins": wins, "ties": 0, "n": GAMES, "our_regime": regime, "games": games}


def _group(units: List[Dict[str, Any]], teamsets: List[str], seeds: Any = SEEDS) -> Dict[str, Any]:
    return {"units": units, "teamsets": teamsets, "unit_seeds": list(seeds), "regime_id": "r" * 16,
            "regime_ids": {ts: "r" * 16 for ts in teamsets}}


class _Spec:
    pass


def _read(tmp_path: Path, per_unit: List[int], *, guard_delta: int = 0, counted: bool = True,
          regime: str = "greedy") -> Dict[str, Any]:
    path, sha = _bank(tmp_path)
    p = _params(path, sha)
    rc = [_unit("rc72", s, w, counted=counted, regime=regime) for s, w in zip(SEEDS, per_unit)]
    guard = [_unit(ts, 1, BANK_WINS[ts] + guard_delta, games=_games(1, BANK_WINS[ts] + guard_delta))
             for ts in (FAM, "forum72")]
    cells = {"anchor": {"rc": _group(rc, ["rc72"]), "guard": _group(guard, [FAM, "forum72"], [1])}}
    doc = RU.inputs("score_bands", cells, p, _Spec())
    return RU.decide("score_bands", json.loads(json.dumps(doc)), p)


@pytest.mark.parametrize("wins, band", [
    ([15, 14, 14, 14], RU.SB_KILL),          # 57 / 100 = 0.57 exactly: >= kill_at
    ([14, 14, 14, 14], RU.SB_UNDECIDED),     # 0.56
    ([13, 13, 13, 14], RU.SB_UNDECIDED),     # 0.53
    ([13, 13, 13, 13], RU.SB_SUPPORT),       # 0.52 exactly: <= support_at
    ([10, 11, 12, 13], RU.SB_SUPPORT),
])
def test_the_band_is_read_exactly_at_its_boundaries(tmp_path, wins, band):
    dec = _read(tmp_path, wins)["decision"]
    assert dec["band"] == band
    assert dec["verdict"].startswith({RU.SB_KILL: "H-CTX KILLED", RU.SB_SUPPORT: "H-CTX SUPPORTED",
                                      RU.SB_UNDECIDED: "H-CTX UNDECIDED"}[band])
    assert f"{sum(wins)}/100" in dec["verdict"]


def test_a_guard_that_does_not_replay_the_banked_games_voids_the_read(tmp_path):
    dec = _read(tmp_path, [15, 15, 15, 15], guard_delta=1)
    assert dec["decision"]["verdict"] == RU.VOID
    assert any("determinism broken" in r for r in dec["decision"]["reasons"])
    assert dec["score"]["n"] == 100, "the score is still reported beside the VOID"


def test_under_counted_units_or_a_wrong_regime_void_the_read(tmp_path):
    dec = _read(tmp_path, [15, 15, 15, 15], counted=False)
    assert dec["decision"]["verdict"] == RU.VOID and "no game counted" in dec["decision"]["reasons"]
    dec = _read(tmp_path, [15, 15, 15, 15], regime="greedy+belief_prior_mix=1")
    assert dec["decision"]["verdict"] == RU.VOID and any("not 'greedy'" in r for r in dec["decision"]["reasons"])


def test_the_descriptive_comparisons_use_the_banked_cells(tmp_path):
    s = _read(tmp_path, [14, 14, 14, 14])["score"]
    assert s["banked_familiar"]["score"] == pytest.approx(0.60)
    assert s["banked_unfamiliar"]["mean"] == pytest.approx(0.48)
    assert s["minus_familiar"]["d"] == pytest.approx(-0.04) and s["minus_unfamiliar"]["d"] == pytest.approx(0.08)
    assert s["premium_kept"] == pytest.approx(0.08 / 0.12)
    assert s["minus_familiar"]["lo"] < -0.04 < s["minus_familiar"]["hi"]
    assert s["dispersion"]["phi"] == pytest.approx(0.0) and s["dispersion"]["units"] == 4


class _A:
    def __init__(self, models: List[str], teamsets: List[str]) -> None:
        self.models, self.teamsets = models, teamsets


class _M:
    def __init__(self, load: Dict[str, Any]) -> None:
        self.load = load


def test_the_params_check_names_each_defect():
    anchor = {"rc": _A(["pstar"], ["rc72"]), "guard": _A(["pstar"], [FAM])}
    models = {"pstar": _M({})}
    ok = _params("x.jsonl", "0" * 64)
    assert RU.RULES["score_bands"].check_params(ok, models=models, anchor=anchor, s={}) == []
    bad = _params("x.jsonl", "0" * 64, kill_at=0.5, support_at=0.52)
    assert any("support_at < kill_at" in m for m in RU.RULES["score_bands"].check_params(
        bad, models=models, anchor=anchor, s={}))
    two = {"rc": _A(["pstar"], ["rc72", FAM]), "guard": _A(["pstar"], [FAM])}
    assert any("exactly" in m for m in RU.RULES["score_bands"].check_params(ok, models=models, anchor=two, s={}))
    off = {"rc": _A(["pstar"], ["rc72"]), "guard": _A(["pstar"], ["rc72"])}
    assert any("banked cell" in m for m in RU.RULES["score_bands"].check_params(ok, models=models, anchor=off, s={}))
