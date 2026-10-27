"""``main.reads.rules`` ``prior_mix_ab`` on constructed cases: the banked control (sha-checked rows), the
game-for-game determinism guard, the VOID branches, each class of the primary, home72's harm margin and D0f's
H3 kill rule."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest

from main.reads import engine as EN
from main.reads import rules as RU

UNF = ["heldout72_hpfix", "forum72", "p72"]
FAM = "home72"
CELLS = {"heldout72_hpfix": "pstar_hpfix", "forum72": "pstar_forum72", "p72": "pstar_p72", "home72": "pstar_home72"}
SEEDS = range(4)
GAMES = 10


def _params(path: str, sha: str) -> Dict[str, Any]:
    return {"arms": {"l025": {"group": "g025", "label": "pstar_l025"}, "l100": {"group": "g100", "label": "pstar_l100"}},
            "primary": "l025",
            "banked": {"path": path, "sha256": sha, "cells": CELLS},
            "guard": {"group": "guard", "label": "pstar"}, "unfamiliar": UNF, "familiar": FAM,
            "heldout": "heldout72_hpfix", "min_unit_frac": 0.9, "home_margin_pp": 3.0, "did_ub_pp": 3.0}


def _games(seed: int, wins: int) -> List[List[Any]]:
    return sorted(([h, i, f"t{seed}_{i}.txt", "win" if i <= wins else "loss", 30 + i]
                   for h in ("ours_challenge",) for i in range(1, GAMES + 1)), key=lambda g: (g[0], g[1]))


def _bank(tmp_path: Path, wins_off: int) -> "tuple[str, str]":
    p = tmp_path / "rows.jsonl"
    with open(p, "w") as f:
        for ts, cell in CELLS.items():
            for s in SEEDS:
                for g in _games(s, wins_off):
                    f.write(json.dumps({"read_cell": cell, "unit_seed": s, "half": g[0], "index": g[1],
                                        "our_team": g[2], "result": g[3], "turns": g[4],
                                        "server_version": X51_SERVER}) + "\n")
        f.write(json.dumps({"read_cell": "toy_home72", "unit_seed": 0, "half": "x", "index": 1, "our_team": "t",
                            "result": "win", "turns": 1}) + "\n")      # another cell: ignored
    return str(p), hashlib.sha256(p.read_bytes()).hexdigest()


def _unit(label: str, ts: str, s: int, wins: int, regime: str, lam: Any, counted: bool = True,
          games: Any = None) -> Dict[str, Any]:
    return {"unit": f"{label}__{ts}__{s:02d}", "label": label, "panel": ts, "unit_seed": s, "counted": counted,
            "wins": wins, "ties": 0, "n": GAMES, "our_regime": regime, "belief_prior_mix": lam, "games": games}


#: X51's rows stamp this playing commit; its engine is E0 (``main.reads.engine.LEGACY_ENGINE``).
X51_SERVER = "ws_frontend[in-process]@63f97a575+rust:cargo:src/rust_sim"


def _group(units: List[Dict[str, Any]], teamsets: List[str], engine: Any = EN.E0) -> Dict[str, Any]:
    return {"units": units, "teamsets": teamsets, "unit_seeds": list(SEEDS), "regime_id": "r" * 16,
            "regime_ids": {ts: "r" * 16 for ts in teamsets}, "engine": engine}


class _Spec:
    class _M:
        def __init__(self, load: Dict[str, Any]) -> None:
            self.load = load

    models = {"pstar_l025": _M({"belief_prior_mix": 0.25}), "pstar_l100": _M({"belief_prior_mix": 1.0}),
              "pstar": _M({})}


def _cells(w025: Dict[str, int], w100: Dict[str, int], wins_off: int, *, guard_wins: int = None,
           counted: bool = True) -> Dict[str, Any]:
    ts_all = UNF + [FAM]
    g025 = [_unit("pstar_l025", ts, s, w025[ts], "greedy+belief_prior_mix=0.25", 0.25, counted) for ts in ts_all
            for s in SEEDS]
    g100 = [_unit("pstar_l100", ts, s, w100[ts], "greedy+belief_prior_mix=1", 1.0) for ts in ts_all for s in SEEDS]
    gw = wins_off if guard_wins is None else guard_wins
    guard = [_unit("pstar", ts, 0, gw, "greedy", None, games=_games(0, gw)) for ts in ts_all]
    return {"anchor": {"g025": _group(g025, ts_all), "g100": _group(g100, ts_all), "guard": _group(guard, ts_all)}}


def _read(tmp_path, w025, w100, wins_off=5, **kw):
    path, sha = _bank(tmp_path, wins_off)
    p = _params(path, sha)
    doc = RU.inputs("prior_mix_ab", _cells(w025, w100, wins_off, **kw), p, _Spec())
    return RU.decide("prior_mix_ab", json.loads(json.dumps(doc)), p), doc


def test_the_banked_control_is_sha_checked_and_reads_only_its_cells(tmp_path):
    path, sha = _bank(tmp_path, 5)
    b = RU.banked_units({"path": path, "sha256": sha, "cells": CELLS})
    assert set(b) == set(CELLS) and set(b["p72"]) == set(SEEDS) and b["p72"][0]["n"] == GAMES
    assert b["p72"][0]["wins"] == 5 and b["p72"][0]["games"] == _games(0, 5)
    with pytest.raises(RU.RuleInputError, match="sha256"):
        RU.banked_units({"path": path, "sha256": "0" * 64, "cells": CELLS})


def test_a_guard_that_does_not_replay_the_banked_games_voids_the_read(tmp_path):
    same = {ts: 5 for ts in UNF + [FAM]}
    dec, _ = _read(tmp_path, same, same, guard_wins=6)
    assert dec["decision"]["verdict"] == RU.VOID
    assert any("determinism broken" in r for r in dec["decision"]["reasons"])


def test_an_under_counted_arm_voids_the_read(tmp_path):
    same = {ts: 5 for ts in UNF + [FAM]}
    dec, _ = _read(tmp_path, same, same, counted=False)
    assert dec["decision"]["verdict"] == RU.VOID and any("units counted" in r for r in dec["decision"]["reasons"])


def _spread(tmp_path, w025, w100, jitter100=True):
    """Per-unit wins vary by seed (a paired interval needs within-panel variance); ``jitter100`` False plays the
    λ = 1 arm exactly at its stated wins."""
    path, sha = _bank(tmp_path, 5)
    p = _params(path, sha)
    cells = _cells({ts: 0 for ts in w025}, {ts: 0 for ts in w100}, 5)
    for g, w, j in (("g025", w025, True), ("g100", w100, jitter100)):
        for u in cells["anchor"][g]["units"]:
            u["wins"] = max(0, min(GAMES, w[u["panel"]] + ((1 if u["unit_seed"] % 2 else -1) if j else 0)))
    doc = RU.inputs("prior_mix_ab", cells, p, _Spec())
    return RU.decide("prior_mix_ab", json.loads(json.dumps(doc)), p)


def test_a_clear_unfamiliar_gain_with_home72_kept_is_prior_helps(tmp_path):
    dec = _spread(tmp_path, {"heldout72_hpfix": 8, "forum72": 8, "p72": 8, FAM: 5},
                  {"heldout72_hpfix": 5, "forum72": 5, "p72": 5, FAM: 2})
    d = dec["decision"]["arms"]
    assert d["l025"]["verdict"] == RU.PM_HELPS and d["l025"]["u_class"] == "GAIN" and d["l025"]["d0f"] is None
    assert dec["arms"]["l025"]["delta_u"]["d"] == pytest.approx(0.3)
    # λ = 1: U null, home72 lost -> the DiD is +0.3 (> +3 pp): H3 NOT killed
    assert d["l100"]["verdict"] == RU.PM_ND and d["l100"]["home_harmed"] is True
    assert d["l100"]["d0f"]["did"] == pytest.approx(0.3) and d["l100"]["d0f"]["h3_killed"] is False
    assert dec["decision"]["verdict"] == "lambda=0.25: PRIOR-HELPS | lambda=1: NOT DETECTED, D0f H3 NOT KILLED"


def test_a_gain_that_costs_home72_more_than_the_margin_is_at_a_cost(tmp_path):
    dec = _spread(tmp_path, {"heldout72_hpfix": 8, "forum72": 8, "p72": 8, FAM: 4},
                  {"heldout72_hpfix": 5, "forum72": 5, "p72": 5, FAM: 5})
    assert dec["decision"]["arms"]["l025"]["verdict"] == RU.PM_HELPS_COST
    assert dec["decision"]["arms"]["l025"]["home_harmed"] is True        # Δ̂_home = -10 pp < -3 pp


def test_a_clear_unfamiliar_loss_is_prior_hurts_and_a_null_d0f_kills_h3(tmp_path):
    dec = _spread(tmp_path, {"heldout72_hpfix": 2, "forum72": 2, "p72": 2, FAM: 5},
                  {"heldout72_hpfix": 5, "forum72": 5, "p72": 5, FAM: 5}, jitter100=False)
    d = dec["decision"]["arms"]
    assert d["l025"]["verdict"] == RU.PM_HURTS
    assert d["l100"]["d0f"]["did"] == pytest.approx(0.0) and d["l100"]["d0f"]["h3_killed"] is True


def test_a_banked_control_on_another_engine_refuses(tmp_path):
    """REVIEW F F-M2: the belief-prior-mix arms played at 4ee655648 (E1) against X51's banked rows at 63f97a575 (E0).
    Under the engine rule the arms, or the guard, on another engine than the bank's refuse before any number."""
    path, sha = _bank(tmp_path, 5)
    p = _params(path, sha)
    for group in ("g025", "guard"):
        same = {ts: 5 for ts in UNF + [FAM]}
        cells = _cells(same, same, 5)
        cells["anchor"][group]["engine"] = EN.E1
        with pytest.raises(RU.RuleInputError, match="never crosses engines"):
            RU.inputs("prior_mix_ab", cells, p, _Spec())
