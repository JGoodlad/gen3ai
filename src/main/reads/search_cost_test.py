"""``main.reads search-cost`` on a hand-built searched cell with an EXCLUSION and a safety-net cut: the search CPU is
every search-path decision's (exclusions included, FINDING SP-2) and splits into searched and excluded; the excluded
decisions' own distribution; the timer replay over the game."""
from __future__ import annotations

import json

from main.reads import search_cost as SC


def _cell(tmp_path):
    cell = tmp_path / "cells" / "pstar_s__forum72__800__g00"
    cell.mkdir(parents=True)
    rows = [{"turn": t, "searched": False, "why": "before_turn", "n_legal": 3, "search": "r0s4_honest_inc@10"}
            for t in range(1, 10)]
    rows += [{"turn": 10, "searched": True, "n_legal": 3, "cpu_s": 6.0, "wall_s": 6.0, "world_cpu_s": 4.0,
              "playout_cpu_s": 2.0, "search": "r0s4_honest_inc@10"},
             {"turn": 11, "searched": False, "why": "excluded", "n_legal": 3, "cpu_s": 54.0, "wall_s": 160.0,
              "world_cpu_s": 54.0, "playout_cpu_s": 0.0, "search": "r0s4_honest_inc@10"},
             {"turn": 12, "searched": True, "why": "deadline", "n_legal": 3, "cpu_s": 12.0, "wall_s": 12.0,
              "world_cpu_s": 9.0, "playout_cpu_s": 3.0, "net": {"cut": True}, "search": "r0s4_honest_inc@10"}]
    (cell / "search_decisions.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (cell / "summary.json").write_text(json.dumps({"status": "OK"}))
    return cell


def test_the_search_cpu_counts_the_exclusions_and_splits_them_out(tmp_path):
    rep = SC.report([_cell(tmp_path)])
    g = rep["per_game"][0]
    assert (g["searched"], g["excluded"], g["net_cut"]) == (2, 1, 1)
    assert g["search_cpu_min"] == round(72.0 / 60, 3) == round(g["searched_cpu_min"] + g["excluded_cpu_min"], 3)
    assert (g["searched_cpu_min"], g["excluded_cpu_min"]) == (round(18.0 / 60, 3), round(54.0 / 60, 3))
    assert rep["game_search_cpu_min"]["median"] == round(72.0 / 60, 3)
    assert rep["game_excluded_cpu_min"]["median"] == round(54.0 / 60, 3)
    assert rep["excluded_decision_cpu_s"]["max"] == 54.0 and rep["excluded_decision_wall_s"]["n"] == 1
    assert rep["searched_decision_cpu_s"]["n"] == 2 and rep["net_cut"] == 1
    assert not any(k in g for k in ("timer_rows", "excluded_cpu_s"))           # pooled, never per game


def test_the_report_replays_the_games_through_the_timer(tmp_path):
    rep = SC.report([_cell(tmp_path)])
    t = rep["timer"]["policies"]
    # the excluded decision's 160-s wall: as played it reaches the 150-s request limit (a forfeit); the bank-aware
    # deadline cuts it at 139 s, leaving a 20-s bank, so the next decision's 12 s passes its 9-s deadline: the cut the
    # record names (``why: deadline``); nothing forfeits
    assert t["as_played"]["forum72"]["counts"]["forfeit"] == 1 and t["as_played"]["U"]["forfeit_share"] == 1.0
    assert t["bank_r10"]["forum72"]["counts"] == {"games": 1, "searched": 2, "forfeit": 0, "cut": 2,
                                                  "cut_searched": 1, "path": 3, "battle_cut": 1}
