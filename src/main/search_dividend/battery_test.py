"""The results file (append / resume), the matched-game derivation, and the fallback fold."""

from __future__ import annotations

import json
import os

import pytest

from main.search_dividend.battery import (Cell, ResultsFile, game_seed, summarize_decisions,
                                          team_pair)
from main.search_dividend.summary import elo_by_arm, per_cell, wilson


def _row(arm="base", budget=0.0, opponent="heuristic", game=0, **kw):
    row = {"v": 1, "arm": arm, "budget": budget, "opponent": opponent, "game": game,
           "result": "win", "finished": 1, "won": 1, "wall_s": 1.0,
           "n_decisions": 10, "n_searched": 8, "n_changed": 3, "fallbacks": {},
           "deadline_truncated": 0, "worlds_gate_failed": 0,
           "realized_mean": {"m_opp": 2.0, "k_worlds": 1.0, "r_dice": 1.0, "arms": 16.0,
                             "elapsed": 0.8}}
    row.update(kw)
    return row


# -- append / resume ----------------------------------------------------------


def test_rows_append_and_resume_picks_up_where_it_stopped(tmp_path):
    path = str(tmp_path / "r.jsonl")
    rf = ResultsFile(path)
    cell = Cell("oracle", 1.0, "heuristic")
    assert rf.done_games(cell) == set()
    for g in (0, 1, 2):
        rf.append(_row(arm="oracle", budget=1.0, game=g))
    assert rf.n_done(cell) == 3

    reopened = ResultsFile(path)                      # a fresh process
    assert reopened.done_games(cell) == {0, 1, 2}
    assert reopened.n_done(Cell("base", 0.0, "heuristic")) == 0


def test_resume_is_keyed_by_the_WHOLE_cell_not_just_the_arm(tmp_path):
    """A budget sweep runs the same arm at four budgets; keying resume on the arm alone would
    make three of the four cells look already-played."""
    path = str(tmp_path / "r.jsonl")
    rf = ResultsFile(path)
    rf.append(_row(arm="honest", budget=0.5, game=0))
    assert rf.n_done(Cell("honest", 0.5, "heuristic")) == 1
    assert rf.n_done(Cell("honest", 1.0, "heuristic")) == 0
    assert rf.n_done(Cell("honest", 0.5, "staller")) == 0


def test_a_truncated_final_line_stops_the_read_rather_than_being_skipped(tmp_path):
    """A row half-written by a kill must not make a SHORT file look complete. Stopping at the bad
    line replays that game; skipping past it would silently drop everything after."""
    path = str(tmp_path / "r.jsonl")
    with open(path, "w") as fh:
        fh.write(json.dumps(_row(game=0)) + "\n")
        fh.write(json.dumps(_row(game=1))[:40])       # killed mid-write
    rf = ResultsFile(path)
    assert rf.done_games(Cell("base", 0.0, "heuristic")) == {0}
    assert len(rf.rows()) == 1


def test_appending_is_durable_and_never_rewrites(tmp_path):
    path = str(tmp_path / "r.jsonl")
    rf = ResultsFile(path)
    rf.append(_row(game=0))
    first = open(path).read()
    rf.append(_row(game=1))
    assert open(path).read().startswith(first), "the file must be append-only"


def test_the_results_dir_is_created_on_demand(tmp_path):
    path = str(tmp_path / "deep" / "nested" / "r.jsonl")
    ResultsFile(path).append(_row())
    assert os.path.exists(path)


# -- matched games ------------------------------------------------------------


def test_the_game_seed_is_independent_of_arm_and_budget():
    """THE property that makes the arms comparable: cell (base,1s,heuristic) game 7 and
    (oracle,3s,heuristic) game 7 are the SAME battle."""
    assert game_seed("heuristic", 7, 42) == game_seed("heuristic", 7, 42)
    assert game_seed("heuristic", 7, 42) != game_seed("heuristic", 8, 42)
    assert game_seed("heuristic", 7, 42) != game_seed("staller", 7, 42)
    assert game_seed("heuristic", 7, 42) != game_seed("heuristic", 7, 43)
    assert game_seed("heuristic", 7, 42).startswith("sodium,")


def test_the_team_draw_is_matched_too():
    """Teams matter as much as dice — the exploiter work measured an 'edge' that was pure
    team-draw and vanished under an equal-pilot mirror."""
    pool = [f"Mon{i}||item|ability|move|Serious|||||" for i in range(20)]
    a = team_pair("heuristic", 3, 99, pool)
    assert a == team_pair("heuristic", 3, 99, pool)
    assert a != team_pair("heuristic", 4, 99, pool)


# -- the decision fold --------------------------------------------------------


def test_fallbacks_are_a_HISTOGRAM_by_reason():
    """'The search fell back' is not a finding; 'every determinized world failed the prefix gate'
    is. A single total would hide which one happened."""
    got = summarize_decisions([
        {"fallback": "prefix_gate_failed"},
        {"fallback": "prefix_gate_failed"},
        {"fallback": "not_move_selection"},
        {"fallback": None, "changed": True, "widths": {"opp_candidates": 3, "worlds_gated_ok": 2,
                                                       "dice": 1, "arms_scored": 24,
                                                       "elapsed_s": 0.9}},
        {"fallback": None, "changed": False, "widths": {"opp_candidates": 1, "worlds_gated_ok": 2,
                                                        "dice": 1, "arms_scored": 8,
                                                        "elapsed_s": 0.4,
                                                        "deadline_truncated": True,
                                                        "worlds_gate_failed": 1}},
    ])
    assert got["fallbacks"] == {"prefix_gate_failed": 2, "not_move_selection": 1}
    assert got["n_decisions"] == 5
    assert got["n_searched"] == 2
    assert got["n_changed"] == 1
    assert got["deadline_truncated"] == 1
    assert got["worlds_gate_failed"] == 1
    assert got["realized_mean"]["m_opp"] == 2.0
    assert got["realized_mean"]["arms"] == 16.0


def test_a_fallback_decision_never_counts_as_a_searched_one():
    """The whole point of counting fallbacks: an arm that fell back on every decision IS the base
    arm, and a change_rate computed over all decisions would hide that."""
    got = summarize_decisions([{"fallback": "deadline"} for _ in range(9)])
    assert got["n_searched"] == 0
    assert got["realized_mean"]["arms"] == 0.0


# -- the report ---------------------------------------------------------------


def test_a_win_rate_is_never_quoted_without_an_interval():
    p, lo, hi = wilson(6, 10)
    assert p == pytest.approx(0.6)
    assert lo < 0.4 and hi > 0.8, "at n=10 the interval must be visibly wide"
    assert wilson(0, 0) == (0.0, 0.0, 0.0)
    assert wilson(10, 10)[2] <= 1.0 and wilson(0, 10)[1] >= 0.0


def test_unfinished_games_are_excluded_from_the_win_rate_not_scored_as_losses():
    """A crash / transport error is never a semantic outcome — the contention lesson, which this
    project has already paid for once (39/40 timeouts reported as a clean PASS)."""
    rows = [_row(game=0, result="win", finished=1, won=1),
            _row(game=1, result="unfinished", finished=0, won=0, error="RuntimeError: boom")]
    (cell,) = per_cell(rows)
    assert cell["games"] == 2
    assert cell["finished"] == 1
    assert cell["errors"] == 1
    assert cell["win_rate"] == 1.0


def test_per_cell_splits_by_arm_budget_and_opponent():
    rows = [_row(arm="base", budget=0.0, opponent="heuristic"),
            _row(arm="oracle", budget=1.0, opponent="heuristic"),
            _row(arm="oracle", budget=1.0, opponent="staller")]
    assert len({(c["arm"], c["budget"], c["opponent"]) for c in per_cell(rows)}) == 3


def test_elo_reports_a_DELTA_vs_base_and_carries_its_caveats():
    rows = ([_row(arm="base", budget=0.0, opponent=o, game=g, won=w, result="win" if w else "loss")
             for o in ("heuristic", "staller") for g, w in enumerate([1, 0, 1, 0])]
            + [_row(arm="oracle", budget=1.0, opponent=o, game=g, won=w,
                    result="win" if w else "loss")
               for o in ("heuristic", "staller") for g, w in enumerate([1, 1, 1, 0])])
    out = elo_by_arm(rows)
    cells = {(c["arm"], c["budget"]): c for c in out["cells"]}
    assert cells[("base", 0.0)]["delta_vs_base"] == 0.0
    assert cells[("oracle", 1.0)]["delta_vs_base"] > 0
    assert out["caveats"], "an ELO must never be published here without its caveats"
