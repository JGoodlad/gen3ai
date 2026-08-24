"""The results file (append / resume), the matched-game derivation, and the fallback fold."""

from __future__ import annotations

import json
import os

import pytest

from main.search_dividend.battery import (Cell, ResultsFile, finalize_row, game_seed,
                                          summarize_decisions, team_pair)
from main.search_dividend.summary import elo_by_arm, mirror_report, per_cell, wilson
from main.search_dividend.__main__ import resolve_side_swap


def _row(arm="base", budget=0.0, opponent="heuristic", game=0, **kw):
    row = {"v": 1, "arm": arm, "budget": budget, "opponent": opponent, "game": game,
           "orientation": 0, "result": "win", "finished": 1, "won": 1, "tied": 0, "wall_s": 1.0,
           "n_decisions": 10, "n_searched": 8, "n_changed": 3, "n_deepened": 0, "fallbacks": {},
           "deadline_truncated": 0, "worlds_gate_failed": 0,
           "realized_mean": {"m_opp": 2.0, "k_worlds": 1.0, "r_dice": 1.0, "arms": 16.0,
                             "elapsed": 0.8, "depth": 1.0, "beam": 0.0}}
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


def test_an_UNFINISHED_game_is_replayed_on_resume_not_counted_done(tmp_path):
    """The battery's unit of account is FINISHED games per cell. Counting a crash row as done
    quietly shrinks a cell's n — measured 2026-08-23, when a pruned worktree killed the bridge
    child mid-battery and 8 straight games recorded as unfinished; a resume that skipped them
    would have reported an honest-arm cell of n=2 posing as n=10."""
    path = str(tmp_path / "r.jsonl")
    rf = ResultsFile(path)
    rf.append(_row(game=0, result="win", finished=1, won=1))
    rf.append(_row(game=1, result="unfinished", finished=0, won=0,
                   error="battle_never_finished: ..."))
    cell = Cell("base", 0.0, "heuristic")
    assert rf.done_games(cell) == {0}
    assert ResultsFile(path).done_games(cell) == {0}, "a reopened file must agree"
    assert len(rf.rows()) == 2, "the evidence row itself is never dropped"


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


# -- side-swap pairing --------------------------------------------------------


def test_orientation_1_SWAPS_the_teams_and_nothing_else():
    """The pair's whole job: identical draw, opposite sides. If orientation changed the draw too,
    the two games would be independent samples and there would be nothing to difference out."""
    pool = [f"Mon{i}||item|ability|move|Serious|||||" for i in range(20)]
    ours, theirs = team_pair("self", 3, 99, pool, 0)
    assert team_pair("self", 3, 99, pool, 1) == (theirs, ours)


def test_both_orientations_of_a_game_share_ONE_pinned_seed():
    """The dice are common to the pair by construction — the seed is derived without orientation,
    so the two orientations of a game are as matched as two different team assignments can be."""
    assert game_seed("self", 5, 42) == game_seed("self", 5, 42)


def test_resume_counts_ORIENTATION_GAMES_so_half_a_finished_pair_is_not_a_finished_pair(tmp_path):
    """Under side-swap a `game` is two battles. Resuming on the game INDEX would skip the missing
    orientation and leave the paired read silently unbalanced — every pair half-played, none of
    them usable, and a `n_pairs` of zero with no explanation."""
    path = str(tmp_path / "r.jsonl")
    rf = ResultsFile(path)
    cell = Cell("oracle", 1.0, "self")
    rf.append(_row(arm="oracle", budget=1.0, opponent="self", game=0, orientation=0))
    assert rf.done_units(cell) == {(0, 0)}
    assert rf.n_done(cell) == 1
    rf.append(_row(arm="oracle", budget=1.0, opponent="self", game=0, orientation=1))
    assert ResultsFile(path).done_units(cell) == {(0, 0), (0, 1)}
    assert ResultsFile(path).done_games(cell) == {0}, "the game-level view still reads"


def test_a_preswap_row_with_no_orientation_field_resumes_as_orientation_zero(tmp_path):
    """The battery is append-only and files outlive schema changes; an old row must keep meaning
    what it meant, or a relaunch silently replays a cell that was already paid for."""
    path = str(tmp_path / "r.jsonl")
    old = _row(game=4)
    old.pop("orientation")
    with open(path, "w") as fh:
        fh.write(json.dumps(old) + "\n")
    assert ResultsFile(path).done_units(Cell("base", 0.0, "heuristic")) == {(4, 0)}


def test_side_swap_defaults_ON_for_the_mirror_and_OFF_against_a_scripted_bot():
    """Against a bot the two sides are not interchangeable, so swapping teams pairs nothing — it
    just plays each draw twice. In a mirror they ARE, which is exactly what makes the pair valid."""
    assert resolve_side_swap(None, "self") is True
    assert resolve_side_swap(None, "heuristic") is False
    assert resolve_side_swap(False, "self") is False, "an explicit flag always wins"
    assert resolve_side_swap(True, "heuristic") is True


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


# -- the outcome-XOR-error invariant ------------------------------------------


def test_a_row_with_no_outcome_and_no_error_is_UNREPRESENTABLE():
    """THE hardening this build was ordered from. A mirror smoke produced a row that recorded no
    win and carried no error — the shape that reads as a played battle to every consumer, dilutes
    the cell's win rate toward whatever it happened to be, and points at no cause at all. It is
    now repaired at the one place rows are made rather than being an expectation about call
    sites."""
    row = finalize_row({"result": None, "finished": 1, "won": 0, "battle_created": True})
    assert row["result"] == "unfinished"
    assert row["finished"] == 0, "an outcome-less game is not a finished game"
    assert "battle_never_finished" in row["error"]


def test_a_battle_that_was_never_CREATED_gets_a_different_name_than_one_that_never_ended():
    """Two failures, two places to look: a bridge child that produced no protocol at all, versus
    one that produced some and then stopped. Folding them would send every reader to the wrong
    half of the pipeline."""
    row = finalize_row({"result": "unfinished", "battle_created": False})
    assert "battle_never_created" in row["error"]


def test_a_real_outcome_is_left_ALONE_including_a_tie():
    for result in ("win", "loss", "tie"):
        row = finalize_row({"result": result, "finished": 1, "won": 0, "error": None})
        assert row["result"] == result and row["error"] is None


def test_an_existing_error_is_never_overwritten_by_the_generic_one():
    """The specific cause outranks the generic one; a `RuntimeError` from the transport tells you
    more than 'no result'."""
    row = finalize_row({"result": "unfinished", "error": "RuntimeError: boom",
                        "battle_created": True})
    assert row["error"] == "RuntimeError: boom"


# -- ties ---------------------------------------------------------------------


def test_a_TIE_is_excluded_from_the_win_rate_denominator_and_reported():
    """A tie scored as a loss is a bias with a direction. Excluding it and printing the count is
    the honest convention — and `decisive` is published so the interval's n is never ambiguous."""
    rows = [_row(game=0, result="win", finished=1, won=1),
            _row(game=1, result="loss", finished=1, won=0),
            _row(game=2, result="tie", finished=1, won=0, tied=1)]
    (cell,) = per_cell(rows)
    assert cell["finished"] == 3 and cell["decisive"] == 2 and cell["tied"] == 1
    assert cell["win_rate"] == 0.5, "1 of 2 DECISIVE games, not 1 of 3"


# -- the mirror report --------------------------------------------------------


def _mirror_rows(pairs, arm="oracle", budget=1.0):
    """``pairs`` = ``[(score_o0, score_o1), ...]`` with 1 = the searched side won."""
    out = []
    for g, (a, b) in enumerate(pairs):
        for o, s in ((0, a), (1, b)):
            if s is None:
                continue
            out.append(_row(arm=arm, budget=budget, opponent="self", game=g, orientation=o,
                            result="win" if s else "loss", finished=1, won=int(s)))
    return out


def test_the_mirror_null_is_stated_as_0_50_and_the_verdict_is_read_off_the_interval():
    """A mirror's no-effect point is 0.50 BY CONSTRUCTION — same network, search off on one side.
    So 'does the interval exclude 0.50' is the whole test, and the report states it rather than
    leaving a reader to eyeball two numbers."""
    out = mirror_report(_mirror_rows([(1, 1)] * 20))
    (cell,) = out["cells"]
    assert out["null"] == 0.5
    assert cell["beats_null"] is True and cell["worse_than_null"] is False

    weak = mirror_report(_mirror_rows([(1, 0)] * 6))
    assert weak["cells"][0]["beats_null"] is False, "a 50% cell must not read as a result"


def test_the_PAIRED_read_differences_out_the_team_draw():
    """The reason side-swap exists. A cell that wins every game on one orientation and loses every
    game on the other has learnt nothing about the search — it has measured which team is better,
    and the paired score says exactly 0.5 while the unpaired win rate also says 0.5 by accident of
    balance. The pair is what makes the reading robust when the orientations are UNBALANCED."""
    out = mirror_report(_mirror_rows([(1, 0)] * 8))
    (cell,) = out["cells"]
    assert cell["n_pairs"] == 8
    assert cell["paired_win_rate"] == 0.5
    assert cell["paired_ci95"] == [0.5, 0.5], "every pair split — zero variance, and honestly so"


def test_a_consistent_search_edge_shows_up_in_BOTH_orientations():
    out = mirror_report(_mirror_rows([(1, 1)] * 10))
    (cell,) = out["cells"]
    assert cell["paired_win_rate"] == 1.0 and cell["n_pairs"] == 10


def test_an_orientation_with_no_partner_is_counted_but_not_paired():
    """A half-played pair must not enter the paired mean — a lone orientation carries the team
    asymmetry it was supposed to cancel."""
    out = mirror_report(_mirror_rows([(1, 1), (1, None), (0, 0)]))
    (cell,) = out["cells"]
    assert cell["n_pairs"] == 2 and cell["unpaired_games"] == 1
    assert cell["finished"] == 5


def test_a_tie_scores_HALF_in_a_pair_because_that_is_the_nulls_prediction():
    rows = [_row(arm="oracle", budget=1.0, opponent="self", game=0, orientation=0,
                 result="tie", finished=1, won=0, tied=1),
            _row(arm="oracle", budget=1.0, opponent="self", game=0, orientation=1,
                 result="tie", finished=1, won=0, tied=1)]
    (cell,) = mirror_report(rows)["cells"]
    assert cell["paired_win_rate"] == 0.5 and cell["ties"] == 2
    assert cell["decisive"] == 0


def test_mirror_cells_are_EXCLUDED_from_the_anchored_elo_fit_with_a_stated_reason():
    """`self` carries no anchor: its rating is the trainee's own, so a mirror cell would enter an
    ANCHORED fit as a free parameter matched against another free parameter and drag every arm it
    shares a node with. The mirror has its own reading and needs no ELO to be one."""
    out = elo_by_arm(_mirror_rows([(1, 0)] * 4))
    assert out["cells"] == []
    assert "mirror" in out["note"]


def test_an_elo_fit_still_works_when_mirror_and_bot_cells_share_a_file():
    """The common case once the flagged mode is used beside the roster: the bots still fit, the
    mirror still reports, and neither reading contaminates the other."""
    rows = (_mirror_rows([(1, 0)] * 4)
            + [_row(arm="base", budget=0.0, opponent=o, game=g, won=w,
                    result="win" if w else "loss")
               for o in ("heuristic", "staller") for g, w in enumerate([1, 0, 1, 0])]
            + [_row(arm="oracle", budget=1.0, opponent=o, game=g, won=w,
                    result="win" if w else "loss")
               for o in ("heuristic", "staller") for g, w in enumerate([1, 1, 1, 0])])
    elo = elo_by_arm(rows)
    assert {(c["arm"], c["budget"]) for c in elo["cells"]} == {("base", 0.0), ("oracle", 1.0)}
    assert mirror_report(rows)["cells"][0]["n_pairs"] == 4


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


# -- the dice-clairvoyance schema break ---------------------------------------


def test_rows_played_under_the_DICE_LEAK_announce_themselves_in_the_report():
    """An append-only file outlives its schema, and here the schema break is a MEANING break.

    Before `ROW_VERSION` 3, dice draw 0 was the sim's own `original` stream — the dice the turn was
    actually about to be resolved with (11 of 12 live decisions reproduced the real turn's protocol
    byte-for-byte). Each arm's score is a mean over R draws, so the leak's share is 1/R and a cell's
    reading tracks its realized `r_dice`; the ORACLE arm, pinned to `k_worlds=1` with dice last in
    `WIDTH_ORDER`, was the only arm that routinely bought R>1 and was the only one to read below the
    mirror null. Those rows stay in the file, so the REPORT is what has to say so — a stale artifact
    that reads like a current one misleads every reader after it.
    """
    from main.search_dividend.battery import ROW_VERSION
    from main.search_dividend.summary import (DICE_LEAK_ROW_VERSION, format_report, leaked_rows)

    assert ROW_VERSION >= DICE_LEAK_ROW_VERSION, "a freshly played row must never look leaked"
    old = [_row(v=2, game=g, won=g % 2, result="win" if g % 2 else "loss") for g in range(4)]
    new = [_row(v=ROW_VERSION, game=g, won=g % 2, result="win" if g % 2 else "loss")
           for g in range(4)]
    assert leaked_rows(old) == 4 and leaked_rows(new) == 0
    assert leaked_rows([{k: v for k, v in r.items() if k != "v"} for r in old]) == 4, \
        "the oldest rows carry no `v` at all and must not read as clean"
    report = format_report(old)
    assert "dice-clairvoyance" in report and "4 of 4 rows" in report
    assert "dice-clairvoyance" not in format_report(new)
