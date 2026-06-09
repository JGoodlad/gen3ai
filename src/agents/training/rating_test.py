"""Tests for the rating-model seam (``agents.training.rating``).

The load-bearing test: ``BradleyTerryRating`` over records from ``eval_rows_to_match_records`` must
reproduce the live ``elo`` anchored fit BIT-FOR-BIT (ratings + SE) — so the seam is a true drop-in,
not a re-implementation that could drift. Plus protocol conformance and the draw convention.
"""
import pytest

from agents.training import elo
from agents.training.rating import (
    BradleyTerryRating, MatchRecord, RatingModel, RatingResult, eval_rows_to_match_records,
)


def _rows():
    return [
        elo.EvalRow(1000, 100, {"random": 0.90, "heuristic": 0.40, "staller": 0.30}, []),
        elo.EvalRow(2000, 100, {"random": 0.95, "heuristic": 0.55, "staller": 0.45}, [(1000, 0.60)]),
        elo.EvalRow(3000, 100, {"random": 0.97, "heuristic": 0.62, "staller": 0.50},
                    [(2000, 0.55), (1000, 0.66)]),
    ]


def test_bradley_terry_matches_live_elo_fit_exactly():
    rows = _rows()
    name_pins = {"random": 1000.0, "heuristic": 1400.0, "staller": 1200.0}   # fit_elo wants names
    key_pins = {elo.bot_key(n): v for n, v in name_pins.items()}             # seam wants player keys

    live = elo.fit_elo(rows, pin_ratings=name_pins)
    res = BradleyTerryRating().fit(eval_rows_to_match_records(rows), pinned=key_pins)

    assert isinstance(res, RatingResult)
    keys = set(live.ratings) | set(res.ratings)
    assert keys == set(live.ratings) == set(res.ratings)
    for k in keys:
        assert res.ratings[k] == pytest.approx(live.ratings[k], abs=1e-9), k
        assert res.uncertainty[k] == pytest.approx(live.se[k], abs=1e-9), k


def test_bradley_terry_is_a_rating_model():
    assert isinstance(BradleyTerryRating(), RatingModel)


def test_eval_rows_to_match_records_mirrors_elo_lowering():
    rows = _rows()
    recs = eval_rows_to_match_records(rows)
    got = sorted((r.player_a, r.player_b, r.n_won_a, r.n_games) for r in recs)
    want = sorted(elo._rows_to_results(rows))
    assert got == want
    # period_id is the cycle step (the natural Glicko-2 rating period)
    assert all(r.period_id in (1000, 2000, 3000) for r in recs)


def test_rating_for_accessor():
    res = BradleyTerryRating().fit(
        eval_rows_to_match_records(_rows()),
        pinned={elo.bot_key("random"): 1000.0})
    assert res.rating_for("missing:player") is None
    rr = res.rating_for(elo.snap_key(3000))
    assert rr is not None and isinstance(rr[0], float)


def test_draws_count_as_half_a_win():
    # A pure-draw record vs a pinned anchor: BT should rate the players equal (a draw ⇒ even).
    m = BradleyTerryRating()
    res = m.fit([MatchRecord("snap:1", "bot:anchor", n_games=100, n_won_a=0, n_drawn=100)],
                pinned={"bot:anchor": 1000.0})
    assert res.ratings["snap:1"] == pytest.approx(1000.0, abs=1.0)
