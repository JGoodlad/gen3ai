"""THE LADDER'S TRANSPORT BOUNDARY (P2, 2026-10-06) — pure: no engine, no model.

Every test here FAILS ON REVERT of the boundary: a row's transport is read (absent = ``python_bridge``), a fit over
both transports is REFUSED unless a transport is named or the mix is consented, the recipe block carries the
transport, and two ladders of different transports are refused side by side."""
from __future__ import annotations

import json

import pytest

from agents.training import snapshot_ladder as sl
from agents.training import snapshot_ladder_play as LP
from agents.training import snapshot_ladder_transport as T


def _rust(run, a, b, wins_a, games, draws=0):
    sl._append_game(run, a, b, wins_a, games, play={
        "wins_a": wins_a, "games": games, "draws": draws, "transport": LP.TRANSPORT, "encoder": LP.ENCODER,
        "protocol": LP.PROTOCOL, "mirrored": True})


@pytest.fixture
def no_anchors(monkeypatch):
    monkeypatch.setattr(sl.elo_mod, "load_bot_anchors", lambda: None)
    monkeypatch.setattr(sl.elo_mod, "load_rows", lambda run_dir, source="log": [])


def test_a_row_without_a_transport_key_is_a_PYTHON_row_and_an_unknown_one_is_refused():
    assert sl.row_transport({"a": 1, "b": 2}) == sl.TRANSPORT_PYTHON
    assert sl.row_transport({"transport": "rust_eval"}) == sl.TRANSPORT_RUST
    with pytest.raises(sl.LadderTransportError):
        sl.row_transport({"transport": "carrier_pigeon"})


def test_a_fit_over_BOTH_transports_is_REFUSED(tmp_path, no_anchors, monkeypatch):
    run = str(tmp_path)
    monkeypatch.setattr(sl, "pool_snapshot_steps", lambda d: [100, 200, 300])
    sl._append_game(run, 200, 100, 60, 100)            # the Python ladder's row
    _rust(run, 300, 200, 55, 100)                       # the Rust ladder's row
    with pytest.raises(sl.LadderTransportError, match="TWO transports"):
        sl.fit_ladder(run, write=False)


def test_naming_a_transport_fits_ONLY_its_rows_and_stamps_it(tmp_path, no_anchors, monkeypatch):
    run = str(tmp_path)
    monkeypatch.setattr(sl, "pool_snapshot_steps", lambda d: [100, 200, 300])
    sl._append_game(run, 200, 100, 60, 100)
    _rust(run, 300, 200, 55, 100)
    lad = sl.fit_ladder(run, write=False, transport=sl.TRANSPORT_RUST)
    assert lad["n_frozen_pairs_measured"] == 1 and set(lad["ratings"]) == {"200", "300"}
    assert lad["recipe"]["transport"] == sl.TRANSPORT_RUST and sl.ladder_transport(lad) == sl.TRANSPORT_RUST
    assert sl.recipe_status(lad)[0] == "current"         # the transport is NOT a recipe change
    mixed = sl.fit_ladder(run, write=False, allow_mixed_transport=True)
    assert mixed["n_frozen_pairs_measured"] == 2 and mixed["recipe"]["transport"] == sl.TRANSPORT_MIXED


def test_a_single_transport_fit_stamps_that_transport(tmp_path, no_anchors, monkeypatch):
    run = str(tmp_path)
    monkeypatch.setattr(sl, "pool_snapshot_steps", lambda d: [100, 200])
    sl._append_game(run, 200, 100, 60, 100)
    assert sl.fit_ladder(run, write=False)["recipe"]["transport"] == sl.TRANSPORT_PYTHON


def test_the_transport_census_skips_pairs_outside_the_fit_and_eval_cycle_rows(tmp_path, no_anchors, monkeypatch):
    """A Python pair OUTSIDE a first_n prefix, or a v2 eval-cycle row, does not make a Rust fit 'mixed'."""
    run = str(tmp_path)
    monkeypatch.setattr(sl, "pool_snapshot_steps", lambda d: [100, 200, 300])
    _rust(run, 200, 100, 55, 100)
    sl._append_game(run, 300, 200, 60, 100)                              # Python, but node 300 is outside first_n=2
    sl._append_game(run, 100, 200, 40, 100, source=sl.EVAL_CYCLE_SOURCE)  # never an edge
    assert sl.fit_ladder(run, first_n=2)["recipe"]["transport"] == sl.TRANSPORT_RUST


def test_draws_are_recorded_and_EXCLUDED_from_the_edge(tmp_path):
    """A Rust row's `games` is the decisive count; `draws` rides beside it and never enters load_games."""
    run = str(tmp_path)
    _rust(run, 200, 100, 55, 96, draws=4)
    row = json.loads(open(sl.games_log_path(run)).read())
    assert row["draws"] == 4 and row["games"] == 96
    assert sl.load_games(run) == {(100, 200): [41, 96]}


def test_two_ladders_on_different_transports_are_refused_side_by_side():
    py = {"recipe": {"name": sl.LADDER_RECIPE_NAME}}                # pre-boundary file: no transport key
    ru = {"recipe": {"name": sl.LADDER_RECIPE_NAME, "transport": sl.TRANSPORT_RUST}}
    assert sl.ladder_transport(py) == sl.TRANSPORT_PYTHON
    assert sl.check_same_transport({"a": ru, "b": dict(ru)}) == []
    with pytest.raises(sl.LadderTransportError, match="different transports"):
        sl.check_same_transport({"a": py, "b": ru})
    lines = sl.check_same_transport({"a": py, "b": ru}, allow=True)
    assert lines and "TRANSPORT MIX ACCEPTED" in lines[0]
    with pytest.raises(sl.LadderTransportError):                   # a `mixed` fit is never quietly comparable
        sl.check_same_transport({"a": {"recipe": {"transport": sl.TRANSPORT_MIXED}}})


def test_the_rust_edge_folds_both_seats_into_a_s_column():
    """`combine`: a-as-player W/L plus b-as-player L/W; draws apart; the seat split is ceil/floor."""
    from types import SimpleNamespace as NS
    s_ab = NS(w=30, l=18, d=2, pair_counts=[0, 1, 2, 3, 4])
    s_ba = NS(w=20, l=27, d=3, pair_counts=[1, 1, 1, 1, 1])        # b won 20 as the player
    e = LP.combine(s_ab, s_ba)
    assert (e["wins_a"], e["losses_a"], e["draws"], e["games"], e["games_played"]) == (57, 38, 5, 95, 100)
    assert e["by_seat"] == {"a_p1": [30, 18, 2], "b_p1": [27, 20, 3]}
    assert LP.seat_split(50) == (25, 25) and LP.seat_split(25) == (13, 12) and LP.pairs_for(100) == 50
    assert LP.schedule_key("x", "y") == LP.schedule_key("y", "x") and LP.schedule_key("x", "y").startswith("ladder:")


def test_the_eval_cycle_constant_is_shared():
    assert T._EVAL_CYCLE == sl.EVAL_CYCLE_SOURCE
