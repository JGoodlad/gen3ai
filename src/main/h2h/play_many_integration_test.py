"""``main.h2h play-many`` (``main.h2h.many``) on the REAL engine (CPU, tiny): many cells on ONE engine play the SAME
games as single-cell ``play`` on the same seeds (row for row, outcome digest for outcome digest, and on the engine
itself game for game), a re-run skips every recorded batch and builds no engine, a longer plan plays only the new
batches, a family reads across the cells, a third-architecture cell is refused before anything plays, and a slot
can never serve the previous cell's weights (a skipped load and a failed copy are both caught, and no row is written).

Seeded PERTURBED-fresh checkpoints (``conftest``): the numbers are not the point, the identity is. The plays are MODULE
fixtures (the tier budget is per test call); the tests only read what they recorded."""
from __future__ import annotations

import pytest

from agents.inference.service import InferenceService
from agents.inference.service.slots import SlotGroup
from agents.inference.service.spec import CopyParityFailure, ParityFailure
from agents.training import eval_ledger as L
from main.h2h import many as MANY
from main.h2h import play as PL

pytestmark = [pytest.mark.sim, pytest.mark.integration]

COMPUTE = PL.Compute(device="cpu", backend="eager", n_envs=8, threads=2, torch_threads=2, front="ffi",
                     profile="selfcheck")
SEED = 3
FAMILY = "h2h_many_test_family"
LOOK = "h2h_many_test_look1"

ROWS = L.ReaderDecl(name="h2h_many_test", purposes=L.ALL_PURPOSES, regime=L.RegimeFilter(), requests="any",
                    selection="include", flags_ok=frozenset(), inference="conditional")
FAMILY_READ = L.ReaderDecl(name="h2h_many_test_family", purposes=frozenset({"ab"}),
                           regime=L.RegimeFilter(protocol=PL.PROTOCOL, play="greedy", opponent_play="greedy",
                                                 mirrored=True),
                           requests="family", selection="include", flags_ok=frozenset(), inference="conditional",
                           decision_kind="ab_verdict")
QUIET = dict(emit=lambda _m: None)


def rows_of(out):
    return list(L.read(ROWS, root=out).rows)


def by_cell_batch(rows):
    return {(r["player"]["sha256"], r["opponent"]["sha256"], r["seed"]["batch"]): r for r in rows}


def specs(a, b):
    return [(a, b), (b, a), (a, a)]


@pytest.fixture(scope="module")
def played(built, checkpoints, tmp_path_factory):
    """The same three cells played (1) single-cell, one ``play_edge`` (one engine) each, and (2) on ONE engine as one
    X5-style LOOK of a registered family; then (3) the many-cell call re-run, and (4) extended by one batch."""
    a, b = checkpoints
    base = tmp_path_factory.mktemp("h2h_many")
    single, many = base / "single", base / "many"
    cells = MANY.resolve_cells(specs(a, b))
    for p, o in cells:
        PL.play_edge(str(single), p, o, pairs=2, batch_pairs=2, schedule_seed=SEED, run_label="test-study",
                     compute=COMPUTE, **QUIET)
    w = L.LedgerWriter(many, producer="h2h")
    w.register_family(FAMILY, decision_kind="ab_verdict", rule="test rule v1", protocol=PL.PROTOCOL)
    w.close()
    kw = dict(batch_pairs=2, schedule_seed=SEED, run_label="test-study", compute=COMPUTE, purpose="ab",
              family=FAMILY, request_id=LOOK, **QUIET)
    first = MANY.play_cells(str(many), cells, pairs=2, **kw)
    rows_first = rows_of(many)
    again = MANY.play_cells(str(many), cells, pairs=2, **kw)
    rows_again = rows_of(many)
    longer = MANY.play_cells(str(many), cells, pairs=4, **kw)
    return {"cells": cells, "single": rows_of(single), "many": many, "first": first, "rows_first": rows_first,
            "again": again, "rows_again": rows_again, "longer": longer, "rows_longer": rows_of(many)}


def test_many_cells_on_one_engine_write_the_rows_single_cell_play_writes(played):
    single, many = by_cell_batch(played["single"]), by_cell_batch(played["rows_first"])
    assert len(single) == len(many) == 3 and set(single) == set(many)
    for k, s in single.items():
        m = many[k]
        assert m["seed"] == s["seed"], "the same per-batch seed block (schedule key, cycle seed)"
        for f in ("counts", "pairs", "teams"):
            assert m[f] == s[f], f"cell batch {k}: {f} differs"
        for f in ("outcome_digest", "near_tie_games", "trainee_decisions", "p2_policy_decisions"):
            assert m["compute"][f] == s["compute"][f], f"cell batch {k}: compute.{f} differs"
        assert m["regime"] == s["regime"]
        assert m["request"]["id"] == LOOK and m["request"]["family"] == FAMILY and m["purpose"] == "ab"
        assert s["request"]["id"] != LOOK, "the single-cell rows are each cell's own default request"


def test_the_row_comparison_is_not_vacuous(played):
    """Cells on different schedule keys play different TEAM PAIRS, so their rows' team sets differ — a multi-cell row
    that came from the wrong cell could not match. (A-vs-B and B-vs-A share a key by design — the same team pairs — so
    whether their counters coincide depends on the outcomes; that pair is not asserted.) A row's outcome digest leaves
    out every game with a decision inside the GPU near-tie bar; the engine-level test below compares EVERY game."""
    rows = played["rows_first"]
    by_key = {}
    for r in rows:
        by_key.setdefault(r["seed"]["schedule_key"], set()).add(frozenset(r["teams"]))
    assert len(by_key) == 2, "(a, b) / (b, a) share one schedule key; (a, a) has its own"
    sets = list(by_key.values())
    assert all(len(v) == 1 for v in sets), "one schedule key, one team set (the same team pairs)"
    assert sets[0] != sets[1], "the two schedule keys drew different team sets"
    assert all(L.validate_row(r) == [] for r in rows)


def test_one_engine_served_every_cell_and_a_swap_costs_no_engine_start(played):
    eng = played["first"]["engine"]
    assert eng["startup_s"] is not None and eng["preflight_s"] is not None
    assert [c["rows"] for c in eng["cells"]] == [1, 1, 1]
    assert eng["cells"][0]["set_cell_s"] == 0.0 and all(c["set_cell_s"] > 0 for c in eng["cells"][1:])
    assert all(c["set_cell_s"] < eng["startup_s"] for c in eng["cells"][1:])


def test_a_rerun_skips_every_recorded_batch_and_builds_no_engine(played):
    assert played["again"]["engine"] == {"startup_s": None, "preflight_s": None, "cells": [], "decl": None}
    assert len(played["rows_again"]) == len(played["rows_first"]) == 3


def test_a_longer_plan_plays_only_the_new_batch_of_every_cell(played):
    eng = played["longer"]["engine"]
    assert [c["rows"] for c in eng["cells"]] == [1, 1, 1]
    rows = played["rows_longer"]
    assert len(rows) == 6 and sorted(r["seed"]["batch"] for r in rows) == [0, 0, 0, 1, 1, 1]
    assert [s["pairs"] for s in played["longer"]["cells"]] == [4, 4, 4]


def test_the_family_reads_across_the_cells_of_its_look(played):
    got = L.read(FAMILY_READ, root=played["many"], family=FAMILY)
    looks = L.looks(got, min_pairs=4)
    assert [rid for rid, _c in looks] == [LOOK]
    cells = looks[0][1]
    want = {(p.sha256, o.sha256) for p, o in played["cells"]}
    assert {(c.player, c.opponent) for c in cells} == want
    assert all(c.verdict == "OK" and c.n_pairs == 4 for c in cells)
    assert [c.verdict for c in L.looks(got, min_pairs=5)[0][1]] == ["INCONCLUSIVE"] * 3


# ------------------------------------------------------------------------------------------------ refusals
def test_a_third_architecture_cell_is_refused_before_any_engine_or_game(built, checkpoints, foreign, third,
                                                                        tmp_path):
    """Two architectures (``blob`` and X5's ``fixed_mass``) make one engine (``play_cross_integration_test.py``); a
    THIRD is refused in the pre-flight, naming it and what differs from each group, before anything plays."""
    a, b = checkpoints
    out = tmp_path / "ledger"
    with pytest.raises(PL.CellArchMismatch) as ei:
        MANY.play_cells(str(out), MANY.resolve_cells([(a, b), (a, foreign), (third, b)]), pairs=2, batch_pairs=2,
                        run_label="s", compute=COMPUTE, **QUIET)
    msg = str(ei.value)
    assert PL.resolve_player(third).id in msg and PL.resolve_player(a).id in msg
    assert PL.resolve_player(foreign).id in msg and "move_prior_fusion" in msg and "belief_tokens" in msg
    assert "matches neither" in msg
    assert rows_of(out) == [], "nothing was played"


def test_a_cell_listed_twice_is_refused(checkpoints):
    a, b = checkpoints
    with pytest.raises(PL.H2HError, match="repeats cell 0"):
        MANY.resolve_cells([(a, b), (b, a), (a, b)])


def test_a_load_that_never_reached_the_slots_is_caught_and_writes_no_row(built, checkpoints, tmp_path, monkeypatch):
    """``InferenceService.load`` made a no-op: the cycle runs on whatever the slots held (the opponent's slot holds the
    PLAYER's template weights), and the engine's own after-cycle check refuses the batch before it is scored."""
    a, b = checkpoints
    out = tmp_path / "ledger"
    monkeypatch.setattr(InferenceService, "load", lambda self, slot, policy, model_id: None)
    with pytest.raises(CopyParityFailure, match="slot 1"):
        MANY.play_cells(str(out), MANY.resolve_cells([(a, b)]), pairs=2, batch_pairs=2, run_label="s",
                        compute=COMPUTE, **QUIET)
    assert rows_of(out) == []


@pytest.fixture(scope="module")
def swapped(built, checkpoints, foreign):
    """ONE engine, driven by hand: cells (a, b) → (b, a) → (a, a) one batch each at the single-cell seeds, then a
    refused foreign cell (another architecture than the ONE this engine declared), then a cell whose slot copy silently does nothing (the service must refuse it and stay
    refused). Records everything; the tests read it."""
    a, b = checkpoints
    (pa, pb), (pb2, pa2), (paa, paa2) = cells = MANY.resolve_cells(specs(a, b))
    out = {"cells": cells, "games": []}
    eng = PL.H2HEngine(pa, pb, COMPUTE, **QUIET)
    try:
        for i, (p, o) in enumerate(cells):
            if i:
                eng.set_cell(p, o)
            games, _st = eng.play_batch(3, PL.cycle_seed(SEED, PL.schedule_key_of(p, o), 0))
            out["games"].append(sorted(games, key=lambda g: g["game"]))
        try:
            eng.set_cell(pa, PL.resolve_player(foreign))
        except PL.CellArchMismatch as e:
            out["refused"] = str(e)
        out["cell_after_refusal"] = (eng.player.sha256, eng.opponent.sha256)
        games, _st = eng.play_batch(3, PL.cycle_seed(SEED, PL.schedule_key_of(paa, paa2), 0))
        out["after_refusal"] = sorted(games, key=lambda g: g["game"])
        eng.set_cell(pa, pb)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(SlotGroup, "copy_in", lambda self, slot, sd: None)
            try:
                eng.play_batch(3, PL.cycle_seed(SEED, PL.schedule_key_of(pa, pb), 0))
            except ParityFailure as e:
                out["stale"] = e
        try:
            eng.play_batch(3, PL.cycle_seed(SEED, PL.schedule_key_of(pa, pb), 0))
        except Exception as e:                                                     # noqa: BLE001 - recorded
            out["after_poison"] = e
    finally:
        eng.close()
    # the LAST cell (after two swaps: both slots reloaded twice) on an engine of its own — the single-cell path;
    # the other cells are compared row for row with single-cell `play` above
    p, o = cells[-1]
    e2 = PL.H2HEngine(p, o, COMPUTE, **QUIET)
    try:
        games, _st = e2.play_batch(3, PL.cycle_seed(SEED, PL.schedule_key_of(p, o), 0))
        out["fresh"] = sorted(games, key=lambda g: g["game"])
    finally:
        e2.close()
    return out


def test_a_swapped_cell_plays_game_for_game_what_a_fresh_engine_plays(swapped):
    swapped_games, fresh_games = swapped["games"][-1], swapped["fresh"]
    assert swapped_games == fresh_games, "the same game log: results, teams, end turns, near-tie counts"
    dig = lambda gs: L.outcome_digest(PL.outcome_vector(gs), [])   # noqa: E731 - EVERY game, near-ties included
    assert dig(swapped_games) == dig(fresh_games)
    assert len({dig(gs) for gs in swapped["games"]}) == 3, "the three cells played different games (not vacuous)"


def test_a_refused_cell_leaves_the_engine_on_its_previous_cell(swapped):
    assert "belief_tokens" in swapped["refused"]
    (_c1, _c2, (paa, paa2)) = swapped["cells"]
    assert swapped["cell_after_refusal"] == (paa.sha256, paa2.sha256)
    assert swapped["after_refusal"] == swapped["games"][2], "the previous cell still plays its own games"


def test_a_slot_copy_that_does_nothing_is_caught_and_poisons_the_service(swapped):
    assert isinstance(swapped.get("stale"), CopyParityFailure), "the stale slot (the previous cell's weights) is caught"
    assert "after_poison" in swapped, "a poisoned service serves no later cell"
