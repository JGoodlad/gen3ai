"""``main.h2h.play`` on the REAL engine (CPU, tiny): the mirror's hand-over, what a checkpoint against ITSELF
does, and what a fixed seed reproduces. Seeded PERTURBED-fresh checkpoints, T2 eager, the in-process core.

WHAT THE MIRROR DOES AND DOES NOT CANCEL (measured, 2026-10-03; ``designs/training/eval_and_rating.md``).
A pair hands the TEAMS over and keeps the player in seat p1 on ONE battle seed. For a player against ITSELF
the second game is the first with the SEATS relabelled, so the pair is exactly (W, L) — IF the engine is seat-
symmetric. It is not always: a speed tie is broken by the RNG in a seat-dependent order, so after the first
such event the two games leave the mirror image of each other (the decision margins agree to 5 digits until
then, and jump after it). So "exactly 0.5 per pair" is NOT an engine invariant, and this file does not assert
it. It asserts the part that IS one: every pair whose two games stayed MIRROR IMAGES decision for decision
(each side's action stream equals the other game's opposite stream, to the same end turn) scores exactly 1/2 —
and that there are enough of them for the assertion to bite. ``play_test.py`` pins the exact cancellation of
the scoring itself on a seat-symmetric toy engine.

The heavy plays are MODULE fixtures (the tier budget is per test call); the tests only read them."""
from __future__ import annotations

import pytest

from main.h2h import play as PL

pytestmark = [pytest.mark.sim, pytest.mark.integration]

COMPUTE = dict(device="cpu", backend="eager", threads=2, torch_threads=2, front="ffi", profile="selfcheck")
SCHEDULE_KEY = "h2h:test"
PTS = {"WIN": 2, "DRAW": 1, "LOSS": 0}


def play(zip_a, zip_b, pairs, n_envs, *, raw=False, times=1):
    """``times`` batches of ``pairs`` pairs at ONE seed on ONE engine: ``([(games, stats)], team table)``."""
    eng = PL.H2HEngine(PL.resolve_player(zip_a), PL.resolve_player(zip_b), PL.Compute(n_envs=n_envs, **COMPUTE),
                       emit=lambda _m: None)
    try:
        seed = PL.cycle_seed(0, SCHEDULE_KEY, 0)
        return [eng.play_batch(pairs, seed, sink=[] if raw else None) for _ in range(times)], list(eng.team_packed)
    finally:
        eng.close()


@pytest.fixture(scope="module")
def self_play(built, checkpoints):
    a, _b = checkpoints
    (run,), teams = play(a, a, 16, 16, raw=True)
    return run[0], run[1], teams


@pytest.fixture(scope="module")
def reproduction(built, checkpoints):
    a, b = checkpoints
    twice, teams8 = play(a, b, 12, 8, times=2)              # one engine, two cycles at one seed
    (other,), teams16 = play(a, b, 12, 16)                  # another engine, another env count
    return twice, teams8, other, teams16


def test_the_two_games_of_every_pair_hand_the_teams_over_and_the_executor_agrees_with_the_log(self_play):
    raw, stats, team_packed = self_play
    by = {g["game"]: g for g in raw}
    assert sorted(by) == list(range(32))
    for k in range(16):
        g1, g2 = by[2 * k], by[2 * k + 1]
        assert (g1["swapped"], g2["swapped"]) == (False, True)
        assert list(g2["teams"]) == list(g1["teams"])[::-1], f"pair {k}: teams not handed over"
    sc = PL.score_games(raw, team_packed, 16)             # raises on any broken pair
    assert stats["executor_pair_counts"] == sc.pair_counts, "the executor's pentanomial is the game log's"
    assert sc.mirror_checked == 16 and sc.w + sc.l + sc.d == 32


def test_a_checkpoint_against_itself_cancels_exactly_in_every_pair_that_stayed_a_mirror_image(self_play):
    raw, _stats, _teams = self_play
    by = {g["game"]: g for g in raw}
    mirror = []
    for k in range(16):
        g1, g2 = by[2 * k], by[2 * k + 1]
        own1, opp1 = list(g1["actions"]), [o[1] for o in g1["opp"]]
        own2, opp2 = list(g2["actions"]), [o[1] for o in g2["opp"]]
        if own1 == opp2 and opp1 == own2 and g1["end_turn"] == g2["end_turn"]:
            mirror.append(PTS[g1["result"]] + PTS[g2["result"]])
    assert len(mirror) >= 8, f"only {len(mirror)} of 16 pairs stayed mirror images: the assertion below would be vacuous"
    assert all(p == 2 for p in mirror), f"a mirror-image pair that did not cancel: {mirror}"


def test_a_fixed_seed_reproduces_every_game_bit_for_bit_on_one_configuration(reproduction):
    (first, second), teams8, _other, _teams16 = reproduction
    assert first[0] == second[0], "the same seed and configuration: the same games, cycle after cycle"
    assert first[1]["executor_pair_counts"] == second[1]["executor_pair_counts"]
    assert len(teams8) > 100, "the team table is the pool's"


def test_a_change_of_env_count_changes_no_game_that_is_clear_of_a_near_tie(reproduction):
    (first, _second), teams8, other, teams16 = reproduction
    assert teams16 == teams8, "the team table does not depend on the env count"
    key = lambda g: (g["game"], g["result"], g["end_turn"], tuple(g["teams"]))   # noqa: E731
    one, three = {g["game"]: g for g in first[0]}, {g["game"]: g for g in other[0]}
    assert sorted(one) == sorted(three) == list(range(24))
    clean = [k for k in one if one[k]["near_ties"] == 0 and three[k]["near_ties"] == 0]
    assert len(clean) >= 12, f"only {len(clean)} of 24 games are clear of a near-tie: the comparison would be vacuous"
    # a decision within a rounding error of a tie may flip with the batch shape (rule 8: those games are
    # EXCLUDED, not tolerated); every other game is the same game whatever the env count
    assert [key(one[k]) for k in clean] == [key(three[k]) for k in clean]
