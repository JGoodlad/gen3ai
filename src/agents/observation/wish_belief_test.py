"""Unit tests for the gen3_wish_wired_v1 pending-Wish reconstruction (observation/wish_belief.py).

The double-Wish, timing, and per-side logic are pinned here against the verified gen3 mechanics
(heal end of cast+1, slot-keyed, second Wish on an occupied slot fails). The end-to-end correctness
vs the real sim is the bridge fuzz (`poke_env_gaps/wish_floating_fuzz_test.py`)."""
from types import SimpleNamespace

from agents.battle.battle_event import BattleEvent, EventKind, OURS, OPP
from agents.observation.wish_belief import build_wish_pending, wish_floating_value, WISH_HEAL_FRACTION


def _wish(seq, turn, side):
    return BattleEvent(seq=seq, turn=turn, kind=EventKind.MOVE, side=side,
                       actor_species="milotic", value={"move_id": "wish"})


def _move(seq, turn, side, mid):
    return BattleEvent(seq=seq, turn=turn, kind=EventKind.MOVE, side=side,
                       actor_species="snorlax", value={"move_id": mid})


def _battle(events, turn):
    return SimpleNamespace(events=events, turn=turn)


def test_pending_iff_cast_last_turn():
    # cast turn 5 → pending at the turn-6 decision (resolves end of turn 6), gone by turn 7
    assert build_wish_pending(_battle([_wish(1, 5, OURS)], 6))[OURS] is True
    assert build_wish_pending(_battle([_wish(1, 5, OURS)], 7))[OURS] is False
    # not pending on the cast turn itself (no post-cast decision normally, but be safe)
    assert build_wish_pending(_battle([_wish(1, 5, OURS)], 5))[OURS] is False


def test_double_wish_second_fails():
    # consecutive-turn wishes: the turn-6 one fails (slot still holds the turn-5 wish), so at the
    # turn-7 decision nothing is pending (the turn-5 wish resolved end of turn 6).
    evs = [_wish(1, 5, OURS), _wish(2, 6, OURS)]
    assert build_wish_pending(_battle(evs, 7))[OURS] is False
    # but the turn-5 wish IS pending at the turn-6 decision
    assert build_wish_pending(_battle(evs, 6))[OURS] is True


def test_spaced_wishes_both_succeed():
    # turn 5 and turn 7 (slot free again by 7) → the turn-7 wish is pending at turn 8
    evs = [_wish(1, 5, OURS), _wish(2, 7, OURS)]
    assert build_wish_pending(_battle(evs, 8))[OURS] is True
    assert build_wish_pending(_battle(evs, 6))[OURS] is True   # the turn-5 one
    assert build_wish_pending(_battle(evs, 7))[OURS] is False  # turn-5 resolved, turn-7 not cast yet at decision


def test_per_side_independent():
    evs = [_wish(1, 5, OURS), _wish(2, 5, OPP)]
    p = build_wish_pending(_battle(evs, 6))
    assert p[OURS] is True and p[OPP] is True
    # opp only
    p = build_wish_pending(_battle([_wish(1, 5, OPP)], 6))
    assert p[OURS] is False and p[OPP] is True


def test_non_wish_moves_ignored():
    evs = [_move(1, 5, OURS, "surf"), _move(2, 5, OPP, "recover")]
    p = build_wish_pending(_battle(evs, 6))
    assert p[OURS] is False and p[OPP] is False


def test_empty_or_no_event_log():
    assert build_wish_pending(_battle([], 6)) == {OURS: False, OPP: False}
    assert build_wish_pending(SimpleNamespace(events=None, turn=6)) == {OURS: False, OPP: False}


def test_floating_value():
    assert wish_floating_value(True) == WISH_HEAL_FRACTION == 0.5
    assert wish_floating_value(False) == 0.0
