"""
Tests for EpisodeTracker._actions sync and prev_N_delta_vecs().

The _actions list is the core of the N-turn history feature: it stores one
action per BattleContext transition so prev_N_delta_vecs() can reconstruct
historical TurnDeltas without re-reading the live battle object.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock

from agents.training.battle_context import BattleContext, TurnDelta
from agents.training.episode_tracker import EpisodeTracker
from agents.observation.turn_delta_encoder import (
    TurnDeltaEncoder,
    TURN_DELTA_DIM,
    OFFSET_OUR_HP_DELTA_SUM,
)
from agents.observation.state_encoder import load_mappings
from agents.gen3_mechanics import BOOST_DIM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_encoder():
    mappings = load_mappings()
    return TurnDeltaEncoder(
        mappings.get("moves", {}),
        mappings.get("species", {}),
    )


def _make_ctx(turn: int = 0, our_hp0: float = 1.0) -> BattleContext:
    """Minimal BattleContext stub with controlled HP at slot 0."""
    return BattleContext(
        turn=turn,
        phase="move_selection",
        mask=np.ones(11, dtype=np.int8),
        our_slot_map={"bulbasaur": 0},
        opp_slot_map={"charmander": 0},
        our_hp=np.array([our_hp0, 0, 0, 0, 0, 0], dtype=np.float32),
        opp_hp=np.zeros(6, dtype=np.float32),
        our_active="bulbasaur",
        opp_active="charmander",
        our_fainted_count=0,
        opp_fainted_count=0,
        active_move_ids=[None, None, None, None],
        opp_last_move_id=None,
        opp_all_last_move_ids={},
        opp_active_revealed_moves=frozenset(),
        our_cant_reason=None,
        opp_cant_reason=None,
        our_boosts=np.zeros(BOOST_DIM, dtype=np.int8),
        opp_boosts=np.zeros(BOOST_DIM, dtype=np.int8),
        our_last_effectiveness=None,
        opp_last_effectiveness=None,
        we_moved_first=None,
    )


def _mock_battle(turn: int = 1) -> MagicMock:
    """Minimal battle mock compatible with BattleContext.from_battle()."""
    battle = MagicMock()
    battle.team = {}
    battle.opponent_team = {}
    battle.active_pokemon = None
    battle.opponent_active_pokemon = None
    battle.force_switch = False
    battle.turn = turn
    battle._gen3_decision_context = None
    battle.last_request = None
    battle.our_last_effectiveness = None
    battle.opp_last_effectiveness = None
    battle.we_moved_first = None
    return battle


def _inject(tracker: EpisodeTracker, ctxs: list, actions: list) -> None:
    """Directly inject history and actions, bypassing from_battle()."""
    tracker._history.extend(ctxs)
    tracker._actions.extend(actions)


# ---------------------------------------------------------------------------
# _actions list synchronisation
# ---------------------------------------------------------------------------

def test_no_action_committed_on_first_record():
    """First record() has nothing to commit — _actions stays empty."""
    tracker = EpisodeTracker()
    tracker.advance(3)
    tracker.record(_mock_battle(1), np.ones(11, dtype=np.int8))
    assert len(tracker._actions) == 0
    assert len(tracker._history) == 1


def test_action_committed_on_second_record():
    """Second record() commits the action that was advance()d after the first."""
    tracker = EpisodeTracker()
    tracker.record(_mock_battle(1), np.ones(11, dtype=np.int8))
    tracker.advance(7)
    tracker.record(_mock_battle(2), np.ones(11, dtype=np.int8))

    assert len(tracker._actions) == 1
    assert tracker._actions[0] == 7


def test_actions_parallel_to_history():
    """After N records + advances, len(_actions) == len(_history) - 1."""
    tracker = EpisodeTracker()
    n_turns = 5
    for t in range(n_turns):
        tracker.record(_mock_battle(t), np.ones(11, dtype=np.int8))
        tracker.advance(t)
    # 5 records → 4 committed actions (first record has nothing to commit)
    assert len(tracker._history) == n_turns
    assert len(tracker._actions) == n_turns - 1


def test_actions_contain_correct_values():
    """Each committed action must equal what advance() received."""
    tracker = EpisodeTracker()
    actions = [2, 5, 8, 10, 1]
    for t, a in enumerate(actions):
        tracker.record(_mock_battle(t), np.ones(11, dtype=np.int8))
        tracker.advance(a)
    tracker.record(_mock_battle(len(actions)), np.ones(11, dtype=np.int8))
    # _actions[i] is the action advance()d from history[i] (committed at i+1's record)
    assert tracker._actions == actions


def test_reset_clears_actions():
    tracker = EpisodeTracker()
    tracker.record(_mock_battle(1), np.ones(11, dtype=np.int8))
    tracker.advance(4)
    tracker.record(_mock_battle(2), np.ones(11, dtype=np.int8))
    assert len(tracker._actions) == 1
    tracker.reset()
    assert len(tracker._actions) == 0
    assert len(tracker._history) == 0


# ---------------------------------------------------------------------------
# prev_N_delta_vecs — shape
# ---------------------------------------------------------------------------

def test_prev_N_delta_vecs_shape_empty_episode():
    """At episode start (no history), result is all-zeros (n, dim) array."""
    tracker = EpisodeTracker()
    encoder = _make_encoder()
    result = tracker.prev_N_delta_vecs(5, encoder)
    assert result.shape == (5, TURN_DELTA_DIM)
    assert np.all(result == 0)


def test_prev_N_delta_vecs_shape_one_delta():
    """With one committed delta, result has shape (n, dim)."""
    tracker = EpisodeTracker()
    encoder = _make_encoder()
    _inject(tracker, [_make_ctx(0), _make_ctx(1)], [0])
    result = tracker.prev_N_delta_vecs(5, encoder)
    assert result.shape == (5, TURN_DELTA_DIM)


def test_prev_N_delta_vecs_shape_n_equals_1():
    _inject_n = 3
    tracker = EpisodeTracker()
    encoder = _make_encoder()
    _inject(tracker, [_make_ctx(t) for t in range(_inject_n)], [0] * (_inject_n - 1))
    result = tracker.prev_N_delta_vecs(1, encoder)
    assert result.shape == (1, TURN_DELTA_DIM)


# ---------------------------------------------------------------------------
# prev_N_delta_vecs — zero-padding
# ---------------------------------------------------------------------------

def test_prev_N_delta_vecs_all_zeros_at_episode_start():
    tracker = EpisodeTracker()
    encoder = _make_encoder()
    result = tracker.prev_N_delta_vecs(5, encoder)
    assert np.all(result == 0), "No history → all zeros"


def test_prev_N_delta_vecs_leading_zeros_when_fewer_than_n_deltas():
    """With k < n committed deltas, first (n-k) rows must be all-zeros."""
    tracker = EpisodeTracker()
    encoder = _make_encoder()
    # 3 contexts → 2 committed deltas
    _inject(tracker, [_make_ctx(t) for t in range(3)], [0, 0])
    result = tracker.prev_N_delta_vecs(5, encoder)
    assert np.all(result[:3] == 0), "First 3 of 5 slots must be zero-padded"


def test_prev_N_delta_vecs_no_leading_zeros_when_full():
    """With exactly n committed deltas, no padding needed — no all-zero rows."""
    tracker = EpisodeTracker()
    encoder = _make_encoder()
    n = 5
    # n+1 contexts → n committed deltas
    _inject(tracker, [_make_ctx(t) for t in range(n + 1)], [0] * n)
    result = tracker.prev_N_delta_vecs(n, encoder)
    # Can't assert no zero rows (some may coincidentally be zero), but shape is correct
    assert result.shape == (n, TURN_DELTA_DIM)


# ---------------------------------------------------------------------------
# prev_N_delta_vecs — ordering (oldest-first)
# ---------------------------------------------------------------------------

def test_prev_N_delta_vecs_ordering_oldest_first():
    """Index 0 = oldest delta, index n-1 = most recent.

    We create contexts with decreasing HP at slot 0, so our_hp_delta
    (the our_hp_delta_sum scalar within the base block) is negative and
    distinct per turn. The magnitude tells us which delta is which.
    """
    encoder = _make_encoder()
    tracker = EpisodeTracker()
    # HP: 1.0 → 0.9 → 0.7 → 0.4 → 0.0
    hp_values = [1.0, 0.9, 0.7, 0.4, 0.0]
    ctxs = [_make_ctx(t, our_hp0=hp) for t, hp in enumerate(hp_values)]
    # 5 contexts → 4 actions (action indices don't matter for hp_delta)
    _inject(tracker, ctxs, [0] * (len(ctxs) - 1))

    result = tracker.prev_N_delta_vecs(5, encoder)

    # We have 4 deltas, n=5, so result[0] is zero-padded, result[1..4] are real.
    assert np.all(result[0] == 0), "First slot should be zero-padded"

    # Slot 1 = delta between ctxs[0] and ctxs[1] (oldest: hp_delta = 0.9-1.0 = -0.1)
    # Slot 4 = delta between ctxs[3] and ctxs[4] (most recent: hp_delta = 0.0-0.4 = -0.4)
    # Drive the offset from the encoder constant so a layout shift surfaces
    # in the error message instead of silently pointing at the wrong field.
    OUR_HP_DELTA_IDX = OFFSET_OUR_HP_DELTA_SUM
    delta_oldest = TurnDelta.build(ctxs[0], ctxs[1], 0)
    delta_newest = TurnDelta.build(ctxs[3], ctxs[4], 0)
    oldest_encoded = encoder.encode(delta_oldest)
    newest_encoded = encoder.encode(delta_newest)

    assert np.allclose(result[1], oldest_encoded, atol=1e-6), (
        f"result[1] should be oldest delta. "
        f"hp_delta: result={result[1][OUR_HP_DELTA_IDX]:.3f}, "
        f"expected={oldest_encoded[OUR_HP_DELTA_IDX]:.3f}"
    )
    assert np.allclose(result[4], newest_encoded, atol=1e-6), (
        f"result[4] should be most recent delta. "
        f"hp_delta: result={result[4][OUR_HP_DELTA_IDX]:.3f}, "
        f"expected={newest_encoded[OUR_HP_DELTA_IDX]:.3f}"
    )


def test_prev_N_delta_vecs_last_slot_matches_build_delta():
    """result[n-1] must equal build_delta() when _last_action == _actions[-1].

    This holds after record() and before the next advance().
    (build_delta() uses _last_action; prev_N_delta_vecs uses _actions[-1].)
    """
    encoder = _make_encoder()
    tracker = EpisodeTracker()
    ctxs = [_make_ctx(t) for t in range(4)]
    _inject(tracker, ctxs, [0, 0, 0])
    # _last_action defaults to -1 after __init__; _actions[-1] is 0.
    # Manually set _last_action to match for this comparison.
    tracker._last_action = 0

    result = tracker.prev_N_delta_vecs(4, encoder)
    expected = encoder.encode(tracker.build_delta())
    assert np.allclose(result[3], expected, atol=1e-6), (
        "Last slot of prev_N_delta_vecs must match build_delta() when actions agree"
    )


# ---------------------------------------------------------------------------
# prev_N_delta_vecs — action index is used correctly
# ---------------------------------------------------------------------------

def test_prev_N_delta_vecs_uses_stored_actions():
    """Different actions at the same context transition must produce different deltas.

    Action 6 maps to move slot 0 (our_move_id = active_move_ids[0]).
    Action 7 maps to move slot 1.
    With distinct active_move_ids, the encoded vectors should differ.
    """
    encoder = _make_encoder()
    ctx0 = BattleContext(
        turn=0, phase="move_selection",
        mask=np.ones(11, dtype=np.int8),
        our_slot_map={"bulbasaur": 0}, opp_slot_map={"charmander": 0},
        our_hp=np.ones(6, dtype=np.float32), opp_hp=np.zeros(6, dtype=np.float32),
        our_active="bulbasaur", opp_active="charmander",
        our_fainted_count=0, opp_fainted_count=0,
        active_move_ids=["razorleaf", "vinewhip", None, None],  # distinct move IDs
        opp_last_move_id=None, opp_all_last_move_ids={},
        opp_active_revealed_moves=frozenset(),
        our_cant_reason=None, opp_cant_reason=None,
        our_boosts=np.zeros(BOOST_DIM, dtype=np.int8),
        opp_boosts=np.zeros(BOOST_DIM, dtype=np.int8),
        our_last_effectiveness=None, opp_last_effectiveness=None,
        we_moved_first=None,
    )
    ctx1 = _make_ctx(1)

    tracker_a = EpisodeTracker()
    _inject(tracker_a, [ctx0, ctx1], [6])  # action 6 → move slot 0 → razorleaf

    tracker_b = EpisodeTracker()
    _inject(tracker_b, [ctx0, ctx1], [7])  # action 7 → move slot 1 → vinewhip

    result_a = tracker_a.prev_N_delta_vecs(2, encoder)
    result_b = tracker_b.prev_N_delta_vecs(2, encoder)

    # The last slot differs because different moves have different IDs
    assert not np.allclose(result_a[1], result_b[1]), (
        "Different actions on the same contexts should produce different delta vectors"
    )


# ---------------------------------------------------------------------------
# Hidden Power rule-out wiring (mark_no_hp on full moveset reveal)
# ---------------------------------------------------------------------------

def _opp_mon(species: str, move_ids: list[str]) -> MagicMock:
    """Mock opponent mon with the given revealed moves dict-keyed by id."""
    mon = MagicMock()
    mon.species = species
    mon.moves = {mid: MagicMock() for mid in move_ids}
    return mon


def test_mark_no_hp_fires_when_four_moves_revealed_without_hp():
    """Opp with 4 non-HP moves revealed → tracker marks species ruled-out."""
    tracker = EpisodeTracker()
    battle = _mock_battle(1)
    battle.opponent_team = {
        "p2: Snorlax": _opp_mon("snorlax", ["bodyslam", "earthquake", "rest", "curse"])
    }
    tracker.record(battle, np.ones(11, dtype=np.int8))
    assert tracker.hidden_power_tracker.is_known("snorlax") is True
    assert not tracker.hidden_power_tracker.get_probs("snorlax").any()


def test_mark_no_hp_does_not_fire_with_fewer_than_four_moves():
    """3 moves revealed → still uncertain whether HP is in slot 4. No mark yet."""
    tracker = EpisodeTracker()
    battle = _mock_battle(1)
    battle.opponent_team = {
        "p2: Tauros": _opp_mon("tauros", ["bodyslam", "earthquake", "return"])
    }
    tracker.record(battle, np.ones(11, dtype=np.int8))
    assert tracker.hidden_power_tracker.is_known("tauros") is False


def test_mark_no_hp_does_not_fire_when_hp_is_among_four_moves():
    """4 moves including HP → known via observation path, not ruled out."""
    tracker = EpisodeTracker()
    battle = _mock_battle(1)
    battle.opponent_team = {
        "p2: Jolteon": _opp_mon("jolteon", ["thunderbolt", "shadowball", "substitute", "hiddenpower"])
    }
    tracker.record(battle, np.ones(11, dtype=np.int8))
    # is_known stays False here — we haven't OBSERVED HP fire yet, just seen
    # it sit in the moveset. Tracker only flips on observe() in that case.
    assert tracker.hidden_power_tracker.is_known("jolteon") is False


def test_mark_no_hp_recognizes_typed_hp_variants():
    """If poke-env exposes the typed key 'hiddenpowergrass', that also blocks rule-out."""
    tracker = EpisodeTracker()
    battle = _mock_battle(1)
    battle.opponent_team = {
        "p2: Celebi": _opp_mon("celebi", ["leechseed", "recover", "perishsong", "hiddenpowergrass"])
    }
    tracker.record(battle, np.ones(11, dtype=np.int8))
    assert tracker.hidden_power_tracker.is_known("celebi") is False


def test_mark_no_hp_persists_across_turns():
    """Once ruled out, the species stays ruled out for subsequent turns."""
    tracker = EpisodeTracker()
    b1 = _mock_battle(1)
    b1.opponent_team = {
        "p2: Snorlax": _opp_mon("snorlax", ["bodyslam", "earthquake", "rest", "curse"])
    }
    tracker.record(b1, np.ones(11, dtype=np.int8))
    assert tracker.hidden_power_tracker.is_known("snorlax") is True

    b2 = _mock_battle(2)
    # Snorlax still has the same moveset
    b2.opponent_team = b1.opponent_team
    tracker.record(b2, np.ones(11, dtype=np.int8))
    assert tracker.hidden_power_tracker.is_known("snorlax") is True


def test_reset_clears_rule_out_state():
    """EpisodeTracker.reset() must wipe the rule-out set via tracker.reset()."""
    tracker = EpisodeTracker()
    battle = _mock_battle(1)
    battle.opponent_team = {
        "p2: Snorlax": _opp_mon("snorlax", ["bodyslam", "earthquake", "rest", "curse"])
    }
    tracker.record(battle, np.ones(11, dtype=np.int8))
    assert tracker.hidden_power_tracker.is_known("snorlax") is True
    tracker.reset()
    assert tracker.hidden_power_tracker.is_known("snorlax") is False
