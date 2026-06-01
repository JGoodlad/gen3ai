"""
Tests for EpisodeTracker._actions sync and prev_N_delta_vecs().

The _actions list is the core of the N-turn history feature: it stores one
action per BattleContext transition so prev_N_delta_vecs() can reconstruct
historical TurnDeltas without re-reading the live battle object.
"""
import numpy as np
import pytest
from unittest.mock import MagicMock

from agents.training.battle_snapshot import BattleContext
from agents.training.turn_delta_legacy import build_legacy
from agents.training.episode_tracker import EpisodeTracker
from agents.action.mask_generator import Gen3ActionMasker
from agents.battle.live_view import (
    LegalActions,
    LegalMove,
    LiveMove,
    LivePokemon,
    LiveSide,
    LiveView,
    LiveWeather,
)
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


def _empty_side() -> LiveSide:
    return LiveSide(team_size=0, active=None, mons=(), side_conditions={})


def _live_view(opp_mons=()) -> LiveView:
    """Real LiveView with the given LivePokemon on the opponent side (our side empty).

    record() now reads current-board state through battle.strict_view().live, so the
    mock must surface a real LiveView for the HP-candidate scan / HP-target lookup."""
    return LiveView(
        turn=1,
        weather=LiveWeather(weather=None, is_permanent=False, turns_active=0),
        ours=_empty_side(),
        opp=LiveSide(
            team_size=len(opp_mons), active=None, mons=tuple(opp_mons),
            side_conditions={},
        ),
    )


def _mock_battle(turn: int = 1, opp_live_mons=()) -> MagicMock:
    """Minimal battle mock compatible with BattleContext.from_battle() and the
    LiveView-backed current-board reads record() makes (battle.strict_view().live)."""
    battle = MagicMock()
    battle.team = {}
    battle.opponent_team = {}
    battle.active_pokemon = None
    battle.opponent_active_pokemon = None
    battle.force_switch = False
    battle.turn = turn
    battle.last_request = None
    battle.our_last_effectiveness = None
    battle.opp_last_effectiveness = None
    battle.we_moved_first = None
    battle.strict_view.return_value.live = _live_view(opp_live_mons)
    return battle


def _inject(tracker: EpisodeTracker, ctxs: list, actions: list) -> None:
    """Directly inject history and actions, bypassing from_battle()."""
    tracker._history.extend(ctxs)
    tracker._actions.extend(actions)


def _lm(move_id, disabled=False):
    return LegalMove(id=move_id, current_pp=16, max_pp=16, disabled=disabled, target="normal")


def _legal(move_slots, struggle=False):
    return LegalActions(
        move_slots=tuple(move_slots), switches=(), force_switch=False, trapped=False,
        maybe_trapped=False, wait=False, struggle=struggle, last_request=None,
    )


# ---------------------------------------------------------------------------
# prev_mask (the "action mask as an obs FEATURE") — end-to-end through the new
# legal-sourced active_move_ids. The MOVE bits must be reordered from
# request/action order into sorted-by-id order (so the validity bit lands on the
# same move the feature extractor reads at that slot). This is the path whose
# source changed: active_move_ids now comes from legal.move_ids.
# ---------------------------------------------------------------------------

def test_prev_mask_reorders_disabled_bit_to_sorted_slot():
    """A disabled move in non-sorted request order must have its 0 bit land on that
    move's SORTED slot in prev_mask — sourced via legal.move_ids → active_move_ids."""
    # request/action order: [thunderbolt, willowisp(DISABLED), icepunch, taunt]
    legal = _legal([
        _lm("thunderbolt"), _lm("willowisp", disabled=True), _lm("icepunch"), _lm("taunt"),
    ])
    mask = Gen3ActionMasker.mask_from_legal(legal)
    assert mask[6:10].tolist() == [1, 0, 1, 1], "willowisp disabled → slot-7 bit clear"

    tracker = EpisodeTracker()
    tracker.record(_mock_battle(1), mask, legal=legal)   # becomes _history[-2]
    tracker.record(_mock_battle(2), mask, legal=legal)   # becomes _history[-1]

    # active order [thunderbolt, willowisp, icepunch, taunt] → sorted
    # [icepunch, taunt, thunderbolt, willowisp]; reordered move bits should be
    # [icepunch=1, taunt=1, thunderbolt=1, willowisp=0].
    prev = tracker.prev_mask  # also runs assert_sorted_validity_correct internally
    assert prev[6:10].tolist() == [1.0, 1.0, 1.0, 0.0]
    assert prev[:6].tolist() == [0.0] * 6      # no switches in this legal
    assert prev[10] == 0.0                     # no struggle


def test_prev_mask_all_legal_is_all_ones():
    legal = _legal([_lm("thunderbolt"), _lm("willowisp"), _lm("icepunch"), _lm("taunt")])
    mask = Gen3ActionMasker.mask_from_legal(legal)
    tracker = EpisodeTracker()
    tracker.record(_mock_battle(1), mask, legal=legal)
    tracker.record(_mock_battle(2), mask, legal=legal)
    assert tracker.prev_mask[6:10].tolist() == [1.0, 1.0, 1.0, 1.0]


def test_prev_mask_struggle_turn_has_no_move_bits():
    """On a forced-struggle turn legal.move_ids is empty (struggle is the flag); the
    move bits must stay clear and only the struggle bit carries through."""
    legal = _legal([], struggle=True)
    mask = Gen3ActionMasker.mask_from_legal(legal)
    assert mask[6:10].tolist() == [0, 0, 0, 0] and mask[10] == 1
    tracker = EpisodeTracker()
    tracker.record(_mock_battle(1), mask, legal=legal)
    tracker.record(_mock_battle(2), mask, legal=legal)
    prev = tracker.prev_mask
    assert prev[6:10].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert prev[10] == 1.0


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
    # _actions[i] is the action advance()d from history[i] (committed at i+1's record).
    # _actions is a deque (bounded when a history_cap is set), so compare contents.
    assert list(tracker._actions) == actions


def test_history_slots_fold_their_own_window():
    """Regression: prev_N_delta_vecs must fold each history slot over ITS OWN decision
    window, not that turn through "now".

    The bug: it fed `events_since(cursor)` (cursor → end of log, no upper bound) for every
    slot, so the fold (which takes the LAST move) made every older slot report the latest
    turn's move. Here turn 1 = Thunderbolt, turn 2 = Ice Beam: the most-recent history slot
    must carry Ice Beam and the previous slot must carry Thunderbolt (not Ice Beam)."""
    import logging
    from agents.battle.gen3_battle import Gen3Battle
    from agents.observation.turn_delta_encoder import OFFSET_OUR_MOVE_BLOCK

    moves = load_mappings()["moves"]
    tbolt = float(moves["thunderbolt"]["num"])
    ibeam = float(moves["icebeam"]["num"])

    def feed(b, lines):
        for ln in lines:
            b.parse_message(ln)

    b = Gen3Battle("battle-gen3ou-window", "p1user", logging.getLogger("ep-win"), gen=3)
    feed(b, [
        ["", "player", "p1", "p1user", "", ""], ["", "player", "p2", "p2user", "", ""],
        ["", "teamsize", "p1", "6"], ["", "teamsize", "p2", "6"], ["", "gen", "3"],
        ["", "start"],
        ["", "switch", "p1a: Zappy", "Zapdos, L100", "100/100"],
        ["", "switch", "p2a: Tyra", "Tyranitar, L100, M", "100/100"], ["", "turn", "1"],
    ])
    enc = _make_encoder()
    m = np.ones(11, dtype=np.int8)
    tr = EpisodeTracker(history_cap=10)

    tr.record(b, m); tr.advance(6)
    feed(b, [
        ["", "move", "p1a: Zappy", "Thunderbolt", "p2a: Tyra"], ["", "-damage", "p2a: Tyra", "70/100"],
        ["", "move", "p2a: Tyra", "Rock Slide", "p1a: Zappy"], ["", "-damage", "p1a: Zappy", "80/100"],
        ["", "turn", "2"],
    ])
    tr.record(b, m); tr.advance(7)
    feed(b, [
        ["", "move", "p1a: Zappy", "Ice Beam", "p2a: Tyra"], ["", "-damage", "p2a: Tyra", "45/100"],
        ["", "move", "p2a: Tyra", "Earthquake", "p1a: Zappy"], ["", "-damage", "p1a: Zappy", "55/100"],
        ["", "turn", "3"],
    ])
    tr.record(b, m)

    vecs = tr.prev_N_delta_vecs(10, enc, battle=b)  # (10, TURN_DELTA_DIM), oldest-first
    assert vecs.shape == (10, TURN_DELTA_DIM)
    assert vecs[-1][OFFSET_OUR_MOVE_BLOCK] == pytest.approx(ibeam)   # most-recent turn
    assert vecs[-2][OFFSET_OUR_MOVE_BLOCK] == pytest.approx(tbolt)   # its OWN turn, not Ice Beam
    assert vecs[-2][OFFSET_OUR_MOVE_BLOCK] != vecs[-1][OFFSET_OUR_MOVE_BLOCK]


def test_history_cache_matches_recompute_and_encodes_once_per_turn():
    """The deque-memoized prev_N_delta_vecs must be BIT-IDENTICAL to recomputing every slot
    from scratch (even past the deque-drop point), while encoding only the newest delta each
    step. Regression for the monotonic-counter bug: using len(history)-1 (which caps with the
    bounded deque) made the cache miss new turns once dropping began."""
    import logging
    from agents.battle.gen3_battle import Gen3Battle

    N = 10
    moves = ["thunderbolt", "icebeam", "hiddenpowergrass", "roar", "rest",
             "protect", "toxic", "spikes", "rockslide", "earthquake"]

    def feed(b, lines):
        for ln in lines:
            b.parse_message(ln)

    def battle():
        b = Gen3Battle("battle-gen3ou-cache", "p1user", logging.getLogger("ep-cache"), gen=3)
        feed(b, [
            ["", "player", "p1", "p1user", "", ""], ["", "player", "p2", "p2user", "", ""],
            ["", "teamsize", "p1", "6"], ["", "teamsize", "p2", "6"], ["", "gen", "3"],
            ["", "start"],
            ["", "switch", "p1a: Zappy", "Zapdos, L100", "100/100"],
            ["", "switch", "p2a: Tyra", "Tyranitar, L100, M", "100/100"],
        ])
        return b

    def play(b, t):
        feed(b, [
            ["", "turn", str(t)],
            ["", "move", "p1a: Zappy", moves[t % len(moves)].title(), "p2a: Tyra"],
            ["", "-damage", "p2a: Tyra", f"{max(5, 100 - t)}/100"],
            ["", "move", "p2a: Tyra", "Rock Slide", "p1a: Zappy"],
            ["", "-damage", "p1a: Zappy", f"{max(5, 100 - t)}/100"],
        ])

    # wrap the cached encoder to count encode() calls
    enc_c = _make_encoder(); calls = [0]
    _orig = enc_c.encode
    enc_c.encode = lambda d: (calls.__setitem__(0, calls[0] + 1), _orig(d))[1]
    enc_r = _make_encoder()  # uncached reference (separate, uncounted)

    def recompute(tr, b, n):
        res = np.zeros((n, enc_r.dimension), dtype=np.float32)
        avail = min(n, len(tr._history) - 1, len(tr._actions), len(tr._cursors))
        for i in range(avail):
            res[n - 1 - i] = tr._encode_delta_slot(i, enc_r, b)
        return res

    b = battle()
    trA = EpisodeTracker(history_cap=N)   # cached path
    trB = EpisodeTracker(history_cap=N)   # uncached reference
    m = np.ones(11, dtype=np.int8)
    T = N + 8  # run PAST the deque-drop point
    for t in range(1, T + 1):
        trA.record(b, m); trA.advance((t % 4) + 6)
        trB.record(b, m); trB.advance((t % 4) + 6)
        cached = trA.prev_N_delta_vecs(N, enc_c, battle=b)
        ref = recompute(trB, b, N)
        assert np.array_equal(cached, ref), f"cache != recompute at turn {t}"
        play(b, t)

    # one encode per completed turn (T-1), not ~N per step
    assert calls[0] == T - 1, f"expected {T - 1} encodes (one per turn), got {calls[0]}"


def test_history_lists_are_bounded_by_cap():
    """With a history_cap, the rolling history/action/cursor deques stay bounded across a
    long episode (the 250-turn-stall memory guard), preserving len(actions)=len(history)-1."""
    tracker = EpisodeTracker(history_cap=10)
    for t in range(50):
        tracker.record(_mock_battle(t), np.ones(11, dtype=np.int8))
        tracker.advance(t % 11)
    assert len(tracker._history) == 11           # N+1
    assert len(tracker._actions) == 10           # N
    assert len(tracker._cursors) == 10
    assert len(tracker._actions) == len(tracker._history) - 1


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
    delta_oldest = build_legacy(ctxs[0], ctxs[1], 0)
    delta_newest = build_legacy(ctxs[3], ctxs[4], 0)
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

def _opp_live_mon(species: str, move_ids: list[str]) -> LivePokemon:
    """Real opponent LivePokemon with the given revealed move ids — the current-board
    form the scan now reads (live.opp.mons / LivePokemon.move_ids). Only the fields the
    scan touches (species, move_ids) carry meaning; the rest are inert placeholders."""
    moves = tuple(LiveMove(id=mid, current_pp=8, max_pp=8) for mid in sorted(move_ids))
    return LivePokemon(
        species=species, active=False, fainted=False, revealed=True,
        hp_fraction=1.0, status=None, types=("normal",), moves=moves,
        item=None, ability=None, boosts={}, volatiles={},
    )


def test_mark_no_hp_fires_when_four_moves_revealed_without_hp():
    """Opp with 4 non-HP moves revealed → tracker marks species ruled-out."""
    tracker = EpisodeTracker()
    battle = _mock_battle(1, opp_live_mons=[
        _opp_live_mon("snorlax", ["bodyslam", "earthquake", "rest", "curse"])
    ])
    tracker.record(battle, np.ones(11, dtype=np.int8))
    assert tracker.hidden_power_tracker.is_known("snorlax") is True
    assert not tracker.hidden_power_tracker.get_probs("snorlax").any()


def test_mark_no_hp_does_not_fire_with_fewer_than_four_moves():
    """3 moves revealed → still uncertain whether HP is in slot 4. No mark yet."""
    tracker = EpisodeTracker()
    battle = _mock_battle(1, opp_live_mons=[
        _opp_live_mon("tauros", ["bodyslam", "earthquake", "return"])
    ])
    tracker.record(battle, np.ones(11, dtype=np.int8))
    assert tracker.hidden_power_tracker.is_known("tauros") is False


def test_mark_no_hp_does_not_fire_when_hp_is_among_four_moves():
    """4 moves including HP → known via observation path, not ruled out."""
    tracker = EpisodeTracker()
    battle = _mock_battle(1, opp_live_mons=[
        _opp_live_mon("jolteon", ["thunderbolt", "shadowball", "substitute", "hiddenpower"])
    ])
    tracker.record(battle, np.ones(11, dtype=np.int8))
    # is_known stays False here — we haven't OBSERVED HP fire yet, just seen
    # it sit in the moveset. Tracker only flips on observe() in that case.
    assert tracker.hidden_power_tracker.is_known("jolteon") is False


def test_mark_no_hp_recognizes_typed_hp_variants():
    """If poke-env exposes the typed key 'hiddenpowergrass', that also blocks rule-out."""
    tracker = EpisodeTracker()
    battle = _mock_battle(1, opp_live_mons=[
        _opp_live_mon("celebi", ["leechseed", "recover", "perishsong", "hiddenpowergrass"])
    ])
    tracker.record(battle, np.ones(11, dtype=np.int8))
    assert tracker.hidden_power_tracker.is_known("celebi") is False


def test_mark_no_hp_persists_across_turns():
    """Once ruled out, the species stays ruled out for subsequent turns."""
    tracker = EpisodeTracker()
    moveset = ["bodyslam", "earthquake", "rest", "curse"]
    b1 = _mock_battle(1, opp_live_mons=[_opp_live_mon("snorlax", moveset)])
    tracker.record(b1, np.ones(11, dtype=np.int8))
    assert tracker.hidden_power_tracker.is_known("snorlax") is True

    # Snorlax still has the same moveset next turn.
    b2 = _mock_battle(2, opp_live_mons=[_opp_live_mon("snorlax", moveset)])
    tracker.record(b2, np.ones(11, dtype=np.int8))
    assert tracker.hidden_power_tracker.is_known("snorlax") is True


def test_reset_clears_rule_out_state():
    """EpisodeTracker.reset() must wipe the rule-out set via tracker.reset()."""
    tracker = EpisodeTracker()
    battle = _mock_battle(1, opp_live_mons=[
        _opp_live_mon("snorlax", ["bodyslam", "earthquake", "rest", "curse"])
    ])
    tracker.record(battle, np.ones(11, dtype=np.int8))
    assert tracker.hidden_power_tracker.is_known("snorlax") is True
    tracker.reset()
    assert tracker.hidden_power_tracker.is_known("snorlax") is False
