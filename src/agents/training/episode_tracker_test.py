"""
Tests for EpisodeTracker._actions sync and prev_N_delta_vecs().

The _actions list is the core of the N-turn history feature: it stores one
action per BattleContext transition so prev_N_delta_vecs() can reconstruct
historical TurnDeltas without re-reading the live battle object.
"""
import numpy as np
from unittest.mock import MagicMock

from agents.training.battle_snapshot import BattleContext
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

# ---------------------------------------------------------------------------
# prev_N_delta_vecs — zero-padding
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# prev_N_delta_vecs — ordering (oldest-first)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# prev_N_delta_vecs — action index is used correctly
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# snapshot() / restore() — the rolling-history rollback that lets the self-play
# opponent RE-DECIDE on a stale request without leaving a phantom turn in its
# turn-history obs. (Driven by RLPlayer.choose_move; see player_test.py for the
# control-flow wiring and the *_fuzz_test for the real-battle invariant.)
# ---------------------------------------------------------------------------

def _hist_state(tr):
    """The full rolling-history state snapshot()/restore() are responsible for, in a
    value-comparable form (BattleContexts compared by turn, arrays by contents)."""
    return (
        [c.turn for c in tr._history],
        list(tr._actions),
        list(tr._cursors),
        tr._n_transitions,
        tr._last_cursor,
        tr._last_action,
    )


def test_snapshot_then_restore_undoes_a_record_exactly():
    legal = _legal([_lm("tackle"), _lm("growl")])
    mask = Gen3ActionMasker.mask_from_legal(legal)
    tr = EpisodeTracker(history_cap=8)
    tr.record(_mock_battle(1), mask, legal=legal); tr.advance(6)
    tr.record(_mock_battle(2), mask, legal=legal); tr.advance(7)
    before = _hist_state(tr)
    snap = tr.snapshot()
    # A "stale" extra record — the would-be decision that gets superseded by a re-decide.
    tr.record(_mock_battle(3), mask, legal=legal); tr.advance(8)
    assert _hist_state(tr) != before        # the record really mutated history + scalars
    tr.restore(snap)
    assert _hist_state(tr) == before         # ... and restore undid every field exactly


def test_restore_recovers_entry_maxlen_would_have_dropped():
    # history_cap=2 → the aux deques (_actions/_cursors) hold maxlen=2.
    # gen3_frame_deletion_v1: `_hist_vec_cache` went with the lag frames it memoized; the
    # rollback claim is unchanged and still worth pinning on the deques that remain.
    # A surgical length-only rollback would lose the dropped-oldest entry; snapshotting the
    # contents restores it exactly even at the cap.
    tr = EpisodeTracker(history_cap=2)
    tr._actions.extend([10, 11])
    tr._cursors.extend([100, 110])
    snap = tr.snapshot()
    tr._actions.append(12); tr._cursors.append(120)
    assert list(tr._actions) == [11, 12]     # oldest (10) dropped by the bounded deque
    tr.restore(snap)
    assert list(tr._actions) == [10, 11]      # dropped entry recovered exactly
    assert list(tr._cursors) == [100, 110]


def test_redecide_pattern_leaves_no_phantom_turn():
    """Mirror RLPlayer.choose_move's rollback: a stale attempt records then restores; only the
    committed (re-decided) turn survives, so the rolling history grows by exactly one."""
    legal = _legal([_lm("tackle"), _lm("growl")])
    mask = Gen3ActionMasker.mask_from_legal(legal)
    tr = EpisodeTracker(history_cap=8)
    tr.record(_mock_battle(1), mask, legal=legal); tr.advance(6)
    tr.record(_mock_battle(2), mask, legal=legal); tr.advance(7)
    n0 = len(tr._history)
    snap = tr.snapshot()
    tr.record(_mock_battle(3), mask, legal=legal)   # stale attempt (turn 3) — superseded
    tr.restore(snap)                                  # rolled back, no phantom
    tr.record(_mock_battle(3), mask, legal=legal)   # the committed re-decided turn
    tr.advance(8)
    assert len(tr._history) == n0 + 1                 # exactly one new turn, not two
    assert [c.turn for c in tr._history] == [1, 2, 3]
    assert tr._n_transitions == 2                      # transitions 1→2 and 2→3 only
