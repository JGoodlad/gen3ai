from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from agents.training.reward_tracker import RewardTracker, RewardTrackingMixin
from agents.training.slot_registry import SlotRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tracker(reward_value=0.5):
    """RewardTracker with a mocked reward function that returns reward_value."""
    reward_fn = MagicMock()
    reward_fn.process_turn_reward.return_value = reward_value
    reward_fn_factory = MagicMock(return_value=reward_fn)

    our_slots = SlotRegistry()
    opp_slots = SlotRegistry()
    tracker = RewardTracker(reward_fn_factory, our_slots, opp_slots)
    return tracker, reward_fn


def _fake_ctx():
    return MagicMock()


def _fake_battle(tag="battle-gen3ou-1"):
    battle = MagicMock()
    battle.battle_tag = tag
    return battle


# ---------------------------------------------------------------------------
# RewardTracker — pending state
# ---------------------------------------------------------------------------

def test_has_pending_false_initially():
    tracker, _ = _make_tracker()
    assert not tracker.has_pending


def test_has_pending_true_after_begin_turn():
    tracker, _ = _make_tracker()
    tracker.begin_turn(_fake_ctx(), 0)
    assert tracker.has_pending


def test_has_pending_false_after_complete_pending():
    tracker, _ = _make_tracker()
    tracker.begin_turn(_fake_ctx(), 0)

    with patch("agents.training.reward_tracker.TurnDelta") as mock_td, \
         patch("agents.training.reward_tracker.BattleContext") as mock_bc:
        mock_td.build.return_value = MagicMock()
        tracker.complete_pending(_fake_ctx(), _fake_battle())

    assert not tracker.has_pending


def test_has_pending_false_after_finalize():
    tracker, _ = _make_tracker()
    tracker.begin_turn(_fake_ctx(), 0)

    with patch("agents.training.reward_tracker.TurnDelta") as mock_td, \
         patch("agents.training.reward_tracker.BattleContext") as mock_bc:
        mock_td.build.return_value = MagicMock()
        mock_bc.from_battle.return_value = _fake_ctx()
        tracker.finalize(_fake_battle())

    assert not tracker.has_pending


# ---------------------------------------------------------------------------
# RewardTracker — ProgressClock wiring (eval/training reward parity)
# ---------------------------------------------------------------------------

def test_progress_clock_wired_to_reward_fn():
    """The server-free reward path must OWN a ProgressClock and hand the SAME instance to the reward
    manager — else no_progress_tax is silently 0 in eval traces (the eval/training reward gap)."""
    import functools
    from agents.training.reward_manager import Gen3RewardManager, RewardConfig

    factory = functools.partial(Gen3RewardManager, config=RewardConfig(all_shaping_pbrs=True))
    tracker = RewardTracker(factory, SlotRegistry(), SlotRegistry())
    assert tracker._progress_clock is not None
    assert tracker._reward_fn.progress_clock is tracker._progress_clock
    assert tracker._progress_clock.no_progress_penalty == pytest.approx(0.15)


def test_advance_clock_charges_a_noop():
    """Advancing the clock on a futile no-op window (RapidSpin into a full-HP wall — no damage event,
    no hazard, no opp switch) charges the no-progress tax, exactly as training does. Before this wiring
    the same eval window scored 0 (clock absent)."""
    import functools
    from agents.training.reward_manager import Gen3RewardManager, RewardConfig
    from agents.training.reward_redesign_test import _delta, _Legal, _full_team_live

    factory = functools.partial(Gen3RewardManager, config=RewardConfig(all_shaping_pbrs=True))
    tracker = RewardTracker(factory, SlotRegistry(), SlotRegistry())

    battle = MagicMock()
    view = MagicMock(); view.live = _full_team_live(); view.legal = _Legal(switches=[1])
    battle.strict_view.return_value = view

    tracker._advance_clock(_delta(our_move_id="rapidspin", our_move_outcome="hit"), battle)
    assert tracker._progress_clock.last_penalty == pytest.approx(-0.15)


# ---------------------------------------------------------------------------
# RewardTracker — reward accumulation
# ---------------------------------------------------------------------------

def test_total_reward_zero_initially():
    tracker, _ = _make_tracker()
    assert tracker.total_reward == 0.0


def test_complete_pending_accumulates_reward():
    tracker, _ = _make_tracker(reward_value=1.0)
    tracker.begin_turn(_fake_ctx(), 0)

    with patch("agents.training.reward_tracker.TurnDelta") as mock_td, \
         patch("agents.training.reward_tracker.BattleContext"):
        mock_td.build.return_value = MagicMock()
        tracker.complete_pending(_fake_ctx(), _fake_battle())

    assert tracker.total_reward == pytest.approx(1.0)


def test_reward_accumulates_across_multiple_turns():
    tracker, _ = _make_tracker(reward_value=0.5)

    with patch("agents.training.reward_tracker.TurnDelta") as mock_td, \
         patch("agents.training.reward_tracker.BattleContext"):
        mock_td.build.return_value = MagicMock()

        tracker.begin_turn(_fake_ctx(), 0)
        tracker.complete_pending(_fake_ctx(), _fake_battle())

        tracker.begin_turn(_fake_ctx(), 1)
        tracker.complete_pending(_fake_ctx(), _fake_battle())

    assert tracker.total_reward == pytest.approx(1.0)


def test_finalize_adds_to_total_reward():
    tracker, _ = _make_tracker(reward_value=2.0)
    tracker.begin_turn(_fake_ctx(), 0)

    with patch("agents.training.reward_tracker.TurnDelta") as mock_td, \
         patch("agents.training.reward_tracker.BattleContext") as mock_bc:
        mock_td.build.return_value = MagicMock()
        mock_bc.from_battle.return_value = _fake_ctx()
        tracker.finalize(_fake_battle())

    assert tracker.total_reward == pytest.approx(2.0)


def test_complete_pending_returns_delta_and_reward():
    tracker, _ = _make_tracker(reward_value=0.7)
    tracker.begin_turn(_fake_ctx(), 0)

    fake_delta = MagicMock()
    with patch("agents.training.reward_tracker.TurnDelta") as mock_td, \
         patch("agents.training.reward_tracker.BattleContext"):
        mock_td.build_from_events.return_value = fake_delta
        delta, reward = tracker.complete_pending(_fake_ctx(), _fake_battle())

    assert delta is fake_delta
    assert reward == pytest.approx(0.7)


def test_finalize_returns_terminal_ctx_delta_reward():
    tracker, _ = _make_tracker(reward_value=1.5)
    tracker.begin_turn(_fake_ctx(), 0)

    fake_terminal = _fake_ctx()
    fake_delta = MagicMock()
    with patch("agents.training.reward_tracker.TurnDelta") as mock_td, \
         patch("agents.training.reward_tracker.BattleContext") as mock_bc:
        mock_td.build_from_events.return_value = fake_delta
        mock_bc.from_battle.return_value = fake_terminal
        terminal_ctx, delta, reward = tracker.finalize(_fake_battle())

    assert terminal_ctx is fake_terminal
    assert delta is fake_delta
    assert reward == pytest.approx(1.5)


def test_record_action_called_with_pending_ctx_and_action():
    tracker, reward_fn = _make_tracker()
    prev_ctx = _fake_ctx()
    tracker.begin_turn(prev_ctx, 3)

    with patch("agents.training.reward_tracker.TurnDelta") as mock_td, \
         patch("agents.training.reward_tracker.BattleContext"):
        mock_td.build.return_value = MagicMock()
        tracker.complete_pending(_fake_ctx(), _fake_battle())

    reward_fn.record_action.assert_called_once_with(prev_ctx, 3)


# ---------------------------------------------------------------------------
# RewardTrackingMixin — via a minimal stub class
# ---------------------------------------------------------------------------

class _CallbackBase:
    """Stand-in for poke-env Player's no-op _battle_finished_callback."""
    def _battle_finished_callback(self, battle):
        pass


class _MixinStub(RewardTrackingMixin, _CallbackBase):
    """Minimal stub: MRO is _MixinStub → RewardTrackingMixin → _CallbackBase → object."""

    def __init__(self, reward_fn_factory):
        self._battles: dict = {}
        self._init_reward_tracking(reward_fn_factory)


def _make_mixin(reward_value=1.0):
    reward_fn = MagicMock()
    reward_fn.process_turn_reward.return_value = reward_value
    factory = MagicMock(return_value=reward_fn)
    stub = _MixinStub(factory)
    return stub, reward_fn


def test_mean_reward_empty_before_battles():
    stub, _ = _make_mixin()
    assert stub.mean_episode_reward == pytest.approx(0.0)


def test_separate_trackers_per_battle_tag():
    stub, _ = _make_mixin()

    battle_a = _fake_battle("tag-A")
    battle_b = _fake_battle("tag-B")
    mask = np.ones(11, dtype=np.float32)

    with patch("agents.training.reward_tracker.BattleContext") as mock_bc, \
         patch("agents.training.reward_tracker.TurnDelta"):
        mock_bc.from_battle.return_value = _fake_ctx()
        stub._track_reward(battle_a, 0, mask)
        stub._track_reward(battle_b, 1, mask)

    assert "tag-A" in stub._reward_trackers
    assert "tag-B" in stub._reward_trackers
    assert stub._reward_trackers["tag-A"] is not stub._reward_trackers["tag-B"]


def test_battle_finished_callback_finalizes_reward():
    stub, _ = _make_mixin(reward_value=3.0)
    mask = np.ones(11, dtype=np.float32)
    battle = _fake_battle("tag-1")

    with patch("agents.training.reward_tracker.BattleContext") as mock_bc, \
         patch("agents.training.reward_tracker.TurnDelta") as mock_td:
        mock_bc.from_battle.return_value = _fake_ctx()
        mock_td.build.return_value = MagicMock()

        stub._track_reward(battle, 0, mask)
        stub._battle_finished_callback(battle)

    assert "tag-1" in stub._episode_rewards
    assert stub._episode_rewards["tag-1"] == pytest.approx(3.0)
    assert stub.mean_episode_reward == pytest.approx(3.0)


def test_reset_clears_all_state():
    stub, _ = _make_mixin(reward_value=1.0)
    mask = np.ones(11, dtype=np.float32)
    battle = _fake_battle("tag-1")

    with patch("agents.training.reward_tracker.BattleContext") as mock_bc, \
         patch("agents.training.reward_tracker.TurnDelta") as mock_td:
        mock_bc.from_battle.return_value = _fake_ctx()
        mock_td.build.return_value = MagicMock()
        stub._track_reward(battle, 0, mask)
        stub._battle_finished_callback(battle)

    stub.reset_reward_tracking()

    assert stub._reward_trackers == {}
    assert stub._episode_rewards == {}
    assert stub.mean_episode_reward == pytest.approx(0.0)


def test_mean_reward_averages_multiple_battles():
    stub, reward_fn = _make_mixin()
    mask = np.ones(11, dtype=np.float32)

    # First battle: reward 2.0, second: 4.0 → mean = 3.0
    reward_fn.process_turn_reward.side_effect = [2.0, 4.0]

    b1 = _fake_battle("tag-1")
    b2 = _fake_battle("tag-2")

    with patch("agents.training.reward_tracker.BattleContext") as mock_bc, \
         patch("agents.training.reward_tracker.TurnDelta") as mock_td:
        mock_bc.from_battle.return_value = _fake_ctx()
        mock_td.build.return_value = MagicMock()

        stub._track_reward(b1, 0, mask)
        stub._battle_finished_callback(b1)

        stub._track_reward(b2, 0, mask)
        stub._battle_finished_callback(b2)

    assert stub.mean_episode_reward == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# RewardTrackingMixin — concurrent-battle interleaving
# ---------------------------------------------------------------------------

def test_interleaved_calls_use_correct_pending_ctx_per_battle():
    """
    A's pending ctx must not be overwritten when B's _track_reward fires between A's turns.
    Sequence: A-turn1, B-turn1, A-turn2 (completing A's turn1 reward).
    The complete_pending for A must use ctx_a1 (not ctx_b1) as prev_ctx.
    """
    stub, reward_fn = _make_mixin()
    mask = np.ones(11, dtype=np.float32)

    ba = _fake_battle("tag-A")
    bb = _fake_battle("tag-B")

    ctx_a1, ctx_b1, ctx_a2 = _fake_ctx(), _fake_ctx(), _fake_ctx()
    record_action_calls = []
    reward_fn.record_action.side_effect = lambda ctx, action: record_action_calls.append((ctx, action))

    with patch("agents.training.reward_tracker.BattleContext") as mock_bc, \
         patch("agents.training.reward_tracker.TurnDelta") as mock_td:
        mock_bc.from_battle.side_effect = [ctx_a1, ctx_b1, ctx_a2]
        mock_td.build.return_value = MagicMock()

        stub._track_reward(ba, 0, mask)  # A: begin_turn(ctx_a1, 0)
        stub._track_reward(bb, 1, mask)  # B: begin_turn(ctx_b1, 1) — must not touch A
        stub._track_reward(ba, 2, mask)  # A: complete_pending(prev=ctx_a1, curr=ctx_a2)

    # record_action was called once (for A's complete_pending) with ctx_a1, not ctx_b1
    assert len(record_action_calls) == 1
    prev_ctx_used, action_used = record_action_calls[0]
    assert prev_ctx_used is ctx_a1
    assert action_used == 0


def test_interleaved_battles_accumulate_independent_rewards():
    """
    Interleaved choose_move() sequence: A1, B1, A2, B2, finish-A, finish-B.
    Each battle's total_reward must reflect only its own turns.
    """
    stub, reward_fn = _make_mixin()
    mask = np.ones(11, dtype=np.float32)
    # turn rewards in call order: A complete→1.0, B complete→2.0, A finalize→3.0, B finalize→4.0
    reward_fn.process_turn_reward.side_effect = [1.0, 2.0, 3.0, 4.0]

    ba = _fake_battle("tag-A")
    bb = _fake_battle("tag-B")

    with patch("agents.training.reward_tracker.BattleContext") as mock_bc, \
         patch("agents.training.reward_tracker.TurnDelta") as mock_td:
        mock_bc.from_battle.return_value = _fake_ctx()
        mock_td.build.return_value = MagicMock()

        stub._track_reward(ba, 0, mask)   # A: begin_turn(ctx_a1, 0)
        stub._track_reward(bb, 1, mask)   # B: begin_turn(ctx_b1, 1)
        stub._track_reward(ba, 2, mask)   # A: complete_pending → reward 1.0; begin_turn(ctx_a2, 2)
        stub._track_reward(bb, 3, mask)   # B: complete_pending → reward 2.0; begin_turn(ctx_b2, 3)
        stub._battle_finished_callback(ba)  # A: finalize → reward 3.0; total_A = 4.0
        stub._battle_finished_callback(bb)  # B: finalize → reward 4.0; total_B = 6.0

    assert stub._episode_rewards["tag-A"] == pytest.approx(4.0)
    assert stub._episode_rewards["tag-B"] == pytest.approx(6.0)
    assert stub.mean_episode_reward == pytest.approx(5.0)  # (4+6)/2


def test_battle_finished_for_a_does_not_finalize_b():
    """Finishing battle A must leave B's tracker still pending."""
    stub, _ = _make_mixin()
    mask = np.ones(11, dtype=np.float32)

    ba = _fake_battle("tag-A")
    bb = _fake_battle("tag-B")

    with patch("agents.training.reward_tracker.BattleContext") as mock_bc, \
         patch("agents.training.reward_tracker.TurnDelta") as mock_td:
        mock_bc.from_battle.return_value = _fake_ctx()
        mock_td.build.return_value = MagicMock()

        stub._track_reward(ba, 0, mask)
        stub._track_reward(bb, 1, mask)
        stub._battle_finished_callback(ba)  # only A finished

    # A is finalized; B still has a pending turn
    assert "tag-A" in stub._episode_rewards
    assert "tag-B" not in stub._episode_rewards
    assert stub._reward_trackers["tag-B"].has_pending


def test_reset_after_concurrent_battles_clears_all():
    """reset_reward_tracking() must clear state from all battles regardless of count."""
    stub, _ = _make_mixin()
    mask = np.ones(11, dtype=np.float32)

    with patch("agents.training.reward_tracker.BattleContext") as mock_bc, \
         patch("agents.training.reward_tracker.TurnDelta") as mock_td:
        mock_bc.from_battle.return_value = _fake_ctx()
        mock_td.build.return_value = MagicMock()

        for tag in ["tag-A", "tag-B", "tag-C"]:
            b = _fake_battle(tag)
            stub._track_reward(b, 0, mask)
            stub._battle_finished_callback(b)

    assert len(stub._episode_rewards) == 3
    stub.reset_reward_tracking()
    assert stub._reward_trackers == {}
    assert stub._episode_rewards == {}
    assert stub.mean_episode_reward == pytest.approx(0.0)
