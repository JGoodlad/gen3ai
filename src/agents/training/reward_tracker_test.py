from unittest.mock import MagicMock, patch, call
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
        mock_td.build.return_value = fake_delta
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
        mock_td.build.return_value = fake_delta
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
