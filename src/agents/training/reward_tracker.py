import numpy as np

from agents.action.ordering_integrity import check_outcome_matches_intent
from agents.training.battle_context import BattleContext, TurnDelta
from agents.training.reward_function import RewardFunction
from agents.training.reward_manager import Gen3RewardManager
from agents.training.slot_registry import SlotRegistry


class RewardTracker:
    """
    Tracks per-turn reward for a single battle without Gen3Env.

    Implements the same deferred pattern as BattleRecorder / Gen3Env:
    - begin_turn(ctx, action): latch pre-action state (≈ embed_battle + tracker.advance)
    - complete_pending(curr_ctx, battle): settle previous turn once next choose_move fires
      (≈ Gen3Env.calc_reward after super().step() returns)
    - finalize(battle): settle last pending turn after battle ends via _battle_finished_callback

    Both complete_pending and finalize call record_action(prev_ctx, action) then
    process_turn_reward(battle, delta) — same order as Gen3Env.step().
    """

    def __init__(self, reward_fn_factory, our_slots: SlotRegistry, opp_slots: SlotRegistry):
        self._reward_fn: RewardFunction = reward_fn_factory()
        self._reward_fn.reset()
        self._our_slots = our_slots
        self._opp_slots = opp_slots
        self._pending_ctx: BattleContext | None = None
        self._pending_action: int = -1
        self._total_reward: float = 0.0

    @property
    def has_pending(self) -> bool:
        return self._pending_ctx is not None

    @property
    def pending_ctx(self) -> BattleContext | None:
        return self._pending_ctx

    @property
    def total_reward(self) -> float:
        return self._total_reward

    def begin_turn(self, ctx: BattleContext, action_idx: int) -> None:
        """Latch the current turn's pre-action state for deferred reward computation."""
        self._pending_ctx = ctx
        self._pending_action = action_idx

    def complete_pending(self, curr_ctx: BattleContext, battle) -> tuple[TurnDelta, float]:
        """
        Settle the previous turn now that the next choose_move() has fired and curr_ctx
        reflects the post-battle state. Returns (delta, reward) so BattleRecorder can
        use them for its JSON outcome entry without re-computing.
        """
        delta = TurnDelta.build(self._pending_ctx, curr_ctx, self._pending_action)
        check_outcome_matches_intent(
            self._pending_ctx.active_move_ids, delta, self._pending_action
        )
        self._reward_fn.record_action(self._pending_ctx, self._pending_action)
        reward = self._reward_fn.process_turn_reward(battle, delta)
        self._total_reward += reward
        self._pending_ctx = None
        return delta, reward

    def finalize(self, battle) -> tuple[BattleContext, TurnDelta, float]:
        """
        Settle the last pending turn after the battle ends. Builds a terminal BattleContext
        with all-zero mask (no valid moves). Returns (terminal_ctx, delta, reward) so
        BattleRecorder can build its terminal JSON entry.
        """
        terminal_ctx = BattleContext.from_battle(
            battle,
            np.zeros(11, dtype=np.float32),
            self._our_slots,
            self._opp_slots,
        )
        delta = TurnDelta.build(self._pending_ctx, terminal_ctx, self._pending_action)
        self._reward_fn.record_action(self._pending_ctx, self._pending_action)
        reward = self._reward_fn.process_turn_reward(battle, delta)
        self._total_reward += reward
        self._pending_ctx = None
        return terminal_ctx, delta, reward


class RewardTrackingMixin:
    """
    Mixin for RLPlayer subclasses that tracks per-episode reward without Gen3Env.

    Hooks into two poke-env Player lifecycle points:
    - choose_move(): call _track_reward(battle, action_idx, mask) after choosing
    - _battle_finished_callback(): auto-finalization (override included here)

    After battle_against() returns, all _battle_finished_callback calls have already
    fired, so mean_episode_reward is ready to read. Call reset_reward_tracking()
    before starting the next batch of battles.
    """

    def _init_reward_tracking(self, reward_fn_factory=Gen3RewardManager) -> None:
        self._reward_fn_factory = reward_fn_factory
        self._reward_trackers: dict[str, RewardTracker] = {}
        self._episode_rewards: dict[str, float] = {}

    def _track_reward(self, battle, action_idx: int, mask: np.ndarray) -> None:
        """Call from choose_move() after computing action_idx."""
        tag = battle.battle_tag
        if tag not in self._reward_trackers:
            our_slots, opp_slots = SlotRegistry(), SlotRegistry()
            self._reward_trackers[tag] = RewardTracker(self._reward_fn_factory, our_slots, opp_slots)
        tracker = self._reward_trackers[tag]
        curr_ctx = BattleContext.from_battle(
            battle, mask, tracker._our_slots, tracker._opp_slots
        )
        if tracker.has_pending:
            tracker.complete_pending(curr_ctx, battle)
        tracker.begin_turn(curr_ctx, action_idx)

    def _battle_finished_callback(self, battle) -> None:
        """Override poke-env's no-op to finalize reward tracking when each battle ends."""
        super()._battle_finished_callback(battle)
        tag = battle.battle_tag
        if tag in self._reward_trackers:
            tracker = self._reward_trackers[tag]
            if tracker.has_pending:
                tracker.finalize(battle)
            self._episode_rewards[tag] = tracker.total_reward

    @property
    def mean_episode_reward(self) -> float:
        if not self._episode_rewards:
            return 0.0
        return sum(self._episode_rewards.values()) / len(self._episode_rewards)

    def reset_reward_tracking(self) -> None:
        self._reward_trackers.clear()
        self._episode_rewards.clear()
