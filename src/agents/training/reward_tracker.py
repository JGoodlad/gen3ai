import numpy as np

from agents.training.battle_snapshot import BattleContext
from agents.training.turn_delta import TurnDelta
from agents.training.reward_function import RewardFunction
from agents.training.reward_manager import Gen3RewardManager
from agents.training.progress_clock import ProgressClock
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
        self._pending_cursor: int = 0   # event_cursor when the pending turn was latched
        self._total_reward: float = 0.0
        # Server-free reward path (BattleRecorder / RewardTrackingMixin) has no Gen3Env to own the
        # EpisodeTracker's ProgressClock, so the reward manager's no_progress_tax would be silently 0
        # (progress_clock=None gates _apply_progress_clock off) — eval traces then understated the
        # training penalty on every stall/no-op turn. Own a per-battle clock here and advance it before
        # each reward, mirroring Gen3Env's embed-time timing (update for the just-completed window →
        # reward reads last_penalty). Only the gate (all_shaping_pbrs / bias_redesign in the reward
        # config) decides whether the tax actually fires, so a default-config run stays a no-op.
        self._progress_clock: ProgressClock | None = None
        if hasattr(self._reward_fn, "progress_clock"):
            cfg = getattr(self._reward_fn, "config", None)
            penalty = getattr(cfg, "no_progress_penalty", 0.15)
            self._progress_clock = ProgressClock(no_progress_penalty=penalty)
            self._reward_fn.progress_clock = self._progress_clock

    @property
    def has_pending(self) -> bool:
        return self._pending_ctx is not None

    @property
    def pending_ctx(self) -> BattleContext | None:
        return self._pending_ctx

    @property
    def total_reward(self) -> float:
        return self._total_reward

    def begin_turn(self, ctx: BattleContext, action_idx: int, cursor: int = 0) -> None:
        """Latch the current turn's pre-action state for deferred reward computation.

        ``cursor`` is ``battle.event_cursor`` at THIS decision — the start of the event
        window the next ``complete_pending`` / ``finalize`` folds the TurnDelta from.
        """
        self._pending_ctx = ctx
        self._pending_action = action_idx
        self._pending_cursor = cursor

    def _window(self, battle) -> list:
        """The event window for the pending turn: everything since it was latched."""
        fn = getattr(battle, "events_since", None)
        return fn(self._pending_cursor) if fn is not None else []

    def _advance_clock(self, delta: TurnDelta, battle) -> None:
        """Fold the just-completed window into the ProgressClock BEFORE the reward reads its
        ``last_penalty`` — same window, same order as Gen3Env (embed-time update → calc_reward read).
        Reads the current board / legality through the StrictBattleView, like the env does. No-op when
        no clock is owned (non-Gen3RewardManager reward fn)."""
        if self._progress_clock is None:
            return
        view = battle.strict_view()
        self._progress_clock.update(delta, view.live, view.legal)

    def complete_pending(self, curr_ctx: BattleContext, battle) -> tuple[TurnDelta, float]:
        """
        Settle the previous turn now that the next choose_move() has fired and curr_ctx
        reflects the post-battle state. Returns (delta, reward) so BattleRecorder can
        use them for its JSON outcome entry without re-computing. Folds the TurnDelta
        from the event window (``events_since`` the pending cursor) — the production path.
        """
        delta = TurnDelta.build_from_events(
            self._pending_ctx, curr_ctx, self._pending_action, self._window(battle)
        )
        self._reward_fn.record_action(self._pending_ctx, self._pending_action)
        self._advance_clock(delta, battle)
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
        events = self._window(battle)
        terminal_ctx = BattleContext.from_battle(
            battle,
            np.zeros(11, dtype=np.float32),
            self._our_slots,
            self._opp_slots,
        )
        delta = TurnDelta.build_from_events(
            self._pending_ctx, terminal_ctx, self._pending_action, events
        )
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
        tracker.begin_turn(curr_ctx, action_idx, getattr(battle, "event_cursor", 0))

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

    @property
    def episode_reward_sum(self) -> float:
        """Σ of per-battle total rewards this matchup — the additive numerator a sharded eval
        pools across workers (parent recovers the mean as Σreward / Σn_episodes, exactly)."""
        return sum(self._episode_rewards.values())

    @property
    def n_reward_episodes(self) -> int:
        """Count of finished battles that contributed a reward (the pooling denominator)."""
        return len(self._episode_rewards)

    def reset_reward_tracking(self) -> None:
        self._reward_trackers.clear()
        self._episode_rewards.clear()
