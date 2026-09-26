"""`Gen3RewardManager` — the per-decision reward, which is the TERMINAL alone.

**The shaped reward path is DELETED** (`gen3_shaped_reward_deletion_v1`, program_rust_core §4 M3
row, owner-approved 2026-09-26): the eight PBRS potentials (`reward_potentials.py`), the ~25 BIAS
terms (`reward_bias_terms.py`), the bias-additivity refund, the no-progress tax, the suppressed-term
fast path and its `GEN3AI_REWARD_VERIFY` shadow twin (`reward_verify.py`), and the 14 flags that
configured them (`designs/deleted_flags.md`). Production had trained on the terminal alone since the
win-prob era (`--critic winprob --terminal-indicator --victory-value 1.0`); `reward_golden_test`
was recorded at the last pre-deletion commit and passes unchanged after it, which is the proof that
production's reward — and the `win_margin` obs key this module publishes — did not move.

A checkpoint that TRAINED with shaping cannot be resumed or forked on this code: see
`agents.model.model_version.shaped_reward`.

What stays: the terminal (indicator or signed, `RewardConfig`), the `reward/` live export
(`reward_term_stats`), the material MARGIN by-product (`material_margin.py` — the `win_margin`
training-only obs key, not a reward term), and the episode counters the `🏁 Episode Finished` line
prints. Every public name of `reward_config` / `reward_composition` / `reward_weights` is
re-exported here, so `from agents.training.reward_manager import RewardConfig` still resolves.
"""
from typing import Optional

from utils.logging.rate_limiter import RateLimitedLogger
from utils.logging.levels import LogLevel
from agents.training.battle_snapshot import BattleContext
from agents.training.material_margin import material_margin as _material_margin
from agents.training.turn_delta import TurnDelta
from agents.training.reward_term_stats import (
    RewardTermAccumulator as _RewardTermAccumulator,
    tracked_terms as _tracked_terms,
)

# --- The reward's DECLARATIONS, CENSUS and MAGNITUDES — re-exported (explicit, so a name dropped
# from the other side is an ImportError at import time). ---
from agents.training.reward_config import (   # noqa: E402,F401 - declared re-export hub
    RewardClass, RewardConfig, RewardBreakdown)
from agents.training.reward_composition import (   # noqa: E402,F401 - declared re-export hub
    _rc, reward_class_composition, reward_config_digest, format_reward_composition,
    inert_reward_flags, reward_composition_block,
)
from agents.training.reward_weights import (   # noqa: E402,F401 - declared re-export hub
    VICTORY_VALUE, _TIMEOUT_TURN_CAP, PBRS_GAMMA)


class Gen3RewardManager:
    """
    Reward calculator for the Gen 3 RL trainee agent: the TERMINAL, per `RewardConfig`.

      - record_action()       — records the chosen action (episode counters only)
      - process_turn_reward() — the turn's reward: `victory_value` on a win; on any other terminal
                                0.0 (`terminal_indicator`) or −victory_value / `draw_penalty` at
                                the turn cap (signed); 0.0 on every non-terminal turn

    Satisfies the RewardFunction protocol. Must only be called for the trainee's battle — the env
    gates this at the call site.
    """
    def __init__(self, log_level: LogLevel = LogLevel.QUIET,
                 config: Optional[RewardConfig] = None):
        self.log_level = log_level
        self.config = config or RewardConfig()
        self.switch_count = 0
        self.forced_switch_count = 0
        self.attack_count = 0
        self.struggle_turns = 0
        self.total_reward = 0.0
        self.logger = RateLimitedLogger(interval_seconds=1.0)
        self.episode_logger = RateLimitedLogger(interval_seconds=5.0)
        self._last_reward_metadata: dict = {}
        # Normalized material margin ∈ [−1,1], recomputed every turn from the LiveView. Read by
        # gen3_env's `win_margin` obs key (the win-prob head's closeness-stratified metrics). NOT a
        # reward term — see `material_margin.py`.
        self._last_material_margin: float = 0.0
        # The `reward/` live export (`gen3_reward_term_export_v1`) — per-decision sums of the
        # tracked terms, drained by `env_method` once per rollout.
        self._term_stats = _RewardTermAccumulator(
            _tracked_terms(reward_class_composition(self.config)))

    def drain_reward_terms(self):
        """Drain-and-zero the `reward/` term accumulator (`env_method` PULL, once per rollout)."""
        return self._term_stats.drain()

    def reset(self):
        self.switch_count = 0
        self.forced_switch_count = 0
        self.attack_count = 0
        self.struggle_turns = 0
        self.total_reward = 0.0
        self._last_reward_metadata = {}
        self._last_material_margin = 0.0

    def record_action(self, ctx: BattleContext, action: int) -> None:
        """Record the action the model chose for this turn (called before the turn is processed,
        with the context the model saw). Feeds only the episode counters of `report_episode`."""
        if action >= 6:
            self.attack_count += 1
            if action == 10:          # struggle — forced by the server when all PP is depleted
                self.struggle_turns += 1
            self._last_reward_metadata = {"type": "ATTACK"}
        elif ctx.phase == "forced_switch":
            self.forced_switch_count += 1
            self._last_reward_metadata = {"type": "FORCED"}
        else:
            # A VOLUNTARY switch is counted at OUTCOME time (`process_turn_reward`): a pressed switch
            # can silently fail to execute (poke-env "gap=0"), and the counter reports switches made.
            self._last_reward_metadata = {"type": "VOLUNTARY"}

    def process_turn_reward(self, battle, delta: TurnDelta) -> float:
        """The reward for a completed turn. Builds the `RewardBreakdown` and stores it on
        ``self._last_breakdown`` so callers (e.g. BattleRecorder) can read it."""
        bd = RewardBreakdown()
        live = battle.live_view()

        # --- TERMINAL: the win/loss ---
        victory = float(self.config.victory_value)
        indicator = bool(getattr(self.config, "terminal_indicator", False))
        won, lost, finished = self._terminal(live)
        if won:
            bd.win_loss = victory
        elif finished and indicator:
            # gen3_winprob_critic_mode_v1: the WIN INDICATOR. Every non-win terminal — decisive
            # loss, pre-cap tie, 250-turn timeout — pays exactly 0.0, so the undiscounted return
            # is `victory_value * 1{win}` and (at victory_value 1.0) V(s) == P(win|s) exactly.
            bd.win_loss = 0.0
        elif finished:
            # SIGNED terminal. A no-progress STALL ends with the trainee FORFEITING at the turn cap
            # (gen3_env issues ForfeitBattleOrder at turn>=cap → lost=True, turn>=cap) — NOT a tie —
            # so the timeout is detected by the turn count. A timeout takes `draw_penalty`; a
            # DECISIVE loss — and the rare PRE-CAP TIE, which shares this branch — −victory_value.
            timed_out = live.turn >= _TIMEOUT_TURN_CAP
            bd.win_loss = self.config.draw_penalty if timed_out else -victory

        # The `win_margin` obs key's source (NOT a reward term; see `material_margin.py`).
        self._last_material_margin = _material_margin(live)

        meta = self._last_reward_metadata
        if meta.get("type") == "VOLUNTARY" and delta.our_switch_to is not None:
            self.switch_count += 1

        self._last_breakdown = bd
        reward = bd.total
        # +REWARD EXPORT: `reward` is passed rather than re-derived, so `reward/untracked_abs_mean`
        # compares against the number training actually saw.
        self._term_stats.observe(bd, reward)
        self.total_reward += reward
        if self.log_level >= LogLevel.DETAILED and self.logger.should_log():
            self.logger.log(f"  [REWARD] Turn {battle.turn} | Terminal: {bd.win_loss:+.4f} | "
                            f"Won: {battle.won}\n", force=True)
        return reward

    @staticmethod
    def _terminal(live):
        """(won, lost, finished) from the LiveView's result meta."""
        return bool(live.won), bool(live.lost), bool(live.finished)

    def report_episode(self, battle):
        if self.log_level < LogLevel.PERIODIC or self.total_reward == 0:
            return

        status = "UNKNOWN"
        our_alive = opp_alive = turns = 0
        if battle:
            # alive counts via the LiveView read-model; win/lost/finished/turn are meta
            # scalars (allowed off the battle directly).
            live = battle.live_view()
            if battle.won: status = "WIN"
            elif battle.lost: status = "LOSS"
            elif battle.finished: status = "TIE/STALL"
            our_alive = sum(1 for m in live.ours.mons if not m.fainted)
            opp_alive = sum(1 for m in live.opp.mons if not m.fainted)
            turns = battle.turn

        self.episode_logger.log(
            f"\n🏁 Episode Finished | Reward: {self.total_reward:6.2f} | Status: {status:4} | "
            f"Mon: {our_alive} vs {opp_alive} | Turns: {turns:3} | "
            f"Attacks: {self.attack_count:2} | Sw(Vol): {self.switch_count:2} | Sw(For): {self.forced_switch_count:2} | "
            f"Struggle: {self.struggle_turns:2}\n",
            force=False
        )
