"""RANK TRIPWIRE (gen3_distill_target_gate_v1; design_advantage_gated_distillation.md §4.1).

`rank/policy_pr` — the participation ratio of `pi_features`, sampled once per `train()` by the
existing `rank_metrics.rank_probe` — is the collapse signature of the five failed distill arms
(21.87 with no KL, 12.5-13.6 with ANY KL: a 38-43% drop, on none of the known-good controls), and
the instrument was already running and was read five days late. This callback makes the reading
STRUCTURAL: an EMA of the existing scalar against the run's own early baseline, a persistence
rule, a WARN band, and an opt-in abort. No new probe and no new forward — it re-reads the value
`train()` already logged, out of `model.logger.name_to_value` at the next rollout boundary
(`_on_rollout_end` runs before the logger dump clears it, so each rollout end sees the PREVIOUS
train() call's reading — a one-iteration lag, same cadence).

The spec (§4.1, implemented verbatim):

* baseline  = median of `rank/policy_pr` over readings [W_SKIP, W_SKIP + W_BASE) — skip the
  resume/compile transient, then 20 readings; logged as `rank/policy_pr_baseline` on every
  subsequent call so a post-hoc reader gets it free.
* statistic = EMA of the reading, half-life 10 train() calls (one-minibatch probe → noisy; the
  EMA plus persistence is what makes a verdict rather than a flicker).
* WARN at `ema < (1 - drop/2)·base` for 3 consecutive readings — one launcher event
  (`main.launcher.ipc.emit`), `rank/policy_pr_ratio` logged.
* TRIP at `ema < (1 - drop)·base` ×3 — loud event + `rank/tripwire_fired = 1` latched; under
  mode="abort" the callback returns False from `_on_step`, so SB3 stops `learn()` cleanly and the
  normal end-of-learn path saves the checkpoint.
* A missing reading is "no reading", never a trip and never an all-clear: logged as
  `rank/tripwire_no_reading`, and neither the EMA nor any persistence counter advances (nor
  resets). A diagnostic must never crash a run, and must never *silently* stop speaking either.

Default drop = 0.20: every known-bad arm fell 38-43% and every known-good control fell 0, so a
20% trip threshold fires on all five and on none of the controls; the 20-38% band is the margin.
"""
import math
import statistics

from stable_baselines3.common.callbacks import BaseCallback

from main.launcher.ipc import emit


class RankTripwireCallback(BaseCallback):
    """§4.1 bookkeeping over the existing `rank/policy_pr` scalar. Pure diagnostic: it folds no
    loss and writes no grad; the ONLY way it can affect a run is mode="abort" stopping `learn()`
    (which changes when training ends, never what a step computes). mode="off" is handled by not
    registering the callback at all (`main.train.callbacks`)."""

    SIGNAL = "rank/policy_pr"
    W_SKIP = 5          # readings skipped before the baseline window (resume/compile transient)
    W_BASE = 20         # readings in the baseline window; baseline = their median
    EMA_HALF_LIFE = 10.0
    PERSISTENCE = 3     # consecutive below-threshold readings before WARN / TRIP

    def __init__(self, mode: str = "warn", drop: float = 0.20, verbose: int = 0):
        super().__init__(verbose)
        if mode not in ("warn", "abort"):
            raise ValueError(f"RankTripwireCallback mode must be 'warn' or 'abort' (got {mode!r}); "
                             "'off' means: do not register the callback")
        self.mode = mode
        self.drop = float(drop)
        self._decay = 0.5 ** (1.0 / self.EMA_HALF_LIFE)
        self._ema: float | None = None
        self._n_readings = 0
        self._window: list[float] = []
        self._baseline: float | None = None
        self._warn_streak = 0
        self._trip_streak = 0
        self._warned = False        # re-arms when the ratio recovers above the WARN band
        self._fired = False         # LATCHED — never clears
        self._abort = False

    @property
    def warn_threshold(self) -> float:
        return 1.0 - self.drop / 2.0

    @property
    def trip_threshold(self) -> float:
        return 1.0 - self.drop

    def _on_step(self) -> bool:
        # The abort channel: a False from any callback's on_step stops rollout collection and
        # learn() returns cleanly, so the run's normal final save still happens.
        return not self._abort

    def _on_rollout_end(self) -> None:
        raw = self.model.logger.name_to_value.get(self.SIGNAL)
        if raw is None or not math.isfinite(float(raw)):
            # NO READING (non-Gen3 extractor, capture failure, or the very first rollout before
            # any train()): counters do not advance and do not reset; the EMA does not move.
            self.logger.record("rank/tripwire_no_reading", 1.0)
            if self._fired:
                self.logger.record("rank/tripwire_fired", 1.0)
            return
        reading = float(raw)
        i = self._n_readings            # 0-based reading index
        self._n_readings += 1
        self._ema = reading if self._ema is None else (
            self._decay * self._ema + (1.0 - self._decay) * reading)
        if self._baseline is None:
            if self.W_SKIP <= i < self.W_SKIP + self.W_BASE:
                self._window.append(reading)
            if len(self._window) == self.W_BASE:
                self._baseline = float(statistics.median(self._window))
        if self._fired:
            self.logger.record("rank/tripwire_fired", 1.0)
        if self._baseline is None or self._baseline <= 0.0:
            return                      # no baseline yet → nothing to judge against
        ratio = self._ema / self._baseline
        self.logger.record("rank/policy_pr_baseline", self._baseline)
        self.logger.record("rank/policy_pr_ratio", float(ratio))
        self._warn_streak = self._warn_streak + 1 if ratio < self.warn_threshold else 0
        self._trip_streak = self._trip_streak + 1 if ratio < self.trip_threshold else 0
        if self._warn_streak == 0:
            self._warned = False        # recovered → a fresh degradation warns again
        if self._trip_streak >= self.PERSISTENCE and not self._fired:
            self._fired = True
            self.logger.record("rank/tripwire_fired", 1.0)
            emit(f"🚨 [RankTripwire] TRIP: rank/policy_pr EMA {self._ema:.2f} < "
                 f"{self.trip_threshold:.2f}× baseline {self._baseline:.2f} (ratio {ratio:.3f}) "
                 f"for {self.PERSISTENCE} consecutive readings — the five-arm collapse signature "
                 f"(design_advantage_gated_distillation.md §4.1)."
                 + (" ABORTING: stopping learn() at the next step (clean stop, checkpoint saved)."
                    if self.mode == "abort" else " mode=warn: training continues."))
            if self.mode == "abort":
                self._abort = True
        elif self._warn_streak >= self.PERSISTENCE and not self._warned and not self._fired:
            self._warned = True
            emit(f"⚠️ [RankTripwire] WARN: rank/policy_pr EMA {self._ema:.2f} < "
                 f"{self.warn_threshold:.2f}× baseline {self._baseline:.2f} (ratio {ratio:.3f}) "
                 f"for {self.PERSISTENCE} consecutive readings.")
