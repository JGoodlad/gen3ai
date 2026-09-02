"""`--adaptive-batch` — a feedback loop on the EFFECTIVE batch, the way `adaptive_lr_callback`
is a feedback loop on the learning rate.

THE QUESTION IT ANSWERS. `train/noise_scale_ratio` = `B_simple / (batch_size·K)` says whether the
batch is the right size *for the gradient being measured*: ≫1 ⇒ noise-limited (each update is
mostly sideways), ≪1 ⇒ over-batched (samples polish an already-clean gradient instead of buying
more steps). Until now that reading was ADVICE — `_noise_scale_advice` printed "raise
--grad-accum-steps ~N×" into the Events panel and a human typed it on the next relaunch. This
closes the loop.

WHY IT MOVES `K` AND NEVER `--batch-size`. Three independent reasons, and all three matter:

  * **Shape stability.** `--compile-trainer` keys graphs on shape and dynamo's `cache_size_limit`
    is 8; a moving `batch_size` is an unbounded shape set, which makes dynamo fall back to eager
    SILENTLY — the invisible ~1.75x regression `check_shape_stability` exists to refuse. Moving K
    leaves every forward shape byte-identical, so the compiled trainer never notices.
  * **Memory.** The activation peak is one MICRO-batch. K is the one batch lever with no VRAM cost.
  * **Exactness.** K micro-batches summed IS the gradient of a `batch_size·K` batch (see
    `PpoHyperparameters.grad_accum_steps`), so the loop changes the batch without approximating it.

WHICH RATIO. `--adaptive-batch {total,policy}`. `policy` reads
`train/noise_scale_ratio_policy` — the SAME estimator run on the clipped surrogate's gradient
alone (`noise_scale_terms.py`) — and is the one to use: on this tree the TOTAL gradient is ~100%
the value term plus a dozen dense supervised auxiliaries whose per-example gradients agree, so the
total reads "over-batched" (0.001-0.06 measured) while the policy term reads "noise-limited"
(2.7-6.2 on the same calls). Sizing the batch on the total in that state shrinks the batch the
policy gradient needed. `total` exists so the legacy scalar can still drive the loop when someone
wants that arm.

IT READS THE EXISTING EMA, IT DOES NOT FORK THE ESTIMATOR.
`NoiseScaleDiagnostics.noise_ratio_sample` returns the smoothed ratio and its sample count
straight from the EMAs `train()` already maintains, so the number the controller acts on is
character-for-character the number TensorBoard shows.

THE TWO-CONTROLLER INTERACTION is documented on `AdaptiveBatchController` and in
`src/agents/training/CLAUDE.md`.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Optional, Set, Tuple

from stable_baselines3.common.callbacks import BaseCallback

#: The hard floor the loop enforces on K regardless of `--adaptive-batch-min-accum`. The
#: noise-scale estimator needs gradient norms at TWO batch sizes and gets the second one from the
#: accumulation group, so it emits NOTHING at K=1 — a loop allowed to reach K=1 would blind the
#: very signal it steers by and could never climb back out. The floor is therefore part of the
#: controller, not advice.
_ACCUM_FLOOR_WHEN_ON = 2

#: Noise-scale EMA folds required before the loop may act at all. Mirrors the NSR advisor's own
#: warm-up (`NoiseScaleDiagnostics._emit_noise_scale_warnings` suppresses the first 20): a
#: single-sample `B_simple` can SIGN-FLIP, so acting on an unwarmed EMA is acting on noise.
_DEFAULT_WARMUP_SAMPLES = 20


@dataclasses.dataclass(frozen=True)
class BatchDecision:
    """One rollout's verdict. `accum` is what K should be AFTER this decision."""

    accum: int
    moved: bool
    reason: str
    ratio: Optional[float] = None


class AdaptiveBatchController:
    """The PURE decision rule — no SB3, no torch, no logging. One `decide()` per rollout.

    THE RULE, in one paragraph. Every rollout the controller is handed the smoothed noise-scale
    ratio of the CHOSEN term and the number of EMA samples behind it. It does nothing at all until
    the EMA is warm (`warmup_samples`, default 20) and at least `every` rollouts have passed since
    the last move. Then, if the ratio has left the band `[target/band, target*band]`, K is DOUBLED
    when the ratio is ABOVE it (critical batch >> effective ⇒ noise-limited ⇒ buy a bigger batch)
    and HALVED when it is BELOW (over-batched ⇒ buy more update steps per sample), clamped into
    `[max(2, min_accum), max_accum]`. Anything else — an unreadable ratio, a cold EMA, a
    within-band reading, a clamp — is a no-op with a named reason.

    WHY THE STEP IS A FACTOR OF 2, AND WHY THE BAND MUST BE AT LEAST sqrt(2). The ratio's
    denominator is `batch_size·K`, so a K move changes the reading INSTANTLY and exactly: doubling
    K halves the ratio. A correction therefore overshoots to the OTHER side of the band only when
    `ratio > target*band` and `ratio/2 < target/band` can both hold, i.e. when `band^2 < 2`. So at
    any `band >= sqrt(2) ~= 1.4142` a single move can never cross the band, and the loop settles
    (a ratio further out than one doubling takes several moves in ONE direction, which is progress,
    not chatter). Below sqrt(2) it can oscillate — measured: `band=1.41` chatters, `band=1.50` does
    not, exactly where the algebra puts the boundary. The default 2.0 sits comfortably above it and
    matches the KL lr controller's `kl_factor`; a narrower band is for a smoke that WANTS movement
    in a handful of rollouts, and `--adaptive-batch-band` below sqrt(2) is a deliberate choice to
    give the guarantee up.

    WHY IT MUST BE SLOWER THAN THE LR LOOP, and why it is. The KL-driven lr controller
    (`adaptive_lr_callback`) and this one are coupled through the update: at a fixed `target_kl`, a
    larger K means each optimizer step consumes more data, so the per-step KL falls, so the lr
    controller RAISES lr. That is the intended division of labour — the batch loop fixes the
    signal-to-NOISE of an update, the lr loop fixes its STEP SIZE — but two controllers chasing
    each other on the same timescale is a classic oscillation. They are separated by their SIGNALS,
    not merely by their cadences: the lr loop reads a KL EMA with `alpha=0.20` (half-life ~3
    rollouts) and this one reads the noise-scale EMA with decay `0.99` (a several-hundred-`train()`
    -call window), so the quantity here is ~30-100x slower-moving by construction. The `every`
    cadence (default 4 rollouts) is the second-order guard on top of that, and the lr loop's own
    7-rollout post-move cooldown means the fast loop has re-settled before the slow one looks
    again.
    """

    def __init__(self, *, mode: str, target: float = 1.0, band: float = 2.0,
                 min_accum: int = 1, max_accum: int = 32, every: int = 4,
                 warmup_samples: int = _DEFAULT_WARMUP_SAMPLES) -> None:
        if mode not in ("off", "total", "policy"):
            raise ValueError(f"--adaptive-batch must be off|total|policy, got {mode!r}")
        if target <= 0.0:
            raise ValueError("--adaptive-batch-target must be > 0")
        if band <= 1.0:
            raise ValueError("--adaptive-batch-band must be > 1 (it is a MULTIPLICATIVE band "
                             "around the target: [target/band, target*band])")
        if min_accum < 1 or max_accum < min_accum:
            raise ValueError("--adaptive-batch-min-accum must be >= 1 and "
                             "<= --adaptive-batch-max-accum")
        if every < 1:
            raise ValueError("--adaptive-batch-every must be >= 1 (rollouts between moves)")
        self.mode = mode
        self.target = float(target)
        self.band = float(band)
        #: The EFFECTIVE floor. See `_ACCUM_FLOOR_WHEN_ON` — a loop that reaches K=1 loses its
        #: own instrument, so the requested floor is raised (and the raise is announced, once).
        self.requested_min_accum = int(min_accum)
        self.min_accum = (max(int(min_accum), _ACCUM_FLOOR_WHEN_ON) if mode != "off"
                          else int(min_accum))
        self.max_accum = max(int(max_accum), self.min_accum)
        self.every = int(every)
        self.warmup_samples = max(2, int(warmup_samples))
        self._since_move = 0

    # ------------------------------------------------------------------
    @property
    def metric_key(self) -> str:
        """The TensorBoard tag whose value this controller steers by."""
        return ("train/noise_scale_ratio" if self.mode == "total"
                else f"train/noise_scale_ratio_{self.mode}")

    @property
    def floor_was_raised(self) -> bool:
        return self.mode != "off" and self.min_accum > self.requested_min_accum

    def clamp(self, accum: int) -> int:
        """Bring a K into the controller's range (used to seed a fresh/resumed run)."""
        return max(self.min_accum, min(int(accum), self.max_accum))

    # ------------------------------------------------------------------
    def decide(self, *, ratio: Optional[float], samples: int, accum: int) -> BatchDecision:
        """One rollout's decision. PURE — same inputs, same output, no state but the cadence count.

        `ratio` is the smoothed noise-scale ratio of the chosen term (`None` when the instrument
        has produced nothing), `samples` how many EMA folds are behind it, `accum` the K in force.
        """
        accum = int(accum)
        if self.mode == "off":
            return BatchDecision(accum, False, "off", ratio)
        self._since_move += 1
        if ratio is None or not math.isfinite(float(ratio)) or float(ratio) <= 0.0:
            # PROTECTION: the chosen term is unreadable — the per-term probe is off, the EMA has
            # not turned positive yet, or the estimate is NaN. A controller that guesses here is
            # worse than one that waits, because the guess is unfalsifiable.
            return BatchDecision(accum, False, "unavailable", None)
        ratio = float(ratio)
        if samples < self.warmup_samples:
            return BatchDecision(accum, False, "warming", ratio)
        if self._since_move < self.every:
            return BatchDecision(accum, False, "cadence", ratio)
        hi = self.target * self.band
        lo = self.target / self.band
        if ratio > hi:
            proposed = accum * 2           # noise-limited: each update is mostly sideways
        elif ratio < lo:
            proposed = max(accum // 2, 1)  # over-batched: buy update steps instead of averaging
        else:
            return BatchDecision(accum, False, "in band", ratio)
        new = self.clamp(proposed)
        if new == accum:
            return BatchDecision(accum, False, "clamped", ratio)
        self._since_move = 0
        return BatchDecision(new, True, "noise-limited" if ratio > hi else "over-batched", ratio)


def _no_move_message(reason: str, controller: "AdaptiveBatchController", samples: int) -> str:
    """The once-per-reason explanation of a loop that is deliberately doing nothing.

    Split out (and pure) because a silent no-op loop is indistinguishable from a broken one, and
    the distinction is exactly what an operator needs the first time `K` never moves.
    """
    if reason == "unavailable":
        return (f"{controller.metric_key} has produced no reading yet — with --adaptive-batch "
                f"policy the per-term probe must be ON ($GEN3AI_NOISE_SCALE_PER_TERM) and K>=2, "
                f"and a strongly noise-limited term is the LAST tag to warm up. Reported once.")
    if reason == "warming":
        return (f"only {samples} EMA folds so far (need {controller.warmup_samples}); a "
                f"single-sample B_simple can sign-flip. Reported once.")
    return (f"the ratio wants K past a bound [{controller.min_accum}, {controller.max_accum}] — "
            f"raise --adaptive-batch-max-accum if this persists. Reported once.")


class AdaptiveBatchCallback(BaseCallback):
    """Drives `AdaptiveBatchController` off the live model, once per rollout.

    Reads the noise-scale EMA the PPO already maintains (never its own estimator), writes the
    resulting K to `model.grad_accum_steps` — the only thing it ever mutates — and records the
    three series an operator reads the loop by:

      `train/grad_accum_steps`            K in force for the `train()` that follows.
      `train/effective_batch`             `batch_size * K`.
      `train/adaptive_batch_ratio_used`   the exact ratio the decision was made on.

    PERSISTENCE across a launcher restart is free and deliberately so: `_model_hparams` already
    writes `grad_accum_steps` into every checkpoint's sidecar, straight off the model attribute
    this callback owns, so a moved K is recorded by the existing checkpointer with no new key and
    no edit to the checkpoint path. `resume_accum` (read back from that sidecar in
    `main.train.callbacks`) is installed in `_on_training_start`, AFTER `model_build` has applied
    the CLI `--grad-accum-steps`, so the controller's own history wins over the launch argv —
    which is the whole point of persisting it.
    """

    def __init__(self, controller: AdaptiveBatchController,
                 resume_accum: Optional[int] = None, verbose: int = 1) -> None:
        super().__init__(verbose)
        self.controller = controller
        self._resume_accum = resume_accum
        self._reasons_logged: Set[str] = set()
        self.moves: int = 0

    # ------------------------------------------------------------------
    def _emit(self, msg: str) -> None:
        """Launcher Events panel when there is one, plain print otherwise."""
        try:
            from main.launcher.ipc import emit
            emit(msg)
        except Exception:                    # noqa: BLE001 — an event must never take a run down
            print(msg, flush=True)

    def _batch_size(self) -> int:
        return int(getattr(self.model, "batch_size", 0) or 0)

    def _accum(self) -> int:
        return max(1, int(getattr(self.model, "grad_accum_steps", 1)))

    def _set_accum(self, k: int) -> None:
        self.model.grad_accum_steps = int(k)

    def _read_ratio(self) -> Tuple[Optional[float], int]:
        """`(ratio, samples)` from the PPO's own noise-scale EMAs, or `(None, 0)`."""
        reader = getattr(self.model, "noise_ratio_sample", None)
        if reader is None:
            return None, 0
        try:
            return reader(self.controller.mode, float(self._batch_size()) * self._accum())
        except Exception:                    # noqa: BLE001 — a diagnostic read is never fatal
            return None, 0

    # ------------------------------------------------------------------
    def _on_training_start(self) -> None:
        c = self.controller
        if self._resume_accum is not None:
            self._set_accum(c.clamp(int(self._resume_accum)))
            if self.verbose >= 1:
                print(f"[AdaptiveBatch] Resumed K={self._accum()} from the checkpoint sidecar "
                      f"(--grad-accum-steps on the command line is the FRESH-run seed only).")
        else:
            self._set_accum(c.clamp(self._accum()))
        if c.floor_was_raised and self.verbose >= 1:
            print(f"[AdaptiveBatch] min accum raised {c.requested_min_accum} -> {c.min_accum}: "
                  f"the gradient-noise-scale estimator needs gradient norms at TWO batch sizes and "
                  f"gets the second from the accumulation group, so at K=1 it emits nothing and "
                  f"the loop would be permanently blind.")
        k = self._accum()
        self._emit(f"📦 [AdaptiveBatch] ON ({c.mode}): K={k}, effective batch "
                   f"{self._batch_size() * k}, holding {c.metric_key} in "
                   f"[{c.target / c.band:.3g}, {c.target * c.band:.3g}] (target {c.target:g}), "
                   f"K in [{c.min_accum}, {c.max_accum}], one move per >={c.every} rollouts.")

    def _on_rollout_end(self) -> None:
        ratio, samples = self._read_ratio()
        accum = self._accum()
        d = self.controller.decide(ratio=ratio, samples=samples, accum=accum)
        if d.moved:
            self.moves += 1
            self._set_accum(d.accum)
            arrow = "UP" if d.accum > accum else "DOWN"
            self._emit(
                f"📦 [AdaptiveBatch] {self.controller.metric_key}={d.ratio:.3g} ({d.reason}) -> "
                f"K {arrow} {accum} -> {d.accum} (effective batch "
                f"{self._batch_size() * accum} -> {self._batch_size() * d.accum}). Updates per "
                f"env-step changed — the DOSE moved; watch train/dose_rate, and expect the KL lr "
                f"controller to follow.")
        elif d.reason in ("unavailable", "warming", "clamped") \
                and d.reason not in self._reasons_logged:
            # Say it ONCE per reason. A loop that is silently doing nothing is indistinguishable
            # from a loop that is broken, and a loop that says so every rollout is noise.
            self._reasons_logged.add(d.reason)
            self._emit(f"📦 [AdaptiveBatch] no move ({d.reason}): "
                       + _no_move_message(d.reason, self.controller, samples))
        k = self._accum()
        self.logger.record("train/grad_accum_steps", int(k))
        self.logger.record("train/effective_batch", self._batch_size() * k)
        if d.ratio is not None:
            self.logger.record("train/adaptive_batch_ratio_used", float(d.ratio))

    def _on_step(self) -> bool:
        return True
