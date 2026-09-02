"""THE FOLD STOP RULE + THE DUAL-ASCENT ANCHOR COEFFICIENT (`gen3_distill_stop_rule_v1`).

**WHY A FOLD NEEDS AN END, and why the signal is now live.** The 2026-09-01 pair
(`designs/research_state/ledger.md`: *"v8's GIFT IS A TRANSIENT HUMP"*, *"WHAT v8's LAST 2.5M
UNDID"*) measured a fold doing two things in two directions. The GIFT — an early off-slice habit
change, PPO-driven, ORTHOGONAL to the teachers' fingerprint (cos 0.14) — landed by ~+3M and was
**92% intact at +15M**. The LEAK — the taught content itself continuing to arrive on untaught
boards, PARALLEL to that fingerprint (cos +0.559, perm p 0.0015) — cost **-5.66pp [-12.1, -0.2]**
on untaught teams while costing nothing on taught ones. v8's untaught gain peaked at **+9.67pp**
around +12.5M and fell to **+4.98pp** by +15.04M, with distillation still running at full strength
against teachers it had already absorbed.

So a fold has an OPTIMAL LENGTH, and running past it pays ~5pp of untaught win rate for content
the student already holds. The ledger's own design consequence (c) names the signal, and both
halves of it are already instrumented in this tree:

    STOP when  `distill/collateral_kl_vs_parent` is RISING
          and  `distill/teacher_agreement_on_slice` has PLATEAUED.

That conjunction is the point. Rising collateral ALONE is an ordinary fold in progress — the leak
and the teaching arrive together, and paying collateral for content is the trade the fold exists to
make. A plateaued agreement ALONE is a fold that has finished absorbing and is now merely idling.
It is the two TOGETHER — displacement still accumulating with nothing left to absorb — that is the
R3-SELF regime seen from the inside: a dense distill term pushing with nothing left to teach.

**TWO MECHANISMS, ONE MODULE, DIFFERENT JOBS.**

  * `AnchorDualAscent` turns `--distill-anchor-coef` from a coefficient nobody can tune into a
    CONSTRAINT with a readable budget: hold the off-slice divergence at `--distill-anchor-target-kl`
    by moving the coefficient, not by guessing it. This is PPO-penalty's adaptive beta (Schulman et
    al. 2017 §4) and MPO's Lagrangian dual (Abdolmaleki et al. 2018) at a different constraint.
  * `FoldStopDetector` + `DistillStopCallback` are the STOP RULE itself: a plateau detector and a
    rise detector, AND-gated, with a persistence count, driving one of three actions.

They are separable — either can run without the other — but they share this file because they are
the same question asked at two timescales: *how hard should the fold be held back right now* and
*should the fold still be running at all*.

🚨 **BOTH ARE OFF BY DEFAULT AND MUST STAY OFF** until the three-dose cell's curves size the window.
`--distill-stop-window 8` and `--distill-stop-eps 0.005` are placeholders derived from nothing but
the shape of the v8 curve at a cadence this tree has never run a fold at; firing a stop rule on an
unsized window would be a new way to lose a training window. `warn` is the mode to run first, and
what it buys is the calibration data the other two modes need.

**RESTART SAFETY IS NOT OPTIONAL HERE, and it is the mirror of the anchor's own restart rule.** A
launcher run restarts every few hours. A detector that re-armed on every launch would need its full
window again each time and — at a 3h restart against an 8-rollout window — might never fire at all,
while reading as ON throughout. An annealed `--distill-coef` is worse: the launcher forwards the
ORIGINAL argv, so an explicit `--distill-coef 0.3` would silently re-install itself at every restart
and undo the wind-down. Both are therefore persisted in the checkpoint sidecar (`_model_hparams`,
the same place `handoff_lr` and `grad_accum_steps` ride) and RE-INSTALLED at `_on_training_start`,
which is the only hook that runs on every launch.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

from stable_baselines3.common.callbacks import BaseCallback

#: `--distill-stop` values. "off" is handled by not registering the callback at all.
STOP_MODES = ("off", "warn", "anneal", "abort")

#: The absorption meter — student<->teacher top-1 agreement on the TAUGHT slice, averaged over
#: active teachers. Recorded by `train()` inside the teacher block, so it exists only while a
#: distill term is actually folded (see `DistillStopCallback` on what an absent reading means).
AGREE_SIGNAL = "distill/teacher_agreement_on_slice"

#: The displacement meter — off-slice `KL(frozen parent || student)`, emitted in EVERY anchor
#: reference mode. This is the number the untaught-team win-rate meter correlates with.
COLLATERAL_SIGNAL = "distill/collateral_kl_vs_parent"

#: The anchor's OWN unweighted KL, against whatever reference the anchor is anchored to. The dual
#: drives on this under a MOVING reference — see `AnchorDualAscent` for why.
ANCHOR_KL_SIGNAL = "distill/anchor_kl"


def _finite(x: Any) -> Optional[float]:
    """``float(x)`` when it is a finite number, else ``None`` — the "no reading" gate.

    A missing series and a NaN are the same fact (nothing was measured this rollout) and both must
    be treated as silence rather than as a zero: a controller that reads an absent meter as 0.0
    would drive its coefficient to a clamp, and a detector that read it as "no improvement" would
    call a plateau on a fold that simply did not log.
    """
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


class AnchorDualAscent:
    """DUAL ASCENT on the off-slice anchor coefficient — the PURE controller (no SB3, no torch).

    THE RULE, in one line::

        kl_ema <- alpha*kl + (1-alpha)*kl_ema
        coef   <- clip( coef * exp(eta * (kl_ema/target - 1)),  coef_min,  coef_max )

    It is gradient ascent on the Lagrange multiplier of ``minimize L(theta)  s.t.  KL <= target``,
    taken in LOG-coefficient space so the multiplier stays positive by construction and so a
    correction is proportional to the RATIO of the violation rather than to its absolute size —
    which matters because collateral KL spans two orders of magnitude across configs and a fixed
    additive step would be a different controller at each of them.

    **WHICH SIGNAL, AND WHY IT DEPENDS ON THE REFERENCE.** The dual variable must be attached to a
    quantity its own lever can MOVE, or it winds up against a clamp and stays there while still
    reading as a live controller:

      * ``--distill-anchor-ref parent`` (the default) -> **`distill/collateral_kl_vs_parent`**, the
        ACCUMULATED-DISPLACEMENT meter. That is the quantity the untaught robbery is made of, and
        under a fixed reference it is exactly what the anchor loss penalises (identical to
        `anchor_kl` under `--distill-anchor-mode off_slice`; under `all` the loss covers more rows
        but still moves this number, so it stays controllable).
      * ``ema`` / ``periodic`` -> **`distill/anchor_kl`**, the anchor's own unweighted KL against
        its moving reference. Under a moving reference the anchor DELIBERATELY does not resist
        parent-displacement — that is the whole point of ACER's average-policy trust region, and the
        reason those modes exist is that a fixed reference taxes v8's +5.4pp GIFT as hard as the
        leak. A dual budgeted on a number the loss is designed not to control is a wound-up
        integrator, not a constraint.

      `collateral_kl_vs_parent` is still LOGGED in every mode; the choice above is only about which
      number the dual acts on.

    **WHY EVERY ROLLOUT, AND WHY NO COOLDOWN.** The KL-driven LR ladder
    (`adaptive_lr_callback`) is bang-bang — a fixed multiplicative step whenever the EMA leaves a
    band — so it COMPOUNDS while the EMA lags, and its 7-rollout cooldown is what stops the
    overshoot. This controller is an INTEGRATOR: the step is proportional to the violation, so it
    shrinks to nothing as the constraint is met and its own `dual_lr` already sets the response
    timescale (at eta=0.1 a sustained 2x overshoot moves the coefficient +10.5% per rollout, ~7
    rollouts to double). Adding a cooldown to an integrator inserts DEAD TIME, which is the classic
    cause of the oscillation a cooldown is meant to prevent. The EMA (alpha=0.2, half-life ~3
    rollouts) is the noise filter, and it is the only smoothing this loop needs.

    **THE CLAMPS.** ``coef_max`` defaults to 10x the starting coefficient — the anchor is documented
    as "a FRACTION of --distill-coef, never near it" (a coefficient at distill scale is R3-SELF,
    which measured -9pp), so an unbounded dual could walk into the exact misuse the feature warns
    about. ``coef_min`` defaults to 0.0, which under a MULTIPLICATIVE update is unreachable from
    above and therefore means "no floor"; it exists so an arm can pin one. A starting coefficient of
    exactly 0 is a FIXED POINT of the update (0 * anything = 0), which is why
    `--distill-anchor-target-kl` refuses `--distill-anchor-coef 0` in `resolve_config` rather than
    running as a silent no-op.
    """

    def __init__(self, *, target_kl: float, coef0: float, dual_lr: float = 0.1,
                 coef_min: float = 0.0, coef_max: Optional[float] = None,
                 ema_alpha: float = 0.20) -> None:
        if target_kl <= 0.0:
            raise ValueError("--distill-anchor-target-kl must be > 0 (0 = off; the dual divides by "
                             "it, and a zero budget is not a constraint but a demand for an "
                             "infinite coefficient)")
        if coef0 <= 0.0:
            raise ValueError("dual ascent needs --distill-anchor-coef > 0: the update is "
                             "MULTIPLICATIVE, so a coefficient of 0 is a fixed point and the "
                             "controller would run forever without moving anything")
        if dual_lr <= 0.0:
            raise ValueError("--distill-anchor-dual-lr must be > 0")
        if not (0.0 < ema_alpha <= 1.0):
            raise ValueError("the dual's EMA alpha must be in (0, 1]")
        self.target_kl = float(target_kl)
        self.dual_lr = float(dual_lr)
        self.ema_alpha = float(ema_alpha)
        self.coef_min = float(coef_min)
        self.coef_max = float(coef_max) if coef_max is not None else 10.0 * float(coef0)
        if self.coef_max < self.coef_min:
            raise ValueError("--distill-anchor-coef-max must be >= --distill-anchor-coef-min")
        self.coef = min(max(float(coef0), self.coef_min), self.coef_max)
        self.kl_ema: Optional[float] = None
        self.n_readings = 0
        self.clamped = False        # did the LAST update hit a bound? (logged, not latched)

    def update(self, kl: Optional[float]) -> float:
        """One rollout. ``kl`` ``None`` (or non-finite) is NO READING: the EMA does not move, the
        count does not advance, and the coefficient is returned unchanged."""
        v = _finite(kl)
        if v is None:
            return self.coef
        self.n_readings += 1
        self.kl_ema = v if self.kl_ema is None else (
            self.ema_alpha * v + (1.0 - self.ema_alpha) * self.kl_ema)
        raw = self.coef * math.exp(self.dual_lr * (self.kl_ema / self.target_kl - 1.0))
        clipped = min(max(raw, self.coef_min), self.coef_max)
        self.clamped = clipped != raw
        self.coef = clipped
        return self.coef

    # ---- persistence -----------------------------------------------------------------------
    def state(self) -> Dict[str, Any]:
        return {"coef": float(self.coef), "kl_ema": self.kl_ema,
                "n_readings": int(self.n_readings)}

    def load_state(self, blob: Optional[Dict[str, Any]]) -> bool:
        """Restore ``coef`` / ``kl_ema`` / ``n_readings``. Returns whether anything was restored.

        Total: a blob from a different build, a half-written dict, or garbage is IGNORED rather than
        half-applied — a controller that resumed with a coefficient and no EMA would take its first
        post-restart step from a cold integrator against a warm coefficient.
        """
        if not isinstance(blob, dict):
            return False
        c = _finite(blob.get("coef"))
        if c is None:
            return False
        self.coef = min(max(c, self.coef_min), self.coef_max)
        self.kl_ema = _finite(blob.get("kl_ema"))
        try:
            self.n_readings = int(blob.get("n_readings", 0) or 0)
        except (TypeError, ValueError):
            self.n_readings = 0
        return True


class FoldStopDetector:
    """THE PLATEAU x RISE DETECTOR — the PURE state machine (no SB3, no torch, no logging).

    Two EMAs (alpha=0.2, matching the dual and the LR ladder, half-life ~3 rollouts) over the two
    live meters, each with a rolling history of the last ``window + 1`` values so a window-length
    comparison is available every rollout.

    **PLATEAU (absorption has stopped).** ``agree_ema[t] - agree_ema[t-window] < eps``. The
    difference is SIGNED on purpose: an agreement that is FALLING is not absorbing either, so it
    counts as a plateau rather than as a special case. ``eps`` is an ABSOLUTE change in a top-1
    agreement rate (default 0.005 = half a percentage point over the window), because that is the
    unit the meter is in and a relative threshold near an agreement of 0.9 would mean something
    different than near 0.4.

    **RISE (displacement is still accumulating).** An ordinary least-squares slope over the same
    window, in units of its own STANDARD ERROR::

        rise  <=>  slope > 0  and  slope > kl_slope_t * se(slope)

    i.e. a one-sided t-test that the trend is positive, at ``--distill-stop-kl-slope`` (default
    **2.0**) standard errors. **The flag is a t-multiple, NOT nats per rollout, and that is a
    deliberate choice.** Collateral KL's absolute scale moves by two orders of magnitude across
    configs (the anchor's own build smoke read 0.00035 early and 0.034 late on toy CPU runs), so no
    absolute slope threshold could be quoted in a help string and be right on the next arm. A
    t-statistic is scale-free and asks the only question that transfers: *is this rising by more
    than its own wobble?* A perfectly linear series has ``se = 0``, and ``slope > 0`` alone then
    decides — which is correct, since a noiseless rise IS a rise.

    🚨 **THE TREND TEST READS THE *RAW* COLLATERAL SERIES, NOT ITS EMA, AND THAT IS A MEASURED
    CORRECTION rather than a preference.** An EMA is a low-pass filter: it makes consecutive points
    strongly autocorrelated, so an OLS fit through it has residuals far smaller than the series' own
    noise and its standard error understates the uncertainty by a large factor. Fitting the EMA
    therefore reports a SIGNIFICANT positive trend on white noise — measured while building this
    detector: a zero-mean wobble (sd 0.004 around a level 0.01) passed ``t > 2`` at a 6-rollout
    window, i.e. the detector would have fired on a fold that was not drifting at all. The raw
    readings are the ones that carry the noise the test is supposed to be measured against, and a
    windowed OLS already averages, so the smoothing bought nothing the fit does not.
    ``kl_ema`` is still maintained and still reported (it is what a human reads on the level), it is
    simply not what the trend test consumes.

    The PLATEAU half keeps the EMA, because it compares two LEVELS rather than fitting a trend, and
    autocorrelation does not bias a level.

    **THE AND-GATE AND THE PERSISTENCE COUNT.** Both must hold on the SAME rollout; the count of
    consecutive such rollouts is ``hold``, and the detector FIRES (latched, forever) at
    ``persist``. Any rollout where either condition fails resets ``hold`` to 0. A rollout with NO
    READING for either meter is silence, exactly as in `rank_tripwire`: no EMA moves, no history
    entry, ``hold`` neither advances NOR resets. (An anneal that has driven `--distill-coef` to 0
    turns the teacher forwards off, so `teacher_agreement_on_slice` stops existing — after a fire
    that is expected and harmless, since the detector is already latched.)

    ``state`` is the number the TB series reports: 0 armed, 1 plateau only, 2 plateau AND rise,
    3 fired.
    """

    ARMED, PLATEAU, PLATEAU_AND_RISE, FIRED = 0, 1, 2, 3

    def __init__(self, *, window: int = 8, eps: float = 0.005, kl_slope_t: float = 2.0,
                 persist: int = 3, ema_alpha: float = 0.20) -> None:
        if window < 2:
            raise ValueError("--distill-stop-window must be >= 2: the rise test is an OLS slope "
                             "over window+1 points and needs at least one residual degree of "
                             "freedom for its standard error to exist")
        if persist < 1:
            raise ValueError("--distill-stop-persist must be >= 1")
        if not (0.0 < ema_alpha <= 1.0):
            raise ValueError("the detector's EMA alpha must be in (0, 1]")
        self.window = int(window)
        self.eps = float(eps)
        self.kl_slope_t = float(kl_slope_t)
        self.persist = int(persist)
        self.ema_alpha = float(ema_alpha)
        self.agree_ema: Optional[float] = None
        self.kl_ema: Optional[float] = None
        self.agree_hist: List[float] = []       # the agreement EMA's history (a LEVEL comparison)
        self.kl_hist: List[float] = []          # the RAW collateral readings (a TREND fit)
        self.hold = 0
        self.fired = False
        self.rollouts_since_fire = 0
        self.n_readings = 0
        self.last_plateau = False
        self.last_rise = False

    # ---- the two tests ---------------------------------------------------------------------
    def _plateau(self) -> bool:
        if len(self.agree_hist) <= self.window:
            return False
        return (self.agree_hist[-1] - self.agree_hist[-1 - self.window]) < self.eps

    def _rise(self) -> bool:
        if len(self.kl_hist) <= self.window:
            return False
        ys = self.kl_hist[-(self.window + 1):]
        slope, se = ols_slope_and_se(ys)
        if slope is None or se is None or slope <= 0.0:
            return False
        return slope > self.kl_slope_t * se

    # ---- one rollout -----------------------------------------------------------------------
    def update(self, agree: Optional[float], kl: Optional[float]) -> int:
        """Fold one rollout's readings; return the state (0/1/2/3).

        A rollout is only a rollout for this detector when BOTH meters read: the AND-gate is over
        two series measured on the same call, and advancing one EMA against a stale other would
        compare an absorption from rollout k with a displacement from rollout k-3.
        """
        if self.fired:
            self.rollouts_since_fire += 1
            return self.FIRED
        a, k = _finite(agree), _finite(kl)
        if a is None or k is None:
            return self.state_code()            # NO READING: freeze, never reset, never fire
        self.n_readings += 1
        self.agree_ema = a if self.agree_ema is None else (
            self.ema_alpha * a + (1.0 - self.ema_alpha) * self.agree_ema)
        self.kl_ema = k if self.kl_ema is None else (
            self.ema_alpha * k + (1.0 - self.ema_alpha) * self.kl_ema)
        self.agree_hist.append(self.agree_ema)
        self.kl_hist.append(k)                  # RAW — see the class docstring on why not the EMA
        keep = self.window + 1
        del self.agree_hist[:-keep]
        del self.kl_hist[:-keep]
        self.last_plateau = self._plateau()
        self.last_rise = self._rise()
        self.hold = self.hold + 1 if (self.last_plateau and self.last_rise) else 0
        if self.hold >= self.persist:
            self.fired = True
            self.rollouts_since_fire = 0
            return self.FIRED
        return self.state_code()

    def state_code(self) -> int:
        """The TB series `distill/stop_state`: 0 armed, 1 plateau only, 2 plateau AND rise, 3 fired."""
        if self.fired:
            return self.FIRED
        if self.last_plateau and self.last_rise:
            return self.PLATEAU_AND_RISE
        return self.PLATEAU if self.last_plateau else self.ARMED

    # ---- persistence -----------------------------------------------------------------------
    def state(self) -> Dict[str, Any]:
        return {
            "agree_ema": self.agree_ema, "kl_ema": self.kl_ema,
            "agree_hist": list(self.agree_hist), "kl_hist": list(self.kl_hist),
            "hold": int(self.hold), "fired": bool(self.fired),
            "rollouts_since_fire": int(self.rollouts_since_fire),
            "n_readings": int(self.n_readings),
            "last_plateau": bool(self.last_plateau), "last_rise": bool(self.last_rise),
        }

    def load_state(self, blob: Optional[Dict[str, Any]]) -> bool:
        """Restore the EMAs, the two histories, the hold count and the latch. Returns whether
        anything was restored; a malformed blob is ignored WHOLE rather than half-applied."""
        if not isinstance(blob, dict):
            return False
        try:
            ah = [float(v) for v in (blob.get("agree_hist") or [])]
            kh = [float(v) for v in (blob.get("kl_hist") or [])]
        except (TypeError, ValueError):
            return False
        if not all(math.isfinite(v) for v in ah + kh):
            return False
        self.agree_ema = _finite(blob.get("agree_ema"))
        self.kl_ema = _finite(blob.get("kl_ema"))
        keep = self.window + 1
        self.agree_hist = ah[-keep:]
        self.kl_hist = kh[-keep:]
        try:
            self.hold = max(0, int(blob.get("hold", 0) or 0))
            self.rollouts_since_fire = max(0, int(blob.get("rollouts_since_fire", 0) or 0))
            self.n_readings = max(0, int(blob.get("n_readings", 0) or 0))
        except (TypeError, ValueError):
            return False
        self.fired = bool(blob.get("fired", False))
        self.last_plateau = bool(blob.get("last_plateau", False))
        self.last_rise = bool(blob.get("last_rise", False))
        return True


def ols_slope_and_se(ys: List[float]):
    """``(slope, standard_error)`` of an ordinary least-squares fit of ``ys`` against ``0..n-1``.

    Returns ``(None, None)`` for fewer than 3 points (no residual degrees of freedom, so the
    standard error is undefined and a "significant" trend cannot be claimed). A perfectly linear
    series yields ``se = 0.0``, which the caller reads as "a noiseless rise IS a rise" rather than
    as a division to guard.

    Pure arithmetic — no numpy, so the controller is importable and testable without a torch/numpy
    stack and so its behaviour is exactly what is written here.
    """
    n = len(ys)
    if n < 3:
        return None, None
    xbar = (n - 1) / 2.0
    ybar = sum(ys) / n
    sxx = sum((i - xbar) ** 2 for i in range(n))
    if sxx <= 0.0:
        return None, None
    sxy = sum((i - xbar) * (ys[i] - ybar) for i in range(n))
    slope = sxy / sxx
    intercept = ybar - slope * xbar
    resid = sum((ys[i] - (intercept + slope * i)) ** 2 for i in range(n))
    se = math.sqrt(max(resid, 0.0) / ((n - 2) * sxx))
    return slope, se


class DistillStopCallback(BaseCallback):
    """The SB3 wrapper around `FoldStopDetector`: read the two meters at each rollout boundary,
    advance the detector, ACT, log, and expose the state for the checkpoint sidecar.

    Modelled on `rank_tripwire.RankTripwireCallback` — same cadence (`_on_rollout_end` sees the
    PREVIOUS `train()` call's `logger.name_to_value`, a one-iteration lag), same "no reading is
    never a verdict" rule, same abort channel (`_on_step` returns False, SB3 stops `learn()`
    cleanly, and the run's normal end-of-learn save still happens, so the process exits COMPLETE
    rather than CRASH and the launcher does not restart-loop).

    THE THREE ACTIONS, in increasing order of how much they take away:

      * ``warn``   — one launcher event, `distill/stop_signal` latched to 1. Changes nothing about
                     the run. **This is the mode to run first**, because the window and eps are
                     not yet sized by anything but the shape of a curve measured at a different
                     cadence, and a mis-sized stop rule that ABORTS is a lost training window.
      * ``anneal`` — plus a geometric decay of `--distill-coef` every subsequent rollout. The fold
                     WINDS DOWN rather than stopping: the teacher terms fade out over ~a dozen
                     rollouts instead of vanishing between two, so nothing about the loss landscape
                     changes discontinuously under an optimizer carrying momentum.
      * ``abort``  — plus stopping `learn()` at the next step.

    **THE ANNEAL'S FLOOR IS EXACTLY 0, AND IT SNAPS.** A geometric decay never reaches zero, and a
    `distill_coef` of 1e-12 still pays a full teacher forward per minibatch per teacher for a term
    that changes nothing. So the decay snaps to exactly 0.0 once it falls below ``1e-6`` of the
    coefficient in force when the rule fired (~39 rollouts at the default factor 0.7). At exactly
    0.0 `train()`'s `distill_on` predicate goes False: the teacher forwards stop, and with them
    `distill/teacher_agreement_on_slice` — the fold really is over. The ANCHOR's meters
    (`collateral_kl_vs_parent`, `on_slice_kl`) keep reading, because they depend on the frozen
    parent and the `distill_mask` obs key, neither of which the coefficient gates.

    🚨 **THE RESTART RULE.** The launcher forwards the ORIGINAL argv on every relaunch, so an
    explicit `--distill-coef 0.3` re-installs itself every few hours and would undo the wind-down
    silently. `_on_training_start` therefore RE-APPLIES the persisted annealed coefficient over
    whatever the argv asked for, and says so on stdout + the launcher event stream — the same
    shape, and the same reason, as `DistillAnchorCallback`'s moving-reference sibling.
    """

    def __init__(self, *, mode: str, window: int = 8, eps: float = 0.005,
                 kl_slope_t: float = 2.0, persist: int = 3,
                 anneal_factor: float = 0.7, resume_state: Optional[Dict[str, Any]] = None,
                 verbose: int = 0) -> None:
        super().__init__(verbose)
        if mode not in ("warn", "anneal", "abort"):
            raise ValueError(f"--distill-stop must be warn|anneal|abort (got {mode!r}); "
                             "'off' means: do not register the callback")
        if not (0.0 < anneal_factor < 1.0):
            raise ValueError("--distill-stop-anneal-factor must be in (0, 1): it is the per-rollout "
                             "geometric decay of --distill-coef after the rule fires")
        self.mode = mode
        self.anneal_factor = float(anneal_factor)
        self.detector = FoldStopDetector(window=window, eps=eps, kl_slope_t=kl_slope_t,
                                         persist=persist)
        self._resume_state = resume_state if isinstance(resume_state, dict) else None
        self._announced = False          # the one-per-event print
        self._abort = False
        self._coef_at_fire: Optional[float] = None
        self._restore_note = "fold start (no persisted detector state)"

    # ---- SB3 hooks -------------------------------------------------------------------------
    def _on_training_start(self) -> None:
        from main.launcher.ipc import emit
        restored = self.detector.load_state((self._resume_state or {}).get("detector"))
        if restored:
            self._restore_note = (
                f"RESTORED (readings={self.detector.n_readings}, hold={self.detector.hold}, "
                f"fired={self.detector.fired}, since_fire={self.detector.rollouts_since_fire})")
        blob = self._resume_state or {}
        self._coef_at_fire = _finite(blob.get("coef_at_fire"))
        self._announced = bool(self.detector.fired)
        # THE RESTART RULE: re-install the annealed coefficient over the argv's. Only under
        # `anneal`, only after a fire, and only when the persisted value is BELOW what this launch
        # asked for — an operator who deliberately raised the coefficient between restarts is not
        # overruled by a stale wind-down.
        annealed = _finite(blob.get("distill_coef_annealed"))
        if self.mode == "anneal" and self.detector.fired and annealed is not None:
            current = float(getattr(self.model, "distill_coef", 0.0) or 0.0)
            if annealed < current:
                self.model.distill_coef = annealed
                self._restore_note += (f"; --distill-coef RE-ANNEALED {current:g} -> {annealed:g} "
                                       f"(the launcher forwards the original argv; without this the "
                                       f"wind-down would restart every few hours)")
        if self.detector.fired and self.mode == "abort":
            # A restart AFTER an abort would otherwise collect one more rollout before stopping
            # again. Refuse at the first step instead — the verdict is latched and final.
            self._abort = True
        self._publish_state()
        emit(f"🛑 [DISTILL-STOP] mode={self.mode} — plateau on {AGREE_SIGNAL} "
             f"(EMA improvement over {self.detector.window} rollouts < {self.detector.eps:g}) AND "
             f"rise on {COLLATERAL_SIGNAL} (OLS slope > {self.detector.kl_slope_t:g} standard "
             f"errors), for {self.detector.persist} consecutive rollouts"
             + (f"; then --distill-coef x{self.anneal_factor:g} per rollout to 0"
                if self.mode == "anneal" else "")
             + ("; then learn() stops cleanly" if self.mode == "abort" else "")
             + f". {self._restore_note}.")

    def _on_step(self) -> bool:
        return not self._abort

    def _on_rollout_end(self) -> None:
        vals = self.model.logger.name_to_value
        agree = vals.get(AGREE_SIGNAL)
        kl = vals.get(COLLATERAL_SIGNAL)
        was_fired = self.detector.fired
        self.detector.update(agree, kl)
        if self.detector.fired and not was_fired:
            self._fire()
        if self.detector.fired and self.mode == "anneal":
            self._anneal_one_rollout()
        self._publish_state()

    # ---- the actions -----------------------------------------------------------------------
    def _fire(self) -> None:
        from main.launcher.ipc import emit
        self._coef_at_fire = float(getattr(self.model, "distill_coef", 0.0) or 0.0)
        d = self.detector
        if not self._announced:
            self._announced = True
            emit(f"🛑 [DISTILL-STOP] FIRED at {int(getattr(self.model, 'num_timesteps', 0))} steps: "
                 f"{AGREE_SIGNAL} EMA {d.agree_ema:.4f} has PLATEAUED (improvement over "
                 f"{d.window} rollouts < {d.eps:g}) while {COLLATERAL_SIGNAL} EMA {d.kl_ema:.5f} is "
                 f"still RISING, for {d.persist} consecutive rollouts. This is the fold's optimal "
                 f"length: v8 lost ~5pp of untaught win rate running past it "
                 f"(ledger 2026-09-01). "
                 + {"warn": "mode=warn: training continues unchanged.",
                    "anneal": (f"mode=anneal: --distill-coef {self._coef_at_fire:g} now decays "
                               f"x{self.anneal_factor:g} per rollout to 0."),
                    "abort": "mode=abort: stopping learn() at the next step "
                             "(clean stop, checkpoint saved)."}[self.mode])
        if self.mode == "abort":
            self._abort = True

    def _anneal_one_rollout(self) -> None:
        """Geometric decay of `--distill-coef`, with the snap to exactly 0 described on the class."""
        cur = float(getattr(self.model, "distill_coef", 0.0) or 0.0)
        if cur <= 0.0:
            return
        nxt = cur * self.anneal_factor
        base = self._coef_at_fire if self._coef_at_fire else cur
        if nxt < 1e-6 * base:
            nxt = 0.0
        self.model.distill_coef = nxt

    # ---- state + logging -------------------------------------------------------------------
    def _publish_state(self) -> None:
        d = self.detector
        self.logger.record("distill/stop_state", float(d.state_code()))
        self.logger.record("distill/stop_signal", 1.0 if d.fired else 0.0)
        self.logger.record("distill/stop_rollouts_since_fire",
                           float(d.rollouts_since_fire) if d.fired else 0.0)
        # The sidecar's view — read by `_model_hparams` at every checkpoint site. Present only
        # while this callback is registered, so an ordinary run's sidecar is unchanged.
        self.model.distill_stop_state = self.sidecar_state()

    def sidecar_state(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "detector": self.detector.state(),
            "coef_at_fire": self._coef_at_fire,
            "distill_coef_annealed": float(getattr(self.model, "distill_coef", 0.0) or 0.0),
        }
