"""The McCandlish gradient-noise-scale estimator, and the advisor that reads it out.

`--grad-accum-steps K >= 2` gives gradient norms at TWO batch sizes for free (one micro-batch and
the accumulated group), which is exactly what the `B_simple = tr(Sigma)/|G|^2` critical-batch-size
estimator needs. `train()` captures the two norms; everything else — the two-point solve, the
separate EMAs, and the rate-limited out-of-band warnings — is here.

The PER-TERM half (`noise_scale_terms.py`) reuses this module's estimator verbatim on
per-loss-GROUP gradients: `_fold_per_term_noise` below is only bookkeeping (its own EMAs, the
`share` denominator) around the SAME `_noise_scale_estimate`. The math is not forked, because
the whole point of the comparison is that the total and the per-term readings are the same
quantity measured on different gradients.
"""
def debiased_ema(prev, n_prev, sample, decay):
    """ONE warm-up fold, shared by the TOTAL reading and the PER-TERM readings.

    A plain `d*prev + (1-d)*x` at a fixed `d = 0.99` that is ANCHORED ON ITS FIRST SAMPLE reads
    `0.99*x1 + 0.01*x2` after two samples — i.e. for the first few hundred calls it reports the
    first sample rather than the average, and a single bad first sample (this estimator's
    single-call `tr(Sigma)`/`|G|^2` can SIGN-FLIP under noise) suppresses the tag for hundreds of
    calls. That is why the first production reading of `train/noise_scale` on R5F15 had to be
    called "provisional, n=2".

    The fix is the same one Adam's `ema / (1 - beta^t)` performs, spelled as an effective decay:
    `d = min(decay, 1 - 1/(n+1))` makes the estimate a plain running MEAN until the `1/(1-decay)`
    window fills and the exponential decay takes over. So at every step the reading is an unbiased
    average of the samples taken so far, and a CONSTANT stream reads its own constant from step 1.

    `prev` is `None` (equivalently `n_prev == 0`) on the first sample. Pure; unit-tested.
    """
    if prev is None or n_prev <= 0:
        return float(sample)
    d = min(float(decay), 1.0 - 1.0 / (float(n_prev) + 1.0))
    return d * float(prev) + (1.0 - d) * float(sample)


class NoiseScaleDiagnostics:
    """The noise-scale EMA state, the pure estimator, and the advisor."""

    # +NOISE-SCALE: running (EMA) estimates of the McCandlish gradient-noise-scale numerator/denominator
    # — tr(Σ) and |G|² — accumulated across train() calls (one sample per call). None until the first
    # measurable call. Only updated when grad_accum_steps >= 2 (the diagnostic needs gradients at TWO
    # batch sizes: one micro-batch = batch_size, and the accumulated group = batch_size·accum). Process-
    # local (reset to None on a launcher restart → re-converges in a few hundred calls); not saved.
    _noise_ema_s: float = None    # EMA of tr(Σ)  (per-example gradient-variance trace)
    _noise_ema_g2: float = None   # EMA of |G|²   (true-gradient squared norm)
    # +WARM-UP: how many samples are behind the two EMAs above — the `n` `debiased_ema` needs.
    # The TOTAL used to anchor on its first sample with no correction, so its first ~hundreds of
    # readings were that sample rather than an average; it now warms up through the SAME helper as
    # the per-term readings, so the two can never disagree about start-up bias again.
    _noise_ema_n: int = 0
    # +PER-TERM: the same two EMAs, per loss GROUP — {group: [ema_tr_sigma, ema_g2, n_samples]}.
    # None until the first sampled call; same process-local, unsaved lifetime as the two scalars
    # above. The third slot drives the debiased warm-up (see `_fold_per_term_noise`).
    _noise_ema_terms: dict = None
    # +PER-TERM: how many train() calls have gone by, for the `_NOISE_PER_TERM_EVERY` cadence.
    _noise_per_term_calls: int = 0

    @staticmethod
    def _global_grad_sq(params) -> float:
        """Squared global L2 norm ‖g‖² of the CURRENT .grad over all params (one device→host sync).
        Mirrors what clip_grad_norm_ computes, but read-only (no clipping)."""
        sq = None
        for p in params:
            g = p.grad
            if g is not None:
                s = g.detach().pow(2).sum()
                sq = s if sq is None else sq + s
        return float(sq) if sq is not None else 0.0

    @staticmethod
    def _noise_scale_estimate(g_small_sq, g_big_sq, b_small, b_big):
        """McCandlish et al. 2018 'simple' gradient-noise-scale building blocks from squared gradient
        norms at TWO batch sizes. Since E‖Ĝ_B‖² = ‖G‖² + tr(Σ)/B, two (B, ‖Ĝ_B‖²) points pin both
        unknowns:
            |G|²   = (b_big·g_big_sq − b_small·g_small_sq) / (b_big − b_small)        # true-grad norm²
            tr(Σ)  = (g_small_sq − g_big_sq) / (1/b_small − 1/b_big)                   # per-example noise
        Returns ``(tr_sigma, g2)`` (single-sample, pre-EMA; either can be negative under noise — the
        caller EMAs them separately before the B_simple = tr(Σ)/|G|² ratio). Pure → unit-testable."""
        g2 = (b_big * g_big_sq - b_small * g_small_sq) / (b_big - b_small)
        tr_sigma = (g_small_sq - g_big_sq) / (1.0 / b_small - 1.0 / b_big)
        return tr_sigma, g2
    # NSR advisor state (class-level; per-process). _nsr_warn_last rate-limits the ⚠️ [NOISE]
    # band (see _noise_scale_advice). Process-local (resets each child — fine, it re-warns).
    _nsr_warn_last: dict = None
    _nsr_samples: int = 0

    #: The band edges the advisor reads `B_simple / effective_batch` against. > HIGH ⇒ noise-limited,
    #: < LOW ⇒ over-batched, between ⇒ in band. Named so the per-term half cannot drift from the total.
    _NSR_HIGH = 2.0
    _NSR_LOW = 0.5
    #: A total-vs-policy gap this many times or more is reported as a DISAGREEMENT even when both
    #: ratios land in the same band (a 5x gap inside "over-batched" is still the finding).
    _NSR_DISAGREE_FACTOR = 3.0

    @classmethod
    def _nsr_band(cls, ratio):
        """`None` / 'noise-limited' / 'in band' / 'over-batched' for one ratio. Pure."""
        if ratio is None:
            return None
        if ratio > cls._NSR_HIGH:
            return "noise-limited"
        if ratio < cls._NSR_LOW:
            return "over-batched"
        return "in band"

    @classmethod
    def _noise_scale_advice(cls, global_ratio, b_eff, policy_ratio=None):
        """PURE advisory logic for the noise-scale ratios → list of (key, warning) pairs; [] when
        healthy. The TUI-warning half of the McCandlish instrumentation: a ratio ≫ 1 means updates
        are noise-dominated (each step's direction is mostly sideways — and under Adam the noise
        still moves params at full speed, so spurious content gets WRITTEN, not just slowed); ≪ 1
        means samples are being spent polishing an already-clean gradient instead of taking more
        steps. Each warning names the concrete fix.

        `policy_ratio` (`train/noise_scale_ratio_policy`, present once the per-term probe has warmed
        up) is the SAME quantity measured on the clipped-surrogate gradient alone. It is quoted
        inside the band warnings, and a DISAGREEMENT between the two gets its own line: the total is
        an average over a dozen dense supervised heads whose per-example gradients agree, so a total
        that reads "over-batched" while the policy term does not is the total being DEFLATED, not
        the batch being too big. Acting on the total in that state shrinks the batch the policy
        gradient needed."""
        import math
        out = []
        pol = "" if policy_ratio is None else (
            f" PPO-policy-term ratio {policy_ratio:.3g} ({cls._nsr_band(policy_ratio)}).")
        if global_ratio is not None:
            if global_ratio > cls._NSR_HIGH:
                out.append(("global_high", (
                    f"⚠️ [NOISE] train/noise_scale_ratio {global_ratio:.1f} — gradient NOISE-LIMITED "
                    f"(critical batch ≈ {global_ratio * b_eff / 1000:.0f}k vs effective {b_eff / 1000:.0f}k; "
                    f"updates are mostly sideways). Fix: raise --grad-accum-steps ~{math.ceil(global_ratio)}× "
                    f"(free — no VRAM/FPS cost, same rollout).{pol}")))
            elif global_ratio < cls._NSR_LOW:
                out.append(("global_low", (
                    f"⚠️ [NOISE] train/noise_scale_ratio {global_ratio:.2f} — OVER-BATCHED (effective "
                    f"{b_eff / 1000:.0f}k is ≫ the critical batch; samples polish an already-clean gradient). "
                    f"Fix: lower --grad-accum-steps for more update steps per sample.{pol}")))
        if global_ratio is not None and policy_ratio is not None:
            gb, pb = cls._nsr_band(global_ratio), cls._nsr_band(policy_ratio)
            far = (max(global_ratio, policy_ratio) >= cls._NSR_DISAGREE_FACTOR
                   * max(min(global_ratio, policy_ratio), 1e-12))
            if gb != pb or far:
                out.append(("total_vs_policy_disagree", (
                    f"⚠️ [NOISE] TOTAL vs POLICY-TERM DISAGREE: train/noise_scale_ratio "
                    f"{global_ratio:.3g} ({gb}) but train/noise_scale_ratio_policy "
                    f"{policy_ratio:.3g} ({pb}). The total is measured on the SUM of every loss "
                    f"term, and the dense supervised aux heads have far lower gradient noise than "
                    f"the clipped surrogate — so a low total can be aux DEFLATION rather than a "
                    f"batch that is too big. Read train/noise_scale_share_* to see which group "
                    f"owns |G|², and size the batch on the term you are trying to train.")))
        return out

    def _emit_noise_scale_warnings(self, global_ratio, b_eff, policy_ratio=None):
        """Rate-limited (30 min per key) Events-panel emit of _noise_scale_advice, after an EMA
        warm-up (first ~20 samples are settling and would false-alarm)."""
        import time
        self._nsr_samples += 1
        if self._nsr_samples < 20:
            return
        if self._nsr_warn_last is None:
            self._nsr_warn_last = {}
        advice = self._noise_scale_advice(global_ratio, b_eff, policy_ratio)
        now = time.time()
        for key, msg in advice:
            if now - self._nsr_warn_last.get(key, 0.0) >= 1800.0:
                self._nsr_warn_last[key] = now
                try:
                    from main.launcher.ipc import emit
                    emit(msg)
                except Exception:
                    print(msg, flush=True)

    def _fold_per_term_noise(self, per_term, b_small, b_big, total_g2):
        """Fold `{group: (g_small_sq, g_big_sq)}` into the per-group EMAs → `{tag: value}` to log.

        Deliberately thin: the two-point solve is `_noise_scale_estimate` (the SAME call the total
        makes), the smoothing is the SAME separately-EMA'd numerator/denominator with the SAME
        decay, and the emit gate is the SAME "both EMAs positive" rule. If the per-term reading
        disagrees with the total, that has to be the GRADIENT differing, not the estimator.

        Three tags per group:
          `train/noise_scale_<g>`        B_simple for that group's gradient alone.
          `train/noise_scale_ratio_<g>`  the same over the effective batch — the actionable read.
          `train/noise_scale_share_<g>`  |G_g|² / |G_total|², i.e. how much of the true gradient's
                                         squared length this group owns.

        ⚠️ The shares do NOT sum to 1, and that is not a bug: `|G_total|² = ‖Σ_g G_g‖²` carries the
        CROSS terms, so groups pulling together sum above 1 and groups fighting sum below it. Read a
        share as "how big is this group's own pull", never as a partition of the total.
        """
        from agents.training.instrumented_ppo.constants import _NOISE_SCALE_EMA_DECAY
        if self._noise_ema_terms is None:
            self._noise_ema_terms = {}
        decay = _NOISE_SCALE_EMA_DECAY
        out = {}
        for group, (small_sq, big_sq) in per_term.items():
            tr_sigma, g2 = self._noise_scale_estimate(small_sq, big_sq, b_small, b_big)
            prev = self._noise_ema_terms.get(group)
            n_prev = int(prev[2]) if prev is not None else 0
            # DEBIASED WARM-UP — the SHARED `debiased_ema`, the identical fold the TOTAL takes.
            # The group that needs it most is exactly the one this module exists to read: a
            # strongly noise-limited policy term has |G|² ≈ 0 at these batch sizes, so its
            # single-sample estimate SIGN-FLIPS, and one negative first sample under an
            # anchor-on-first-sample fold would suppress the tag for hundreds of calls.
            ema_s = debiased_ema(None if prev is None else prev[0], n_prev, tr_sigma, decay)
            ema_g2 = debiased_ema(None if prev is None else prev[1], n_prev, g2, decay)
            self._noise_ema_terms[group] = [ema_s, ema_g2, n_prev + 1]
            if ema_g2 > 1e-12 and ema_s > 0.0:
                b_simple = ema_s / ema_g2
                out[f"train/noise_scale_{group}"] = float(b_simple)
                out[f"train/noise_scale_ratio_{group}"] = float(b_simple / b_big)
            if total_g2 is not None and total_g2 > 1e-12 and ema_g2 > 0.0:
                out[f"train/noise_scale_share_{group}"] = float(ema_g2 / total_g2)
        return out

    def _per_term_ratio(self, group, b_big):
        """`B_simple(group) / b_big` from the smoothed per-group EMAs, or None if not yet readable.

        Read from the EMA state rather than from the fold's return value on purpose: under a
        `_NOISE_PER_TERM_EVERY > 1` cadence most `train()` calls take no per-term sample, but the
        advisor still fires on them, and a warning that silently drops the policy-term half on 3
        calls in 4 would be read as "no per-term reading exists" rather than "not sampled here".
        """
        ema = (self._noise_ema_terms or {}).get(group)
        if not ema or not (ema[1] > 1e-12 and ema[0] > 0.0) or not b_big:
            return None
        return float((ema[0] / ema[1]) / b_big)

    def _total_ratio(self, b_big):
        """`B_simple / b_big` from the TOTAL gradient's smoothed EMAs, or None if not readable.

        The same emit gate `train()` applies before recording `train/noise_scale_ratio` (both EMAs
        positive), so this returns exactly the value TensorBoard carries — never a value the
        series itself withheld."""
        if self._noise_ema_s is None or self._noise_ema_g2 is None or not b_big:
            return None
        if not (self._noise_ema_g2 > 1e-12 and self._noise_ema_s > 0.0):
            return None
        return float((self._noise_ema_s / self._noise_ema_g2) / b_big)

    def noise_ratio_sample(self, kind, b_big):
        """`(ratio, n_samples)` for a CONSUMER of the noise scale — `'total'`, or a term group.

        The read seam for `--adaptive-batch` (`agents.training.adaptive_batch_callback`), which
        steers `grad_accum_steps` by one of these ratios. It exists so the controller reads the
        EMA this class already maintains instead of estimating anything itself: a control loop and
        the TensorBoard series it is judged by MUST be the same number, or a post-mortem compares
        the controller against a quantity it never saw.

        `n_samples` is the fold count behind the ratio — the warm-up gate. `_nsr_samples` counts
        the calls on which the TOTAL was readable (it is incremented by
        `_emit_noise_scale_warnings`, which fires only then); a per-term group carries its own
        count in slot 2 of its EMA triple. Returns `(None, 0)` for an unknown kind rather than
        raising, because a diagnostic read must never be able to take a run down."""
        if kind == "total":
            return self._total_ratio(b_big), int(self._nsr_samples)
        ema = (self._noise_ema_terms or {}).get(kind)
        n = int(ema[2]) if ema else 0
        return self._per_term_ratio(kind, b_big), n
