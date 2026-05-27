import math

from stable_baselines3.common.callbacks import BaseCallback


class TwoPhaseLRCallback(BaseCallback):
    """KL-driven adaptive LR, then cosine decay.

    Phase 1 (``num_timesteps < anneal_start_steps``):
        KL-driven adjustment identical to :class:`AdaptivePPOCallback` —
        an EMA of ``train/approx_kl`` nudges LR up/down by ``lr_factor``
        whenever it leaves ``target_kl * (1 ± kl_tolerance)``. Bounded by
        ``[min_lr, max_lr]``.

    Phase 2 (``num_timesteps >= anneal_start_steps``):
        Pure cosine decay from ``handoff_lr`` to ``anneal_min_lr`` over
        ``[anneal_start_steps, total_steps]``. KL adjustment is OFF. The
        schedule is fully deterministic — given ``handoff_lr`` and the
        current step it returns the same LR every time, so launcher
        restarts reproduce the schedule exactly.

    handoff_lr is captured ONCE, at the first rollout where ``num_timesteps``
    crosses ``anneal_start_steps``, as the optimizer's current LR — i.e.,
    the value Phase 1 dynamically settled on. It must be persisted across
    launcher restarts (via the checkpoint sidecar) so a resumed run picks
    up the same cosine. Pass the persisted value via the ``handoff_lr=``
    constructor arg when resuming.

    If a resumed run finds itself already in Phase 2 (``num_timesteps >=
    anneal_start_steps``) but no ``handoff_lr`` was passed in, the
    optimizer's current LR is used as a graceful fallback (e.g. for runs
    that pre-date the two-phase callback).

    Args:
        initial_lr:         LR at the start of a fresh run (Phase 1 seed).
                            Ignored on resume — Phase 1 reads the optimizer's
                            saved LR instead.
        total_steps:        Step at which the cosine reaches ``anneal_min_lr``.
                            Usually equals ``--steps``.
        anneal_start_steps: Step at which Phase 1 ends and Phase 2 begins.
                            Set to 0 to start the cosine immediately.
        anneal_min_lr:      LR floor reached at ``total_steps``.
        min_lr:             Phase 1 LR lower bound.
        max_lr:             Phase 1 LR upper bound. Defaults to ``2 × initial_lr``.
        target_kl:          Phase 1 KL setpoint.
        kl_tolerance:       Phase 1 KL band as a fraction of ``target_kl``.
        lr_factor:          Phase 1 LR step (multiplicative).
        ema_alpha:          Phase 1 EMA smoothing on ``approx_kl``. With α=0.20
                            the EMA half-life is ~3 rollouts.
        cooldown_rollouts:  After every LR adjustment, skip this many rollouts
                            before considering another adjustment. The EMA
                            continues to update during cooldown, so when the
                            next decision fires it sees current state rather
                            than stale state. Also seeded at startup so a
                            fresh callback (e.g. after a launcher restart)
                            cannot fire on the very first rollout's noise.
                            Default 7 (~2× the EMA half-life at α=0.20).
        handoff_lr:         If not None, treat the run as having already
                            crossed into Phase 2 with this as the cosine
                            start. Used on resume.
    """

    def __init__(
        self,
        initial_lr: float,
        total_steps: int,
        anneal_start_steps: int,
        anneal_min_lr: float,
        min_lr: float = 1e-5,
        max_lr: float | None = None,
        target_kl: float = 0.01,
        kl_tolerance: float = 0.3,
        lr_factor: float = 1.2,
        ema_alpha: float = 0.20,
        cooldown_rollouts: int = 7,
        handoff_lr: float | None = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        # Phase 1 state
        self._current_lr = initial_lr
        self.min_lr = min_lr
        self.max_lr = max_lr if max_lr is not None else initial_lr * 2.0
        self.target_kl = target_kl
        self.kl_tolerance = kl_tolerance
        self.lr_factor = lr_factor
        self.ema_alpha = ema_alpha
        self.cooldown_rollouts = cooldown_rollouts
        self._kl_ema: float | None = None
        # Seed cooldown so the first ``cooldown_rollouts`` rollouts after
        # startup/restart also let the EMA settle before any adjustment fires.
        # Without this, a launcher restart would immediately act on whatever
        # KL the first rollout happens to log, which is exactly the chatter
        # the cooldown is meant to suppress.
        self._cooldown_remaining: int = cooldown_rollouts

        # Phase 2 state
        self._total_steps = total_steps
        self._anneal_start_steps = anneal_start_steps
        self._anneal_min_lr = anneal_min_lr
        self._handoff_lr: float | None = handoff_lr

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def current_lr(self) -> float:
        return self._current_lr

    @property
    def handoff_lr(self) -> float | None:
        """Cosine starting LR. None while still in Phase 1."""
        return self._handoff_lr

    def phase(self, t: int) -> int:
        return 1 if t < self._anneal_start_steps else 2

    # ------------------------------------------------------------------
    # Cosine math
    # ------------------------------------------------------------------

    def _cosine_lr_at(self, t: int) -> float:
        """Cosine decay from ``handoff_lr`` → ``anneal_min_lr`` over
        ``[anneal_start_steps, total_steps]``. Clamped at ``anneal_min_lr``
        for ``t > total_steps``. Caller must ensure ``handoff_lr`` is set.
        """
        assert self._handoff_lr is not None, "handoff_lr must be set before entering Phase 2"
        duration = self._total_steps - self._anneal_start_steps
        if duration <= 0:
            return self._anneal_min_lr
        x = max(0.0, min(1.0, (t - self._anneal_start_steps) / duration))
        return self._anneal_min_lr + (self._handoff_lr - self._anneal_min_lr) * 0.5 * (1.0 + math.cos(math.pi * x))

    # ------------------------------------------------------------------
    # Callback hooks
    # ------------------------------------------------------------------

    def _on_training_start(self) -> None:
        t = self.model.num_timesteps

        # Resumed past anneal_start with no persisted handoff_lr (legacy run
        # or first run after the two-phase upgrade landed). Use the optimizer's
        # current LR as the cosine start.
        if self.phase(t) == 2 and self._handoff_lr is None:
            self._handoff_lr = self.model.policy.optimizer.param_groups[0]["lr"]
            if self.verbose >= 1:
                print(
                    f"[TwoPhaseLR] No persisted handoff_lr on resume at step {t:,}; "
                    f"using optimizer LR {self._handoff_lr:.2e} as cosine start."
                )

        # Set the initial LR for whatever phase we're starting in.
        self._apply_lr(t)
        if self.verbose >= 1:
            if self.phase(t) == 1:
                print(
                    f"[TwoPhaseLR] Phase 1 (adaptive) at step {t:,}: "
                    f"LR={self._current_lr:.2e}, "
                    f"bounds=[{self.min_lr:.2e}, {self.max_lr:.2e}], "
                    f"anneal_start={self._anneal_start_steps:,}"
                )
            else:
                print(
                    f"[TwoPhaseLR] Phase 2 (cosine) at step {t:,}: "
                    f"LR={self._current_lr:.2e}, "
                    f"handoff={self._handoff_lr:.2e} → "
                    f"min={self._anneal_min_lr:.2e} at step {self._total_steps:,}"
                )

    def _apply_lr(self, t: int) -> None:
        """Compute and install the LR for step ``t`` based on phase."""
        if self.phase(t) == 1:
            lr = self._current_lr
        else:
            lr = self._cosine_lr_at(t)
            self._current_lr = lr  # keep public ``current_lr`` in sync for checkpoint metadata
        self.model.lr_schedule = lambda _: lr

    def _on_rollout_end(self) -> None:
        t = self.model.num_timesteps

        # Detect Phase 1 → Phase 2 crossing. Capture the optimizer's current
        # LR as handoff_lr — this is the value Phase 1 actually settled on
        # (may differ slightly from _current_lr if SB3's _update_learning_rate
        # smoothed it). Sidecar persistence happens via record_checkpoint,
        # which reads this via the handoff_lr_fn passed to the checkpoint
        # callback.
        if self.phase(t) == 2 and self._handoff_lr is None:
            self._handoff_lr = self.model.policy.optimizer.param_groups[0]["lr"]
            if self.verbose >= 1:
                print(
                    f"[TwoPhaseLR] Phase 1 → 2 at step {t:,}: "
                    f"handoff_lr = {self._handoff_lr:.2e}, cosine target "
                    f"{self._anneal_min_lr:.2e} at step {self._total_steps:,}"
                )

        if self.phase(t) == 1:
            self._adapt_lr_from_kl()
        else:
            lr = self._cosine_lr_at(t)
            self._current_lr = lr
            self.model.lr_schedule = lambda _: lr
            self.logger.record("train/lr_annealing", lr)

        self.logger.record("train/n_epochs", self.model.n_epochs)

    def _adapt_lr_from_kl(self) -> None:
        """Phase 1: nudge LR based on EMA of approx_kl, with post-adjustment cooldown.

        The EMA is updated every rollout (so it tracks current state during
        the cooldown), but LR adjustments only fire when no cooldown is
        pending. Each adjustment arms ``cooldown_rollouts`` rollouts of
        suppression — preventing the per-rollout compounding that would
        otherwise overshoot when the EMA lags reality.
        """
        kl = self.model.logger.name_to_value.get("train/approx_kl")
        if kl is None:
            return

        # Always update EMA so it tracks during cooldown.
        if self._kl_ema is None:
            self._kl_ema = kl
        else:
            self._kl_ema = self.ema_alpha * kl + (1.0 - self.ema_alpha) * self._kl_ema

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return

        lo = self.target_kl * (1.0 - self.kl_tolerance)
        hi = self.target_kl * (1.0 + self.kl_tolerance)

        if self._kl_ema > hi:
            new_lr = max(self._current_lr / self.lr_factor, self.min_lr)
        elif self._kl_ema < lo:
            new_lr = min(self._current_lr * self.lr_factor, self.max_lr)
        else:
            return

        if new_lr != self._current_lr:
            direction = "↓" if new_lr < self._current_lr else "↑"
            if self.verbose >= 1:
                print(
                    f"[AdaptiveLR] approx_kl={kl:.4f} ema={self._kl_ema:.4f} "
                    f"(target={self.target_kl:.3f}) → LR {direction} "
                    f"{self._current_lr:.2e} → {new_lr:.2e} "
                    f"(cooldown {self.cooldown_rollouts} rollouts)"
                )
            self._current_lr = new_lr
            self.model.lr_schedule = lambda _: new_lr
            self._cooldown_remaining = self.cooldown_rollouts

    def _on_step(self) -> bool:
        return True


class AdaptivePPOCallback(BaseCallback):
    """Adjusts LR after each rollout based on an EMA of approx_kl.

    Reads approx_kl logged after the previous train() call, folds it into an
    exponential moving average, and nudges the LR up or down to keep the EMA
    inside [target_kl * (1 - tolerance), target_kl * (1 + tolerance)].

    Using an EMA means a single outlier rollout only nudges the average by
    `ema_alpha` of its distance, so sustained drift over several rollouts is
    required before a LR change fires.

    Args:
        initial_lr:         Starting LR (also the basis for max_lr default).
        target_kl:          Desired approx_kl. Healthy Gen3OU range is
                            0.005–0.015; default 0.01 centres the ±30% band
                            at [0.007, 0.013].
        kl_tolerance:       Fractional band around target before adjusting
                            (0.3 = ±30%).
        lr_factor:          Multiply/divide LR by this per adjustment.
        min_lr:             Hard lower bound on LR.
        max_lr:             Hard upper bound on LR. Defaults to 2× initial_lr.
        ema_alpha:          EMA smoothing factor (0, 1]. With α=0.20 the EMA
                            half-life is ~3 rollouts.
        cooldown_rollouts:  After every adjustment, skip this many rollouts
                            before considering another. The EMA still updates
                            so the next decision sees current state, not stale
                            state. Also seeded at startup so a fresh callback
                            cannot fire on the first rollout's noise.
                            Default 7 (~2× the EMA half-life at α=0.20).
    """

    def __init__(
        self,
        initial_lr: float,
        target_kl: float = 0.01,
        kl_tolerance: float = 0.3,
        lr_factor: float = 1.2,
        min_lr: float = 1e-5,
        max_lr: float | None = None,
        ema_alpha: float = 0.20,
        cooldown_rollouts: int = 7,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self._current_lr = initial_lr
        self.target_kl = target_kl
        self.kl_tolerance = kl_tolerance
        self.lr_factor = lr_factor
        self.min_lr = min_lr
        self.max_lr = max_lr if max_lr is not None else initial_lr * 2.0
        self.ema_alpha = ema_alpha
        self.cooldown_rollouts = cooldown_rollouts
        self._kl_ema: float | None = None
        # Seed cooldown so launcher restarts and fresh runs both give the EMA
        # ``cooldown_rollouts`` to settle before any adjustment fires.
        self._cooldown_remaining: int = cooldown_rollouts

    @property
    def current_lr(self) -> float:
        return self._current_lr

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        kl = self.model.logger.name_to_value.get("train/approx_kl")
        if kl is None:
            return

        # Always update EMA so it tracks during cooldown.
        if self._kl_ema is None:
            self._kl_ema = kl
        else:
            self._kl_ema = self.ema_alpha * kl + (1.0 - self.ema_alpha) * self._kl_ema

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            self.logger.record("train/n_epochs", self.model.n_epochs)
            return

        lo = self.target_kl * (1.0 - self.kl_tolerance)
        hi = self.target_kl * (1.0 + self.kl_tolerance)

        if self._kl_ema > hi:
            new_lr = max(self._current_lr / self.lr_factor, self.min_lr)
        elif self._kl_ema < lo:
            new_lr = min(self._current_lr * self.lr_factor, self.max_lr)
        else:
            return

        if new_lr != self._current_lr:
            direction = "↓" if new_lr < self._current_lr else "↑"
            if self.verbose >= 1:
                print(
                    f"[AdaptiveLR] approx_kl={kl:.4f} ema={self._kl_ema:.4f} (target={self.target_kl:.3f}) "
                    f"→ LR {direction} {self._current_lr:.2e} → {new_lr:.2e} "
                    f"(cooldown {self.cooldown_rollouts} rollouts)"
                )
            self._current_lr = new_lr
            self.model.lr_schedule = lambda _: new_lr
            self._cooldown_remaining = self.cooldown_rollouts

        self.logger.record("train/n_epochs", self.model.n_epochs)


# Alias so existing references still resolve.
AdaptiveLRCallback = AdaptivePPOCallback
