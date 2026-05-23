from stable_baselines3.common.callbacks import BaseCallback


class AdaptivePPOCallback(BaseCallback):
    """Adjusts LR and n_epochs after each rollout based on two independent signals.

    clip_fraction → LR:    high clip = step size too large; low clip = room to grow
    approx_kl     → n_epochs: high KL = too many passes; low KL = room to add one

    Both levers read signals logged after the previous train() call and apply
    changes before the next train() call. One-rollout lag is fine at this timescale.

    Args:
        initial_lr:     Starting LR (also the basis for max_lr default).
        initial_epochs: Starting n_epochs.
        clip_lo:        clip_fraction below this → LR up.
        clip_hi:        clip_fraction above this → LR down.
        lr_factor:      Multiply/divide LR by this per adjustment.
        min_lr:         Hard lower bound on LR.
        max_lr:         Hard upper bound on LR. Defaults to 2× initial_lr.
        kl_lo:          approx_kl below this → add an epoch.
        kl_hi:          approx_kl above this → drop an epoch.
        epochs_step:    How many epochs to add/drop per adjustment.
        min_epochs:     Hard lower bound on n_epochs.
        max_epochs:     Hard upper bound on n_epochs.
    """

    def __init__(
        self,
        initial_lr: float,
        initial_epochs: int = 10,
        clip_lo: float = 0.07,
        clip_hi: float = 0.15,
        lr_factor: float = 1.2,
        min_lr: float = 1e-5,
        max_lr: float | None = None,
        kl_lo: float = 0.010,
        kl_hi: float = 0.020,
        epochs_step: int = 1,
        min_epochs: int = 2,
        max_epochs: int = 12,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self._current_lr = initial_lr
        self._current_epochs = initial_epochs
        self.max_lr = max_lr if max_lr is not None else initial_lr * 2.0
        self.clip_lo = clip_lo
        self.clip_hi = clip_hi
        self.lr_factor = lr_factor
        self.min_lr = min_lr
        self.kl_lo = kl_lo
        self.kl_hi = kl_hi
        self.epochs_step = epochs_step
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs

    @property
    def current_lr(self) -> float:
        return self._current_lr

    @property
    def current_epochs(self) -> int:
        return self._current_epochs

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        log = self.model.logger.name_to_value
        clip = log.get("train/clip_fraction")
        kl = log.get("train/approx_kl")

        new_lr = self._current_lr
        new_epochs = self._current_epochs

        if clip is not None:
            if clip > self.clip_hi:
                new_lr = max(self._current_lr / self.lr_factor, self.min_lr)
            elif clip < self.clip_lo:
                new_lr = min(self._current_lr * self.lr_factor, self.max_lr)

        if kl is not None:
            if kl > self.kl_hi:
                new_epochs = max(self._current_epochs - self.epochs_step, self.min_epochs)
            elif kl < self.kl_lo:
                new_epochs = min(self._current_epochs + self.epochs_step, self.max_epochs)

        lr_changed = new_lr != self._current_lr
        epochs_changed = new_epochs != self._current_epochs

        if lr_changed or epochs_changed:
            if self.verbose >= 1:
                parts = []
                if lr_changed:
                    direction = "↓" if new_lr < self._current_lr else "↑"
                    parts.append(f"clip={clip:.3f} → LR {direction} {self._current_lr:.2e} → {new_lr:.2e}")
                if epochs_changed:
                    direction = "↓" if new_epochs < self._current_epochs else "↑"
                    parts.append(f"kl={kl:.4f} → epochs {direction} {self._current_epochs} → {new_epochs}")
                print(f"[AdaptivePPO] {' | '.join(parts)}")

            self._current_lr = new_lr
            self._current_epochs = new_epochs
            self.model.lr_schedule = lambda _: new_lr
            self.model.n_epochs = new_epochs

        self.logger.record("train/n_epochs", self._current_epochs)


# Alias so existing references (e.g. train_rl_agent.py resume seeding) still resolve.
AdaptiveLRCallback = AdaptivePPOCallback
