from stable_baselines3.common.callbacks import BaseCallback


class AdaptivePPOCallback(BaseCallback):
    """Adjusts LR after each rollout based on approx_kl.

    Reads approx_kl logged after the previous train() call and nudges the LR
    up or down to keep KL inside [target_kl * (1 - tolerance), target_kl * (1 + tolerance)].

    Args:
        initial_lr:    Starting LR (also the basis for max_lr default).
        target_kl:     Desired approx_kl. Healthy PPO range is 0.01–0.02.
        kl_tolerance:  Fractional band around target before adjusting (0.3 = ±30%).
        lr_factor:     Multiply/divide LR by this per adjustment.
        min_lr:        Hard lower bound on LR.
        max_lr:        Hard upper bound on LR. Defaults to 2× initial_lr.
    """

    def __init__(
        self,
        initial_lr: float,
        target_kl: float = 0.015,
        kl_tolerance: float = 0.3,
        lr_factor: float = 1.5,
        min_lr: float = 1e-5,
        max_lr: float | None = None,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self._current_lr = initial_lr
        self.target_kl = target_kl
        self.kl_tolerance = kl_tolerance
        self.lr_factor = lr_factor
        self.min_lr = min_lr
        self.max_lr = max_lr if max_lr is not None else initial_lr * 2.0

    @property
    def current_lr(self) -> float:
        return self._current_lr

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        kl = self.model.logger.name_to_value.get("train/approx_kl")
        if kl is None:
            return

        lo = self.target_kl * (1.0 - self.kl_tolerance)
        hi = self.target_kl * (1.0 + self.kl_tolerance)

        if kl > hi:
            new_lr = max(self._current_lr / self.lr_factor, self.min_lr)
        elif kl < lo:
            new_lr = min(self._current_lr * self.lr_factor, self.max_lr)
        else:
            return

        if new_lr == self._current_lr:
            return

        direction = "↓" if new_lr < self._current_lr else "↑"
        if self.verbose >= 1:
            print(
                f"[AdaptiveLR] approx_kl={kl:.4f} (target={self.target_kl:.3f}) "
                f"→ LR {direction} {self._current_lr:.2e} → {new_lr:.2e}"
            )

        self._current_lr = new_lr
        self.model.lr_schedule = lambda _: new_lr


# Alias so existing references still resolve.
AdaptiveLRCallback = AdaptivePPOCallback
