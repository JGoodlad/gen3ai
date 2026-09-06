"""`SignalMetricsCallback` — the OUTCOME-ENTROPY half of the `signal/` TensorBoard group.

The other half (advantage density, `signal/adv_*`) is read inside `train()` off the rollout
buffer; see `instrumented_ppo/signal_metrics.py` for the module docstring that explains why the
two must be read TOGETHER (the mirror paradox), and the PopArt units caveat.

**Where the outcomes come from.** `MaskableAgentWrapper.step` already publishes
``info["win_outcome"]`` (1.0 win / 0.0 loss-or-tie) and ``info["opponent_class"]`` at every episode
end. This callback watches the `done` infos as they stream past during collection and pushes them
into rolling per-kind windows — no extra battles, no `env_method` round trip, no env state.

**Both rollout paths are covered.** The stock `collect_rollouts` publishes ``infos``/``dones`` in
the callback locals; `collect_rollouts_async` (`--async-rollout`) publishes ``wave_infos`` /
``wave_dones`` instead (a wave is a macro-step over whichever envs came ready). We read whichever
pair is present. Unlike `WinProbLabelCallback` — which needs the (step, env) BUFFER ROW and so
cannot use the wave batching — outcome entropy is a per-episode aggregate with no row alignment, so
the wave form carries everything it needs.

ALWAYS ON: three numpy means over ≤200-element deques, once per rollout. No flag.
"""
from __future__ import annotations


from stable_baselines3.common.callbacks import BaseCallback

from agents.training.instrumented_ppo.signal_metrics import (   # noqa: F401 — declared re-export
    OPP_CLASS_SUFFIX,
    OutcomeEntropyTracker,
)

# `OPP_CLASS_SUFFIX` moved to `instrumented_ppo.signal_metrics` (2026-09-06) and is RE-EXPORTED
# here so every existing import path still resolves. It had to move: `instrumented_ppo.ppo` needs
# the same map for the `win_prob/start_*` per-class split, and importing it from this callback
# would put a back-edge from the package into a module that imports the package.



class SignalMetricsCallback(BaseCallback):
    """Rolling `signal/outcome_entropy*` from the episode outcomes the training loop already sees."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.tracker = OutcomeEntropyTracker()

    def _consume(self, infos, dones) -> None:
        for info, done in zip(infos, dones):
            if not done or not isinstance(info, dict) or "win_outcome" not in info:
                continue
            kind = OPP_CLASS_SUFFIX.get(info.get("opponent_class"))
            self.tracker.observe(float(info["win_outcome"]) >= 0.5, kind)

    def _on_step(self) -> bool:
        # SYNC keys first; the async collector's wave batching uses different names (see docstring).
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        if infos is None or dones is None:
            infos = self.locals.get("wave_infos")
            dones = self.locals.get("wave_dones")
        if infos is not None and dones is not None:
            self._consume(infos, dones)
        return True

    def _on_rollout_end(self) -> None:
        for key, val in self.tracker.metrics().items():
            self.logger.record(f"signal/{key}", val)

    # `signal/outcome_entropy_rung` is NOT emitted here. It is emitted by `ExploiterLadderCallback`,
    # which owns the number: the ladder SWAPS the target's weights mid-run, so the `_target` window
    # above straddles two different opponents across a promotion, whereas the ladder's own `_last_wr`
    # is by construction the window it zeroes on every swap — the rung being fought NOW. Reaching
    # across the CallbackList for it would also be a lie about the callback graph (SB3 sets `.parent`
    # only under an EventCallback, never inside a plain CallbackList).
