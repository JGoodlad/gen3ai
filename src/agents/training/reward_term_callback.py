"""`RewardTermMetricsCallback` — the `reward/` TensorBoard group.

`gen3_reward_term_export_v1`. The math is pure and lives in `reward_term_stats`; this file is the
transport only: once per rollout, PULL each worker's drained per-term sums through `env_method`,
merge them, and record the scalars.

**Why an `env_method` PULL and not an info dict** — the same reason `TeamWinRateCallback` gives.
The reward is computed in the ENV WORKER, and under `--async-rollout` the callback's step locals
arrive wave-batched with no way to recover which buffer row a step landed on. `env_method` is
drain-safe on `AsyncSubprocVecEnv` (it stashes in-flight step results before the barrier RPC), so
ONE seam covers both collectors, and the accumulator's `drain()` zeroes the window, so a rollout
boundary that pulls twice cannot double-count.

**ALWAYS ON, no flag.** The accumulator folds only the ACTIVE terms (9 of 35 under the production
composition) and the pull is one small dict per worker per rollout, so there is nothing an opt-out
would buy. A run whose env has no Gen3 reward manager returns `None` from every worker and this
callback records nothing at all — an absent curve, never a zero.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from stable_baselines3.common.callbacks import BaseCallback

from agents.training.reward_term_stats import merge_drained, reward_term_metrics


class RewardTermMetricsCallback(BaseCallback):
    """Per-rollout `reward/<term>_mean` / `_abs_share` + the class rollups.

    ``term_class`` is the ``{term -> rollup}`` map built from the run's own
    `reward_class_composition` census (see `reward_term_stats.term_class_map`), passed in rather
    than re-derived here so the exported grouping and the startup banner read one declaration.
    """

    def __init__(self, term_class: Optional[Dict[str, str]] = None, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.term_class: Dict[str, str] = dict(term_class or {})

    def _drain(self) -> List[Optional[dict]]:
        # The `self.training_env` READ is inside the guard deliberately: SB3's property ASSERTS
        # when `model.get_env()` is None rather than returning it, so a callback that reads it
        # outside a try can raise from the accessor itself — and a diagnostic must never be what
        # takes down a run.
        try:
            env = self.training_env
            if env is None or not hasattr(env, "env_method"):
                return []
            return list(env.env_method("drain_reward_terms"))
        except Exception:
            return []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        merged = merge_drained(self._drain())
        for key, val in reward_term_metrics(merged, self.term_class).items():
            self.logger.record(f"reward/{key}", val)
