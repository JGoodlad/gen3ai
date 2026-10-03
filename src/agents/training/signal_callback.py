"""`SignalMetricsCallback` — the OUTCOME-ENTROPY half of the `signal/` TensorBoard group.

The other half (advantage density, `signal/adv_*`) is read inside `train()` off the rollout
buffer; see `instrumented_ppo/signal_metrics.py` for the module docstring that explains why the
two must be read TOGETHER (the mirror paradox), and the units caveat.

**Where the outcomes come from.** The Rust collector publishes each finished game's
``info["win_outcome"]`` (1.0 win / 0.0 loss-or-tie), ``info["win_draw"]`` and ``info["opponent_class"]``
in the step's declared locals (``infos`` / ``dones``, `loop_callbacks.STEP_LOCALS`). This callback
watches the `done` infos as they stream past during collection and pushes them into rolling per-kind
windows — no extra battles, no `env_method` round trip, no env state.

ALWAYS ON: three numpy means over ≤200-element deques, once per rollout. No flag.
"""
from __future__ import annotations


from agents.training.loop_callbacks import BaseCallback

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
        # +DRAW RATE (gen3_winprob_critic_mode_v1, design §3.2 / gap B9). Counted here rather than
        # in `WinProbLabelCallback` for two reasons: this callback is UNCONDITIONAL (it is in
        # `build_callbacks`' base list, where the label callback is registered only when the
        # win-prob head is on), and it already scans every terminal `info` on both rollout paths,
        # so the count costs one branch instead of a second scan. Reset per rollout, so the tag is
        # this rollout's rate rather than a run-long average that flattens a late-onset stall.
        self._draws = 0
        self._terminals = 0

    def _consume(self, infos, dones) -> None:
        for info, done in zip(infos, dones):
            if not done or not isinstance(info, dict) or "win_outcome" not in info:
                continue
            kind = OPP_CLASS_SUFFIX.get(info.get("opponent_class"))
            self.tracker.observe(float(info["win_outcome"]) >= 0.5, kind)
            # `win_draw` is published beside `win_outcome` by the Rust collector — a draw or
            # the 250-turn timeout, i.e. `battle.won is None`. It is SCORED as a not-win by
            # decision, never dropped, and this is the tag that makes that decision's frequency a
            # fact rather than an inference. An older wrapper that predates the key is read as 0
            # draws, which is the honest reading of "this run cannot tell me".
            self._terminals += 1
            self._draws += int(float(info.get("win_draw", 0.0)) >= 0.5)

    def _on_step(self) -> bool:
        # the step's declared locals (`loop_callbacks.STEP_LOCALS`), supplied by the Rust collector
        infos = self.locals.get("infos")
        dones = self.locals.get("dones")
        if infos is not None and dones is not None:
            self._consume(infos, dones)
        return True

    def _on_rollout_end(self) -> None:
        for key, val in self.tracker.metrics().items():
            self.logger.record(f"signal/{key}", val)
        # ⚠️ `signal/draw_rate`, NOT the `train/draw_rate` the design's §3.2 proposes — and the
        # reason is worth stating, because a reader will look for the documented name. This
        # callback carries a PINNED prefix contract (`test_sync_locals_are_consumed_and_recorded_
        # under_the_signal_prefix` asserts every row it emits starts with `signal/`), and on the
        # merits `signal/` is the right group anyway: the draw rate is an OUTCOME statistic whose
        # literal siblings are `signal/outcome_win_rate` and `signal/outcome_entropy`, computed
        # from the same terminal `info` in the same loop. Emitting it under `train/` would put one
        # outcome statistic in a different group from the other two for no reason but a name.
        #
        # It is emitted ONLY when this rollout closed an episode: a rollout that finished none has
        # no rate, and a 0.0 there would read as "no draws" rather than "no data".
        if self._terminals:
            self.logger.record("signal/draw_rate", self._draws / self._terminals)
            self.logger.record("signal/n_terminals", float(self._terminals))
        self._draws = self._terminals = 0

    # `signal/outcome_entropy_rung` (the entropy of the exploiter-ladder rung being fought NOW) was emitted
    # by `ExploiterLadderCallback`, which was DELETED with the ladder (deletion pass L4); the series is
    # retired with it.
