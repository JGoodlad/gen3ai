"""WHEN an update fires (M5 Lane G; order constraint 6 of ``program_rust_core.md``).

:class:`SampleTrigger` — the OWNER'S COLLECTOR (2026-09-29): the update fires once the buffer holds
at least ``target`` rows of COMPLETED games, and consumes exactly ``target`` of them. ``target`` is
a whole multiple of ``quantum`` = lcm(the learner's micro-batch, ``n_envs``): no ragged micro-batch
ever reaches the one compiled learner graph, and the buffer keeps its ``[n_steps, n_envs]`` shape.
(A target that is not also a multiple of micro-batch × accumulation K takes a smaller last optimizer
step each epoch — the recipe review's "ragged step", PROPOSED there as Stage 0.2 and not imposed
here; ``ragged_accumulation`` names it.) ``target`` sizes the row arena at startup.

The trigger never drops a row: rows beyond an update carry to the next one
(``FillReport.carry_rows``). It is the ONLY trigger: the n_steps-per-env WINDOW schedule (the
rollout-level parity tool of the Python core's era) and the adaptive-batch ``set_target`` hook (a hook
with no controller) were DELETED with ``--rollout-trigger`` / ``--rollout-target-band``, deletion pass
P11c.
"""
from __future__ import annotations

from dataclasses import dataclass


class TriggerError(ValueError):
    """A trigger declaration the quantum does not admit."""


@dataclass
class SampleTrigger:
    target: int
    quantum: int

    def __post_init__(self) -> None:
        if self.quantum < 1:
            raise TriggerError(f"SampleTrigger: quantum must be >= 1, got {self.quantum}")
        v = int(self.target)
        if v < self.quantum or v % self.quantum:
            raise TriggerError(f"SampleTrigger: target {v} must be a positive multiple of the quantum "
                               f"{self.quantum} (lcm of the micro-batch and n_envs)")

    def ready(self, completed_rows: int) -> bool:
        return int(completed_rows) >= self.target

    def take(self) -> int:
        return int(self.target)

    def describe(self) -> str:
        return (f"complete-game buffer, update at >= {self.target:,} completed-game rows "
                f"(quantum {self.quantum:,})")


def ragged_accumulation(target: int, micro_batch: int, accum: int) -> bool:
    """True when ``target`` rows take a smaller last optimizer step per epoch (the recipe review's
    §3.3 finding: 98,304 = 1.5 x 65,536 at the live shape). Reported, never refused."""
    return int(target) % (int(micro_batch) * max(1, int(accum))) != 0


def update_rows(*, n_envs: int, n_steps: int, target: int = 0) -> int:
    """The rows ONE update trains on, from the trigger declaration alone (the one definition: the
    trigger below and `compile_trainer.check_shape_stability`'s callers both use it). ``target``
    defaults (0) to ``n_steps * n_envs`` (the rollout size ``--n-steps`` x ``--n-envs`` names) — the
    recipe sets the target independently of both, so it is NOT their product in general."""
    return int(target) or int(n_steps) * int(n_envs)


def trigger_for(*, n_envs: int, n_steps: int, micro_batch: int, target: int = 0) -> SampleTrigger:
    """The declared trigger for a run: ``target`` defaults to ``n_steps * n_envs`` (`update_rows`); the
    quantum is lcm(micro_batch, n_envs)."""
    from math import gcd

    quantum = int(micro_batch) * int(n_envs) // gcd(int(micro_batch), int(n_envs))
    return SampleTrigger(target=update_rows(n_envs=n_envs, n_steps=n_steps, target=target), quantum=quantum)
