"""WHEN an update fires (M5 Lane G; order constraint 6 of ``program_rust_core.md``).

* :class:`SampleTrigger` — the OWNER'S COLLECTOR (2026-09-29): the update fires once the buffer holds
  at least ``target`` rows of COMPLETED games, and consumes exactly ``target`` of them. ``target`` is
  a DECLARED parameter inside a declared band ``[lo, hi]``, a whole multiple of ``quantum`` = lcm(the
  learner's micro-batch, ``n_envs``): no ragged micro-batch ever reaches the one compiled learner graph,
  and the buffer keeps its ``[n_steps, n_envs]`` shape. (A target that is not also a multiple of
  micro-batch × accumulation K takes a smaller last optimizer step each epoch — the recipe review's
  "ragged step", PROPOSED there as Stage 0.2 and not imposed here; ``ragged_accumulation`` names it.) ``hi`` sizes the row
  arena at startup — nothing is allocated when the target moves.

  THE ADAPTIVE-BATCH HOOK: ``set_target(rows)`` is the one entry a controller calls between updates
  (the adaptive accumulation count K at a fixed compiled micro-batch; the SIZING study, order
  constraint 5, chooses the band). A value outside the band or off the quantum is REFUSED by name,
  never rounded silently; an accepted move is COUNTED (``moves``) and logged by the collector.

* :class:`WindowTrigger` — TODAY'S SCHEDULE, kept for the rollout-level parity gate: the update fires
  when every env holds ``n_steps`` rows (and the bootstrap row after them).

Neither trigger drops a row: rows beyond an update carry to the next one (``FillReport.carry_rows``).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


class TriggerError(ValueError):
    """A trigger declaration or a target move the declared band does not admit."""


@dataclass
class SampleTrigger:
    target: int
    quantum: int
    lo: int
    hi: int
    moves: int = 0
    history: List[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.quantum < 1:
            raise TriggerError(f"SampleTrigger: quantum must be >= 1, got {self.quantum}")
        for name in ("target", "lo", "hi"):
            v = int(getattr(self, name))
            if v < self.quantum or v % self.quantum:
                raise TriggerError(f"SampleTrigger: {name} {v} must be a positive multiple of the quantum "
                                   f"{self.quantum} (lcm of the micro-batch and n_envs)")
        if not self.lo <= self.target <= self.hi:
            raise TriggerError(f"SampleTrigger: target {self.target} outside the band [{self.lo}, {self.hi}]")
        self.history.append(int(self.target))

    @property
    def mode(self) -> str:
        return "complete_game"

    def ready(self, completed_rows: int) -> bool:
        return int(completed_rows) >= self.target

    def take(self) -> int:
        return int(self.target)

    def set_target(self, rows: int) -> int:
        """The adaptive-batch controller's hook (module docs). Returns the accepted target."""
        rows = int(rows)
        if rows % self.quantum or not self.lo <= rows <= self.hi:
            raise TriggerError(f"set_target({rows}): must be a multiple of {self.quantum} inside the declared "
                               f"band [{self.lo}, {self.hi}] — the band sizes the arena at startup")
        if rows != self.target:
            self.moves += 1
            self.target = rows
            self.history.append(rows)
        return rows

    def describe(self) -> str:
        return (f"complete-game buffer, update at >= {self.target:,} completed-game rows "
                f"(band [{self.lo:,}, {self.hi:,}], quantum {self.quantum:,})")


@dataclass
class WindowTrigger:
    n_steps: int

    def __post_init__(self) -> None:
        if self.n_steps < 1:
            raise TriggerError("WindowTrigger: n_steps must be >= 1")

    @property
    def mode(self) -> str:
        return "window"

    def describe(self) -> str:
        return f"window of {self.n_steps} decisions per env (today's schedule; the parity tool)"


def ragged_accumulation(target: int, micro_batch: int, accum: int) -> bool:
    """True when ``target`` rows take a smaller last optimizer step per epoch (the recipe review's
    §3.3 finding: 98,304 = 1.5 x 65,536 at the live shape). Reported, never refused."""
    return int(target) % (int(micro_batch) * max(1, int(accum))) != 0


def update_rows(mode: str, *, n_envs: int, n_steps: int, target: int = 0) -> int:
    """The rows ONE update trains on, from the trigger declaration alone (the one definition: the
    trigger below and `compile_trainer.check_shape_stability`'s callers both use it). ``window``:
    ``n_steps * n_envs``. ``complete_game``: ``target``, which defaults (0) to ``n_steps * n_envs``
    (today's rollout size, so the switch changes WHEN rows are trained on, not how many) — the
    recipe sets the target independently of both, so it is NOT their product in general."""
    if mode == "window":
        return int(n_steps) * int(n_envs)
    if mode != "complete_game":
        raise TriggerError(f"unknown rollout trigger {mode!r}")
    return int(target) or int(n_steps) * int(n_envs)


def trigger_for(mode: str, *, n_envs: int, n_steps: int, micro_batch: int, target: int = 0,
                band_lo: int = 0, band_hi: int = 0):
    """The declared trigger for a run. ``complete_game``: ``target`` defaults to ``n_steps * n_envs``
    (`update_rows`); the band defaults to the target alone; the quantum is lcm(micro_batch,
    n_envs)."""
    from math import gcd

    if mode == "window":
        return WindowTrigger(int(n_steps))
    t = update_rows(mode, n_envs=n_envs, n_steps=n_steps, target=target)     # refuses an unknown mode
    quantum = int(micro_batch) * int(n_envs) // gcd(int(micro_batch), int(n_envs))
    return SampleTrigger(target=t, quantum=quantum, lo=int(band_lo) or t, hi=int(band_hi) or t)
