"""The launcher's INHERITED-NICE PREFLIGHT: a real GPU training launch from a de-prioritised shell is refused.

THE INCIDENT (2026-11-10). A waiter armed under the agent-CPU prefix (``nice -n 19``) would have passed nice 19 down to
every GPU training launch it fired. ``run._apply_nice`` only moves the launcher's own nice TOWARD ``DEFAULT_NICE`` (10),
and an unprivileged process cannot LOWER its nice, so those runs would have trained at nice 19 beside agent work that
was meant to yield to them. It was caught by hand. The launcher can read the inherited nice from the OS alone, so it
refuses here, naming the number and the fix.

Scope: only a REAL training launch, i.e. an explicit ``--device cuda`` / ``cuda:N`` without ``--debug``. ``--dry-run``
reports the same verdict as a refusal line (the waiters' dry-run checks catch it before the real launch). A cpu / auto
launch, a ``--debug`` smoke and an agent's own CPU work are never refused. The ceiling is ``max(DEFAULT_NICE, --nice)``:
a typed ``--nice 19`` is a deliberate request and its own inherited 19 is allowed.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

DEFAULT_NICE = 10   # asserted equal to ``run.DEFAULT_NICE`` by nice_gate_test (this module must stay import-light)


def inherited_nice() -> Optional[int]:
    """This process's current nice, read-only (``getpriority``), or None where the OS has no such call."""
    try:
        return int(os.getpriority(os.PRIO_PROCESS, 0))
    except (AttributeError, OSError):
        return None


@dataclass(frozen=True)
class NiceVerdict:
    device: str
    debug: bool
    inherited: Optional[int]
    ceiling: int
    applies: bool          # a real GPU training launch
    refused: bool

    @property
    def reason(self) -> str:
        return (f"this shell runs at nice {self.inherited}, which a GPU training launch would inherit: the launcher can "
                f"only raise its nice toward {DEFAULT_NICE}, never lower it, so the run and its ~940 processes would "
                f"train at nice {self.inherited}, starved by agent work. Fix: launch from a nice-0 (or <= "
                f"{self.ceiling}) shell, not one under `nice -n {self.inherited}` or the agent-CPU prefix")

    def lines(self) -> List[str]:
        if self.refused:
            return [f"inherited nice : ✗ REFUSED (FATAL_CONFIG): {self.reason}"]
        if not self.applies:
            return [f"inherited nice : n/a ({self.inherited}; not a real GPU training launch)"]
        return [f"inherited nice : ✓ {self.inherited} (<= {self.ceiling})"]


def verdict_for_launch(device: Optional[str], *, debug: bool, nice_arg: int,
                       inherited: Optional[int] = None) -> NiceVerdict:
    """The verdict for a launch with ``--device device`` / ``--debug`` / ``--nice nice_arg``.

    ``inherited`` defaults to the OS's reading (``inherited_nice()``); tests pass it."""
    dev = str(device or "auto").strip().lower()
    cur = inherited_nice() if inherited is None else inherited
    ceiling = max(DEFAULT_NICE, int(nice_arg))
    applies = dev.startswith("cuda") and not debug
    return NiceVerdict(dev, bool(debug), cur, ceiling, applies,
                       refused=applies and cur is not None and cur > ceiling)
