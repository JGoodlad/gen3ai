"""The DECLARED-LIFECYCLE marker (M5, `designs/endstate/program_rust_core.md`, "M5 DESIGN PRINCIPLE —
a DECLARED LIFECYCLE"): a training process acquires everything at STARTUP, then FREEZES; the steady
state acquires nothing.

Two enforcers read this marker:

* the STATIC gate `src/learner_lifecycle_gate_test.py` (routine tier, EMPTY allowlist) — it fails an
  optimizer / ``nn.Parameter`` / ``nn.Module`` construction in a training-step code path unless the
  enclosing function is decorated :func:`startup_builder` (read by NAME, so an alias is invisible)
  or is a class's ``__init__`` / ``_build`` / ``_setup_model``;
* the RUNTIME freeze guard — the only thing that proves a builder actually RAN before the freeze. The
  decorator is a declaration, not a proof: calling a ``@startup_builder`` from a training step passes
  the static gate and is the runtime guard's to catch.
"""
from __future__ import annotations

from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable[..., object])


def startup_builder(fn: F) -> F:
    """Marks ``fn`` as a DECLARED STARTUP BUILDER: it runs before the learner freezes (model build,
    `_setup_model`, the trainer's startup path), so constructing an optimizer / Parameter / Module
    inside it is legal. The static gate `src/learner_lifecycle_gate_test.py` reads this decorator by
    NAME; the runtime freeze guard is what proves it actually ran before the freeze."""
    fn.__gen3_startup_builder__ = True  # type: ignore[attr-defined]
    return fn
