"""THE LOOP'S DECLARED HOOKS (`gen3_declared_loop_hooks_v1`; `designs/endstate/design_own_ppo_loop.md` stage 2).

Before this module, two subsystems hooked the PPO loop by REASSIGNING the learner's bound methods as
instance attributes — `learner_lifecycle.attach` (K6's freeze guard + CUDA memory watch) and
`CompileControl.attach` (the compile sentinel's lock / guards / canary / per-update record) — and the
nesting order between them was whichever attached last. Now the loop owns its hook points and the
order is a TABLE:

    HOOK_POINTS = ("learn", "collect", "update")     # where `OwnedLoop.learn` opens a hook scope
    HOOK_OWNERS = ("learner_freeze", "compile_sentinel")   # who may hook, OUTERMOST first

A hook is an AROUND hook — a context-manager factory, entered before the phase and exited after it;
code after its `yield` runs only when the phase returned normally, exactly like the wrapper bodies it
replaces. Rules, each a typed `LoopHookError` (a `FatalConfigError`: FATAL_CONFIG, not restarted):
an owner or point not in the table; a second registration for one (owner, point); ANY registration
after the table FROZE — at `learn()`'s training start, the declared lifecycle's "nothing is acquired
after startup".

A duck-typed model with no table (a test stub, a tool driving `collect_rollouts` / `train` directly)
gets the SAME hook bodies installed as instance-attribute wrappers by `install` — the adapter the old
attach code was, kept for callers that are not an `OwnedLoop`.
"""
from __future__ import annotations

import contextlib
from typing import Any, Callable, ContextManager, Dict, Iterator, Mapping, Optional, Tuple

from main.exit_codes import FatalConfigError

#: The loop's hook points, in the order `learn()` nests them (learn > collect | update).
HOOK_POINTS: Tuple[str, ...] = ("learn", "collect", "update")
#: Who may hook the loop, OUTERMOST FIRST. The freeze guard is outermost: it freezes before the
#: sentinel's first-rollout work and checks after the sentinel's checks.
HOOK_OWNERS: Tuple[str, ...] = ("learner_freeze", "compile_sentinel")
#: The bound method each point wraps on a model with no table (the adapter path).
_METHOD = {"learn": "learn", "collect": "collect_rollouts", "update": "train"}

Factory = Callable[[], ContextManager[Any]]


class LoopHookError(FatalConfigError):
    """An undeclared, duplicate or late loop hook."""


class LoopHooks:
    """The table. One per learner, built in `_setup_model` (startup), frozen at training start."""

    def __init__(self) -> None:
        self._hooks: Dict[str, Dict[str, Factory]] = {p: {} for p in HOOK_POINTS}
        self.frozen_at: Optional[str] = None

    def register(self, owner: str, point: str, factory: Factory) -> None:
        if owner not in HOOK_OWNERS:
            raise LoopHookError(f"[LoopHooks] FATAL: undeclared hook owner {owner!r} (declared, outermost "
                                f"first: {HOOK_OWNERS}) — add it to loop_hooks.HOOK_OWNERS at its nesting place")
        if point not in HOOK_POINTS:
            raise LoopHookError(f"[LoopHooks] FATAL: unknown hook point {point!r} (want one of {HOOK_POINTS})")
        if self.frozen_at is not None:
            raise LoopHookError(f"[LoopHooks] FATAL: {owner!r} hooked {point!r} after the table froze at "
                                f"{self.frozen_at} — loop hooks are acquired at startup, never later")
        if owner in self._hooks[point]:
            raise LoopHookError(f"[LoopHooks] FATAL: {owner!r} already hooks {point!r}")
        self._hooks[point][owner] = factory

    def freeze(self, where: str) -> None:
        if self.frozen_at is None:
            self.frozen_at = where

    def owners(self, point: str) -> Tuple[str, ...]:
        """The owners hooking ``point``, outermost first."""
        return tuple(o for o in HOOK_OWNERS if o in self._hooks[point])

    @contextlib.contextmanager
    def around(self, point: str) -> Iterator[None]:
        """Enter every hook on ``point``, outermost first; exit in reverse."""
        with contextlib.ExitStack() as stack:
            for owner in self.owners(point):
                stack.enter_context(self._hooks[point][owner]())
            yield


def install(model: Any, owner: str, factories: Mapping[str, Factory]) -> None:
    """Hook ``model``'s loop as ``owner``: into its `LoopHooks` table when it has one (every
    `InstrumentedMaskablePPO`), else as instance-attribute wrappers over its bound methods (the adapter
    for a duck-typed caller — nested in call order, so attach the OUTER owner last, as before)."""
    table = getattr(model, "_loop_hooks", None)
    if isinstance(table, LoopHooks):
        for point, factory in factories.items():
            table.register(owner, point, factory)
        return
    for point, factory in factories.items():
        if point not in _METHOD:
            raise LoopHookError(f"[LoopHooks] FATAL: unknown hook point {point!r}")
        name = _METHOD[point]
        orig = getattr(model, name)

        def wrapped(*a: Any, _orig: Any = orig, _factory: Factory = factory, **k: Any) -> Any:
            with _factory():
                return _orig(*a, **k)

        setattr(model, name, wrapped)
