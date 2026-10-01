"""An eval cycle's logger DUMP must not take the last update's scalars with it (`gen3_eval_dump_isolation_v1`).

THE DEFECT (`designs/endstate/design_own_ppo_loop.md` §2.1, ledger 2026-10-01). The PPO loop dumps
the logger BEFORE each update, so update k's `train/*` scalars wait in `logger.name_to_value` until the
next iteration's dump — and the KL->LR controller, RankTripwire, DistillStop, the DistillAnchor dual
and the launcher's metrics pipe all READ them from there at the next `on_rollout_end`. Both eval
callbacks publish a finished cycle with `logger.dump(step)` from inside `_on_step` — mid-rollout, at
the eval's SNAPSHOT step. sb3's `dump` writes and CLEARS everything pending, so on every eval cycle the
previous update's `train/*` went out at the (earlier) eval step and the controller found nothing: one
KL reading in 20 lost on every run with a live controller (N0: 37 of 739 updates). Both env cores:
the Python core collects in a later `_on_step`, the Rust core's blocking in-process cycle in the
launching one — both through `_collect_pending`.

THE FIX. `isolated_dump` wraps a method that records and dumps its OWN scalars: everything pending
when it starts is held aside, the method records and dumps exactly what it wrote, and the held values
are put back — so the eval series is unchanged (same keys, same step) and the update's scalars reach
their own dump and every reader of the bus. A REGIME BOUNDARY for runs with a live controller.
"""
from __future__ import annotations

import functools
from typing import Any, Callable, Dict, Tuple, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

#: Set on every method `isolated_dump` wraps — the structural pin reads it.
ISOLATED_ATTR = "_gen3_isolated_dump"

Held = Tuple[Dict[str, Any], Dict[str, int], Dict[str, Any]]


def hold(logger: Any) -> Held:
    """Take everything pending off `logger` (sb3's `Logger`: value, mean-count and exclusion per key)."""
    held = (dict(logger.name_to_value), dict(logger.name_to_count), dict(logger.name_to_excluded))
    logger.name_to_value.clear()
    logger.name_to_count.clear()
    logger.name_to_excluded.clear()
    return held


def restore(logger: Any, held: Held) -> None:
    """Put the held values back. A key the scope itself left pending (recorded, not yet dumped) keeps the
    held value — the held one is the update's, which is what the next dump must carry."""
    values, counts, excluded = held
    logger.name_to_value.update(values)
    logger.name_to_count.update(counts)
    logger.name_to_excluded.update(excluded)


def isolated_dump(fn: F) -> F:
    """Method decorator for a callback (`self.logger`) whose body records + dumps its own scalars."""

    @functools.wraps(fn)
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        logger = self.logger
        held = hold(logger)
        try:
            return fn(self, *args, **kwargs)
        finally:
            restore(logger, held)

    setattr(wrapper, ISOLATED_ATTR, True)
    return wrapper  # type: ignore[return-value]
