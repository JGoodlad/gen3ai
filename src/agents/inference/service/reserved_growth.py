"""``gen3_t2_reserved_after_freeze_v1`` — the ONE implementation of the reserved-byte growth policy that
three lifecycle checks share: T2's ``InferenceService._frozen_guard`` (which MEASURES the growth from
``torch.cuda.memory_stats``), and the rust-rollout collector's and the rust-eval executor's
``check_lifecycle`` (which read T2's cumulative counter through `ReservedGrowthWatch`).

Why it exists: under ``expandable_segments:True`` (the launcher's setting for every child) the
``segment.all.allocated`` counter reads 0 forever, so the segment check is blind; the cumulative
``reserved_bytes.all.allocated`` counter still sees an expandable segment's newly mapped pages.
A growth is COUNTED and WARNED (at growths 1, 2, 4, 8, ...), never fatal while
`RESERVED_GROWTH_POISONS` is off. Pure Python (no torch import), so the downstream checks stay light.
"""
from __future__ import annotations

import sys
from typing import Dict, Mapping, MutableMapping, Optional

#: The switch, for ALL three sites. Off: a growth is counted and warned. On: the caller fails
#: (T2 poisons, the collector / executor raise their `LifecycleViolation`). Arm it only after
#: ``service_cuda_test.py``'s expandable test passes on a leased card — it proves a healthy flush
#: grows the reservation by 0 bytes on every backend (else arming would stop a healthy run).
RESERVED_GROWTH_POISONS = False

BYTES_KEY = "cuda_reserved_bytes_after_freeze"
GROWTHS_KEY = "cuda_reserved_growths_after_freeze"


def note_growth(counters: MutableMapping[str, int], where: str, dr: int) -> Optional[str]:
    """Count a reservation growth of ``dr`` bytes into ``counters`` (`BYTES_KEY`, `GROWTHS_KEY`).

    Returns the violation text when `RESERVED_GROWTH_POISONS` is armed (the caller fails, nothing is
    printed); otherwise warns on stderr at the 1st, 2nd, 4th, 8th, ... growth and returns ``None``.
    ``dr <= 0`` is no growth: nothing is counted."""
    if dr <= 0:
        return None
    counters[BYTES_KEY] = int(counters.get(BYTES_KEY, 0)) + int(dr)
    counters[GROWTHS_KEY] = n = int(counters.get(GROWTHS_KEY, 0)) + 1
    if RESERVED_GROWTH_POISONS:
        return (f"{where} grew the CUDA reservation by {dr} bytes after the freeze "
                f"(the declared lifecycle allows none)")
    if n & (n - 1) == 0:                              # the 1st, 2nd, 4th, 8th, ... growth
        print(f"[T2] WARN {where} grew the CUDA reservation by {dr / 2**20:.1f} MiB after the "
              f"freeze (growth #{n}, {counters[BYTES_KEY] / 2**20:.1f} MiB in all; counted, not "
              f"fatal: gen3_t2_reserved_after_freeze_v1)", file=sys.stderr, flush=True)
    return None


class ReservedGrowthWatch:
    """A downstream lifecycle check's view of T2's CUMULATIVE `BYTES_KEY` counter: each `check` sees
    the bytes T2 counted since the previous check and passes them to `note_growth` (so a check
    that saw growth is one "growth" here, warned on the same 1, 2, 4, 8 spacing)."""

    def __init__(self, where: str):
        self.where = where
        self.seen = 0
        self.counters: Dict[str, int] = {BYTES_KEY: 0, GROWTHS_KEY: 0}

    def check(self, svc_counters: Mapping[str, int]) -> Optional[str]:
        now = int(svc_counters.get(BYTES_KEY, 0))
        dr, self.seen = now - self.seen, max(self.seen, now)
        return note_growth(self.counters, self.where, dr)
