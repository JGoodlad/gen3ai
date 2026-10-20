"""``gen3_t2_reserved_after_freeze_v1`` — the ONE implementation of the reserved-byte growth policy that
three lifecycle checks share: T2's ``InferenceService._frozen_guard`` (which MEASURES the growth from
``torch.cuda.memory_stats``), and the rust-rollout collector's and the rust-eval executor's
``check_lifecycle`` (which read T2's cumulative counter through `ReservedGrowthWatch`).

Why it exists: under ``expandable_segments:True`` (the launcher's setting for every child) the
``segment.all.allocated`` counter reads 0 forever, so the segment check is blind; the cumulative
``reserved_bytes.all.allocated`` counter still sees an expandable segment's newly mapped pages.
A growth is COUNTED and WARNED (at growths 1, 2, 4, 8, ...), never fatal while
`RESERVED_GROWTH_POISONS` is off. Pure Python (no torch import), so the downstream checks stay light.

``gen3_t2_stream_attribution_v1`` (`StreamAttribution`): the counters above are PROCESS-WIDE, so with X14's
overlap on (T2 flushing on the collector's worker stream while the learner runs ``train()`` on the default stream)
a learner allocation inside a T2 window was booked as T2's (core review 2026-10-20 C-1 / X14 finding 6): a learner
SEGMENT was a FATAL ``cuda_segments_after_freeze``, a learner reservation a counted T2 growth. The guard now books to
T2 only what grew on T2's OWN streams (its lanes + the calling thread's current stream), read from the allocator's
per-segment ``stream`` field (``torch.cuda.memory_snapshot()``); the rest of a window's growth is counted as
FOREIGN, never fatal. The caching allocator never lends a block across streams (a block, and an expandable segment,
belong to the stream they were allocated on), so a growth on a stream T2 does not run on cannot be T2's.
"""
from __future__ import annotations

import sys
from typing import (
    Any, Callable, Dict, FrozenSet, Iterable, Mapping, MutableMapping, NamedTuple, Optional, Tuple,
)

#: The switch, for ALL three sites. Off: a growth is counted and warned. On: the caller fails
#: (T2 poisons, the collector / executor raise their `LifecycleViolation`). Arm it only after
#: ``service_cuda_test.py``'s expandable test passes on a leased card — it proves a healthy flush
#: grows the reservation by 0 bytes on every backend (else arming would stop a healthy run).
RESERVED_GROWTH_POISONS = False

BYTES_KEY = "cuda_reserved_bytes_after_freeze"
GROWTHS_KEY = "cuda_reserved_growths_after_freeze"


def new_mapping_floor(alloc_bytes: int, reusable_bytes: int) -> int:
    """The least a ``reserved_bytes.all.allocated`` counter MUST grow when ``alloc_bytes`` is requested
    while ``reusable_bytes`` (``memory_reserved() - memory_allocated()``, read just before the request) is
    already mapped and free.

    Why not ``alloc_bytes``: under ``expandable_segments`` the allocator first reuses mapped-but-free
    pages (it extends the segment's free tail, mapping only the shortfall), so a 256 MiB request read a
    growth of 251,658,240 B = 240 MiB on the 2026-10-19 battery -- the other 16 MiB was already mapped.
    Everything beyond the reusable bytes MUST be newly mapped, so the floor is ``alloc - reusable``
    (never below 0): deterministic whatever slack the warm-up left, and still > 0 whenever the request
    is larger than the slack, i.e. the counter must see a new mapping the segment counter cannot."""
    return max(0, int(alloc_bytes) - max(0, int(reusable_bytes)))


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


# ---------------------------------------------------------------- gen3_t2_stream_attribution_v1
#: Counters `StreamAttribution` keeps in the service's counter dict (all in ``InferenceService.stats()``).
#: FOREIGN_*: a window's growth on streams that are not T2's (the learner's, under overlap) — observed, never fatal.
FOREIGN_BYTES_KEY = "cuda_reserved_bytes_foreign_in_window"
FOREIGN_SEGMENTS_KEY = "cuda_segments_foreign_in_window"
#: How many allocator snapshots the attribution took (it takes one only when a counter moved).
SNAPSHOTS_KEY = "cuda_attribution_snapshots"
#: Windows whose growth was booked process-wide (to T2) because a snapshot could not be read — today's rule.
FALLBACKS_KEY = "cuda_attribution_fallbacks"
#: Windows that grew AND released (a concurrent ``empty_cache`` / an OOM retry's cache release): the per-stream
#: growth is NET, so T2's bytes in such a window are a LOWER bound.
RELEASED_KEY = "cuda_attribution_released_in_window"
ATTRIBUTION_KEYS = (FOREIGN_BYTES_KEY, FOREIGN_SEGMENTS_KEY, SNAPSHOTS_KEY, FALLBACKS_KEY, RELEASED_KEY)


class Marks(NamedTuple):
    """Three CUMULATIVE process-wide ``torch.cuda.memory_stats`` counters, read in ONE call: ``segment.all.allocated``
    (``cudaMalloc`` segments; 0 forever under ``expandable_segments``), ``reserved_bytes.all.allocated`` (every byte
    ever added to the reservation, expandable pages included) and ``reserved_bytes.all.freed`` (every byte ever
    released). Equal marks at two instants mean NO segment was mapped or unmapped anywhere in between."""

    segments: int
    reserved: int
    released: int


#: Per stream (the ``cudaStream_t`` as an int, ``torch.cuda.Stream.cuda_stream``; 0 = the legacy default stream):
#: (mapped bytes, the addresses of its NON-expandable segments).
Ledger = Dict[int, Tuple[int, FrozenSet[int]]]
_EMPTY: Tuple[int, FrozenSet[int]] = (0, frozenset())


def stream_ledger(segments: Iterable[Mapping[str, Any]]) -> Ledger:
    """Fold ``torch.cuda.memory_snapshot()``'s segments (already filtered to one device) into a `Ledger`. An expandable
    segment is reported once per contiguous MAPPED range, so summing ``total_size`` per stream is that stream's
    reservation under either allocator mode; only a non-expandable segment's address names a ``cudaMalloc``."""
    acc: Dict[int, Tuple[int, set]] = {}
    for seg in segments:
        s = int(seg["stream"])
        n, addrs = acc.setdefault(s, (0, set()))
        if not seg.get("is_expandable", False):
            addrs.add(int(seg["address"]))
        acc[s] = (n + int(seg["total_size"]), addrs)
    return {s: (n, frozenset(a)) for s, (n, a) in acc.items()}


class Attributed(NamedTuple):
    own_bytes: int          # the window's growth on T2's streams
    own_segments: int       # new non-expandable segments on T2's streams
    foreign_bytes: int      # the rest of the process-wide growth
    foreign_segments: int


class StreamAttribution:
    """Split a T2 window's process-wide growth into T2's OWN (on ``streams``) and FOREIGN.

    Cost: the window's two `Marks` reads are the guard's own ``memory_stats`` calls; a snapshot is taken only when a
    mark MOVED — at ``enter`` when anything was mapped or released since the last snapshot (the baseline must be the
    window's start), at ``exit`` when the window grew. A steady state where nothing grows anywhere takes none.

    Why the baseline is exact: a stream's segments change only by an allocation ON that stream or by a release.
    ``enter`` re-reads the ledger whenever the marks differ from the last snapshot's, so the baseline is the ledger at
    the window's start for every stream; what grew on T2's streams by ``exit`` was allocated inside the window by the
    only code that runs on them. A snapshot that raises books the whole window to T2 (today's rule; `FALLBACKS_KEY`)."""

    def __init__(self, snapshot: Callable[[], Iterable[Mapping[str, Any]]],
                 counters: MutableMapping[str, int]):
        self._snapshot = snapshot
        self.counters = counters
        for k in ATTRIBUTION_KEYS:
            counters.setdefault(k, 0)
        self._base: Optional[Ledger] = None
        self._base_marks: Optional[Marks] = None

    def _take(self) -> Optional[Ledger]:
        self.counters[SNAPSHOTS_KEY] += 1
        try:
            return stream_ledger(self._snapshot())
        except Exception:               # never let the instrument kill a flush: book process-wide instead
            return None

    def enter(self, m0: Marks) -> None:
        if self._base is None or self._base_marks != m0:
            self._base, self._base_marks = self._take(), m0

    def exit(self, m0: Marks, m1: Marks, streams: FrozenSet[int]) -> Attributed:
        dr, ds = m1.reserved - m0.reserved, m1.segments - m0.segments
        if dr <= 0 and ds <= 0:
            return Attributed(0, 0, 0, 0)          # nothing mapped anywhere; a release re-reads at the next enter
        base, after = self._base, self._take()
        self._base, self._base_marks = after, m1
        if base is None or after is None:
            self.counters[FALLBACKS_KEY] += 1
            return Attributed(max(dr, 0), max(ds, 0), 0, 0)
        if m1.released != m0.released:
            self.counters[RELEASED_KEY] += 1
        own_dr = own_ds = 0
        for s in streams:
            (b0, a0), (b1, a1) = base.get(s, _EMPTY), after.get(s, _EMPTY)
            own_dr += max(0, b1 - b0)
            own_ds += len(a1 - a0)
        out = Attributed(own_dr, own_ds, max(0, dr - own_dr), max(0, ds - own_ds))
        self.counters[FOREIGN_BYTES_KEY] += out.foreign_bytes
        self.counters[FOREIGN_SEGMENTS_KEY] += out.foreign_segments
        return out
