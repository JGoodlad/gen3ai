"""``gen3_t2_guard_attribution_v1`` — T2's frozen guard books a dynamo compile or a matmul-precision change to T2 only
when T2's OWN thread made it. The sibling of ``reserved_growth.StreamAttribution`` (gen3_t2_stream_attribution_v1),
which does the same for CUDA memory; the guard's three process-wide reads were the same defect.

The defect (core review 2026-10-20, X14 finding 6): with X14's overlap on, T2 flushes on the collector's worker thread
while the learner runs on the main thread. The guard diffed two PROCESS-WIDE values around the flush:

* ``counters['stats']['unique_graphs']`` (dynamo's graph count): a learner recompile inside the window bumped it and
  was a FATAL ``compiles_after_freeze`` on a healthy run;
* ``torch.get_float32_matmul_precision()``: a precision change by any thread was a FATAL on T2.

The fix attributes both by the THREAD that acted, the thread that entered the guard being T2's:

* **Graphs.** ``compile_control.thread_graph_marks()`` is a per-thread tally of the ``unique_graphs`` increments (dynamo
  bumps the counter on the compiling thread). `attribute_compiles` books only the guard thread's share to T2; the
  rest is FOREIGN (``compiles_foreign_in_window``), never fatal. A window the tally cannot split (the instrument is
  unavailable, was rebuilt mid-window, or the counter moved without the tally) is booked process-wide, the old rule, and
  counted (``compile_attribution_fallbacks``).
* **Matmul precision.** torch's precision is one process global, so the SETTER is recorded: `PrecisionLedger` wraps
  ``torch.set_float32_matmul_precision`` (idempotent, installed at T2's startup) and keeps who set which value.
  `judge_precision` fails the guard when the precision left the startup value and T2's own thread did it (during the
  window, even if it set it back, or in the state it carried in), is FOREIGN when only other threads did (counted and
  warned, not fatal: a graph-backend flush replays kernels captured at startup, an eager one is the learner's to
  answer for), and falls back to the old rule (fatal) when the setter was not seen (``torch.backends.cuda.matmul.
  allow_tf32``, a C++ setter, a ``from torch import set_float32_matmul_precision`` bound before the wrap), counted
  (``precision_attribution_fallbacks``).

Production arms do not run overlap today: serial runs have ONE thread, so every compile and every change is T2's, as
before. Pure Python beyond the one torch hook.
"""
from __future__ import annotations

import collections
import functools
import sys
import threading
from typing import Any, Deque, List, MutableMapping, NamedTuple, Optional, Sequence, Set

FOREIGN_COMPILES_KEY = "compiles_foreign_in_window"
COMPILE_FALLBACKS_KEY = "compile_attribution_fallbacks"
FOREIGN_PRECISION_KEY = "precision_foreign_in_window"
PRECISION_FALLBACKS_KEY = "precision_attribution_fallbacks"
#: Counters the service keeps (all in ``InferenceService.stats()``).
GUARD_ATTRIBUTION_KEYS = (FOREIGN_COMPILES_KEY, COMPILE_FALLBACKS_KEY, FOREIGN_PRECISION_KEY, PRECISION_FALLBACKS_KEY)


def note_foreign(counters: MutableMapping[str, int], key: str, text: str) -> None:
    """Count one foreign event under ``key``; warn on stderr at the 1st, 2nd, 4th, 8th, ... (never fatal)."""
    counters[key] = n = int(counters.get(key, 0)) + 1
    if n & (n - 1) == 0:
        print(f"[T2] WARN {text} (foreign #{n}: counted, not fatal: gen3_t2_guard_attribution_v1)",
              file=sys.stderr, flush=True)


# ---------------------------------------------------------------- dynamo graphs
class CompileShare(NamedTuple):
    own: int            # graphs compiled on T2's thread (plus any the tally could not place): the guard's fatal count
    foreign: int        # graphs compiled on other threads inside the window
    fallback: bool      # the window could not be split: `own` is the process-wide count (the old rule)


def attribute_compiles(before: Any, after: Any, tid: int, process_wide: int) -> CompileShare:
    """Split a window's dynamo graph compiles between thread ``tid`` (T2's) and the rest.

    ``before`` / ``after`` are ``compile_control.GraphMarks`` (or ``None``: the instrument was unavailable) read at the
    window's two ends; ``process_wide`` is the plain ``unique_graphs`` diff, used only when they cannot be trusted.
    ``total - sum(by_thread)`` is an increment that bypassed the tally: it is booked to T2 (conservative) and flagged."""
    if before is None or after is None or before.generation != after.generation:
        return CompileShare(max(process_wide, 0), 0, process_wide > 0)
    total = after.total - before.total
    if total <= 0:
        return CompileShare(0, 0, False)
    own = max(0, after.by_thread.get(tid, 0) - before.by_thread.get(tid, 0))
    attributed = sum(after.by_thread.values()) - sum(before.by_thread.values())
    unplaced = max(0, total - attributed)
    return CompileShare(own + unplaced, max(0, attributed - own), unplaced > 0)


# ---------------------------------------------------------------- fp32 matmul precision
class PrecisionEvent(NamedTuple):
    seq: int
    tid: int            # the thread whose ``torch.set_float32_matmul_precision`` call CHANGED the value
    value: str          # the value it left


class PrecisionRead(NamedTuple):
    value: str          # ``torch.get_float32_matmul_precision()``
    seq: int            # the ledger's event count at this read
    owner: Optional[int]    # the thread that last set ``value`` (None: the last recorded set was another value, or none)


OK, FOREIGN, OWN, UNKNOWN = "ok", "foreign", "own", "unknown"


class PrecisionVerdict(NamedTuple):
    kind: str           # OK | FOREIGN | OWN | UNKNOWN (UNKNOWN is fatal like OWN: the old rule)
    detail: str


def judge_precision(baseline: str, before: PrecisionRead, after: PrecisionRead,
                    events: Optional[Sequence[PrecisionEvent]], own_tid: int) -> PrecisionVerdict:
    """Who is answerable for the precision having left ``baseline`` in a window (``before`` .. ``after``)?

    The causes: the state carried IN (``before``) when it is not the baseline, every set INSIDE the window to a value
    that is not the baseline (a set-and-restore by T2's own thread still ran its flush under another precision), and
    the state left OUT (``after``). ``events`` are the window's recorded sets (``None``: the ledger dropped some)."""
    causes: Set[Optional[int]] = set()
    values: Set[str] = set()
    if before.value != baseline:
        causes.add(before.owner)
        values.add(before.value)
    if events is None:
        causes.add(None)
    else:
        for e in events:
            if e.value != baseline:
                causes.add(e.tid)
                values.add(e.value)
    if after.value != baseline:
        causes.add(after.owner)
        values.add(after.value)
    if not causes:
        return PrecisionVerdict(OK, "")
    change = f"fp32 matmul precision changed after the freeze ({baseline!r} -> {', '.join(sorted(map(repr, values))) or '?'})"
    if own_tid in causes:
        return PrecisionVerdict(OWN, f"{change}, by T2's own thread")
    if None in causes:
        return PrecisionVerdict(UNKNOWN, f"{change}, by a setter the ledger did not see")
    return PrecisionVerdict(FOREIGN, f"{change}, by thread(s) {sorted(c for c in causes if c is not None)}, not T2's")


_HOOK = "_gen3_precision_ledger"


class PrecisionLedger:
    """Who set ``torch.set_float32_matmul_precision`` to what. `install` wraps the setter (once per process); `read`
    returns the current value with its recorded setter. The wrapper and every read share one lock, so a read never
    sees the value changed without its event."""

    def __init__(self, maxlen: int = 256):
        self._lock = threading.Lock()
        self._events: Deque[PrecisionEvent] = collections.deque(maxlen=maxlen)
        self._seq = 0

    def install(self) -> bool:
        """Wrap ``torch.set_float32_matmul_precision`` to record its caller (idempotent). Returns False when the
        setter is not a plain callable we can wrap (the verdict then falls back to the old rule)."""
        import torch

        with self._lock:
            orig = getattr(torch, "set_float32_matmul_precision", None)
            if not callable(orig):
                return False
            hooked = getattr(orig, _HOOK, None)
            if hooked is not None:                                 # already wrapped: by this ledger (done) or another (never stack)
                return bool(hooked is self)
            ledger = self

            @functools.wraps(orig)
            def set_float32_matmul_precision(precision: Any) -> None:
                with ledger._lock:
                    before = torch.get_float32_matmul_precision()
                    orig(precision)
                    now = torch.get_float32_matmul_precision()     # the ACTUAL value: torch WARNS and keeps the old one for a bad name
                    if now != before:                              # a no-op set changes nothing, so it owns nothing
                        ledger._seq += 1
                        ledger._events.append(PrecisionEvent(ledger._seq, threading.get_ident(), now))

            setattr(set_float32_matmul_precision, _HOOK, self)
            torch.set_float32_matmul_precision = set_float32_matmul_precision
            return True

    def read(self) -> PrecisionRead:
        import torch

        with self._lock:
            value = torch.get_float32_matmul_precision()
            last = self._events[-1] if self._events else None
            return PrecisionRead(value, self._seq, last.tid if last is not None and last.value == value else None)

    def events_after(self, seq: int) -> Optional[List[PrecisionEvent]]:
        """The sets recorded after event ``seq``; ``None`` when the bounded log already dropped some of them."""
        with self._lock:
            if self._events and self._events[0].seq > seq + 1:
                return None
            return [e for e in self._events if e.seq > seq]

    def verdict(self, baseline: str, before: PrecisionRead, own_tid: int) -> PrecisionVerdict:
        """The window that began at ``before`` and ends now."""
        after = self.read()
        events = self.events_after(before.seq) if after.seq != before.seq else []
        return judge_precision(baseline, before, after, events, own_tid)


#: The process's one ledger (the wrapped setter is process-global, so is its record).
PRECISION = PrecisionLedger()
