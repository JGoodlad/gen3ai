"""Per-battle event ring buffer for root-causing the self-play stale-decision race.

OFF by default (a single bool check → zero production cost). Enable with `GEN3_RACE_TRACE=1`
on a debug run. When on, every protocol line parsed on POKE_LOOP and every decision event
(embed / serialize / assert) on the training thread is appended to a per-battle ring with a
global monotonic sequence number + the thread name. On a `StaleDecisionError`, the crashing
battle's ring is dumped into the exception message — so the crash file shows the EXACT
interleaving that advanced the battle between the snapshot and the serialize, across threads.

This is debugging infrastructure: it is intentionally checked in (gated off) so we can flip it
on for a live run, capture the sequence, and root-cause the race. Remove once that's done.
"""

from __future__ import annotations

import os
import threading
from collections import deque
from typing import Deque, Dict, Tuple

# Resolved once at import. Production never sets it → trace()/dump() are cheap no-ops.
ENABLED: bool = bool(os.environ.get("GEN3_RACE_TRACE"))

_MAXLEN = 220  # ~10-20 turns of protocol + decision events per battle
_traces: Dict[str, "Deque[Tuple[int, str, str]]"] = {}
_seq = 0
_lock = threading.Lock()


def trace(tag: str, event: str) -> None:
    """Append (global_seq, thread_name, event) to ``tag``'s ring. Cross-thread safe; the
    global seq is what lets the dump reconstruct the POKE_LOOP-vs-training-thread ordering."""
    if not ENABLED or not tag:
        return
    global _seq
    with _lock:
        _seq += 1
        seq = _seq
        dq = _traces.get(tag)
        if dq is None:
            dq = deque(maxlen=_MAXLEN)
            _traces[tag] = dq
        dq.append((seq, threading.current_thread().name, event))


def dump(tag: str) -> str:
    """Format ``tag``'s ring (oldest→newest) for inclusion in a crash message."""
    if not ENABLED:
        return ""
    dq = _traces.get(tag)
    if not dq:
        return f"\n--- RACE TRACE [{tag}]: (empty — GEN3_RACE_TRACE on but no events) ---"
    out = [f"\n--- RACE TRACE [{tag}] — last {len(dq)} events (seq | thread | event) ---"]
    for seq, thr, ev in dq:
        out.append(f"  {seq:>7} | {thr:<20.20} | {ev}")
    out.append("--- END RACE TRACE ---")
    return "\n".join(out)
