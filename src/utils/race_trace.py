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
# Bound the number of per-battle rings retained. A long GEN3_RACE_TRACE run plays thousands of
# battles (one new tag each), so without a cap _traces grows unboundedly. We only ever dump the
# most-recently-active battle on a crash, so retaining a few hundred recent tags is plenty; the
# oldest (finished, chronologically earliest) tag is evicted past this.
_MAX_TAGS = 256
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
            if len(_traces) >= _MAX_TAGS:
                # dict preserves insertion order → first key is the oldest battle (finished).
                _traces.pop(next(iter(_traces)))
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


def dump_recent(n: int = 2) -> str:
    """Format the ``n`` most-recently-active battles' rings, newest-active first. Used on a
    crash path that doesn't know the wedged battle's tag (e.g. ``race_get``'s silent-stall guard,
    which lives on the queue, not the player). A SubprocVecEnv worker hosts one battle at a time,
    so the most-recently-active tag IS the wedged battle; ``n>1`` adds a little prior context."""
    if not ENABLED:
        return ""
    with _lock:
        # Rank tags by their newest seq (last appended). Don't call dump() under the lock —
        # threading.Lock is non-reentrant and dump() would re-acquire it.
        ranked = sorted(
            _traces.items(),
            key=lambda kv: (kv[1][-1][0] if kv[1] else 0),
            reverse=True,
        )
        tags = [t for t, _ in ranked[:n]]
    if not tags:
        return "\n--- RACE TRACE: (none — GEN3_RACE_TRACE on but no battles traced) ---"
    # Emit oldest-active first so the WEDGED (most-recently-active) battle's newest events land
    # at the very END — the launcher's per-crash file keeps only the last ~100 lines of child
    # output, so the wedge point must be last to survive that tail-capture. (launcher_child.log
    # keeps the full message, including the older battle's context.)
    return "".join(dump(t) for t in reversed(tags))
