"""THE ANCHOR LANE'S BOUNDED CONCURRENCY: up to K units in flight at once (``designs/tools/reads.md`` §3).

SCHEDULING ONLY. A unit's play is ``main.x35.kakuna.run_unit``, called exactly as before: its own short-lane
``mem_cap.sh`` job, its own 95xx port claimed with a held flock (``kakuna.claim_port``: a flock is per open file, so two
claims in this process cannot take one port), its own cell dir and log. This module decides only WHEN a unit starts. It
keeps up to K of them in flight, each on a thread that waits on its child process (a thread plays nothing), and hands
every finished unit back to the supervisor loop, which counts it, retries it and falls back to the long lane exactly as
the one-at-a-time lane did. At K = 1 the unit runs INLINE on the supervisor's own thread: the one-at-a-time lane, no
thread. Nothing a unit reads depends on another unit or on K, so the units, the seeds and the per-unit records are the
same at every K; only the wall clock moves.

K is capped at the box's heavy-slot count (``utils.heavy_slots.slot_count``): every in-flight unit is a short-lane job,
and while a training run is live or the GPU is leased each one takes a slot, so more units than slots would only queue.
"""
from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

#: How long the supervisor waits for any in-flight unit to end before it looks at its own claims again, s.
POLL_S = 60.0

Key = Tuple[str, str]


def slot_cap() -> int:
    """The most units the lane may run at once: the box's heavy-slot count (``$GEN3AI_HEAVY_SLOTS`` overrides)."""
    from utils import heavy_slots

    return heavy_slots.slot_count()


class UnitPool:
    """Up to ``k`` units in flight. ``play(state, unit)`` is the unit runner (``kakuna.run_unit``); ``ended(state,
    unit)`` is called on the SUPERVISOR's thread for each unit that returned (never for one that raised). ``log``
    reports a unit error that is not the one re-raised."""

    def __init__(self, k: int, play: Callable[[Path, Any], Any], ended: Callable[[Path, Any], None],
                 log: Callable[[str], None] = print) -> None:
        if k < 1:
            raise ValueError(f"a pool runs at least one unit, not {k}")
        self.k, self._play, self._ended, self._log = k, play, ended, log
        self._flight: Dict[Key, Tuple[Path, Any]] = {}
        self._finished: "queue.Queue[Tuple[Key, Optional[BaseException]]]" = queue.Queue()

    @staticmethod
    def key(state: Path, unit: Any) -> Key:
        return str(state), str(unit.name)

    def __len__(self) -> int:
        return len(self._flight)

    def full(self) -> bool:
        return len(self._flight) >= self.k

    def free(self) -> int:
        return self.k - len(self._flight)

    def busy(self, state: Path, unit: Any) -> bool:
        return self.key(state, unit) in self._flight

    def submit(self, state: Path, unit: Any) -> None:
        """Start ``unit`` (it must not be in flight, and a slot of the pool must be free). At K = 1 it runs to its end
        right here, and an exception it raises propagates at once, as it always did."""
        if self.busy(state, unit) or self.full():
            raise RuntimeError(f"cannot start {unit.name}: in flight or the pool of {self.k} is full")
        if self.k == 1:
            self._play(state, unit)
            self._ended(state, unit)
            return
        key = self.key(state, unit)
        self._flight[key] = (state, unit)
        threading.Thread(target=self._work, args=(key, state, unit), name=f"unit-{unit.name}", daemon=True).start()

    def _work(self, key: Key, state: Path, unit: Any) -> None:
        err: Optional[BaseException] = None
        try:
            self._play(state, unit)
        except BaseException as exc:                      # handed to the supervisor's thread, never lost
            err = exc
        self._finished.put((key, err))

    def _settle(self, items: "List[Tuple[Key, Optional[BaseException]]]") -> Optional[BaseException]:
        """Count each ended unit; the first unit exception is returned (the later ones are logged)."""
        first: Optional[BaseException] = None
        for key, err in items:
            state, unit = self._flight.pop(key)
            if err is None:
                self._ended(state, unit)
            elif first is None:
                first = err
            else:
                self._log(f"[reads] unit {unit.name} also raised {type(err).__name__}: {err}")
        return first

    def _collect(self, *, block: bool, timeout: Optional[float]) -> "List[Tuple[Key, Optional[BaseException]]]":
        items = []
        try:
            items.append(self._finished.get(block, timeout))
            while True:
                items.append(self._finished.get_nowait())
        except queue.Empty:
            pass
        return items

    def wait_one(self, timeout: float = POLL_S) -> bool:
        """Wait up to ``timeout`` s for an in-flight unit to end; count every one that has. True if any ended. A unit
        that RAISED (a preflight refusal, say) lets the other in-flight units finish first, then re-raises here."""
        items = self._collect(block=True, timeout=timeout)
        if not items:
            return False
        err = self._settle(items)
        if err is not None:
            self.close()
            raise err
        return True

    def drain(self) -> None:
        """Wait for every in-flight unit to end and count it (a lane never stops with a unit of its own in flight). The
        first unit exception is re-raised once all have ended."""
        first: Optional[BaseException] = None
        while self._flight:
            err = self._settle(self._collect(block=True, timeout=None))
            first = first or err
        if first is not None:
            raise first

    def close(self) -> None:
        """Drain, counting what ended and ignoring unit errors (the caller is already leaving by its own)."""
        while self._flight:
            self._settle(self._collect(block=True, timeout=None))

    def abandon(self) -> None:
        """Leave the in-flight units running (a KeyboardInterrupt): each child is in its own session and finishes and
        counts on resume, exactly as when the one-at-a-time lane is interrupted mid-unit."""
        self._flight.clear()
