"""THE DEFERRED ABORT (`gen3_deferred_abort_v1`; P10 review F1): a stop signal is honoured at a SAFE POINT.

Before this module the SIGINT / SIGTERM / SIGHUP handlers ran the whole abort — the pending-scalar dump,
the checkpoint save, the eval drain, `os._exit(15)` — INSIDE the handler, i.e. at whatever bytecode
boundary the main thread happened to be on. Three defects followed from that one shape:

* **F1 (confirmed): the dump deadlocked.** TensorBoard's `_AsyncWriter` takes a NON-reentrant lock in
  both `write` and `flush`. A signal that landed while the main thread was inside a dump's `add_scalar`
  / `flush` ran the handler's own dump on the same thread, which blocked on that lock forever: no
  `final_model_interrupted.zip`, and the launcher SIGKILLed the child at the end of its grace, losing
  everything since the last periodic checkpoint.
* **A torn checkpoint (plausible):** `model.save` at an arbitrary boundary could run in the middle of an
  update (between `optimizer.step`s, mid gradient accumulation) — a checkpoint of no state the learner
  was ever in, which the launcher then resumed from.
* **`print("[ABORT]")` in the handler** could raise "reentrant call" if stdout was mid-write.

**The shape now.** A handler only RECORDS the request (`request`: plain attribute stores, one
`os.write` to stderr, one byte to a wake pipe — no Python lock, no buffered stream). The main thread
runs the abort (`abort`: dump + save + drain + exit 15) at the next SAFE POINT — the run's
`GracefulRestartCallback` calls `safe_point()` at every callback event the loop fires
(`training_start`, `rollout_start`, every collector `step`, `rollout_end`, `training_end`), none of
which runs inside an update or inside a logger dump. A signal that lands mid-update therefore lets the
update FINISH (it is bounded: `train_ms` 41-48 s at the production recipe, 2026-10-02 / 10-03) and the abort runs
at the next rollout start.

**The bounded fallback.** A watchdog thread, started at startup with the handlers (declared lifecycle),
wakes on the request and waits `SAFE_POINT_DEADLINE_SEC`. If no safe point has claimed the exit by
then, it EXITS 15 WITHOUT SAVING: a save from that thread, at a moment the main thread has not
reached a safe point, is exactly the torn checkpoint above; no save leaves `latest.txt` on the last
periodic checkpoint, which the launcher resumes from. The deadline sits under the launcher's SIGKILL
grace (`main.launcher.run.KILL_GRACE_SECONDS`) by `SAVE_BUDGET_SEC`, so a safe point reached just
before it still has time to finish its save.

**One exit, claimed once.** Every way out — the safe-point abort, the watchdog's fallback, the normal
end of training (`stand_down`) — first CLAIMS the exit under one lock; a loser never writes anything
(the main thread parks until the winner exits it). So the fallback can never fire into a save that a
safe point (or the normal end) has started.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Callable, Optional

from main.exit_codes import TrainExitCode

#: How long after a stop signal the watchdog waits for a safe point before exiting WITHOUT a save.
SAFE_POINT_DEADLINE_SEC = 75.0
# Measured 2026-10-03 (a real `--arch production` launch, SIGTERMed as its first update began): the
# update took 47.7 s (`train/train_ms`), the abort's dump + 42 MB save ~1 s; 48.5 s SIGTERM -> exit.
#: What a safe point reached at the deadline may still spend on its dump + save inside the launcher's
#: SIGKILL grace. `deferred_abort_test` pins DEADLINE + BUDGET <= `main.launcher.run.KILL_GRACE_SECONDS`.
SAVE_BUDGET_SEC = 15.0


def _write_stderr(msg: str) -> None:
    """Async-signal-safe enough for a Python signal handler: a raw `os.write`, no buffered stream, no lock."""
    try:
        os.write(2, msg.encode("utf-8", "replace"))
    except OSError:
        pass


class DeferredAbort:
    """The stop request, the safe point and the bounded fallback (module docstring).

    ``commit(reason)`` is the abort's body (dump, save, drain) and RETURNS; ``abort`` then calls
    ``exit_fn(TrainExitCode.INTERRUPTED)``. ``wait_fn(seconds)`` is how the watchdog waits out the
    deadline (default: an interruptible wait on the close event); a test injects one it controls."""

    def __init__(self, commit: Callable[[str], None], *,
                 deadline_sec: float = SAFE_POINT_DEADLINE_SEC,
                 exit_fn: Callable[[int], None] = os._exit,
                 wait_fn: Optional[Callable[[float], None]] = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._commit = commit
        self.deadline_sec = float(deadline_sec)
        self._exit = exit_fn
        self._clock = clock
        self._closed = threading.Event()
        self._wait = wait_fn if wait_fn is not None else (lambda s: self._closed.wait(s))
        # The request: written ONLY by `request` (a signal handler), read by the main thread + watchdog.
        self.reason: Optional[str] = None
        self.requested_at: Optional[float] = None
        # The exit claim: taken by exactly one of {safe point, fallback, normal end}. Never from a handler.
        self._claim_lock = threading.Lock()
        self.claimed_by: Optional[str] = None
        self._wake_r, self._wake_w = os.pipe()
        os.set_blocking(self._wake_w, False)
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ the signal-handler side
    def request(self, reason: str) -> None:
        """Record a stop request. Called FROM A SIGNAL HANDLER: attribute stores and raw `os.write`s only."""
        if self.reason is not None:
            _write_stderr(f"\n[ABORT] {reason} — an abort is already pending ({self.reason})\n")
            return
        self.requested_at = self._clock()
        self.reason = reason
        _write_stderr(f"\n[ABORT] {reason} — stopping at the next safe point (rollout / step boundary; "
                      f"no save after {self.deadline_sec:.0f}s without one)\n")
        try:
            os.write(self._wake_w, b"!")
        except OSError:
            pass

    # ------------------------------------------------------------------ the main thread
    def _claim(self, who: str) -> bool:
        with self._claim_lock:
            if self.claimed_by is None:
                self.claimed_by = who
                return True
            return False

    def _park(self) -> None:
        """Another path claimed the exit and is ending the process; write nothing, wait for it."""
        threading.Event().wait()

    def safe_point(self, where: str = "") -> None:
        """At a safe point (no update, no logger dump in progress): run the abort iff one was requested.
        ``where`` (the loop event) is named in the abort's first line."""
        if self.reason is not None:
            self.abort(f"{self.reason} — at the safe point: {where}" if where else self.reason)

    def abort(self, reason: str) -> None:
        """THE canonical abort: dump + save + drain, then exit 15. Main thread, at a safe point only (the
        graceful restart's rollout end, or `safe_point`). Does not return."""
        if not self._claim("safe point"):
            self._park()
        self._commit(reason)
        self._exit(int(TrainExitCode.INTERRUPTED))

    def stand_down(self) -> bool:
        """The normal end of training (after `learn()` returned): claim the exit so the fallback can never
        fire into the final save. False iff the fallback already claimed it (the caller then parks)."""
        return self._claim("training complete")

    # ------------------------------------------------------------------ the watchdog
    def start(self) -> None:
        """Start the watchdog (startup, with the handlers — never lazily)."""
        if self._thread is None:
            self._thread = threading.Thread(target=self._watch, name="deferred-abort-watchdog", daemon=True)
            self._thread.start()

    def _watch(self) -> None:
        while not self._closed.is_set():
            try:
                os.read(self._wake_r, 1)          # blocks until a request (or `close`) writes a byte
            except OSError:
                return
            if self._closed.is_set():
                return
            if self.reason is None or self.requested_at is None:
                continue
            remaining = self.deadline_sec - (self._clock() - self.requested_at)
            if remaining > 0:
                self._wait(remaining)
            if self._closed.is_set():
                return
            if self._claim("fallback"):
                _write_stderr(f"\n[ABORT] no safe point within {self.deadline_sec:.0f}s of '{self.reason}' — "
                              "exiting WITHOUT a save (a save now could be a torn checkpoint); the launcher "
                              "resumes from the last periodic checkpoint\n")
                self._exit(int(TrainExitCode.INTERRUPTED))
            return

    def close(self) -> None:
        """Stop the watchdog (tests; a process that exits needs nothing). Idempotent."""
        if self._wake_w < 0:
            return
        self._closed.set()
        try:
            os.write(self._wake_w, b"!")
        except OSError:
            pass
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                return
        fds, self._wake_r, self._wake_w = (self._wake_r, self._wake_w), -1, -1
        for fd in fds:
            if fd >= 0:
                os.close(fd)
