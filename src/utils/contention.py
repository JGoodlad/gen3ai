"""Contention-robust timeouts — ``gen3_contention_robust_timeouts_v1``.

This box normally has a production training run on it (48 env workers + eval workers),
so any test or tool that bounds work with a fixed WALL-CLOCK timeout is measuring the
box's spare capacity, not the code. Three separate investigations were voided by exactly
that (see the memory note ``project_rust_bridge_training_enablement`` → OPS): the same
tests passed in isolation and passed on the main checkout, and the only difference was a
load average of 35 on 16 cores.

The discipline that produced was "check ``uptime`` before running a heavy suite", which is
a human remembering to do something. This module is the structural version.

The design rests on one distinction:

    CONTENTION stretches DURATION.   A WEDGE stops PROGRESS.

A starved bridge battle still emits protocol lines, just slower — every line proves the
child is alive and the handshake is advancing. A deadlocked one emits nothing at all
(the project's own ``IDLE cpu = DEADLOCK, not slowness`` lesson). So the bound that
separates the failure we want to catch from the noise we want to ignore is **time since
the last observed progress**, not total elapsed time. A total-duration cap conflates
them by construction: it fires on a slow-but-healthy run and a wedged one alike, and the
only way to stop the false positive is to raise the cap until it stops catching the real
bug too.

Hence, in preference order:

1. :class:`ProgressDeadline` — the primary tool. Bound the IDLE gap, not the total. Under
   20x oversubscription the inter-line gap barely moves (each line is microseconds of
   work); under a wedge it goes to infinity. The signal-to-noise is enormous.
2. :func:`scale_timeout` — for the cases where only a total bound is available (a
   third-party API taking ``timeout=``). Scales by measured contention so the bound
   tracks the capacity actually available.
3. :func:`describe_contention` — attach to EVERY timeout message. The three void
   investigations each cost hours because the failure did not say "you were starved";
   a timeout that reports the load average and the top CPU consumer diagnoses itself.

Deliberately NOT here: retry-on-timeout. Retrying converts a deterministic wedge into a
slow flake and destroys the evidence — the wedge is the thing worth catching.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from typing import Optional

__all__ = [
    "cpu_contention_factor",
    "scale_timeout",
    "describe_contention",
    "warn_if_contended",
    "ProgressDeadline",
    "ProgressTimeout",
]

# A single-threaded process on a box with load L and N cpus receives roughly ``min(1, N/L)``
# of a core, so its wall-clock slowdown is about ``max(1, L/N)`` — which is exactly the
# factor below. It is a fair-share approximation, not a guarantee (it ignores memory
# bandwidth, BLAS thread thrash, and IO), which is why callers should still prefer a
# progress bound over a scaled duration wherever they can.
_MAX_FACTOR = 12.0

# ``os.getloadavg()`` reads /proc every call and is a 1-minute EMA, so sub-second
# resampling buys nothing. Cache briefly; re-reading every few seconds is what lets a
# long-lived ProgressDeadline ADAPT when a training run starts mid-test (the lag that a
# sample-once-at-construction design would bake in permanently).
_CACHE_TTL_S = 5.0

_lock = threading.Lock()
_cached: Optional[tuple[float, float]] = None  # (monotonic_at_sample, factor)


def _cpu_count() -> int:
    """CPUs this process may actually run on.

    ``sched_getaffinity`` over ``cpu_count`` on purpose: a pinned or cpuset-confined
    process is starved relative to ITS OWN affinity mask, and dividing the box-wide load
    average by the box-wide CPU count would hide that.
    """
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):  # pragma: no cover - non-Linux
        return max(1, os.cpu_count() or 1)


def cpu_contention_factor(*, refresh: bool = False) -> float:
    """How many times slower a CPU-bound thread should expect to run right now.

    ``1.0`` on an idle box; ``~2.2`` beside a 48-env training run on 16 cores; clamped to
    ``_MAX_FACTOR`` so a runaway load cannot turn a bounded wait into an unbounded one.

    ``GEN3AI_TIMEOUT_SCALE`` overrides the measurement outright — set it when you KNOW the
    regime and do not want to depend on the load average's one-minute lag (e.g. a CI box
    that starts a trainer at the same moment as the suite).
    """
    override = os.environ.get("GEN3AI_TIMEOUT_SCALE")
    if override:
        try:
            return max(1.0, min(_MAX_FACTOR, float(override)))
        except ValueError:
            pass  # A malformed override must not break the caller; fall through to measuring.

    global _cached
    now = time.monotonic()
    with _lock:
        if not refresh and _cached is not None and (now - _cached[0]) < _CACHE_TTL_S:
            return _cached[1]
        try:
            load = os.getloadavg()[0]
        except OSError:  # pragma: no cover - /proc unavailable
            factor = 1.0
        else:
            factor = max(1.0, min(_MAX_FACTOR, load / _cpu_count()))
        _cached = (now, factor)
        return factor


def scale_timeout(base_seconds: float) -> float:
    """Stretch a wall-clock timeout to match the CPU share actually available.

    Use ONLY where a total-duration bound is forced on you (an API that takes
    ``timeout=``). Where you can observe incremental progress, :class:`ProgressDeadline`
    is strictly better: it does not have to guess how long the whole job should take.
    """
    return base_seconds * cpu_contention_factor()


def describe_contention() -> str:
    """One line of self-diagnosis to append to a timeout message.

    A timeout that says only "timed out after 60s" sends the reader to re-run the test
    and wonder; one that says "load average 35.2 on 16 cpus (2.2x slower)" ends the
    investigation immediately. That difference is the entire reason this function exists.
    """
    try:
        one, five, fifteen = os.getloadavg()
    except OSError:  # pragma: no cover
        return "contention: unknown (no load average available)"
    ncpu = _cpu_count()
    factor = cpu_contention_factor()
    verdict = "box looks idle" if factor < 1.25 else f"CPU-STARVED, expect ~{factor:.1f}x slower"
    return (
        f"contention: load average {one:.2f}/{five:.2f}/{fifteen:.2f} on {ncpu} cpus "
        f"({verdict}). If this is a timeout and a training run is live, it is probably "
        f"starvation, not a bug — confirm with: ps -eo pcpu,pid,args --sort=-pcpu | head"
    )


def warn_if_contended(what: str = "benchmark", *, threshold: float = 1.25) -> bool:
    """Loudly flag a load-sensitive MEASUREMENT taken on a busy box. Returns True if contended.

    Benchmarks get the opposite treatment from tests. A test wants to survive contention, so its
    timeouts are scaled; a benchmark's *output is the measurement*, so stretching its bounds only
    buys a confidently-reported wrong number. The failure this prevents is real and recorded: a
    node-vs-rust throughput comparison (node 798 vs rust 427 fps at 8 envs) was measured on a
    CPU-saturated box and had to be superseded once it was re-run idle — with the conclusion
    REVERSED. Nothing in the output said the box had been busy.

    Deliberately a warning, not a refusal: a relative A/B run back-to-back under identical load is
    often still informative, and the caller is better placed to judge that than this helper. What
    it must not do is stay silent.
    """
    factor = cpu_contention_factor(refresh=True)
    if factor < threshold:
        return False
    print(
        f"\n⚠️  [{what}] THE BOX IS BUSY — these numbers are not comparable to an idle-box "
        f"baseline.\n    {describe_contention()}\n    Absolute timings are inflated ~{factor:.1f}x. "
        f"Re-run idle before recording a result, or treat this as a same-load A/B only.\n",
        file=sys.stderr,
    )
    return True


class ProgressTimeout(TimeoutError):
    """No progress for the whole idle budget — a genuine wedge, not mere slowness.

    A distinct type on purpose. The worst contention bug this repo has had was
    ``bridge_impl_parity_test`` folding a per-battle timeout into its "unmodeled move"
    SKIP bucket: 39 of 40 battles timed out under load and the run reported a clean pass
    with a big skip count. A timeout is never a semantic outcome — catch this type
    explicitly if you must, but never let it fall into an ``except Exception`` that
    classifies results.
    """


class ProgressDeadline:
    """Fails when work STOPS, not when it merely runs slowly.

    Call :meth:`progress` on every observable sign of life (a protocol line read, a turn
    completed, a battle finished) and :meth:`check` wherever you would have tested a
    total-duration deadline. It raises only after ``idle_budget_s`` of *contention-scaled*
    wall-clock with no progress at all.

    Two bounds, because they answer different questions:

    - ``idle_budget_s`` (required) — the real detector. Sized to "the longest plausible
      gap between two signs of life", which is a property of the protocol and is roughly
      invariant to load, so it can be set tight without flaking.
    - ``total_budget_s`` (optional) — a backstop against livelock, where a component
      chatters forever without converging and so keeps resetting the idle bound. Scaled
      too, and left off by default: on most waits a livelock is not a real risk and the
      extra bound only re-introduces the duration sensitivity we are removing.

    Both are re-scaled at each :meth:`check`, so a trainer that starts mid-wait widens the
    budget rather than manufacturing a failure.
    """

    def __init__(
        self,
        idle_budget_s: float,
        *,
        total_budget_s: Optional[float] = None,
        what: str = "operation",
    ) -> None:
        if idle_budget_s <= 0:
            raise ValueError(f"idle_budget_s must be positive, got {idle_budget_s}")
        self.idle_budget_s = float(idle_budget_s)
        self.total_budget_s = float(total_budget_s) if total_budget_s else None
        self.what = what
        now = time.monotonic()
        self._started = now
        self._last_progress = now
        self._progress_count = 0

    def progress(self) -> None:
        """Record a sign of life. Cheap enough for a per-protocol-line hot path."""
        self._last_progress = time.monotonic()
        self._progress_count += 1

    @property
    def idle_seconds(self) -> float:
        return time.monotonic() - self._last_progress

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._started

    @property
    def progress_count(self) -> int:
        return self._progress_count

    def expired(self) -> bool:
        """True if the wait should give up. Never raises — for use in a loop condition."""
        return self._reason() is not None

    def _reason(self) -> Optional[str]:
        factor = cpu_contention_factor()
        idle = self.idle_seconds
        if idle > self.idle_budget_s * factor:
            return (
                f"no progress for {idle:.1f}s (budget {self.idle_budget_s:.1f}s x {factor:.1f} "
                f"contention scale) after {self._progress_count} progress event(s)"
            )
        if self.total_budget_s is not None:
            elapsed = self.elapsed_seconds
            if elapsed > self.total_budget_s * factor:
                return (
                    f"exceeded total budget: {elapsed:.1f}s (budget {self.total_budget_s:.1f}s "
                    f"x {factor:.1f} contention scale) with {self._progress_count} progress "
                    f"event(s) — livelock, not a stall"
                )
        return None

    def check(self) -> None:
        """Raise :class:`ProgressTimeout` if the wait has genuinely wedged."""
        reason = self._reason()
        if reason is not None:
            raise ProgressTimeout(f"{self.what}: {reason}. {describe_contention()}")
