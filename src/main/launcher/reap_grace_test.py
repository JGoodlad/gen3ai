"""`_reap` (the abnormal-app-exit teardown) gives the child the SAME SIGKILL grace as the supervisor
(P10-A2). The child saves only at its next SAFE POINT (`main.train.deferred_abort`), up to an update away;
the old 10 s grace here SIGKILLed it mid-update and lost everything since its last periodic checkpoint.

Driven by a fake child whose `wait` advances a fake clock — the outcome is decided by the grace
arithmetic, never by a real wait."""
from __future__ import annotations

import importlib
import signal
import subprocess
from types import SimpleNamespace
from typing import List, Optional

from main.launcher.run import KILL_GRACE_SECONDS, _reap

launcher_run = importlib.import_module("main.launcher.run")   # (the package re-exports a FUNCTION `run`)


class _FakeChild:
    """Never exits on its own until `exits_at` (fake seconds after the SIGTERM); `wait` advances the clock."""

    def __init__(self, clock: SimpleNamespace, exits_at: Optional[float]) -> None:
        self.pid = 424242
        self.clock = clock
        self.exits_at = exits_at
        self.signals: List[tuple] = []
        self.dead = False

    def poll(self):
        return 0 if self.dead else None

    def wait(self, timeout: Optional[float] = None):
        if self.dead:
            return 0
        step = 1.0 if timeout is None else float(timeout)
        self.clock.t += step
        if self.exits_at is not None and self.clock.t >= self.exits_at:
            self.dead = True
            return 15
        raise subprocess.TimeoutExpired("child", step)


def _wire(monkeypatch, exits_at: Optional[float]) -> _FakeChild:
    clock = SimpleNamespace(t=0.0)
    child = _FakeChild(clock, exits_at)

    def kill(pid: int, sig: int) -> None:
        assert pid == child.pid
        child.signals.append((sig, clock.t))
        if sig == signal.SIGKILL:
            child.dead = True

    monkeypatch.setattr(launcher_run, "time", SimpleNamespace(monotonic=lambda: clock.t))
    monkeypatch.setattr(launcher_run.os, "kill", kill)
    return child


def test_the_reap_grace_is_the_supervisors_kill_grace():
    import inspect

    assert inspect.signature(_reap).parameters["grace"].default == KILL_GRACE_SECONDS


def test_a_child_that_saves_after_an_update_is_not_killed(monkeypatch):
    # stops 100 s after the SIGTERM: a slow update's safe point + save — under the old 10 s it was SIGKILLed
    child = _wire(monkeypatch, exits_at=100.0)
    _reap([child])
    assert child.signals == [(signal.SIGTERM, 0.0)], child.signals


def test_a_wedged_child_is_killed_at_the_grace(monkeypatch):
    child = _wire(monkeypatch, exits_at=None)
    _reap([child])
    assert [s for s, _t in child.signals] == [signal.SIGTERM, signal.SIGKILL]
    assert child.signals[1][1] == KILL_GRACE_SECONDS     # integer fake-second steps land on it exactly
