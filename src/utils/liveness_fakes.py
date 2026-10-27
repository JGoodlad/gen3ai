"""Test support for "is this pid alive" (convergence stage 1, ``designs/tools/convergence_2026-10-26.md``).

``kill(pid, 0)`` is a bad liveness probe in three ways, and every site that asks "is this pid alive" must be immune to
all three. :func:`assert_pid_liveness` is the one contract each site's test runs, so no site can be fixed for one and
left exposed to another:

* a ZOMBIE (exited, not yet reaped) is signalable, so ``kill`` reads it as ALIVE; ``/proc`` says state ``Z``;
* inside a sandboxed agent shell ``kill`` reads a LIVE pid as dead (``ESRCH``, acf2193f);
* for a process we may not signal, ``kill`` raises ``EPERM`` while ``/proc`` simply lists it.

The signal answers are RIGGED (:func:`rig_kill`), not provoked, so the check passes or fails the same on every box and
for every user. Each cell of the contract fails on a revert to ``os.kill``: the zombie under every mode, the live pid
under ``esrch``, and (in the worktree owner check) a reused pid under ``eperm``.
"""
from __future__ import annotations

import contextlib
import os
import subprocess
from typing import Callable, Iterator

import pytest

#: how ``os.kill(pid, 0)`` answers: truthfully, as in the sandbox (a LIVE pid reads as gone), or as for a process we
#: may not signal.
KILL_MODES = ("honest", "esrch", "eperm")


@contextlib.contextmanager
def zombie() -> Iterator[int]:
    """The pid of a child that has EXITED and is NOT reaped. ``waitid(WNOWAIT)`` returns once the child is a zombie and
    leaves it one, so the state is certain (no polling, no sleep); the child is reaped on the way out."""
    child = subprocess.Popen(["true"])
    try:
        os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOWAIT)
        yield child.pid
    finally:
        child.wait()


def reaped_pid() -> int:
    """A pid that existed a moment ago and is gone (a child that was started and reaped)."""
    child = subprocess.Popen(["true"])
    child.wait()
    return child.pid


def rig_kill(monkeypatch: "pytest.MonkeyPatch", mode: str) -> None:
    """Make ``os.kill(pid, 0)`` answer per ``mode`` (see :data:`KILL_MODES`); every other signal is untouched."""
    if mode == "honest":
        return
    err = {"esrch": ProcessLookupError(3, "No such process"), "eperm": PermissionError(1, "Operation not permitted")}[mode]
    real_kill = os.kill

    def kill(pid: int, sig: int) -> None:
        if sig == 0:
            raise err
        real_kill(pid, sig)

    monkeypatch.setattr(os, "kill", kill)


def assert_pid_liveness(is_alive: Callable[[int], bool], monkeypatch: "pytest.MonkeyPatch") -> None:
    """``is_alive`` answers from ``/proc``, whatever ``os.kill`` says: a live pid is alive, a zombie and a reaped pid
    are not, under every :data:`KILL_MODES` answer."""
    for mode in KILL_MODES:
        with monkeypatch.context() as m:
            rig_kill(m, mode)
            assert is_alive(os.getpid()) is True, f"a live pid read as dead when kill says {mode!r}"
            with zombie() as z:
                assert is_alive(z) is False, f"an exited, unreaped child (a zombie) read as alive when kill says {mode!r}"
            assert is_alive(reaped_pid()) is False, f"a reaped pid read as alive when kill says {mode!r}"
