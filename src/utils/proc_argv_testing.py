"""Test support: a stand-in process whose ``/proc/<pid>/cmdline`` ALREADY shows its argv when the test reads it.

Test support only (like ``utils.fake_proc_testing``): production code never imports it.

Why it exists. ``Popen`` returns once the exec can no longer fail, an instant BEFORE the kernel publishes the new
argv: until then ``/proc/<pid>/cmdline`` reads ``''`` (measured 2026-11-16: 4-5 % of direct ``Popen(["sleep", ...])``
reads right after the call, on the agent cores). A reader that matches the cmdline (``procfs.cmdline``,
``procwatch.signal_if``, ``armend.wait_end``'s ``main.launcher`` identity check) calls that an identity MISMATCH, so
the test flaked at that rate (``armend_test`` 2026-11-16, ``procwatch_test`` 2026-11-10). A ``bash -c "exec -a NAME
sleep"`` stand-in is worse: it shows ``bash``'s argv for a further stretch.

So a stand-in is spawned with ONE of these helpers, which (a) set ``argv[0]`` directly (``Popen(executable=...)``,
no ``bash`` stage) and (b) WAIT, on the observable condition and not on a clock, until ``/proc`` shows that argv. The
wait is bounded and its expiry FAILS LOUDLY as a fixture error (:class:`FixtureError`), never as the test's own
semantic check.
"""
from __future__ import annotations

import shutil
import subprocess
import time
from typing import Optional

from utils import procfs


class FixtureError(RuntimeError):
    """The stand-in never showed the argv it was spawned with: the FIXTURE is broken, not the code under test."""


def wait_argv(p: subprocess.Popen, equals: Optional[str] = None, contains: Optional[str] = None,
              timeout_s: float = 20.0, proc: str = "/proc") -> str:
    """Block until ``p``'s ``/proc`` cmdline (argv joined by spaces) is non-empty and, if given, equals ``equals`` /
    contains ``contains``; return it. Bounded: on expiry, or if ``p`` already exited, kill and reap it (by its own
    handle) and raise :class:`FixtureError` naming the last argv read."""
    t_end = time.monotonic() + timeout_s
    seen = ""
    while True:
        seen = procfs.cmdline(p.pid, proc)
        if seen and (equals is None or seen == equals) and (contains is None or contains in seen):
            return seen
        if time.monotonic() > t_end or p.poll() is not None:
            gone = p.poll() is not None
            if not gone:
                p.kill()
            p.wait()
            raise FixtureError(f"pid {p.pid} {'exited' if gone else f'did not show its argv within {timeout_s} s'}: "
                               f"wanted equals={equals!r} contains={contains!r}, last /proc cmdline {seen!r}")
        time.sleep(0.001)


def spawn_sleeper(secs: float = 30, name: Optional[str] = None, timeout_s: float = 20.0) -> subprocess.Popen:
    """A ``sleep <secs>`` child whose argv ``/proc`` already shows. With ``name`` its ``argv[0]`` IS ``name`` (the
    ``exec -a`` of a shell, without the shell): the cmdline then reads ``"<name> <secs>"``. The caller kills and reaps it."""
    argv = [name or "sleep", str(secs)]
    exe = shutil.which("sleep")
    if exe is None:
        raise FixtureError("no `sleep` on PATH")
    p = subprocess.Popen(argv, executable=exe, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)
    wait_argv(p, equals=" ".join(argv), timeout_s=timeout_s)
    return p
