"""``utils.procgroup`` against REAL processes: the command leads a group (or a session) of its own, a signal sent to
the wrapper reaches the whole group, and nothing outside it. (``gpu_lock_test.py`` pins the same through the CLI and
a terminal; ``main/launcher/gpu_window/guest_group_test.py`` pins the T46 guest.)

``integration``: spawns and signals its own processes, ~3 s. Every process it makes is reaped by explicit pid / pgid.
"""
from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from pathlib import Path

import pytest

from utils import procfs
from utils.procgroup import group_members, run_in_own_group

pytestmark = pytest.mark.integration


_alive = procfs.pid_alive      # one atomic /proc read; a zombie or a vanished pid is not alive


def _wait_for(pred, what: str, timeout: float = 15.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = pred()
        if v:
            return v
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {what}")


def _kill_group(pgid: int) -> None:
    if pgid in (0, 1, os.getpgrp(), os.getsid(0)):          # a mutated implementation must never be able to kill the runner
        return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


_GRANDCHILD = "sleep 60 & echo $! > {f}; wait"


@pytest.mark.parametrize("session", [False, True], ids=["group", "session"])
def test_the_command_leads_a_group_of_its_own_and_a_signal_to_the_wrapper_reaches_its_grandchild(tmp_path, session):
    """The wrapper's SIGTERM is forwarded to the WHOLE group: the command (bash) AND the sleep it started. A plain
    ``Popen`` + ``send_signal`` reaches bash only and orphans the sleep (the pre-2026-10-25 behaviour)."""
    f = tmp_path / "gc.pid"
    seen: dict = {}

    def on_start(pgid: int) -> None:
        seen["pgid"] = pgid
        seen["pgrp_of_leader"] = procfs.read_stat(pgid).pgrp
        seen["sid_of_leader"] = procfs.read_stat(pgid).session

    def sender() -> None:
        _wait_for(lambda: f.exists() and f.read_text().strip(), "the grandchild's pid")
        os.kill(os.getpid(), signal.SIGTERM)             # the stop request lands on the WRAPPER, not on the group
        if not done.wait(20):                            # a forward that never came must FAIL the test, not hang it
            _kill_group(seen["pgid"])

    grandchild = None
    done = threading.Event()
    t = threading.Thread(target=sender, daemon=True)
    t.start()
    try:
        rc = run_in_own_group(["bash", "-c", _GRANDCHILD.format(f=f)], session=session, on_start=on_start)
        done.set()
        t.join(5)
        grandchild = int(f.read_text())
        assert rc == 128 + signal.SIGTERM, "bash was killed by the forwarded SIGTERM"
        assert seen["pgrp_of_leader"] == seen["pgid"] != os.getpgrp(), "the command leads a group of its own"
        assert (seen["sid_of_leader"] == seen["pgid"]) == session, "session=True: a session of its own as well"
        _wait_for(lambda: not _alive(grandchild), "the grandchild to die of the forwarded signal", timeout=10)
    finally:
        if grandchild is not None:
            _kill_group(seen["pgid"])


@pytest.mark.parametrize("session", [True, False], ids=["session", "group"])
def test_a_session_forward_reaches_a_nested_group_and_a_group_forward_does_not(tmp_path, session):
    """``set -m`` gives the sleep a process group of its own (what a ``gpu_lock.sh`` inside a job does). The T46 guest
    (``session=True``) signals every group of the job's session, so it is reached; ``gpu_lock`` (``session=False``)
    forwards to its command's group and relies on a nested wrapper to forward further: the documented limit."""
    f = tmp_path / "nested.pid"
    seen: dict = {}

    def sender() -> None:
        _wait_for(lambda: f.exists() and f.read_text().strip(), "the nested child's pid")
        seen["nested_pgrp"] = procfs.read_stat(int(f.read_text())).pgrp         # BEFORE the signal can kill it
        os.kill(os.getpid(), signal.SIGTERM)
        if not done.wait(20):
            _kill_group(seen["pgid"])

    done = threading.Event()
    t = threading.Thread(target=sender, daemon=True)
    t.start()
    nested = None
    try:
        run_in_own_group(["bash", "-c", "set -m; " + _GRANDCHILD.format(f=f)], session=session,
                         on_start=lambda pgid: seen.update(pgid=pgid))
        done.set()
        t.join(5)
        nested = int(f.read_text())
        assert seen["nested_pgrp"] == nested != seen["pgid"], "the sleep leads a group of its own"
        if session:
            _wait_for(lambda: not _alive(nested), "the nested group's sleep to receive the forwarded SIGTERM", timeout=10)
        else:
            time.sleep(0.3)
            assert _alive(nested), "a plain group forward does not reach a nested group (nested wrappers forward on)"
    finally:
        if nested is not None and _alive(nested):
            os.kill(nested, signal.SIGKILL)                      # our own sleep, by explicit pid


def test_a_signal_death_is_128_plus_n_and_the_exit_code_passes_through():
    assert run_in_own_group(["bash", "-c", "exit 7"]) == 7
    assert run_in_own_group(["bash", "-c", "kill -KILL $$"]) == 128 + signal.SIGKILL


def test_a_failed_on_start_kills_the_group_and_propagates(tmp_path):
    """A job whose group cannot be recorded is a job nobody can preempt: it must not be left running."""
    seen: dict = {}

    def on_start(pgid: int) -> None:
        seen["pgid"] = pgid
        raise RuntimeError("could not record the group")

    with pytest.raises(RuntimeError, match="could not record"):
        run_in_own_group(["sleep", "60"], on_start=on_start)
    _wait_for(lambda: not group_members(seen["pgid"]), "the group to be gone", timeout=10)


def test_group_members_lists_only_that_groups_live_processes(tmp_path):
    f = Path(tmp_path / "pid")
    p = subprocess.Popen(["bash", "-c", _GRANDCHILD.format(f=f)], process_group=0)
    bystander = subprocess.Popen(["sleep", "60"])                      # the test's own group
    try:
        _wait_for(lambda: f.exists() and f.read_text().strip(), "the grandchild")
        pids = {m.pid for m in group_members(p.pid)}
        assert pids == {p.pid, int(f.read_text())}, pids
        assert bystander.pid not in pids
    finally:
        _kill_group(p.pid)
        p.wait(timeout=10)
        bystander.kill()
        bystander.wait(timeout=10)
