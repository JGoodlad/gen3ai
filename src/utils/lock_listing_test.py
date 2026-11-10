"""``procfs.LockTable`` against REAL flocks under a listing that MISSES a live holder (``utils.lock_listing_testing``):
the rows removed, and the real tear forced through the kernel. Temp lock files only.

Load-bearing pins, each FAILS on a revert of what it guards:
* a holder the listing omits is still named, by a pid the caller knows and by the all-process walk;
* THE FORCED TEAR really makes one read of ``/proc/locks`` skip a live holder on this kernel (the incident of
  2026-11-09, reproduced), and the verified reading names that holder all the same.

``integration``: spawns and reaps its own holder processes. ~3 s.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from typing import Iterator

import pytest

from utils import lock_listing_testing as T
from utils import procfs as P

pytestmark = pytest.mark.integration

_HOLD = ("import fcntl, os, sys, time\n"
         "fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT)\n"
         "fcntl.flock(fd, fcntl.LOCK_EX)\n"
         "print('held', flush=True)\n"
         "time.sleep(300)\n")


@pytest.fixture
def held(tmp_path) -> Iterator[tuple]:
    """``(holder pid, lock key)``: a child of this test holding a flock on a temp file; killed by pid at the end."""
    lock = tmp_path / "a.lock"
    p = subprocess.Popen([sys.executable, "-c", _HOLD, str(lock)], stdout=subprocess.PIPE, text=True)
    try:
        assert p.stdout is not None and p.stdout.readline().strip() == "held"
        key = P.file_key(str(lock))
        assert key is not None
        yield p.pid, key
    finally:
        if p.poll() is None:
            os.kill(p.pid, signal.SIGKILL)
        p.wait()


def test_a_holder_the_listing_omits_is_still_named(held, monkeypatch):
    pid, key = held
    assert P.lock_holders(key) == [pid]
    monkeypatch.setattr(P, "_read_listing", T.without(pid))
    assert T.listed(key) == [], "the forced miss is in place"
    assert P.LockTable().holders(key, known=[pid]) == [pid]
    assert P.LockTable().holders(key) == [pid] and P.lock_holders(key) == [pid]
    monkeypatch.setattr(P, "_read_listing", T.without())             # a listing that names nobody at all
    assert P.lock_holders(key) == [pid] and P.holds_lock(pid, key)
    os.kill(pid, signal.SIGKILL)                                     # our own child, by pid
    end = time.monotonic() + 20
    while P.pid_alive(pid) and P.read_stat(pid) is not None and P.read_stat(pid).state != "Z" and time.monotonic() < end:
        time.sleep(0.02)
    assert P.lock_holders(key, known=[pid]) == [], "a lock nobody holds must still read FREE"


def test_the_forced_tear_skips_a_live_holder_and_the_table_still_names_it(held, tmp_path, monkeypatch):
    """THE INCIDENT, on the real listing: locks released between two reads of ``/proc/locks`` make it skip a holder that
    was there throughout. The raw reading misses it (``Torn.hides``); ``LockTable`` must not."""
    pid, key = held
    cpus = os.sched_getaffinity(0)
    torn = T.Torn(tmp_path, key)
    if not torn.hides(pid):
        pytest.skip("INCONCLUSIVE: one read of /proc/locks could not be made to skip a live lock on this kernel")
    monkeypatch.setattr(P, "_read_listing", torn)
    before = torn.hid
    assert T.listed(key) == [] and torn.hid == before + 1, "this read was torn: the holder is not listed"
    assert P.LockTable().holders(key) == [pid] and torn.hid == before + 2, "found under a torn read of its own"
    assert P.lock_holders(key) == [pid] and P.holds_lock(pid, key)
    assert os.sched_getaffinity(0) == cpus, "the torn reader must hand back the CPUs it pinned itself to"
