"""``utils.heavy_slots._alive``: a waiter's / bypasser's liveness is read from ``/proc`` through ``procfs.pid_alive``.

It was ``os.path.exists("/proc/<pid>")``, which reads an exited-but-unreaped ZOMBIE as alive: a zombie waiter's ticket kept
its place in the FIFO queue (a free slot was held for it), and its record kept showing as a waiter. Each test fails on a
revert to the existence test. Fast and process-light: no flock is taken, the lock dir is a temp dir.
"""
from __future__ import annotations

import os

import pytest

from utils import heavy_slots as HS
from utils import liveness_fakes as LF


@pytest.fixture
def slots(tmp_path, monkeypatch):
    d = tmp_path / "heavy"
    d.mkdir()
    monkeypatch.setenv("GEN3AI_HEAVY_SLOT_DIR", str(d))
    return d


def test_alive_is_a_proc_read_a_zombie_is_gone(monkeypatch):
    LF.assert_pid_liveness(HS._alive, monkeypatch)


def test_a_zombie_waiters_ticket_leaves_the_queue_and_a_live_ones_stays(slots):
    live = HS.take_ticket(slots, os.getpid())
    with LF.zombie() as z:
        dead = slots / f"ticket-1-{z}.tk"
        dead.write_text("")
        assert [pid for _seq, pid, _p in HS._tickets(slots)] == [os.getpid()]
        assert not dead.exists(), "a zombie waiter's ticket kept its place in the queue"
    assert live.exists()


def test_collect_drops_a_zombie_waiters_record_and_lists_a_live_one(slots):
    """The status read (`collect`) unlinks a dead waiter's / bypasser's record; a zombie is dead."""
    def put(pid):
        return HS._write(slots, HS.Record(pid=pid, kind="wait", since=1.0, cwd="/", cmd=["x"]))
    live = put(os.getpid())
    with LF.zombie() as z:
        dead = put(z)
        out = HS.collect(slots, 2, now=2.0)
        assert not dead.exists(), "a zombie waiter's record was kept"
    assert [w["pid"] for w in out["waiting"]] == [os.getpid()] and live.exists()
