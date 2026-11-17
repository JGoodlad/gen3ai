"""The routine gate's PLACE IN THE QUEUE (2026-11-17): the order of the two waits a gate has, pinned on fake slot state.

A gate waits for the GATE LOCK (one gate at a time) and for a HEAVY SLOT (admitted by memory). It never holds one while it
waits for the other: it takes the heavy slot only after the gate lock, non-blocking, and a gate that finds none releases the
gate lock and waits holding nothing (``test_a_gate_waiting_for_a_heavy_slot_does_not_hold_the_gate_lock``,
``heavy_slots_test.py``: the convoy of 2026-10-19/20). What was left was the AGE of its place: the heavy ticket was taken only
once the gate lock was won, so a gate that waited 30 min for the lock then queued for a slot BEHIND every long job that arrived
meanwhile. ``heavy_slots.park_ticket`` takes the place at arrival, NOT polling (the queue skips it: nobody behind a gate that
is blocked on the lock is held back), and the gate beats it the moment it holds the lock.

Every state here is a flock this test holds itself (the gate lock, the one heavy slot) and a ``sleep`` standing in for the long
job that arrives later; nothing is the real box's. Unmarked (in-process threads, ~1 s).
"""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Iterator, List

import pytest

from utils import gate_lock as GL
from utils import heavy_slots as HS
from utils.contention import scale_timeout


def _wait_for(pred, what: str, timeout: float = 15.0) -> None:
    end = time.monotonic() + scale_timeout(timeout)
    while time.monotonic() < end:
        if pred():
            return
        time.sleep(0.01)
    raise AssertionError(f"timed out waiting for {what}")


@pytest.fixture
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[dict]:
    gate_d, heavy_d = tmp_path / "gate", tmp_path / "heavy"
    gate_d.mkdir()
    heavy_d.mkdir()
    for k, v in {GL.DIR_ENV: str(gate_d), GL.SLOTS_ENV: "1", HS.DIR_ENV: str(heavy_d), HS.SLOTS_ENV: "1",
                 "GEN3AI_GPU_LOCK": str(tmp_path / "gpu.lock")}.items():
        monkeypatch.setenv(k, v)
    for k in (GL.HELD_ENV, HS.HELD_ENV, HS.BLOCKING_ENV):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(HS, "wanted", lambda **_: "a training run is live (test)")      # rule 5b applies
    monkeypatch.setattr(HS.GL, "gpu_job", lambda: None)
    sleepers: List[subprocess.Popen] = []

    def hold(path: Path) -> subprocess.Popen:
        """Another PROCESS holds the file (the gate lock's own self-deadlock check reads a holder that is this process or
        its ancestor as a wait nothing can end): it prints once it has the flock, and releases it when it is killed."""
        p = subprocess.Popen(["python3", "-c",
                              "import fcntl, os, sys, time\n"
                              "fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o666)\n"
                              "fcntl.flock(fd, fcntl.LOCK_EX)\n"
                              "print('held', flush=True)\n"
                              "time.sleep(300)\n", str(path)], stdout=subprocess.PIPE, text=True)
        sleepers.append(p)
        assert p.stdout is not None and p.stdout.readline().strip() == "held"
        return p

    def later_long_job() -> subprocess.Popen:
        p = subprocess.Popen(["sleep", "120"])
        sleepers.append(p)
        return p

    try:
        yield {"gate": gate_d, "heavy": heavy_d, "hold": hold, "later": later_long_job}
    finally:
        for p in sleepers:
            p.kill()
            p.wait()


def _gate(ran: dict, cmd=("mem_cap.sh", "16", "pytest")) -> threading.Thread:
    def go() -> None:
        try:
            with GL.gate_slot(poll_s=0.02, report_every_s=3600.0, cmd=list(cmd), log=lambda _m: None):
                ran["ran"] = True
        except BaseException as exc:                    # noqa: BLE001
            ran["error"] = exc

    t = threading.Thread(target=go, daemon=True)
    t.start()
    return t


def test_a_gate_takes_its_heavy_place_at_arrival_and_keeps_it_while_it_waits_for_the_lock_and_then_for_a_slot(world) -> None:
    """The gate lock is held (an earlier gate running) and the one heavy slot is held (a long job). A gate arrives, a long job
    arrives after it. The gate's place is the FIRST: before it holds the gate lock, while it is blocked on it, and after it
    released the lock to wait for the slot. And a job behind a gate blocked on the lock is never held back for it.
    Fails on a revert of the arrival ticket (the gate then has no place until it wins the lock and queues at the back)."""
    heavy = world["heavy"]
    lock_holder = world["hold"](GL.slot_path(0, world["gate"]))
    slot_holder = world["hold"](HS.slot_path(0, heavy))
    ran: dict = {}
    gate = _gate(ran)
    me = os.getpid()
    try:
        _wait_for(lambda: HS.has_ticket(heavy, me), "the gate to take its place in the heavy queue while it waits for the lock")
        newcomer = world["later"]()
        HS.take_ticket(heavy, newcomer.pid)                                        # a long job that arrives AFTER the gate
        order = [pid for _seq, pid, _p in HS._tickets(heavy)]
        assert order == [me, newcomer.pid], "the gate's place is its arrival order, not the moment it won the gate lock"
        # not polling: the gate is blocked on the lock, so the job behind it is not held back (no counted slot, no memory
        # reserved for a gate that cannot run) - the queue reads the gate's place as parked
        assert [t.pid for t in HS.grant_order(heavy, newcomer.pid)] == [newcomer.pid]
        assert HS.queue_ahead(heavy, newcomer.pid) == 0
        gate.join(0.3)
        assert gate.is_alive() and "ran" not in ran and "error" not in ran, ("the gate must still wait for the lock", ran)

        _end(lock_holder)                                                           # the earlier gate ends
        # the gate wins the lock, finds the heavy slot held, releases the lock and waits for the slot holding NOTHING: it now
        # polls, and it is AHEAD of the job that arrived later
        _wait_for(lambda: HS.queue_ahead(heavy, newcomer.pid) == 1, "the gate to poll for the slot ahead of the later job")
        probe = os.open(str(GL.slot_path(0, world["gate"])), os.O_RDWR)
        try:
            _wait_for(lambda: _free(probe), "the gate to wait for its slot holding no gate lock")
        finally:
            os.close(probe)
        assert "ran" not in ran
        time.sleep(0.2)                                                             # a floor under what the gate waited for it

        _end(slot_holder)                                                           # the long job ends: the gate is next
        gate.join(scale_timeout(30))
        assert not gate.is_alive() and ran.get("ran") is True, ran
        assert not HS.has_ticket(heavy, me), "the gate's place outlived its slot"
    finally:
        _end(lock_holder, slot_holder)
    ev = [json.loads(x) for x in (heavy / HS.EVENTS_FILE).read_text().splitlines()]
    (g,) = [e for e in ev if e["kind"] == "gate"]
    # the two waits, each with a floor the test itself holds open (0.3 s at the lock above, 0.2 s at the slot)
    assert g["cycles"] == 2 and g["cap_gb"] == 16.0 and g["waited_gate_s"] >= 0.3 and g["waited_heavy_s"] >= 0.2, g


def _end(*procs: subprocess.Popen) -> None:
    for p in procs:
        if p.poll() is None:
            p.kill()
        p.wait()


def _free(fd: int) -> bool:
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    fcntl.flock(fd, fcntl.LOCK_UN)
    return True


def test_the_gate_line_carries_the_static_precheck_seconds_the_ship_gate_handed_down(world, monkeypatch) -> None:
    monkeypatch.setenv(GL.PRECHECK_ENV, "41.37")
    _gate({}).join(scale_timeout(30))
    (g,) = [json.loads(x) for x in (world["heavy"] / HS.EVENTS_FILE).read_text().splitlines() if '"gate"' in x]
    assert g["precheck_s"] == 41.4, g
    monkeypatch.setenv(GL.PRECHECK_ENV, "soon")
    assert GL.precheck_s() == 0.0


def test_a_gate_that_finds_everything_free_takes_its_slot_at_once_and_leaves_no_place_behind(world) -> None:
    ran: dict = {}
    _gate(ran).join(scale_timeout(30))
    assert ran == {"ran": True}
    assert not HS.has_ticket(world["heavy"], os.getpid())
    ev = [json.loads(x) for x in (world["heavy"] / HS.EVENTS_FILE).read_text().splitlines()]
    (g,) = [e for e in ev if e["kind"] == "gate"]
    assert g["cycles"] == 1 and g["waited_heavy_s"] == 0.0 and g["waited_gate_s"] == 0.0, g


def test_park_ticket_keeps_the_first_place_and_takes_none_when_no_slot_is_wanted(world, monkeypatch) -> None:
    heavy, me = world["heavy"], os.getpid()
    assert HS.park_ticket() is True and HS.has_ticket(heavy, me)
    (t1,) = [p for _s, pid, p in HS._tickets(heavy) if pid == me]
    assert time.time() - t1.stat().st_mtime > HS.TICKET_STALE_S, "a parked place must read as not polling"
    HS._beat(t1)                                                                    # the gate holds the lock and polls
    assert HS.park_ticket() is True
    assert time.time() - t1.stat().st_mtime < HS.TICKET_STALE_S, "parking again must not take back a polling place"
    HS.drop_ticket(heavy)
    monkeypatch.setattr(HS, "wanted", lambda **_: None)                              # no run live, no lease: no slot at all
    assert HS.park_ticket() is False and not HS.has_ticket(heavy, me)
    monkeypatch.setattr(HS, "wanted", lambda **_: "rule 5b")
    monkeypatch.setenv(HS.BLOCKING_ENV, "the orchestrator's mark")                   # BLOCKING bypasses the queue
    assert HS.park_ticket() is False and not HS.has_ticket(heavy, me)
