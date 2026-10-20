"""``utils.heavy_slots`` (rule 5b's "<= 2 heavy jobs") against REAL processes — never the real slots
(every test points ``$GEN3AI_HEAVY_SLOT_DIR`` at a temp dir and strips an inherited holder marker, since
the gate running THIS file may itself hold a real heavy slot).

Load-bearing pins, each FAILS on revert of what it guards:
* a THIRD taker waits while two hold (N = 2), says so, and gets in when a holder is SIGKILLed;
* ``--no-wait`` refuses at once (exit 87) and ``mem_cap.sh`` takes the slot (its wiring);
* a BLOCKING job bypasses a full house and is listed as a bypasser;
* a child of a holder takes no second slot, and a routine gate holds ONE heavy slot that a
  ``mem_cap.sh`` inside it re-enters (a gate under mem_cap inside gate_lock never takes two);
* lock order: a gate take that would WAIT while an ancestor holds a heavy slot is refused (exit 5).

``integration``: spawns and kills its own processes, by explicit PID.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time

import pytest

from utils import gate_lock as GL
from utils import heavy_slots as HS
from utils import procfs
from utils.contention import scale_timeout
from utils.paths import repo_path, src_root

pytestmark = pytest.mark.integration

_GATE_SH = str(repo_path("scripts", "ops", "gate_lock.sh"))
# A taker: takes a heavy slot (forced, as if a run were live), touches a marker, then sleeps.
_TAKER = ("import os, pathlib, sys, time\n"
          "from utils.heavy_slots import heavy_slot, SlotsFull, EXIT_SLOTS_FULL\n"
          "marker, hold, wait = sys.argv[1], float(sys.argv[2]), sys.argv[3] == 'wait'\n"
          "try:\n"
          "    with heavy_slot(['taker', marker], force=True, wait=wait, poll_s=0.05, report_every_s=0.2) as s:\n"
          "        pathlib.Path(marker).write_text(str(s)); time.sleep(hold)\n"
          "except SlotsFull as e:\n"
          "    print(e, file=sys.stderr); sys.exit(EXIT_SLOTS_FULL)\n")


@pytest.fixture
def slots(tmp_path, monkeypatch):
    d = tmp_path / "heavy"
    monkeypatch.setenv(HS.DIR_ENV, str(d))
    monkeypatch.setenv(HS.SLOTS_ENV, "2")
    monkeypatch.delenv(HS.HELD_ENV, raising=False)
    monkeypatch.delenv(HS.BLOCKING_ENV, raising=False)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(src_root()), os.environ.get("PYTHONPATH", "")]))
    return d


def _taker(marker, hold=60.0, wait=True, **kw) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", _TAKER, str(marker), str(hold), "wait" if wait else "nowait"],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **kw)


def _wait_for(pred, what: str, timeout: float = 15.0) -> None:
    end = time.monotonic() + scale_timeout(timeout)
    while time.monotonic() < end:
        if pred():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


def _kill(*procs) -> None:
    for p in procs:
        if p.poll() is None:
            os.kill(p.pid, signal.SIGKILL)
        p.wait()


def test_a_third_taker_waits_visibly_and_a_crashed_holders_slot_frees(slots, tmp_path):
    a, b, c = (tmp_path / x for x in "abc")
    pa, pb = _taker(a), _taker(b)
    pc = None
    try:
        _wait_for(lambda: a.exists() and b.exists(), "two holders")
        pc = _taker(c, hold=0.0)
        _wait_for(lambda: any(w["pid"] == pc.pid for w in HS.collect()["waiting"]), "the third to queue")
        assert not c.exists(), "a third heavy job ran while both slots were held"
        st = HS.collect()
        assert sorted(p for r in st["slots"] for p in r["pids"]) == sorted([pa.pid, pb.pid])
        assert "queue (1" in HS.format_human(st) and "taker" in HS.format_human(st)
        _kill(pa)                                      # a crash frees its slot (the kernel drops the flock)
        _wait_for(c.exists, "the third to get the freed slot")
        _, err = pc.communicate(timeout=scale_timeout(15))
        assert pc.returncode == 0 and "waiting" in err and "took heavy slot" in err, err
        assert not HS.collect()["waiting"], "a waiter's record outlived its wait"
    finally:
        _kill(*(p for p in (pa, pb, pc) if p is not None))


def test_no_wait_refuses_at_once_and_mem_cap_is_where_the_slot_is_taken(slots, tmp_path, monkeypatch):
    monkeypatch.setenv(HS.SLOTS_ENV, "1")
    pa = _taker(tmp_path / "a")
    try:
        _wait_for((tmp_path / "a").exists, "a holder")
        r = subprocess.run([sys.executable, "-c", _TAKER, str(tmp_path / "x"), "0", "nowait"],
                           capture_output=True, text=True, timeout=scale_timeout(15))
        assert r.returncode == HS.EXIT_SLOTS_FULL and "REFUSING (--no-wait)" in r.stderr
        # mem_cap.sh's own wiring: --slot --no-wait refuses BEFORE any systemd scope (the command never runs)
        r = subprocess.run([str(repo_path("scripts", "ops", "mem_cap.sh")), "--slot", "--no-wait", "1",
                            "touch", str(tmp_path / "ran")], capture_output=True, text=True,
                           timeout=scale_timeout(30))
        assert r.returncode == HS.EXIT_SLOTS_FULL, r.stderr
        assert not (tmp_path / "ran").exists()
    finally:
        _kill(pa)


def test_a_BLOCKING_job_bypasses_a_full_house_and_is_listed(slots, tmp_path, monkeypatch):
    monkeypatch.setenv(HS.SLOTS_ENV, "1")
    pa = _taker(tmp_path / "a")
    pb = None
    try:
        _wait_for((tmp_path / "a").exists, "a holder")
        pb = _taker(tmp_path / "b", hold=60.0, wait=False,
                    env={**os.environ, HS.BLOCKING_ENV: "X34 plateau tick"})
        _wait_for((tmp_path / "b").exists, "the BLOCKING job to run at once")
        assert (tmp_path / "b").read_text() == "None", "a BLOCKING job must take no slot"
        st = HS.collect()
        assert [x["pid"] for x in st["bypassing"]] == [pb.pid] and st["bypassing"][0]["why"] == "X34 plateau tick"
    finally:
        _kill(*(p for p in (pa, pb) if p is not None))


def test_no_live_run_and_no_slot_asked_takes_nothing(slots):
    with HS.heavy_slot(["x"], runs={}) as s:
        assert s is None and HS.HELD_ENV not in os.environ
    assert HS.wanted(runs={"/m/rb_x34": 1}) and HS.wanted(force=True, runs={})


def test_a_gate_holds_ONE_heavy_slot_and_mem_cap_inside_it_takes_none(slots, tmp_path, monkeypatch):
    """The ship gate's order (gate_lock -> mem_cap): the gate takes the heavy slot; everything inside
    it re-enters. With N = 1 a second take would refuse (exit 87); re-entrance exits 0."""
    monkeypatch.setenv(HS.SLOTS_ENV, "1")
    monkeypatch.setenv(GL.DIR_ENV, str(tmp_path / "gate"))
    monkeypatch.delenv(GL.HELD_ENV, raising=False)
    monkeypatch.setattr(procfs, "live_training_runs", lambda *a, **k: {"/m/rb_x34_test": 1})
    with GL.gate_slot(log=lambda _m: None):
        held = HS.verified_holder()
        assert held is not None and held[0] == os.getpid(), "a gate beside a live run holds a heavy slot"
        r = subprocess.run([sys.executable, "-c", _TAKER, str(tmp_path / "inner"), "0", "nowait"],
                           capture_output=True, text=True, timeout=scale_timeout(15))
        assert r.returncode == 0, f"a job inside the gate took (or waited for) a SECOND heavy slot: {r.stderr}"
    assert HS.HELD_ENV not in os.environ and not any(r["pids"] for r in HS.collect()["slots"])


def test_lock_order_a_gate_wait_under_a_heavy_holder_is_refused(slots, tmp_path, monkeypatch):
    gate_d = tmp_path / "gate"
    monkeypatch.setenv(GL.DIR_ENV, str(gate_d))
    monkeypatch.setenv(GL.SLOTS_ENV, "1")
    monkeypatch.delenv(GL.HELD_ENV, raising=False)
    gate_holder = subprocess.Popen([_GATE_SH, sys.executable, "-c",
                                    f"import pathlib, time; pathlib.Path({str(tmp_path / 'g')!r}).touch(); "
                                    "time.sleep(60)"], stderr=subprocess.DEVNULL)
    try:
        _wait_for((tmp_path / "g").exists, "the gate holder")
        with HS.heavy_slot(["outer mem_cap"], force=True, log=lambda _m: None):
            r = subprocess.run([_GATE_SH, "--timeout-s", "5", "true"], capture_output=True, text=True,
                               timeout=scale_timeout(30))
        assert r.returncode == 5 and "lock order" in r.stderr, (r.returncode, r.stderr)
    finally:
        gate_holder.terminate()                        # gate_lock forwards SIGTERM to its child
        gate_holder.wait(timeout=scale_timeout(15))


# ── the CONVOY (2026-10-19/20): a gate waiting for a heavy slot must not hold the gate lock ───────────────────
#
# gate_lock took its heavy slot AFTER the gate lock and WAITED for it, holding the gate lock: pid 971798 held it
# for 1 h 10 min with no children while a 28 h research job held a heavy slot, and 8 gates queued behind it.

_GATE_DRIVER = (
    "import pathlib, sys, time\n"
    "from utils import procfs, gate_lock as GL\n"
    "procfs.live_training_runs = lambda *a, **k: {'/m/rb_convoy_test': 1}      # rule 5b: a run is 'live'\n"
    "marker = sys.argv[1]\n"
    "with GL.gate_slot(poll_s=0.05, report_every_s=0.5, cmd=['convoy-gate']) as s:\n"
    "    pathlib.Path(marker).write_text('ran'); time.sleep(float(sys.argv[2]))\n"
)


def _gate_proc(marker, hold=0.0, env=None) -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", _GATE_DRIVER, str(marker), str(hold)], env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def _gate_lock_is_free(gate_dir) -> bool:
    import fcntl

    fd = os.open(str(gate_dir / "slot0.lock"), os.O_RDWR | os.O_CREAT, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False
    finally:
        os.close(fd)


def test_a_gate_waiting_for_a_heavy_slot_does_not_hold_the_gate_lock(slots, tmp_path, monkeypatch):
    """Both heavy slots held by long jobs (fake slots, real holders); a gate arrives and cannot get one. It
    must hold NOTHING while it waits (the gate lock is free for the next gate), and it runs once a slot frees.
    Fails on a revert of gate_lock's non-blocking heavy take: the waiting gate then sits on the gate lock."""
    gate_dir = tmp_path / "gate"
    env = {**os.environ, GL.DIR_ENV: str(gate_dir), GL.SLOTS_ENV: "1", HS.SLOTS_ENV: "2", HS.DIR_ENV: str(slots)}
    env.pop(GL.HELD_ENV, None)
    env.pop(HS.HELD_ENV, None)
    monkeypatch.setenv(GL.DIR_ENV, str(gate_dir))
    h1 = _taker(tmp_path / "h1", hold=120.0, env=env)
    h2 = _taker(tmp_path / "h2", hold=120.0, env=env)
    gate = None
    try:
        _wait_for((tmp_path / "h1").exists, "the first long heavy job")
        _wait_for((tmp_path / "h2").exists, "the second long heavy job")
        gate = _gate_proc(tmp_path / "ran", env=env)
        _wait_for(lambda: any(w["pid"] == gate.pid for w in HS.collect()["waiting"]), "the gate to queue for a slot")
        time.sleep(1.0)                                       # past the gate's first (failed) slot attempt
        assert gate.poll() is None and not (tmp_path / "ran").exists(), "the gate ran with both slots held"
        assert _gate_lock_is_free(gate_dir), "a gate WAITING for a heavy slot is holding the gate lock (the convoy)"
        # a second gate, with the heavy slots bypassed (the orchestrator's BLOCKING mark), gets through at once
        g2 = _gate_proc(tmp_path / "ran2", env={**env, HS.BLOCKING_ENV: "convoy test"})
        _wait_for((tmp_path / "ran2").exists, "a gate behind the waiting one to run", timeout=30.0)
        g2.wait(timeout=scale_timeout(30))
        assert g2.returncode == 0
        _kill(h1)                                             # a slot frees: the waiting gate runs
        _wait_for((tmp_path / "ran").exists, "the waiting gate to run once a slot frees", timeout=30.0)
        gate.wait(timeout=scale_timeout(30))
        assert gate.returncode == 0
    finally:
        _kill(*(p for p in (h1, h2, gate) if p is not None))
