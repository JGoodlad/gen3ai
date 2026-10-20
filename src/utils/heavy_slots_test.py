"""``utils.heavy_slots`` (rule 5b's "<= 2 heavy jobs") against REAL processes — never the real slots
(every test points ``$GEN3AI_HEAVY_SLOT_DIR`` at a temp dir and strips an inherited holder marker, since
the gate running THIS file may itself hold a real heavy slot).

Load-bearing pins, each FAILS on revert of what it guards:
* a THIRD taker waits while two hold (N = 2), says so, and gets in when a holder is SIGKILLed;
* the queue is FIFO: waiters get the slot in arrival order, a dead / stale head is skipped, a gate keeps its place;
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
import threading
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
    monkeypatch.setenv("GEN3AI_GPU_LOCK", str(tmp_path / "gpu.lock"))     # never the box's real lease
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
# ── D-3: a job that already holds the GPU takes no CPU slot (core review 2026-10-20) ──────────────────────────
#
# `gpu_lock.sh mem_cap.sh …` (the documented order, the GPU-window battery's, cuda_build_validate's, …) held the GPU
# idle while the job queued for a CPU heavy slot. The job now takes none under `gpu_lock`, in either wrapper order.

def _full_house(tmp_path, monkeypatch):
    monkeypatch.setenv(HS.SLOTS_ENV, "1")
    pa = _taker(tmp_path / "a", hold=60.0)
    _wait_for((tmp_path / "a").exists, "the only slot's holder")
    return pa


def _try_take() -> "int | None | str":
    try:
        with HS.heavy_slot(["job"], force=True, wait=False, log=lambda _m: None) as s:
            return s
    except HS.SlotsFull:
        return "FULL"


def test_a_job_under_the_gpu_flock_bypasses_a_full_house_but_the_control_is_refused(slots, tmp_path, monkeypatch):
    from utils import gpu_lock as GPUL

    monkeypatch.setenv(GPUL.GPU_LOCK_ENV, str(tmp_path / "gpu.lock"))
    for k in (GPUL.HELD_ENV, GPUL.GPU_JOB_ENV):
        monkeypatch.delenv(k, raising=False)
    pa = _full_house(tmp_path, monkeypatch)
    try:
        assert _try_take() == "FULL", "control: with no GPU held, a full house refuses (--no-wait)"
        with GPUL.gpu_lock():
            assert GPUL.gpu_job() == os.getpid()
            assert _try_take() is None, "a job holding the GPU queued for / was refused a CPU slot (D-3)"
        assert GPUL.GPU_JOB_ENV not in os.environ and _try_take() == "FULL"
    finally:
        _kill(pa)


def test_a_job_under_the_lease_pass_through_bypasses_too(slots, tmp_path, monkeypatch):
    """The lease owner's `gpu_lock` takes no flock and exports no HELD_ENV: it still marks the GPU job."""
    from types import SimpleNamespace

    from utils import gpu_lock as GPUL

    monkeypatch.setenv(GPUL.GPU_LOCK_ENV, str(tmp_path / "gpu.lock"))
    for k in (GPUL.HELD_ENV, GPUL.GPU_JOB_ENV):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(GPUL.L, "read_lease", lambda _p: SimpleNamespace(pid=os.getpid()))
    monkeypatch.setattr(GPUL.L, "token_matches", lambda _l: True)
    pa = _full_house(tmp_path, monkeypatch)
    try:
        with GPUL.gpu_lock():
            assert GPUL.HELD_ENV not in os.environ and GPUL.gpu_job() == os.getpid()
            assert _try_take() is None
        assert _try_take() == "FULL"
    finally:
        _kill(pa)


def test_a_stale_gpu_job_marker_is_ignored(slots, tmp_path, monkeypatch):
    from utils import gpu_lock as GPUL

    pa = _full_house(tmp_path, monkeypatch)
    try:
        monkeypatch.setenv(GPUL.GPU_JOB_ENV, str(pa.pid))   # alive, but a CHILD of this process, not an ancestor
        assert GPUL.gpu_job() is None
        assert _try_take() == "FULL"
    finally:
        _kill(pa)


# ── rule 5b is keyed to the GPU LEASE too (orchestrator 2026-10-20) ───────────────────────────────────────────

def test_a_held_gpu_lease_wants_the_slots_with_no_run_live_and_a_read_error_never_fails_a_job(slots, monkeypatch):
    """While the GPU is leased (a measurement battery under mem_cap in the same memory slice) the slots are ON
    even though no training run is live - else the jobs queued behind a run flood the window the moment it ends.
    Fails on a revert of `lease_reason`."""
    from types import SimpleNamespace

    from utils import gpu_lock as GPUL

    assert HS.wanted(runs={}) is None, "precondition: no run, no lease, not forced"
    lease = SimpleNamespace(describe=lambda: "lease owner 'battery'")
    monkeypatch.setattr(GPUL.L, "read_lease", lambda _p: lease)
    why = HS.wanted(runs={})
    assert why is not None and "LEASED" in why and "battery" in why, why
    assert HS.wanted(runs={}, lease=False) is None                   # the run-only policy is still expressible
    assert "training run is live" in HS.wanted(runs={"/m/rb_x": 1})   # a live run keeps its own reason

    def boom(_p):
        raise OSError("lease file unreadable")

    monkeypatch.setattr(GPUL.L, "read_lease", boom)
    assert HS.wanted(runs={}) is None, "a lease read error must not fail (or slot) a job"


def test_a_job_behind_a_held_lease_queues_for_a_slot_but_the_lease_owners_gpu_job_does_not(slots, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from utils import gpu_lock as GPUL

    monkeypatch.setattr(GPUL.L, "read_lease", lambda _p: SimpleNamespace(pid=os.getpid(), describe=lambda: "x"))
    monkeypatch.setattr(GPUL.L, "token_matches", lambda _l: False)    # an ordinary agent job, not the owner
    monkeypatch.setattr(procfs, "live_training_runs", lambda *a, **k: {})
    monkeypatch.setenv(HS.SLOTS_ENV, "1")
    pa = _taker(tmp_path / "a", hold=60.0)
    try:
        _wait_for((tmp_path / "a").exists, "the only slot's holder")

        def try_take():
            try:
                with HS.heavy_slot(["job"], wait=False, log=lambda _m: None) as s:
                    return s
            except HS.SlotsFull:
                return "FULL"

        assert try_take() == "FULL", "an agent job beside a held lease took no slot"
        monkeypatch.setattr(GPUL.L, "token_matches", lambda _l: True)
        monkeypatch.delenv(GPUL.GPU_JOB_ENV, raising=False)
        with GPUL.gpu_lock():                                         # the lease owner's battery cell
            assert try_take() is None, "the lease owner's GPU cell must not queue behind the slots"
    finally:
        _kill(pa)


# ── the chunk policy (orchestrator 2026-10-20): a slot per bounded chunk, never per run ───────────────────────

def test_a_holder_past_the_long_hold_bar_is_flagged_in_status_and_warned_once(slots, monkeypatch):
    """Two research jobs each held a slot for a multi-hour run and froze every other agent. Nothing is killed;
    `--status` flags the holder and the holder gets ONE stderr line. Fails on a revert of the flag / timer."""
    monkeypatch.setattr(HS, "LONG_HOLD_S", 0.3)
    lines: list[str] = []
    with HS.heavy_slot(["a", "long", "job"], force=True, log=lines.append):
        st = HS.collect(now=time.time() + 0.1)
        assert st["slots"][0].get("long_hold") is False and HS.LONG_HOLD_TAG not in HS.format_human(st)
        time.sleep(0.9)
        warns = [ln for ln in lines if HS.LONG_HOLD_TAG in ln]
        assert len(warns) == 1, lines                                  # ONE warning, to the holder
        st = HS.collect(now=time.time() + 1.0)
        row = st["slots"][0]
        assert row["long_hold"] is True
        assert HS.LONG_HOLD_TAG in HS.format_human(st) and "LONG HOLD (policy: <= 45 min per chunk)" in HS.format_human(st)
    assert not any(HS.LONG_HOLD_TAG in ln for ln in lines[len(warns) + 1:])      # released: the timer is gone


def test_a_waiter_proceeds_when_the_run_it_waited_behind_ends(slots, tmp_path, monkeypatch):
    """`await_free_slot` re-checks the policy while it waits: a gate queued behind a run that then ended must not
    sit out the other holders (tonight a gate waited ~90 min after the run it queued for was gone)."""
    monkeypatch.setenv(HS.SLOTS_ENV, "1")
    pa = _taker(tmp_path / "a", hold=60.0)
    try:
        _wait_for((tmp_path / "a").exists, "the only slot's holder")
        runs = {"/m/rb_x": 1}
        calls = {"n": 0}

        def wanted_then_over(**_kw):
            calls["n"] += 1
            return "a run is live" if calls["n"] < 3 else None

        monkeypatch.setattr(HS, "wanted", wanted_then_over)
        t0 = time.monotonic()
        HS.await_free_slot(["gate"], poll_s=0.02, log=lambda _m: None, runs=runs)
        assert calls["n"] >= 3 and time.monotonic() - t0 < scale_timeout(20)
    finally:
        _kill(pa)


# ── the queue is FIFO (2026-10-20: a gate starved 3.4 h behind later arrivals; waiters used to poll at random) ──

# a waiter that records WHEN it got the slot (ns), holds briefly, then leaves
_FIFO_TAKER = ("import pathlib, sys, time\n"
               "from utils.heavy_slots import heavy_slot\n"
               "with heavy_slot(['w', sys.argv[1]], force=True, poll_s=0.02, log=lambda m: None):\n"
               "    pathlib.Path(sys.argv[1]).write_text(str(time.time_ns())); time.sleep(0.15)\n")


def _sleeper() -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])


def test_waiters_get_the_slot_in_arrival_order_not_poll_luck(slots, tmp_path, monkeypatch):
    """ONE slot, a holder, then SIX waiters that arrive one after another; when the holder dies they must run in
    arrival order. Fails on a revert to polling the slots at random (1/720 to pass by luck)."""
    monkeypatch.setenv(HS.SLOTS_ENV, "1")
    holder = _taker(tmp_path / "h")
    waiters: list = []
    try:
        _wait_for((tmp_path / "h").exists, "the holder")
        for k in range(6):
            marker = tmp_path / f"w{k}"
            waiters.append((marker, subprocess.Popen([sys.executable, "-c", _FIFO_TAKER, str(marker)])))
            _wait_for(lambda: len(HS._tickets(slots)) == k + 1, f"waiter {k}'s ticket")   # arrival order is fixed
        _kill(holder)
        for _marker, p in waiters:
            assert p.wait(timeout=scale_timeout(60)) == 0
        stamps = [int(m.read_text()) for m, _ in waiters]
        assert stamps == sorted(stamps), f"waiters ran out of arrival order: {stamps}"
        assert not HS._tickets(slots), "a finished waiter's ticket outlived it"
    finally:
        _kill(holder, *(p for _, p in waiters))


def test_the_queue_head_skips_dead_and_silent_tickets_and_a_free_slot_per_place(slots, tmp_path, monkeypatch):
    """The rank rule, with no timing: tickets of real (sleeping) pids; a DEAD pid's ticket is skipped and removed, a
    STALE one (owner not polling) is skipped but keeps its place, and N free slots entitle the N oldest."""
    monkeypatch.setenv(HS.SLOTS_ENV, "2")
    a, b, c = _sleeper(), _sleeper(), _sleeper()
    h = _taker(tmp_path / "h")       # the holder starts BEFORE any ticket exists, else it queues behind b and c and never runs
    try:
        _wait_for((tmp_path / "h").exists, "a holder")
        for p in (a, b, c):
            HS.take_ticket(slots, p.pid)
        t = {pid: path for _, pid, path in HS._tickets(slots)}
        assert [pid for _, pid, _p in HS._tickets(slots)] == [a.pid, b.pid, c.pid]
        assert HS.take_ticket(slots, b.pid) == t[b.pid], "a second take must keep the ticket's age"
        now = time.time()
        assert [HS.queue_ahead(slots, p.pid, now=now) for p in (a, b, c)] == [0, 1, 2]
        _kill(a)                                           # a dead head: skipped, and its ticket is removed
        assert HS.queue_ahead(slots, b.pid, now=now) == 0 and HS.queue_ahead(slots, c.pid, now=now) == 1
        assert not t[a.pid].exists()
        later = now + HS.TICKET_STALE_S + 5                # b stops polling (a gate blocked on the gate lock)
        os.utime(t[c.pid], (later, later))
        assert HS.queue_ahead(slots, c.pid, now=later) == 0, "a silent head must not block the queue"
        assert HS.queue_ahead(slots, b.pid, now=later) == 0, "...and it keeps its own place when it returns"
        assert HS.queue_ahead(slots, b.pid, now=now) == 0 and HS.queue_ahead(slots, c.pid, now=now) == 1
        # one free slot entitles only the head, two free slots the two oldest (entitled() reads the real locks)
        assert HS.free_count(slots, 2) == 1
        assert HS.entitled(slots, 2, b.pid, now=now) and not HS.entitled(slots, 2, c.pid, now=now)
        _kill(h)
        _wait_for(lambda: HS.free_count(slots, 2) == 2, "the holder's slot to free")
        assert HS.entitled(slots, 2, b.pid, now=now) and HS.entitled(slots, 2, c.pid, now=now)
    finally:
        _kill(h, b, c)


def test_a_gate_keeps_its_place_between_its_wait_and_its_take(slots, tmp_path, monkeypatch):
    """`await_free_slot` returns with the ticket still held, so a newcomer that arrives between the wait and the
    gate's `heavy_slot(wait=False, keep_ticket=True)` queues BEHIND the gate. Fails if the ticket is dropped on the
    return, or the take ignores the queue (the newcomer then takes the freed slot and the gate is refused)."""
    monkeypatch.setenv(HS.SLOTS_ENV, "1")
    holder = _taker(tmp_path / "h")
    newcomer = None
    try:
        _wait_for((tmp_path / "h").exists, "the holder")
        done = threading.Event()
        err: list = []

        def gate_wait():
            try:
                HS.await_free_slot(["gate"], force=True, poll_s=0.02, log=lambda _m: None)
            except BaseException as e:          # noqa: BLE001
                err.append(e)
            done.set()

        th = threading.Thread(target=gate_wait, daemon=True)
        th.start()
        _wait_for(lambda: HS.has_ticket(slots), "the gate to queue")
        newcomer = _taker(tmp_path / "n", hold=60.0)
        _wait_for(lambda: len(HS._tickets(slots)) == 2, "the newcomer to queue behind the gate")
        _kill(holder)                                            # the slot frees: the gate is the head
        assert done.wait(scale_timeout(15)) and not err, err
        time.sleep(0.5)                                          # the newcomer polls at 0.05 s: it must NOT jump the gate
        assert not (tmp_path / "n").exists(), "a newcomer took the slot the gate was next in line for"
        with HS.heavy_slot(["gate"], force=True, wait=False, keep_ticket=True, log=lambda _m: None) as s:
            assert s == 0
        assert not HS.has_ticket(slots), "the gate's ticket outlived its slot"
        _wait_for((tmp_path / "n").exists, "the newcomer to run after the gate")
    finally:
        _kill(*(p for p in (holder, newcomer) if p is not None))
