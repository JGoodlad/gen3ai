"""``liveness`` never pronounces a live supervisor dead on one racy read (the 10-27 12:43:42 incident).

The monitor paged "SUPERVISOR DEAD: NO supervisor while arm … RUNNING is mid-flight; its last beat 12:43:24" about a
supervisor that was alive (same start ticks, a beat 18 s old, the lock file unchanged; ``status`` read ALIVE a moment
later). ``/proc/locks`` is built by the kernel while other processes take and drop locks, so one read can skip a row that
was there all along. Each test below fails when its half of the fix is reverted:

* the "nobody holds it" answer is re-read (``lock_holder``);
* a heartbeat inside its declared deadline, whose pid is alive, overrides a missed lock read (``probe``);
* a supervisor whose pid is gone from ``/proc`` is DEAD at once (``Liveness.final``), but a ``/proc`` read that merely
  FAILS is not a death;
* every other DEAD / STALE needs two probes at least ``CONFIRM_GAP_S`` apart (``Confirm`` for the monitor's passes,
  ``probe_confirmed`` for ``status``).

The ``/proc`` here is a fake tree (``live_test.FakeProc``) plus a ``locks`` file; the lock file is a real file, so the
device and inode the probe keys on are real.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import pytest

from main.ops.arm_chain import liveness
from main.ops.arm_chain.live_test import FakeProc
from utils import procfs

SUP_PID = 4242
SUP_START_S = 400
T0 = 1_000_000.0
FLIGHT = {"history": [{"at": "x"}], "items": {"arm:r1": {"phase": "RUNNING", "hold": None, "data": {}}}}


class LockWorld:
    """A fake ``/proc`` with a supervisor process, a real lock file and a ``locks`` table the test edits."""

    def __init__(self, base):
        base.mkdir(parents=True, exist_ok=True)
        self.fp = FakeProc(base)
        self.proc = self.fp.root
        self.lock = str(base / "chain.state.json.lock")
        open(self.lock, "w").close()
        st = os.stat(self.lock)
        self.key = f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}"
        self.fp.add(SUP_PID, ["python", "-m", "main.ops.arm_chain", "run", "chain.json"], start_s=SUP_START_S)
        self.sleeps: List[float] = []
        self.set_locks(False)

    def set_locks(self, held: bool, pid: int = SUP_PID) -> None:
        rows = ["1: FLOCK  ADVISORY  WRITE 99 00:58:86 0 EOF"]      # another process's lock: never ours
        if held:
            rows.append(f"2: FLOCK  ADVISORY  WRITE {pid} {self.key} 0 EOF")
        with open(os.path.join(self.proc, "locks"), "w") as f:
            f.write("\n".join(rows) + "\n")
        fd = os.path.join(self.proc, str(pid), "fd", "9")
        if held and os.path.isdir(os.path.dirname(fd)) and not os.path.lexists(fd):
            os.symlink(self.lock, fd)                              # the supervisor holds its lock file open

    def kill_supervisor(self) -> None:
        import shutil
        shutil.rmtree(os.path.join(self.proc, str(SUP_PID)))

    def sleep(self, s: float) -> None:
        self.sleeps.append(s)

    def state(self, *, beat_age: float = 5.0, bound: float = 600.0, exit: Optional[Any] = None,
              start_ticks: Optional[int] = None, flight: bool = True) -> Dict[str, Any]:
        sup: Dict[str, Any] = {"pid": SUP_PID, "start_ticks": SUP_START_S * procfs.HZ if start_ticks is None else start_ticks,
                               "last_beat_epoch": T0 - beat_age, "stale_after_epoch": T0 - beat_age + bound,
                               "doing": "arm:r1 RUNNING: waiting"}
        if exit is not None:
            sup.update(exit=exit, exited_at="x")
        return {**(FLIGHT if flight else {"history": [], "items": {}}), "supervisor": sup}

    def probe(self, state: Dict[str, Any], now: float = T0, **kw: Any) -> liveness.Liveness:
        kw.setdefault("sleep", self.sleep)
        return liveness.probe(state, self.lock, now, self.proc, **kw)


@pytest.fixture
def lw(tmp_path) -> LockWorld:
    return LockWorld(tmp_path / "w")


# -- the lock read is re-read -----------------------------------------------------------------------------------------

def test_a_lock_read_that_misses_once_is_re_read_and_finds_the_holder(lw):
    """The kernel skips the supervisor's row on the first read and has it on the second. One read (the old code) read it
    DEAD; a supervisor older than the heartbeat (no ``supervisor`` block) has nothing else to vouch for it."""
    def table_whole_again(s: float) -> None:
        lw.set_locks(True)
    assert liveness.lock_holder(lw.lock, lw.proc, reads=1, sleep=table_whole_again) is None       # the old single read
    assert liveness.lock_holder(lw.lock, lw.proc, reads=3, gap_s=0.2, sleep=table_whole_again) == SUP_PID
    lw.set_locks(False)
    lv = lw.probe({**FLIGHT}, sleep=table_whole_again)             # mid-flight, no heartbeat
    assert (lv.verdict, lv.pid, lv.loud) == ("ALIVE", SUP_PID, False)


def test_a_lock_nobody_holds_is_read_reads_times_before_it_is_believed(lw):
    assert liveness.lock_holder(lw.lock, lw.proc, reads=3, gap_s=0.2, sleep=lw.sleep) is None
    assert lw.sleeps == [0.2, 0.2]


def test_an_unreadable_locks_table_is_unknown_never_dead(lw):
    os.remove(os.path.join(lw.proc, "locks"))
    lv = lw.probe({**FLIGHT})
    assert (lv.verdict, lv.loud, lv.final) == ("UNKNOWN", False, False)


# -- a fresh heartbeat overrides a missed lock read ---------------------------------------------------------------------

def test_a_missed_lock_read_with_a_fresh_heartbeat_and_a_live_pid_reads_alive(lw):
    """The 12:43:42 incident: the lock row is missing on every read, the supervisor's pid is alive (same start ticks) and
    its beat is 18 s old, inside its declared deadline."""
    lv = lw.probe(lw.state(beat_age=18.0))
    assert (lv.verdict, lv.pid, lv.loud, lv.final) == ("ALIVE", SUP_PID, False, False)
    assert "18 s ago" in lv.detail and "transient miss" in lv.detail
    assert liveness.Confirm(0.0).confirmed(lv, T0) is False


def test_a_fresh_heartbeat_does_not_save_a_pid_that_is_gone(lw):
    """NEVER suppress a real death: the heartbeat is fresh because the supervisor was killed 5 s ago."""
    lw.kill_supervisor()
    lv = lw.probe(lw.state(beat_age=5.0))
    assert (lv.verdict, lv.loud, lv.final) == ("DEAD", True, True) and "is mid-flight" in lv.detail


def test_a_fresh_heartbeat_does_not_save_a_re_used_pid(lw):
    lv = lw.probe(lw.state(beat_age=5.0, start_ticks=SUP_START_S * procfs.HZ + 7))
    assert (lv.verdict, lv.loud, lv.final) == ("DEAD", True, True)


def test_a_zombie_supervisor_is_gone(lw):
    stat = os.path.join(lw.proc, str(SUP_PID), "stat")
    text = open(stat).read().replace(f"{SUP_PID} (python) S ", f"{SUP_PID} (python) Z ")
    open(stat, "w").write(text)
    lv = lw.probe(lw.state(beat_age=5.0))
    assert (lv.verdict, lv.loud, lv.final) == ("DEAD", True, True)


def test_a_supervisor_that_recorded_its_exit_is_not_vouched_for_by_its_beat(lw):
    lv = lw.probe(lw.state(beat_age=5.0, exit="died: OSError"))
    assert (lv.verdict, lv.loud, lv.final) == ("DEAD", True, True) and "it exited died: OSError" in lv.detail


# -- a death that is certain pages at once; every other one needs two probes -----------------------------------------------

def test_a_dead_pid_is_dead_at_once_on_the_first_probe(lw):
    lw.kill_supervisor()
    lv = lw.probe(lw.state(beat_age=60.0))
    conf = liveness.Confirm()
    assert lv.final and conf.confirmed(lv, T0) is True


def test_a_proc_read_that_fails_is_not_a_death(lw):
    """EMFILE / ENOMEM / EACCES under load: ``/proc/<pid>/stat`` unreadable is ``unknown``, not ``gone``. It neither
    vouches for the supervisor nor pronounces it dead, so it needs the two probes."""
    stat = os.path.join(lw.proc, str(SUP_PID), "stat")
    os.remove(stat)
    os.mkdir(stat)                                                 # open() raises IsADirectoryError, the entry exists
    lv = lw.probe(lw.state(beat_age=5.0))
    assert (lv.verdict, lv.loud, lv.final) == ("DEAD", True, False)
    conf = liveness.Confirm()
    assert conf.confirmed(lv, T0) is False and conf.confirmed(lv, T0 + 10.0) is True


def test_two_consecutive_misses_with_a_stale_heartbeat_alert_and_one_does_not(lw):
    """The pid is alive but its beat is past its declared deadline and nobody holds the lock: DEAD, on the second probe
    10 s later, not the first."""
    state = lw.state(beat_age=2000.0, bound=600.0)
    lv1, lv2 = lw.probe(state, now=T0), lw.probe(state, now=T0 + 10.0)
    assert (lv1.verdict, lv1.loud, lv1.final) == ("DEAD", True, False) and lv1.key == lv2.key
    conf = liveness.Confirm()
    assert conf.confirmed(lv1, T0) is False
    assert conf.confirmed(lv2, T0 + 10.0) is True


def test_the_second_probe_must_be_at_least_the_gap_after_the_first(lw):
    state = lw.state(beat_age=2000.0)
    lv = lw.probe(state)
    conf = liveness.Confirm(10.0)
    assert [conf.confirmed(lv, T0 + t) for t in (0.0, 4.0, 9.9, 10.0)] == [False, False, False, True]


def test_a_quiet_probe_in_between_starts_the_count_again(lw):
    stale = lw.probe(lw.state(beat_age=2000.0))                    # DEAD, loud, not certain
    quiet = lw.probe(lw.state(beat_age=5.0), now=T0 + 10.0)        # a fresh beat: ALIVE
    assert (stale.loud, quiet.loud) == (True, False)
    conf = liveness.Confirm(10.0)
    assert conf.confirmed(stale, T0) is False
    assert conf.confirmed(quiet, T0 + 10.0) is False
    assert conf.confirmed(stale, T0 + 20.0) is False               # the first probe of a new run of two
    assert conf.confirmed(stale, T0 + 30.0) is True


def test_a_stale_heartbeat_needs_two_probes_and_a_beat_in_between_cancels_it(lw):
    lw.set_locks(True)
    old = lw.probe(lw.state(beat_age=2000.0, bound=600.0))
    assert (old.verdict, old.loud, old.final) == ("STALE", True, False)
    conf = liveness.Confirm(10.0)
    assert conf.confirmed(old, T0) is False
    fresh_beat = lw.probe(lw.state(beat_age=1.0, bound=600.0), now=T0 + 10.0)
    assert fresh_beat.verdict == "ALIVE" and conf.confirmed(fresh_beat, T0 + 10.0) is False
    assert conf.confirmed(old, T0 + 20.0) is False                 # the old STALE case starts over
    assert conf.confirmed(old, T0 + 30.0) is True


# -- status: probe_confirmed -----------------------------------------------------------------------------------------------

def _confirmed(lw: LockWorld, state: Dict[str, Any], on_sleep=None, reload=None) -> liveness.Liveness:
    clock = {"t": T0}

    def sleep(s: float) -> None:
        if s >= liveness.CONFIRM_GAP_S:
            clock["t"] += s
            if on_sleep is not None:
                on_sleep()
        lw.sleep(s)
    return liveness.probe_confirmed(state, lw.lock, lambda: clock["t"], lw.proc, reload=reload, sleep=sleep)


def test_status_re_probes_a_loud_verdict_and_reports_the_second(lw):
    """A first probe that read DEAD, with the lock back by the second: ALIVE, with a note saying so."""
    state = lw.state(beat_age=2000.0)
    lv = _confirmed(lw, state, on_sleep=lambda: lw.set_locks(True), reload=lambda: lw.state(beat_age=1.0))
    assert (lv.verdict, lv.loud) == ("ALIVE", False) and "a first probe read DEAD" in lv.detail
    assert liveness.CONFIRM_GAP_S in lw.sleeps


def test_status_keeps_a_verdict_that_held_on_both_probes(lw):
    lv = _confirmed(lw, lw.state(beat_age=2000.0))
    assert (lv.verdict, lv.loud) == ("DEAD", True) and liveness.CONFIRM_GAP_S in lw.sleeps


def test_status_does_not_wait_for_a_certain_death(lw):
    lw.kill_supervisor()
    lv = _confirmed(lw, lw.state(beat_age=5.0))
    assert (lv.verdict, lv.final) == ("DEAD", True) and liveness.CONFIRM_GAP_S not in lw.sleeps


def test_status_does_not_wait_on_a_quiet_verdict(lw):
    lw.set_locks(True)
    lv = _confirmed(lw, lw.state(beat_age=5.0))
    assert (lv.verdict, lv.loud) == ("ALIVE", False) and liveness.CONFIRM_GAP_S not in lw.sleeps
