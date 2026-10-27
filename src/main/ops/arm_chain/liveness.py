"""Is the chain's supervisor alive? Its lock, its heartbeat and its identity, read without touching either.

THE CLASS BUG (review 2026-10-26, finding 2): nothing said whether a supervisor was running. ``status`` showed phases
only, so an OOM-killed or torn-down supervisor read ``RUNNING`` while its run trained to an END nobody handed off,
and the GPU idled until a human looked.

Three readings, none of which takes the lock (a ``status`` that flocked, even for a moment, could make a supervisor
starting at that instant exit 4):

* **the lock holder**: the ``FLOCK`` entry on ``<state>.lock``'s device and inode in ``/proc/locks`` whose pid still
  has that file open (the supervisor holds the fd for its whole life). Every supervisor version takes this lock, so
  it also sees a supervisor older than the heartbeat (``93a05fe91``, which writes only its pid into the lock file);
* **the heartbeat**: ``state["supervisor"]`` (pid, start ticks, ``last_beat``, and ``stale_after_epoch``: the beat
  BEFORE each blocking step declares how long that step may take, plus :data:`BEAT_SLACK_S`), rewritten by the
  supervisor's main thread, so a wedged loop goes STALE where a liveness thread would keep beating;
* **mid-flight**: an item started, not terminal and not held: the case where a dead supervisor idles the GPU.

Verdicts: ``ALIVE``, ``STALE`` (the lock is held but the heartbeat is past its own deadline), ``DEAD`` (no lock
holder), ``UNKNOWN`` (``/proc/locks`` unreadable).

A SECOND CLASS BUG (10-27 12:43:42): ONE probe read DEAD and the monitor paged "SUPERVISOR DEAD" about a supervisor that
was alive (same start ticks, a beat 18 s old, the lock file untouched; ``status`` read ALIVE a moment later). The kernel
builds ``/proc/locks`` while other processes take and drop locks (this box runs ``gpu_lock``, ``gate_lock`` and every
test's flocks), so one read can skip a row that was there all along, and the holder's ``/proc/<pid>/fd`` walk can fail the
same way. A single read of a racy file must not pronounce a death. So:

* **a "nobody holds it" answer is re-read** :data:`LOCK_READS` times, :data:`LOCK_READ_GAP_S` apart, before it is believed;
* **a fresh heartbeat overrides a missed lock read**: the heartbeat's own pid is alive (same start ticks), the
  supervisor has not recorded an ``exit``, and now is not past the heartbeat's declared deadline;
* **DEAD / STALE must hold on two probes at least** :data:`CONFIRM_GAP_S` **apart** (:class:`Confirm` for the monitor's
  passes, :func:`probe_confirmed` for ``status``), the one exception being a death that is certain:
* **a supervisor whose pid is gone from ``/proc`` (or that recorded its own ``exit``) is DEAD at once**
  (:attr:`Liveness.final`). "Gone" is a definite answer only (ENOENT / ESRCH, a zombie, other start ticks): a ``/proc``
  read that fails any other way under load is ``unknown`` and can neither vouch for a supervisor nor pronounce it dead.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional

from utils import procfs

from main.ops.arm_chain.procwatch import ProcRef, alive, ref_from_pidfile

#: added to every declared bound: covers the unwrapped short steps (git, the kit probe, a CPU-gate sample)
BEAT_SLACK_S = 600.0
#: a beat that changes nothing but the time is persisted at most this often
BEAT_EVERY_S = 30.0
TERMINAL = ("DONE", "DONE_FAILED")
#: ``/proc/locks`` is a seq_file walked while other processes take and drop locks: one read can skip a live row
#: (10-27 12:43:42). A "nobody holds it" answer is read this many times, :data:`LOCK_READ_GAP_S` apart, before it stands.
LOCK_READS = 3
LOCK_READ_GAP_S = 0.2
#: a loud DEAD / STALE that is not already certain must hold on :data:`CONFIRM_PROBES` probes at least this far apart
CONFIRM_GAP_S = 10.0
CONFIRM_PROBES = 2


def _iso(t: float) -> str:
    return dt.datetime.fromtimestamp(t).astimezone().isoformat(timespec="seconds")


def _lock_holder_once(lock_path: str, proc: str) -> Optional[int]:
    """One read of ``/proc/locks`` and of the holder's ``/proc/<pid>/fd``. Raises ``OSError`` when ``/proc/locks``
    cannot be read."""
    try:
        st = os.stat(lock_path)
    except FileNotFoundError:
        return None
    key = f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}"
    with open(os.path.join(proc, "locks")) as f:
        rows = f.read().splitlines()
    real = os.path.realpath(lock_path)
    for row in rows:
        parts = row.split()
        if len(parts) < 6 or parts[1] != "FLOCK" or parts[5] != key:
            continue
        try:
            pid = int(parts[4])
        except ValueError:
            continue
        fd_dir = os.path.join(proc, str(pid), "fd")
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue
        for fd in fds:
            try:
                if os.path.realpath(os.readlink(os.path.join(fd_dir, fd))) == real:
                    return pid
            except OSError:
                continue
    return None


def lock_holder(lock_path: str, proc: str = "/proc", reads: int = LOCK_READS, gap_s: float = LOCK_READ_GAP_S,
                sleep: Callable[[float], None] = time.sleep) -> Optional[int]:
    """The pid holding a ``flock`` on ``lock_path`` (and still holding it open), None if nobody does on ANY of
    ``reads`` reads ``gap_s`` apart (a holder is returned at the first read that sees one). Raises ``OSError`` when
    ``/proc/locks`` could not be read and no read found a holder (the caller says UNKNOWN, never DEAD)."""
    if not os.path.exists(lock_path):
        return None
    failed: Optional[OSError] = None
    for i in range(max(1, reads)):
        if i:
            sleep(gap_s)
        try:
            pid = _lock_holder_once(lock_path, proc)
        except OSError as e:
            failed = e
            continue
        if pid is not None:
            return pid
    if failed is not None:
        raise failed
    return None


def _heartbeat_ref(sup: Mapping[str, Any]) -> Optional[ProcRef]:
    try:
        return ProcRef(int(sup["pid"]), int(sup["start_ticks"]))
    except (KeyError, TypeError, ValueError):
        return None


def _pid_state(ref: ProcRef, proc: str) -> str:
    """``alive`` | ``gone`` | ``unknown`` for the process the heartbeat names. ``gone`` is a DEFINITE answer only: the
    process's ``/proc`` entry is absent, a zombie, or has other start ticks (a re-used pid). A read that fails any
    other way (EMFILE, ENOMEM, EACCES under load) is ``unknown``: it neither vouches for a supervisor nor pronounces
    it dead. (Start ticks < 0 are a supervisor that could not read its own: the pid alone identifies it.)"""
    st = procfs.read_stat(ref.pid, proc)
    if st is not None:
        same = ref.start_ticks < 0 or st.starttime == ref.start_ticks
        return "alive" if st.state not in ("Z", "X") and same else "gone"
    try:
        os.stat(os.path.join(proc, str(ref.pid)))
    except FileNotFoundError:
        return "gone"
    except OSError:
        return "unknown"
    return "unknown"


def mid_flight(state: Mapping[str, Any]) -> List[str]:
    """``<key> <phase>`` of each started item that is neither terminal nor held."""
    if not state.get("history"):
        return []
    return [f"{k} {r['phase']}" for k, r in state.get("items", {}).items()
            if r["phase"] not in TERMINAL and not r.get("hold") and (r["phase"] != "PENDING" or _started(state, k))]


def _started(state: Mapping[str, Any], key: str) -> bool:
    """A PENDING item counts once the chain reached it: every item before it is terminal."""
    for k, r in state.get("items", {}).items():
        if k == key:
            return True
        if r["phase"] not in TERMINAL:
            return False
    return False


@dataclass(frozen=True)
class Liveness:
    verdict: str                 # ALIVE | STALE | DEAD | UNKNOWN
    pid: Optional[int]
    detail: str
    loud: bool                   # DEAD with an item mid-flight, or STALE: the GPU may idle unpaged
    key: str                     # one alert per supervisor instance and verdict
    final: bool = False          # a death that is certain (pid gone from /proc, or its own recorded exit): no debounce

    def line(self) -> str:
        head = f"supervisor: {self.verdict}" + (f" pid {self.pid}" if self.pid else "")
        return ("!!! " if self.loud else "") + f"{head} — {self.detail}"


def probe(state: Mapping[str, Any], lock_path: str, now: float, proc: str = "/proc",
          launcher_pidfiles: Optional[Mapping[str, str]] = None, *, reads: int = LOCK_READS,
          read_gap_s: float = LOCK_READ_GAP_S, sleep: Callable[[float], None] = time.sleep) -> Liveness:
    """One probe: the lock (re-read when it says nobody), then the heartbeat. A loud DEAD / STALE from ONE probe is
    still only a candidate: :class:`Confirm` / :func:`probe_confirmed` ask for a second, unless ``final`` is set."""
    sup: Dict[str, Any] = dict(state.get("supervisor") or {})
    try:
        holder = lock_holder(lock_path, proc, reads, read_gap_s, sleep)
    except OSError as e:
        return Liveness("UNKNOWN", None, f"cannot read {proc}/locks ({e})", False, "unknown")
    flight = mid_flight(state)
    beat = float(sup.get("last_beat_epoch") or 0)
    until = float(sup.get("stale_after_epoch") or 0)
    doing = str(sup.get("doing") or "?")
    code = f", code {sup['code']}" if sup.get("code") else ""
    if holder is not None:
        if sup.get("pid") != holder:
            return Liveness("ALIVE", holder, "holds the chain's lock; it writes no heartbeat (a supervisor older than "
                                             "the heartbeat, or one that has not beaten yet)", False, f"alive:{holder}")
        ref = ProcRef(int(sup["pid"]), int(sup.get("start_ticks", -1)))
        if not alive(ref, proc):
            return Liveness("ALIVE", holder, f"holds the chain's lock; its heartbeat names a different process "
                                             f"(start ticks {ref.start_ticks}){code}", False, f"alive:{holder}")
        if now > until:
            return Liveness("STALE", holder, f"last beat {(now - beat) / 60:.0f} min ago ({_iso(beat)}), past its own "
                                             f"deadline {_iso(until)} while '{doing}'{code}: wedged? (it still holds "
                                             "the lock)", True, f"stale:{holder}:{ref.start_ticks}:{int(until)}")
        return Liveness("ALIVE", holder, f"last beat {(now - beat):.0f} s ago, '{doing}'{code}", False,
                        f"alive:{holder}")
    # nobody holds the lock on every read. Ask the heartbeat's own process before believing it.
    hb = _heartbeat_ref(sup)
    pst = _pid_state(hb, proc) if hb is not None else "none"
    exited = sup.get("exit") is not None
    if hb is not None and pst == "alive" and not exited and now <= until:
        return Liveness("ALIVE", hb.pid, f"last beat {(now - beat):.0f} s ago, '{doing}'{code}: its pid is alive and the "
                                         f"beat is inside its declared deadline {_iso(until)}, so the {reads} "
                                         f"{proc}/locks reads that showed no holder were a transient miss", False,
                        f"alive:{hb.pid}")
    final = exited or pst == "gone"
    last = (f"; its last beat {_iso(beat)} (pid {sup.get('pid')}, '{sup.get('doing')}')"
            if sup.get("last_beat_epoch") else "")
    ended = f"; it exited {sup['exit']} at {sup.get('exited_at')}" if exited else ""
    if not flight:
        return Liveness("DEAD", None, f"no supervisor holds the lock (nothing is mid-flight){ended}{last}", False,
                        f"dead-idle:{sup.get('pid')}:{sup.get('start_ticks')}", final)
    lives = []
    for key, pf in (launcher_pidfiles or {}).items():
        lref = ref_from_pidfile(pf, proc)
        if lref is not None and "main.launcher" in procfs.cmdline(lref.pid, proc):
            lives.append(f"{key}'s launcher pid {lref.pid} is ALIVE")
    return Liveness("DEAD", None, f"NO supervisor while {', '.join(flight)} is mid-flight"
                                  + (f" ({'; '.join(lives)})" if lives else "")
                    + f"{ended}{last}: nobody hands off at its END — restart the supervisor", True,
                    f"dead:{sup.get('pid')}:{sup.get('start_ticks')}:{','.join(flight)}", final)


def probe_confirmed(state: Mapping[str, Any], lock_path: str, now: Callable[[], float], proc: str = "/proc",
                    launcher_pidfiles: Optional[Mapping[str, str]] = None, *,
                    reload: Optional[Callable[[], Mapping[str, Any]]] = None, gap_s: float = CONFIRM_GAP_S,
                    sleep: Callable[[float], None] = time.sleep, reads: int = LOCK_READS,
                    read_gap_s: float = LOCK_READ_GAP_S) -> Liveness:
    """``status``'s probe: a loud verdict that is not certain is probed AGAIN ``gap_s`` later (``reload`` re-reads the
    state, so a beat that landed in between is seen); only a verdict that held on both stands, and a first loud
    reading the second did not repeat is returned as the second one, with a note. A certain death (``final``) and
    every quiet verdict are one probe."""
    kw: Dict[str, Any] = dict(reads=reads, read_gap_s=read_gap_s, sleep=sleep)
    lv = probe(state, lock_path, now(), proc, launcher_pidfiles, **kw)
    if not lv.loud or lv.final:
        return lv
    sleep(gap_s)
    again = probe(reload() if reload is not None else state, lock_path, now(), proc, launcher_pidfiles, **kw)
    if again.loud:
        return again
    return dataclasses.replace(again, detail=f"{again.detail} [a first probe read {lv.verdict}, which did not hold "
                                             f"{gap_s:g} s later: a transient read]")


class Confirm:
    """The monitor's debounce across its passes: a loud verdict is acted on only once the SAME case (the verdict's
    ``key``) has been read on :data:`CONFIRM_PROBES` consecutive probes whose first and last are at least ``gap_s``
    apart. A quiet or different probe in between starts the count again. A ``final`` verdict (a certain death) is
    acted on at once."""

    def __init__(self, gap_s: float = CONFIRM_GAP_S, probes: int = CONFIRM_PROBES):
        self.gap_s, self.probes = gap_s, probes
        self._key: Optional[str] = None
        self._first = 0.0
        self._n = 0

    def confirmed(self, lv: Liveness, now: float) -> bool:
        if not lv.loud:
            self._key, self._n = None, 0
            return False
        if lv.final:
            return True
        if lv.key != self._key:
            self._key, self._first, self._n = lv.key, now, 0
        self._n += 1
        return self._n >= self.probes and now - self._first >= self.gap_s
