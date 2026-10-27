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
"""
from __future__ import annotations

import datetime as dt
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from utils import procfs

from main.ops.arm_chain.procwatch import ProcRef, alive, ref_from_pidfile

#: added to every declared bound: covers the unwrapped short steps (git, the kit probe, a CPU-gate sample)
BEAT_SLACK_S = 600.0
#: a beat that changes nothing but the time is persisted at most this often
BEAT_EVERY_S = 30.0
TERMINAL = ("DONE", "DONE_FAILED")


def _iso(t: float) -> str:
    return dt.datetime.fromtimestamp(t).astimezone().isoformat(timespec="seconds")


def lock_holder(lock_path: str, proc: str = "/proc") -> Optional[int]:
    """The pid holding a ``flock`` on ``lock_path`` (and still holding it open), None if nobody does. Raises
    ``OSError`` when ``/proc/locks`` cannot be read (the caller says UNKNOWN, never DEAD)."""
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

    def line(self) -> str:
        head = f"supervisor: {self.verdict}" + (f" pid {self.pid}" if self.pid else "")
        return ("!!! " if self.loud else "") + f"{head} — {self.detail}"


def probe(state: Mapping[str, Any], lock_path: str, now: float, proc: str = "/proc",
          launcher_pidfiles: Optional[Mapping[str, str]] = None) -> Liveness:
    sup: Dict[str, Any] = dict(state.get("supervisor") or {})
    try:
        holder = lock_holder(lock_path, proc)
    except OSError as e:
        return Liveness("UNKNOWN", None, f"cannot read {proc}/locks ({e})", False, "unknown")
    flight = mid_flight(state)
    if holder is not None:
        if sup.get("pid") != holder:
            return Liveness("ALIVE", holder, "holds the chain's lock; it writes no heartbeat (a supervisor older than "
                                             "the heartbeat, or one that has not beaten yet)", False, f"alive:{holder}")
        ref = ProcRef(int(sup["pid"]), int(sup.get("start_ticks", -1)))
        beat = float(sup.get("last_beat_epoch") or 0)
        until = float(sup.get("stale_after_epoch") or 0)
        doing = str(sup.get("doing") or "?")
        code = f", code {sup['code']}" if sup.get("code") else ""
        if not alive(ref, proc):
            return Liveness("ALIVE", holder, f"holds the chain's lock; its heartbeat names a different process "
                                             f"(start ticks {ref.start_ticks}){code}", False, f"alive:{holder}")
        if now > until:
            return Liveness("STALE", holder, f"last beat {(now - beat) / 60:.0f} min ago ({_iso(beat)}), past its own "
                                             f"deadline {_iso(until)} while '{doing}'{code}: wedged? (it still holds "
                                             "the lock)", True, f"stale:{holder}:{ref.start_ticks}:{int(until)}")
        return Liveness("ALIVE", holder, f"last beat {(now - beat):.0f} s ago, '{doing}'{code}", False,
                        f"alive:{holder}")
    last = (f"; its last beat {_iso(float(sup['last_beat_epoch']))} (pid {sup.get('pid')}, '{sup.get('doing')}')"
            if sup.get("last_beat_epoch") else "")
    ended = f"; it exited {sup['exit']} at {sup.get('exited_at')}" if sup.get("exit") is not None else ""
    if not flight:
        return Liveness("DEAD", None, f"no supervisor holds the lock (nothing is mid-flight){ended}{last}", False,
                        f"dead-idle:{sup.get('pid')}:{sup.get('start_ticks')}")
    lives = []
    for key, pf in (launcher_pidfiles or {}).items():
        lref = ref_from_pidfile(pf, proc)
        if lref is not None and "main.launcher" in procfs.cmdline(lref.pid, proc):
            lives.append(f"{key}'s launcher pid {lref.pid} is ALIVE")
    return Liveness("DEAD", None, f"NO supervisor while {', '.join(flight)} is mid-flight"
                                  + (f" ({'; '.join(lives)})" if lives else "")
                    + f"{ended}{last}: nobody hands off at its END — restart the supervisor", True,
                    f"dead:{sup.get('pid')}:{sup.get('start_ticks')}:{','.join(flight)}")
