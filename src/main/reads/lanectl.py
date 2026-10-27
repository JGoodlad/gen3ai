"""LANE CONTROL: a supervisor's own pid file, and the hold / release of a lane (``designs/tools/reads.md`` §3).

The first live reads (belief-prior-mix 71c76dfb FINDINGS 2 and 5) found two rough edges in how an operator drives a lane:

* ``setsid nohup python -m main.reads run ... &`` leaves ``$!`` naming the ``setsid`` wrapper, gone within seconds, so a
  watcher armed on it watched nothing. :func:`write_pidfile` has the supervisor write ITS OWN pid and start ticks to
  ``<state>/<lane>/supervisor.pid`` once it holds the lane; :func:`pid_status` reads it back (checked against
  ``/proc``: the pid must be live and carry the recorded start ticks, so a reused pid is not mistaken for it); ``status``
  and the ``pid`` verb print it. The file is removed when the supervisor ends (a killed one leaves it, and
  :func:`pid_status` then reports it ``alive: false``).
* A HOLD was killing the supervisor by pid by hand. :func:`hold` writes ``<state>/<lane>/hold.json`` and appends to
  ``<state>/holds.jsonl`` (APPEND-ONLY); the lane's supervisor reads the file at the top of every loop turn, starts no
  new unit, waits for the units in flight, logs ``held`` and exits 0. ``run`` refuses a held lane; :func:`release`
  (the ``resume`` verb's first step) clears the hold, and only once no supervisor of the lane is still alive (one that is
  would be draining; a second supervisor is refused by the path lock anyway).

A supervisor of OLDER code writes no pid file and never reads the hold: :func:`hold` reports which live supervisors
honor it (those with a valid pid file) and which do not, so an operator is never told a lane is held when it is not.

This module is OFF the play path (``main.reads.playpath``): a repin to a commit that only changes it is accepted.
"""
from __future__ import annotations

import fcntl
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from main.reads import layout as LY
from main.reads import supervisors as SV
from utils import procfs

PID_SCHEMA = "gen3_read_supervisor_pid_v1"
HOLD_SCHEMA = "gen3_read_hold_v1"
LANES = SV.LANES
#: How often a waiting supervisor looks at its hold file, s (a supervisor mid-unit looks between units).
HOLD_POLL_S = 30.0
#: The longest ``pid --wait`` waits, s.
MAX_WAIT_S = 600.0


class LaneError(RuntimeError):
    """A lane verb that is refused (named; nothing is changed)."""


def _now_iso(t: Optional[float] = None) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(time.time() if t is None else t))


def _atomic(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def _check_lane(lane: str) -> None:
    if lane not in LANES:
        raise LaneError(f"lane {lane!r} is none of {LANES}")


# ------------------------------------------------------------------------------------------------ the pid file
def write_pidfile(state: Path, lane: str, *, stop_epoch: float, pid: Optional[int] = None) -> Dict[str, Any]:
    """The supervisor's own pid and start ticks (``/proc/<pid>/stat`` field 22) at ``<state>/<lane>/supervisor.pid``."""
    _check_lane(lane)
    me = os.getpid() if pid is None else int(pid)
    st = procfs.read_stat(me)
    if st is None:
        raise LaneError(f"pid {me}: no /proc entry; cannot record the supervisor's start ticks")
    doc = {"schema": PID_SCHEMA, "pid": me, "start_ticks": st.starttime, "lane": lane, "state": SV.canonical(state),
           "stop_epoch": int(stop_epoch), "started": _now_iso()}
    _atomic(LY.supervisor_pid_file(state, lane), doc)
    return doc


def read_pidfile(state: Path, lane: str) -> Optional[Dict[str, Any]]:
    """The pid file's document, ``None`` when there is none; a damaged one is a :class:`LaneError`."""
    p = LY.supervisor_pid_file(state, lane)
    try:
        doc = json.loads(p.read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise LaneError(f"{p}: damaged supervisor pid file ({exc})") from None
    if not (isinstance(doc, dict) and isinstance(doc.get("pid"), int) and isinstance(doc.get("start_ticks"), int)):
        raise LaneError(f"{p}: damaged supervisor pid file (no integer pid / start_ticks)")
    return doc


def pid_status(state: Path, lane: str, *, proc: str = "/proc") -> Optional[Dict[str, Any]]:
    """What a watcher needs: ``{pid, start_ticks, alive, stale, stop_epoch, started}`` of the lane's recorded supervisor, or
    ``None`` when no pid file exists. ``alive`` = that pid is a running process with exactly the recorded start ticks
    (read from ``/proc``, never ``kill -0``); ``stale`` = the file outlived its supervisor."""
    try:
        doc = read_pidfile(state, lane)
    except LaneError as exc:
        return {"error": str(exc), "alive": False, "stale": True}
    if doc is None:
        return None
    st = procfs.read_stat(doc["pid"], proc)
    alive = st is not None and st.state not in ("Z", "X") and st.starttime == doc["start_ticks"]
    return {"pid": doc["pid"], "start_ticks": doc["start_ticks"], "alive": alive, "stale": not alive,
            "stop_epoch": doc.get("stop_epoch"), "started": doc.get("started")}


def live_pid(state: Path, lane: str, *, proc: str = "/proc") -> Optional[int]:
    s = pid_status(state, lane, proc=proc)
    return int(s["pid"]) if s and s.get("alive") else None


def remove_pidfile(state: Path, lane: str, *, pid: Optional[int] = None) -> bool:
    """Remove the lane's pid file if it names ``pid`` (default: this process); a successor's file is never removed."""
    try:
        doc = read_pidfile(state, lane)
        if doc is None or doc["pid"] != (os.getpid() if pid is None else int(pid)):
            return False
        LY.supervisor_pid_file(state, lane).unlink()
        return True
    except (LaneError, OSError):
        return False


def wait_pid(state: Path, lane: str, wait_s: float, *, sleep: Callable[[float], None] = time.sleep,
             clock: Callable[[], float] = time.monotonic) -> int:
    """The lane's live supervisor pid, waiting up to ``wait_s`` seconds (a supervisor takes seconds to import and lock);
    a refusal when none is alive by then."""
    if not 0 <= wait_s <= MAX_WAIT_S:
        raise LaneError(f"--wait {wait_s!r}: 0 to {MAX_WAIT_S:g} seconds (a wait is always bounded)")
    end = clock() + wait_s
    while True:
        pid = live_pid(state, lane)
        if pid is not None:
            return pid
        if clock() >= end:
            raise LaneError(f"{lane}: no live supervisor has written {LY.supervisor_pid_file(state, lane)} "
                            f"(waited {wait_s:g} s)")
        sleep(min(1.0, max(0.0, end - clock())))


# ------------------------------------------------------------------------------------------------ the hold
def read_hold(state: Path, lane: str) -> Optional[Dict[str, Any]]:
    """The lane's hold (``by``, ``why``, ``time``), ``None`` when it is not held. A damaged hold file is a refusal, not a
    free lane: a guard that skipped it would be a guard that guessed."""
    p = LY.hold_file(state, lane)
    try:
        doc = json.loads(p.read_text())
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        raise LaneError(f"{p}: damaged hold file ({exc}); the lane is treated as held until it is fixed") from None
    if not (isinstance(doc, dict) and doc.get("by") and doc.get("why")):
        raise LaneError(f"{p}: damaged hold file (no by / why); the lane is treated as held until it is fixed")
    return doc


def refuse_if_held(state: Path, lane: str) -> None:
    h = read_hold(state, lane)
    if h is not None:
        raise LaneError(f"the {lane} lane is HELD by {h['by']} since {h.get('time')} ({h['why']}): "
                        f"`python -m main.reads resume --state {state} --lane {lane} --hours H --by WHO` releases the "
                        "hold and restarts the supervisor")


def wait_unless_held(state: Path, lane: str, seconds: float, sleep: Callable[[float], None]) -> None:
    """Sleep ``seconds`` in :data:`HOLD_POLL_S` steps, returning early as soon as the lane is held."""
    left = float(seconds)
    while left > 0:
        step = min(HOLD_POLL_S, left)
        sleep(step)
        left -= step
        if read_hold(state, lane) is not None:
            return


def _log(state: Path, ev: Dict[str, Any]) -> None:
    fd = os.open(LY.holds_file(state), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    try:
        os.write(fd, (json.dumps(ev, sort_keys=True, default=str) + "\n").encode())    # one write: whole lines only
        os.fsync(fd)
    finally:
        os.close(fd)


def load_holds(state: Path) -> List[Dict[str, Any]]:
    """Every hold / release event of ``state``, in order (``holds.jsonl``)."""
    p = LY.holds_file(state)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def _planned_state(state: Path, lane: str) -> Path:
    from main.reads.spec import SpecError, load as load_spec

    state = LY.check_state_dir(state)
    if not LY.spec_file(state).exists():
        raise LaneError(f"{state}: no spec.json (a hold works on a planned state)")
    try:
        spec = load_spec(LY.spec_file(state))
    except SpecError as exc:
        raise LaneError(f"{state}: its spec does not load ({exc})") from None
    if not (spec.anchor if lane == "anchor" else spec.s):
        raise LaneError(f"the spec has no {lane} group, so there is no {lane} lane to hold")
    return state


def supervisors_of(state: Path, lane: str) -> Dict[str, List[int]]:
    """``{live: [pids by the /proc scan], honoring: [those whose pid file names them], not_honoring: [the rest]}``. A
    supervisor without a valid pid file runs code that predates the hold, and will not stop for one."""
    live = SV.live_supervisors(state, lane=lane, exclude=SV.self_and_ancestors()).get(lane, [])
    mine = live_pid(state, lane)
    return {"live": live, "honoring": [p for p in live if p == mine], "not_honoring": [p for p in live if p != mine]}


def hold(state: Path, lane: str, *, by: str, why: str, now: Optional[float] = None) -> Dict[str, Any]:
    """HOLD ``lane``: its supervisor starts no new unit, finishes the ones in flight, logs ``held`` and exits 0; ``run``
    then refuses the lane until :func:`release`. Returns the hold, plus ``supervisors`` (:func:`supervisors_of`) and
    ``warnings``. A repeat on a held lane is a no-op (``already: True``) that keeps the first ``by`` / ``why``."""
    _check_lane(lane)
    if not (isinstance(by, str) and by.strip()):
        raise LaneError("--by: who is holding the lane (a name) is required")
    if not (isinstance(why, str) and why.strip()):
        raise LaneError("--why: the reason is required")
    state = _planned_state(state, lane)
    with open(LY.hold_lock(state), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        cur = read_hold(state, lane)
        t = time.time() if now is None else now
        doc = cur or {"schema": HOLD_SCHEMA, "lane": lane, "by": by.strip(), "why": why.strip(), "time": _now_iso(t),
                      "t": t}
        if cur is None:
            _atomic(LY.hold_file(state, lane), doc)
            _log(state, dict(doc, event="hold"))
        sup = supervisors_of(state, lane)
    warns = []
    if sup["not_honoring"]:
        warns.append(f"supervisor pid(s) {sup['not_honoring']} run code that predates the hold (no supervisor.pid): "
                     "they will NOT stop for it; stop them by pid, or let them reach their registered stop")
    if not sup["live"]:
        warns.append("no live supervisor: the hold only keeps `run` from starting one (`resume` releases it)")
    return dict(doc, held=True, already=cur is not None, supervisors=sup, warnings=warns)


def release(state: Path, lane: str, *, by: Optional[str] = None, why: Optional[str] = None,
            now: Optional[float] = None) -> Dict[str, Any]:
    """Clear ``lane``'s hold (``held: False, released: False`` when it was not held). A held lane is released only with
    ``by`` (attributed in ``holds.jsonl``) and only once its supervisor has ended: a live one is still draining, and
    clearing the hold under it would let a supervisor that has not looked yet carry on."""
    _check_lane(lane)
    state = LY.check_state_dir(state)
    if not LY.spec_file(state).exists():
        raise LaneError(f"{state}: no spec.json (a hold works on a planned state)")
    with open(LY.hold_lock(state), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        cur = read_hold(state, lane)
        if cur is None:
            return {"lane": lane, "held": False, "released": False}
        if not (isinstance(by, str) and by.strip()):
            raise LaneError(f"the {lane} lane is HELD by {cur['by']} ({cur['why']}): releasing it needs --by WHO")
        live = SV.live_supervisors(state, lane=lane, exclude=SV.self_and_ancestors()).get(lane, [])
        if live:
            raise LaneError(f"the {lane} lane's supervisor pid(s) {live} are still alive (draining the units in flight "
                            "after the hold): wait until they exit (/proc/<pid>), then resume")
        t = time.time() if now is None else now
        ev = {"schema": HOLD_SCHEMA, "event": "release", "lane": lane, "by": by.strip(),
              "why": (why or "resume").strip(), "time": _now_iso(t), "t": t, "held_by": cur["by"],
              "held_since": cur.get("time")}
        LY.hold_file(state, lane).unlink()
        _log(state, ev)
    return dict(ev, held=False, released=True)


# ------------------------------------------------------------------------------------------------ a registered stop
def supervisor_events(state: Path, lane: str) -> List[Dict[str, Any]]:
    p = LY.supervisor_log(state, lane)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def stop_evidence(state: Path, lane: str) -> Dict[str, Any]:
    """Proof that ``lane`` ended at its REGISTERED stop with work left: no supervisor of it is alive, and the last event of
    its log is ``stop_at`` (which a supervisor writes only once its ``--hours`` / ``--stop-at-epoch`` has passed). A lane that
    never ran, that was held, killed, refused or finished is not at a stop. Refuses otherwise."""
    _check_lane(lane)
    live = SV.live_supervisors(state, lane=lane, exclude=SV.self_and_ancestors()).get(lane, [])
    if live:
        raise LaneError(f"the {lane} lane has a live supervisor (pid {live}): its stop is not reached yet")
    evs = supervisor_events(state, lane)
    if not evs:
        raise LaneError(f"the {lane} lane has no supervisor log ({LY.supervisor_log(state, lane)}): it never ran")
    last = evs[-1]
    if last.get("event") != "stop_at":
        raise LaneError(f"the {lane} lane's last supervisor event is {last.get('event')!r}, not a registered stop "
                        "(`stop_at`): a read is recorded partial only at a stop, never after a hold, a kill or a refusal")
    start = next((e for e in reversed(evs) if e.get("event") == "start"), {})
    return {"event": "stop_at", "t": last.get("t"), "time": _now_iso(last["t"]) if last.get("t") else None,
            "left": last.get("left"), "stop_epoch": start.get("stop_epoch")}
