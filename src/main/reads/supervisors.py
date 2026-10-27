"""ONE SUPERVISOR PER LANE PER STATE, keyed by the state's PATH (``designs/tools/reads.md`` §3).

The D3 reader (2026-10-26) ran two anchor supervisors on one read: its state dir was renamed aside, the lock file
``<state>/<lane>.lock`` went with it, ``plan`` wrote a new state at the same path, and a second supervisor started
there while the first one lived (still writing through the old path's NAME). A lock INSIDE the state cannot see that.
So a supervisor is refused when either of two independent signals says another one owns the path:

* a flock on ``<lock dir>/<sha256(canonical path, lane)>.lock``, OUTSIDE the state (``~/.cache/gen3ai/reads_locks``;
  ``$GEN3AI_READS_LOCK_DIR`` overrides), written with the holder's pid;
* a ``/proc`` scan: a live process whose argv is ``... -m main.reads run|resume ... --state S --lane L`` with S
  (resolved against that process's own cwd) naming the same canonical path. This one also sees a supervisor started
  by older code, which never took the path lock.

The standalone Kakuna queue (``python -m main.x35.kakuna run --state S``) is a supervisor of the same kind, in the one
lane :data:`KAKUNA_LANE` (it has no ``--lane``): it takes the same path lock and runs the same scan, so two of them on
one kakuna state are refused, whatever became of the in-state ``supervisor.lock`` (N1-P unit, 2026-10-26).

A running supervisor also pins its state dir's IDENTITY (device, inode) at start: if the path stops naming that dir
(renamed aside, replaced), it stops instead of writing into whatever now sits at the path (:func:`check_same_dir`).

NESTED supervisors (the duplicate-play class by another route, N1-P / D3 look 1, 2026-10-26): the anchor lane of a reads
state ``S`` plays its units on the kakuna states ``S/anchor/<group>/<label>/``, so a standalone ``kakuna run`` on one of
them plays the SAME units, and neither path above names the other (different paths). Three claims close it:

* :class:`Hold` of a ``kakuna`` queue refuses when a live ``main.reads`` ANCHOR supervisor names a state that CONTAINS the
  queue's state (the ``/proc`` scan), or when an ancestor's anchor path lock is held (:func:`nested_supervisors`);
* :class:`Hold` of an ``anchor`` lane refuses when a live standalone ``kakuna run`` names a state INSIDE its own;
* the anchor lane holds every kakuna state it plays (:class:`SubStateClaims`): that state's ``kakuna``-lane path lock
  and its in-state ``supervisor.lock``, the two a ``kakuna run`` takes, so a queue started AFTER the lane (of this code or
  older) is refused by the lock it already honors, and a queue already on the state refuses the lane.

Liveness is read from ``/proc/<pid>`` only (``utils.procfs``), never ``kill -0`` (acf2193f: inside a sandboxed agent
shell ``kill -0`` read a live supervisor as dead).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from utils import procfs
from utils.filelock import try_exclusive

LOCK_DIR_ENV = "GEN3AI_READS_LOCK_DIR"
#: The verbs that ARE a lane supervisor (``s-unit`` is a child inside a heavy slot, not a supervisor).
SUPERVISOR_VERBS = ("run", "resume")
LANES = ("anchor", "s")
#: The standalone Kakuna queue: ``python -m main.x35.kakuna run --state S`` (no ``--lane``; one lane of its own).
KAKUNA_MODULE = "main.x35.kakuna"
KAKUNA_VERBS = ("run",)
KAKUNA_LANE = "kakuna"
#: ``{module: verbs}`` of every process that IS a supervisor.
_SUPERVISOR_MODULES = {"main.reads": SUPERVISOR_VERBS, KAKUNA_MODULE: KAKUNA_VERBS}


class SupervisorError(RuntimeError):
    """Another supervisor owns this state's path (named, with its pid)."""


def lock_dir() -> Path:
    return Path(os.environ.get(LOCK_DIR_ENV) or Path.home() / ".cache" / "gen3ai" / "reads_locks")


def canonical(state: "str | os.PathLike[str]", cwd: Optional[str] = None) -> str:
    """The state's canonical path: ``~`` expanded, made absolute against ``cwd`` (default ours), symlinks resolved."""
    p = os.path.expanduser(str(state))
    if not os.path.isabs(p):
        p = os.path.join(cwd or os.getcwd(), p)
    return os.path.realpath(p)


def lock_path(state: "str | os.PathLike[str]", lane: str) -> Path:
    key = hashlib.sha256(f"{canonical(state)}\0{lane}".encode()).hexdigest()[:24]
    return lock_dir() / f"{key}.lock"


def _holder(path: Path) -> str:
    try:
        d = json.loads(path.read_text() or "{}")
        return f"pid {d.get('pid')} (since {d.get('since')}{', ' + d['via'] if d.get('via') else ''})"
    except (OSError, ValueError):
        return "an unknown pid"


def acquire(state: "str | os.PathLike[str]", lane: str, *, via: Optional[str] = None) -> Any:
    """The path-keyed lane lock (held open for the supervisor's life; closing it releases it). ``via``: who holds it on
    the lane's behalf (the anchor lane holds its kakuna states' ``kakuna`` locks), written in the holder record."""
    p = lock_path(state, lane)
    p.parent.mkdir(parents=True, exist_ok=True)
    fh = try_exclusive(p, "a+")          # "a+": a refused caller reads the holder's record, so never truncate
    if fh is None:
        raise SupervisorError(f"another {lane} supervisor holds the path lock of {canonical(state)} "
                              f"({p}: {_holder(p)})")
    fh.seek(0)
    fh.truncate()
    fh.write(json.dumps({"pid": os.getpid(), "state": canonical(state), "lane": lane,
                         "since": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **({"via": via} if via else {})}) + "\n")
    fh.flush()
    return fh


def parse_argv(argv: Sequence[str]) -> Optional[Tuple[str, Optional[str], Optional[str]]]:
    """``(verb, state, lane)`` of a supervisor argv, else ``None``: ``main.reads run|resume`` (its ``--lane``) or
    ``main.x35.kakuna run`` (lane :data:`KAKUNA_LANE`). The ``-m <module>`` pair may sit anywhere (an interpreter flag
    or wrapper before it); ``--state X`` and ``--state=X`` both parse."""
    a = list(argv)
    for i in range(len(a) - 2):
        if a[i] == "-m" and a[i + 1] in _SUPERVISOR_MODULES:
            module, verb = a[i + 1], a[i + 2]
            if verb not in _SUPERVISOR_MODULES[module]:
                return None
            opts: Dict[str, Optional[str]] = {"--state": None, "--lane": None}
            rest = a[i + 3:]
            for j, tok in enumerate(rest):
                for k in opts:
                    if tok == k and j + 1 < len(rest):
                        opts[k] = rest[j + 1]
                    elif tok.startswith(k + "="):
                        opts[k] = tok[len(k) + 1:]
            return verb, opts["--state"], (KAKUNA_LANE if module == KAKUNA_MODULE else opts["--lane"])
    return None


def _scan(proc: str, exclude: Iterable[int]) -> List[Tuple[int, str, str]]:
    """``(pid, lane, canonical state path)`` of every LIVE supervisor in ``/proc`` (the argv's ``--state`` resolved
    against that process's own cwd), pids ascending. ``exclude``: pids to skip."""
    skip = set(exclude)
    out: List[Tuple[int, str, str]] = []
    for pid in sorted(procfs.all_pids(proc)):
        if pid in skip:
            continue
        raw = procfs.read_text(pid, "cmdline", proc)
        if not any(m in raw for m in _SUPERVISOR_MODULES):
            continue
        got = parse_argv([t for t in raw.split("\0") if t != ""])
        if got is None:
            continue
        _verb, st, ln = got
        if st is None:
            continue
        cwd = procfs.readlink(pid, "cwd", proc) or None
        if not procfs.pid_alive(pid, proc):
            continue
        out.append((pid, str(ln), canonical(st, cwd=cwd)))
    return out


def live_supervisors(state: "str | os.PathLike[str]", *, lane: Optional[str] = None, proc: str = "/proc",
                     exclude: Iterable[int] = ()) -> Dict[str, List[int]]:
    """``{lane: [pid, ...]}`` of every LIVE supervisor (``main.reads run|resume``, or the standalone
    ``main.x35.kakuna run`` as lane :data:`KAKUNA_LANE`) naming ``state``'s canonical path (each argv's ``--state``
    resolved against that process's own ``cwd``), from ``/proc`` alone. ``exclude``: pids to skip (the
    caller and its ancestors, whose argv may carry the same words)."""
    want = canonical(state)
    out: Dict[str, List[int]] = {}
    for pid, ln, st in _scan(proc, exclude):
        if st == want and (lane is None or ln == lane):
            out.setdefault(ln, []).append(pid)
    return out


def _within(child: str, parent: str) -> bool:
    """``child`` is a path strictly inside ``parent`` (both canonical)."""
    return child != parent and child.startswith(parent.rstrip(os.sep) + os.sep)


def nested_supervisors(state: "str | os.PathLike[str]", lane: str, *, proc: str = "/proc",
                       exclude: Iterable[int] = ()) -> List[Tuple[int, str, str]]:
    """The live supervisors that would play the SAME kakuna units as a ``lane`` supervisor at ``state``, by another path
    (module docstring, NESTED): ``(pid, lane, their canonical state)``. For a ``kakuna`` queue: the ``main.reads`` ANCHOR
    supervisors whose state CONTAINS ``state``. For an ``anchor`` lane: the standalone ``kakuna`` queues INSIDE
    ``state``. Empty for any other lane (the S lane plays no kakuna unit)."""
    me = canonical(state)
    if lane == KAKUNA_LANE:
        return [(p, ln, st) for p, ln, st in _scan(proc, exclude) if ln == "anchor" and _within(me, st)]
    if lane == "anchor":
        return [(p, ln, st) for p, ln, st in _scan(proc, exclude) if ln == KAKUNA_LANE and _within(st, me)]
    return []


def held_ancestor_locks(state: "str | os.PathLike[str]") -> List[Tuple[str, str]]:
    """``(ancestor path, holder)`` of every proper ancestor of ``state`` whose ANCHOR-lane path lock is held right now:
    a reads anchor supervisor on that ancestor, of THIS code (older code took no path lock; the ``/proc`` scan sees it).
    The lock is taken and released at once; a lock file that does not exist is not created."""
    out: List[Tuple[str, str]] = []
    for anc in Path(canonical(state)).parents:
        p = lock_path(anc, "anchor")
        if not p.exists():
            continue
        fh = try_exclusive(p, "a+")
        if fh is None:
            out.append((str(anc), _holder(p)))
        else:
            fh.close()
    return out


def self_and_ancestors() -> List[int]:
    me = os.getpid()
    return [me, *procfs.ancestors(me)]


def dir_identity(state: "str | os.PathLike[str]") -> Tuple[int, int]:
    st = os.stat(state)
    return st.st_dev, st.st_ino


def check_same_dir(state: "str | os.PathLike[str]", ident: Tuple[int, int]) -> None:
    """Refuse when ``state``'s path no longer names the dir the supervisor started on (renamed aside or replaced)."""
    try:
        now = dir_identity(state)
    except OSError as exc:
        raise SupervisorError(f"{state}: the state dir is gone from its path ({exc}); this supervisor stops") from None
    if now != ident:
        raise SupervisorError(f"{state}: the path now names ANOTHER dir (inode {now[1]}, started on {ident[1]}): "
                              "the state was moved or replaced under this supervisor, which stops")


class Hold:
    """Every claim a lane supervisor holds: the path lock, the ``/proc`` check passed, the dir identity, and (module
    docstring, NESTED) no supervisor that plays the same kakuna units from a state containing / inside this one."""

    def __init__(self, state: Path, lane: str) -> None:
        self.state, self.lane = state, lane
        self.fh = acquire(state, lane)
        try:
            others = live_supervisors(state, lane=lane, exclude=self_and_ancestors())
            if others:
                raise SupervisorError(f"a live {lane} supervisor already names {canonical(state)}: pid(s) "
                                      f"{others[lane]} (/proc scan)")
            nested = nested_supervisors(state, lane, exclude=self_and_ancestors())
            if nested:
                raise SupervisorError(_nested_message(state, lane, nested))
            if lane == KAKUNA_LANE:
                locked = held_ancestor_locks(state)
                if locked:
                    raise SupervisorError(
                        f"{canonical(state)} lies inside {locked[0][0]}, whose anchor lane holds its path lock "
                        f"({locked[0][1]}): that lane plays this state's units")
            self.ident = dir_identity(state)
        except Exception:
            self.fh.close()
            raise

    def check(self) -> None:
        check_same_dir(self.state, self.ident)

    def close(self) -> None:
        self.fh.close()


def _nested_message(state: "str | os.PathLike[str]", lane: str, nested: Sequence[Tuple[int, str, str]]) -> str:
    who = ", ".join(f"pid {p} ({ln} supervisor of {st})" for p, ln, st in nested)
    if lane == KAKUNA_LANE:
        return (f"{canonical(state)} lies inside a state whose reads ANCHOR lane plays its units: {who} (/proc scan). "
                "A second supervisor on a kakuna state plays every unit twice; stop one of them (by pid)")
    return (f"a live standalone kakuna queue plays a state INSIDE {canonical(state)}: {who} (/proc scan). The anchor "
            "lane would play the same units twice; let the queue finish or stop it (by pid)")


class SubStateClaims:
    """The claims the ANCHOR lane holds on the kakuna states it plays, ``<state>/anchor/<group>/<label>/`` (module
    docstring, NESTED). Per state: no live standalone ``kakuna run`` names it (the ``/proc`` scan), its ``kakuna``-lane
    path lock (outside the state) and its in-state ``supervisor.lock``, the two a ``kakuna run`` honors. Held until
    :meth:`close` (the lane's end): a queue started later is refused by the lock it takes first. :meth:`claim` raises
    :class:`SupervisorError` naming the state when another supervisor holds it, and holds nothing of that state then."""

    def __init__(self) -> None:
        self._held: Dict[str, Tuple[Any, Any]] = {}

    def __contains__(self, kakuna_state: "str | os.PathLike[str]") -> bool:
        return canonical(kakuna_state) in self._held

    def claim(self, kakuna_state: Path) -> None:
        key = canonical(kakuna_state)
        if key in self._held:
            return
        others = live_supervisors(kakuna_state, lane=KAKUNA_LANE, exclude=self_and_ancestors()).get(KAKUNA_LANE, [])
        if others:
            raise SupervisorError(f"a live standalone kakuna queue already plays {key}: pid(s) {others} (/proc scan); the "
                                  "anchor lane would play its units twice")
        path_fh = acquire(kakuna_state, KAKUNA_LANE, via="main.reads anchor lane")
        lock = try_exclusive(Path(kakuna_state) / "supervisor.lock", "w")
        if lock is None:
            path_fh.close()
            raise SupervisorError(f"another supervisor holds {Path(kakuna_state) / 'supervisor.lock'}: it plays {key}, "
                                  "which the anchor lane plays too")
        self._held[key] = (path_fh, lock)

    def close(self) -> None:
        held, self._held = list(self._held.values()), {}
        for path_fh, lock in held:
            lock.close()
            path_fh.close()
