"""Linux ``/proc`` reads for the ops layer: the process table, ancestry, and who holds which lock.

STDLIB ONLY, and nothing here imports the rest of the tree. It is shared by two callers that must
stay cheap and dependency-free:

* ``scripts/ops/idle_waiter_watchdog.py`` — the 15-minute cron that flags agent waiters making no
  progress and lock SELF-DEADLOCKS (a waiter blocked on a lock its own ancestor holds);
* ``utils.gpu_lock`` — the one helper that takes the box's GPU lock, re-entrantly.

Both need the same two answers and must agree on them: *who is my ancestor* (the ``ppid`` chain) and
*who holds this lock file* (``/proc/locks``, keyed by the file's ``major:minor:inode``, where a line
``N: -> FLOCK ...`` is a BLOCKED waiter and a line without the arrow is a HOLDER).

The ``pid`` column of ``/proc/locks`` is the process that TOOK the lock. For ``flock(1)`` that is the
``flock`` process itself — its child inherits the descriptor but is not the one listed — which is
exactly what a self-deadlock check wants: the holder is an ANCESTOR of the command it runs.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

HZ = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
_PROC = "/proc"


@dataclass(frozen=True)
class Proc:
    """One ``/proc/<pid>/stat`` row. CPU fields are in clock ticks (``HZ`` per second);
    ``starttime`` is ticks since boot, so ``(pid, starttime)`` is unique across PID reuse."""

    pid: int
    comm: str
    state: str
    ppid: int
    utime: int
    stime: int
    cutime: int
    cstime: int
    starttime: int

    @property
    def key(self) -> str:
        return f"{self.pid}:{self.starttime}"

    @property
    def cpu(self) -> int:
        """Own CPU plus that of every REAPED child — a child that exits moves its time here."""
        return self.utime + self.stime + self.cutime + self.cstime


def read_stat(pid: int, proc: str = _PROC) -> Optional[Proc]:
    """The process's stat row, or ``None`` if it is gone (or unreadable)."""
    try:
        with open(f"{proc}/{pid}/stat", "rb") as f:
            raw = f.read().decode("utf-8", "replace")
    except OSError:
        return None
    # comm is parenthesised and may itself contain ") " — split on the LAST ')'.
    lp, rp = raw.find("("), raw.rfind(")")
    if lp < 0 or rp < 0:
        return None
    rest = raw[rp + 2:].split()
    try:
        return Proc(pid=pid, comm=raw[lp + 1:rp], state=rest[0], ppid=int(rest[1]),
                    utime=int(rest[11]), stime=int(rest[12]), cutime=int(rest[13]), cstime=int(rest[14]),
                    starttime=int(rest[19]))
    except (IndexError, ValueError):
        return None


def all_pids(proc: str = _PROC) -> List[int]:
    return [int(n) for n in os.listdir(proc) if n.isdigit()]


def snapshot(proc: str = _PROC) -> Dict[int, Proc]:
    """Every live process's stat row, read once."""
    out: Dict[int, Proc] = {}
    for pid in all_pids(proc):
        p = read_stat(pid, proc)
        if p is not None:
            out[pid] = p
    return out


def ancestors(pid: int, procs: Optional[Dict[int, Proc]] = None) -> List[int]:
    """The ``ppid`` chain above ``pid``, nearest first, excluding ``pid`` itself and pid 0.
    With ``procs`` the snapshot is used; without it each step is read live."""
    out: List[int] = []
    seen: Set[int] = {pid}
    cur = pid
    while True:
        p = procs.get(cur) if procs is not None else read_stat(cur)
        if p is None or p.ppid <= 0 or p.ppid in seen:
            return out
        out.append(p.ppid)
        seen.add(p.ppid)
        cur = p.ppid


def children_map(procs: Dict[int, Proc]) -> Dict[int, List[int]]:
    kids: Dict[int, List[int]] = {}
    for p in procs.values():
        kids.setdefault(p.ppid, []).append(p.pid)
    return kids


def descendants(pid: int, kids: Dict[int, List[int]]) -> List[int]:
    """Every descendant of ``pid`` (not ``pid`` itself), breadth-first."""
    out: List[int] = []
    stack = list(kids.get(pid, ()))
    seen: Set[int] = set()
    while stack:
        c = stack.pop()
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
        stack.extend(kids.get(c, ()))
    return out


def subtree_cpu(procs: Dict[int, Proc], kids: Dict[int, List[int]]) -> Dict[int, int]:
    """Per pid: the CPU ticks of the process and of every live descendant (each including its
    reaped children). Iterative post-order, so a deep tree cannot hit the recursion limit."""
    total: Dict[int, int] = {}
    for root in procs:
        if root in total:
            continue
        stack: List[Tuple[int, bool]] = [(root, False)]
        while stack:
            pid, done = stack.pop()
            if done:
                total[pid] = procs[pid].cpu + sum(total.get(c, 0) for c in kids.get(pid, ()) if c in procs)
                continue
            if pid in total:
                continue
            stack.append((pid, True))
            stack.extend((c, False) for c in kids.get(pid, ()) if c in procs and c not in total)
    return total


def read_text(pid: int, name: str, proc: str = _PROC) -> str:
    try:
        with open(f"{proc}/{pid}/{name}", "rb") as f:
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def cmdline(pid: int, proc: str = _PROC) -> str:
    """The argv joined by spaces ('' for a kernel thread or a vanished process)."""
    return read_text(pid, "cmdline", proc).replace("\0", " ").strip()


def wchan(pid: int, proc: str = _PROC) -> str:
    """The kernel symbol the process sleeps in ('0' / '' when running or hidden)."""
    return read_text(pid, "wchan", proc).strip()


def readlink(pid: int, name: str, proc: str = _PROC) -> str:
    try:
        return os.readlink(f"{proc}/{pid}/{name}")
    except OSError:
        return ""


PAGE_SIZE = os.sysconf("SC_PAGE_SIZE") if hasattr(os, "sysconf") else 4096


def rss_bytes(pid: int, proc: str = _PROC) -> int:
    """Resident set size in bytes from ``statm`` (anon + file + shmem; 0 when gone or a kernel thread)."""
    try:
        return int(read_text(pid, "statm", proc).split()[1]) * PAGE_SIZE
    except (IndexError, ValueError):
        return 0


def status_kb(pid: int, field: str, proc: str = _PROC) -> Optional[int]:
    """One ``kB`` field of ``/proc/<pid>/status`` (``RssAnon``, ``VmHWM``, ...) in BYTES, or ``None``."""
    for line in read_text(pid, "status", proc).splitlines():
        if line.startswith(field + ":"):
            try:
                return int(line.split()[1]) * 1024
            except (IndexError, ValueError):
                return None
    return None


def cgroup_of(pid: int, proc: str = _PROC) -> str:
    """The process's cgroup-v2 path (``/user.slice/...``), '' when unknown."""
    for line in read_text(pid, "cgroup", proc).splitlines():
        if line.startswith("0::"):
            return line[3:]
    return ""


def cgroup_memory_limit(cg: str, root: str = "/sys/fs/cgroup") -> Optional[Tuple[str, int]]:
    """The nearest ``memory.max`` below ``max`` at or above the cgroup ``cg`` (a ``cgroup_of`` path),
    as ``(cgroup path, bytes)``; ``None`` when every level up to the root is unlimited (or unreadable)."""
    cur = cg.rstrip("/")
    while cur:
        try:
            with open(f"{root}{cur}/memory.max") as f:
                raw = f.read().strip()
        except OSError:
            raw = "max"
        if raw != "max":
            try:
                return cur, int(raw)
            except ValueError:
                pass
        cur = cur.rsplit("/", 1)[0]
    return None


def mem_total_bytes(path: str = f"{_PROC}/meminfo") -> int:
    """``MemTotal`` in bytes (0 if unreadable)."""
    try:
        with open(path) as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0


def uptime_s() -> float:
    with open(f"{_PROC}/uptime") as f:
        return float(f.read().split()[0])


def age_s(p: Proc, uptime: float) -> float:
    return uptime - p.starttime / HZ


# ------------------------------------------------------------------------------------ /proc/locks

@dataclass(frozen=True)
class LockEntry:
    """One ``/proc/locks`` line. ``file`` is ``"MAJ:MIN:INODE"`` exactly as the kernel prints it
    (major and minor in two-digit hex, inode decimal); ``blocked`` marks a ``->`` waiter line."""

    kind: str       # FLOCK / POSIX / OFDLCK / LEASE / DELEG
    mode: str       # READ / WRITE
    pid: int        # -1 for an OFD lock
    file: str
    blocked: bool

    @property
    def inode(self) -> int:
        return int(self.file.rsplit(":", 1)[1])


_LOCK_RE = re.compile(r"^\s*\d+:\s+(->\s+)?(\S+)\s+\S+\s+(\S+)\s+(-?\d+)\s+([0-9a-f]+:[0-9a-f]+:\d+)\s")


def read_locks(path: str = f"{_PROC}/locks") -> List[LockEntry]:
    out: List[LockEntry] = []
    try:
        with open(path) as f:
            lines = f.readlines()
    except OSError:
        return out
    for line in lines:
        m = _LOCK_RE.match(line)
        if m:
            out.append(LockEntry(kind=m.group(2), mode=m.group(3), pid=int(m.group(4)),
                                 file=m.group(5), blocked=bool(m.group(1))))
    return out


def file_key(path: str) -> Optional[str]:
    """``path``'s ``/proc/locks`` key (``"MAJ:MIN:INODE"``), or ``None`` if it does not exist."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    return f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}"


def holders(key: str, locks: Optional[Iterable[LockEntry]] = None) -> List[int]:
    """The pids HOLDING any lock on the file ``key`` (blocked waiters excluded)."""
    locks = read_locks() if locks is None else locks
    return sorted({e.pid for e in locks if e.file == key and not e.blocked and e.pid > 0})


def waiters(key: str, locks: Optional[Iterable[LockEntry]] = None) -> List[int]:
    locks = read_locks() if locks is None else locks
    return sorted({e.pid for e in locks if e.file == key and e.blocked and e.pid > 0})


def fd_path_for(pid: int, key: str, proc: str = _PROC) -> str:
    """The path by which ``pid`` holds the file ``key`` open ('' when no fd matches) — how a
    ``/proc/locks`` inode is turned back into a name a human can read."""
    d = f"{proc}/{pid}/fd"
    try:
        fds = os.listdir(d)
    except OSError:
        return ""
    ino = key.rsplit(":", 1)[1]
    for fd in fds:
        try:
            st = os.stat(f"{d}/{fd}")
        except OSError:
            continue
        if str(st.st_ino) == ino and f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{ino}" == key:
            try:
                return os.readlink(f"{d}/{fd}")
            except OSError:
                return ""
    return ""
