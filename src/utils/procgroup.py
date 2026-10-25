"""Run ONE command as the leader of its OWN process group (or session) and signal it as a unit.

WHY (backlog rows "``gpu_lock.sh`` runs its command as a child of its own python process" and "the GPU window (T46)
judges its guest by the WRAPPER pid", 2026-10-25). A wrapper that starts its command with a plain ``Popen`` leaves
the command in the WRAPPER's group, which is the launching tool's group: a signal sent to the wrapper's PID reaches
one process, and one sent to the group reaches whatever else the caller started. Here the command is the LEADER of a
group of its own (``pgid == the command's pid``), so ``kill -TERM -- -<pgid>`` reaches the command and every
descendant that stayed in the group, and nothing else. A descendant that ``setsid()``s itself leaves the group.

Two shapes, one engine (:func:`run_in_own_group`):

* ``session=False`` - ``utils.gpu_lock``: ``setpgid``, a new GROUP in the caller's session. When this process's group
  is a terminal's FOREGROUND group, the command's group takes the terminal over for the run, exactly as a shell does
  for a job: its reads work, ^C / ^\\ / ^Z reach it alone, and a ^Z stops this wrapper too (so the shell sees ITS job
  stopped) and a ``fg`` hands the terminal back. Without a terminal nothing of that happens.
* ``session=True`` - the T46 guest (``main.launcher.gpu_window``): ``setsid``, a new SESSION, so the whole job tree
  (nested groups included, a ``gpu_lock.sh`` inside it) is one session a preempt can enumerate.

SIGINT / SIGTERM / SIGHUP reaching THIS process are forwarded to the command's whole group (``session=True``: to every
group of its session, so one a descendant made of its own is reached too); the handlers are installed BEFORE the
command starts, and one that arrives in the window is delivered the moment the group exists. The group is never signalled
once the command is reaped. SIGKILL of the wrapper cannot be forwarded: preempt by ``killpg(<pgid>)``, not by the
wrapper's pid. Linux only.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
from typing import Callable, List, Optional, Sequence

from utils import procfs

#: Signals this process forwards to the command's group (a stop request is a request to the whole job).
STOP_SIGNALS = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)


def killpg_quiet(pgid: int, sig: int) -> bool:
    """``killpg(pgid, sig)``; False if the group is gone (or has nobody we may signal)."""
    try:
        os.killpg(pgid, sig)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def session_groups(sid: int, procs: "Optional[dict]" = None) -> "List[int]":
    """The process-group ids of every LIVE process in session ``sid`` (``sid`` itself first): a session made by
    :func:`run_in_own_group` holds the command's group and any group a descendant made of its own."""
    procs = procfs.snapshot() if procs is None else procs
    groups = {p.pgrp for p in procs.values() if p.session == sid and p.state not in ("Z", "X")}
    return sorted(groups, key=lambda g: (g != sid, g))


def group_members(pgid: int, procs: "Optional[dict]" = None) -> "List[procfs.Proc]":
    """Every LIVE (non-zombie) process whose process group is ``pgid``."""
    procs = procfs.snapshot() if procs is None else procs
    return sorted((p for p in procs.values() if p.pgrp == pgid and p.state not in ("Z", "X")), key=lambda p: p.pid)


# ------------------------------------------------------------------------------------------ the terminal

def _tcgetpgrp(fd: int) -> int:
    try:
        return os.tcgetpgrp(fd)
    except OSError:
        return -1


def _foreground_tty() -> Optional[int]:
    """An fd on the controlling terminal iff this process's group is its FOREGROUND group, else ``None``."""
    try:
        fd = os.open("/dev/tty", os.O_RDWR | os.O_NOCTTY | os.O_CLOEXEC)
    except OSError:
        return None
    if _tcgetpgrp(fd) == os.getpgrp():
        return fd
    os.close(fd)
    return None


def _set_foreground(fd: int, pgid: int) -> bool:
    """Make ``pgid`` the terminal's foreground group. A background caller is stopped by SIGTTOU for this call
    unless the signal is blocked, so it is blocked around the call (a blocked SIGTTOU makes the call legal and
    leaves nothing pending)."""
    old = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGTTOU})
    try:
        os.tcsetpgrp(fd, pgid)
        return True
    except OSError:
        return False
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, old)


def _child_stopped(sig: int, pgid: int, tty: Optional[int]) -> None:
    """The command's group STOPPED (``waitpid(WUNTRACED)``). With no terminal handed over there is nothing to do but
    keep waiting. With one: SIGTTIN / SIGTTOU while the terminal already is the group's means it read or wrote in
    the instant before the hand-over: continue it. Anything else is a ^Z (or a SIGSTOP) of the foreground job:
    take the terminal back, stop THIS process too so the shell sees its job stopped, and on the shell's SIGCONT
    hand the terminal over again (if it was a ``fg``) and continue the group."""
    if tty is None:
        return
    if sig in (signal.SIGTTIN, signal.SIGTTOU) and _tcgetpgrp(tty) == pgid:
        killpg_quiet(pgid, signal.SIGCONT)
        return
    if _tcgetpgrp(tty) == pgid:
        _set_foreground(tty, os.getpgrp())
    os.kill(os.getpid(), signal.SIGSTOP)
    if _tcgetpgrp(tty) == os.getpgrp():
        _set_foreground(tty, pgid)
    killpg_quiet(pgid, signal.SIGCONT)


def _wait(child: "subprocess.Popen", tty: Optional[int]) -> int:
    """Reap ``child`` (handling job-control stops on the way); its exit code, or 128+N when signal N killed it."""
    while True:
        try:
            _, status = os.waitpid(child.pid, os.WUNTRACED)
        except ChildProcessError:           # SIGCHLD ignored by the caller: the child was auto-reaped
            rc = child.wait()
            return 128 - rc if rc < 0 else rc
        if os.WIFSTOPPED(status):
            _child_stopped(os.WSTOPSIG(status), child.pid, tty)
            continue
        rc = os.waitstatus_to_exitcode(status)
        child.returncode = rc               # Popen would otherwise try to reap an already reaped child
        return 128 - rc if rc < 0 else rc


# ----------------------------------------------------------------------------------------------- the run

def run_in_own_group(cmd: Sequence[str], *, session: bool = False,
                     on_start: Optional[Callable[[int], None]] = None, **popen_kw) -> int:
    """Run ``cmd`` as the leader of a NEW process group (``session=True``: a new session) and wait for it.

    ``on_start(pgid)`` runs right after the command exists (``pgid`` is its pid); if it raises, the group is
    SIGKILLed and the exception propagates (a job whose group could not be recorded is a job nobody can preempt).
    Returns the command's exit code, or 128+N when signal N killed it. ``OSError`` from the spawn propagates.
    Must run in the main thread (it installs signal handlers)."""
    pgid_box: List[Optional[int]] = [None]
    pending: List[int] = []

    def forward(n: int, _frame: object) -> None:
        pgid = pgid_box[0]
        if pgid is None:
            pending.append(n)                # before the group exists: delivered once it does
        elif session:
            for g in session_groups(pgid) or [pgid]:         # the job's nested groups too (a gpu_lock.sh inside it)
                killpg_quiet(g, n)
        else:
            killpg_quiet(pgid, n)

    saved = {s: signal.signal(s, forward) for s in STOP_SIGNALS}
    tty: Optional[int] = None
    handed = False
    try:
        tty = None if session else _foreground_tty()
        spawn = {"start_new_session": True} if session else (
            {"process_group": 0} if sys.version_info >= (3, 11) else {"preexec_fn": os.setpgrp})
        child = subprocess.Popen(list(cmd), **spawn, **popen_kw)
        pgid_box[0] = child.pid
        if tty is not None:
            handed = _set_foreground(tty, child.pid)
        try:
            if on_start is not None:
                on_start(child.pid)
        except BaseException:
            killpg_quiet(child.pid, signal.SIGKILL)
            child.wait()
            raise
        for n in pending:
            forward(n, None)
        try:
            return _wait(child, tty if handed else None)
        finally:
            pgid_box[0] = None
    finally:
        pgid_box[0] = None
        if tty is not None:
            if handed:
                _set_foreground(tty, os.getpgrp())
            os.close(tty)
        for s, h in saved.items():
            signal.signal(s, h)
