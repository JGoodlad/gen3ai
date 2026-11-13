"""A child a TEST starts dies with the test, and so does everything under it: :func:`guarded_popen` / :func:`guarded_run`.

WHY (the 2026-11-13 incident). A peer ran ``timeout 900 pytest …`` over ``main/b1/resume_check_test.py``'s two-segment
smoke. The test started ``python -m main.launcher`` by a plain ``Popen``; the launcher starts the trainer, the trainer
its ``rust_env_proc`` workers. The timeout killed pytest, not that tree: an orphaned trainer ran on for three hours, the
BOUNDARY-1 driver read "something is live" at the next run boundary and HELD, and the GPU sat idle.

THE CLASS. A trainer (or any long-lived child) a test spawns must not outlive the test process, whether pytest ends by
an exception, a ``timeout``'s SIGTERM or a SIGKILL. ``try/finally: proc.kill()`` covers the first only, and kills only
the DIRECT child (the launcher; its trainer is a grandchild).

THE MECHANISM, in two parts, with no change to what a launcher or a trainer does:

* the child leads a SESSION of its own (``start_new_session``): the launcher keeps its trainer in its own session
  (``main.launcher.child``) and ``rust_env_proc`` stays in the trainer's, so the whole tree is ONE session that
  ``utils.procgroup.session_groups`` enumerates. (A descendant that ``setsid()``s itself leaves it.)
* a WATCHER (a tiny detached process, in a session of its own so a group signal meant for the test never reaches it)
  holds the read end of a LEASH pipe whose write end only the test process keeps. The leash is cut when the test process
  dies by ANY signal, and the watcher also polls its own parent pid, so a child the test process forked cannot keep the
  leash alive. It then SIGTERMs every group of the child's session and SIGKILLs what is left after ``grace_s``. It exits
  by itself once the session is empty, and when the child's ``Popen`` is dropped.

NOT ``PR_SET_PDEATHSIG`` (``utils.mem_cap`` uses it for its one supervisor, a direct child): the kernel clears it across
``fork``, so it would reach the launcher and not its trainer, and it fires when the parent's THREAD that made the child
exits, not the process. A pytest-xdist worker spawns from threads of its own.

Use it for every test-side spawn of a trainer or launcher (``child_guard_test`` scans the tree and refuses a raw
``subprocess`` call beside the trainer's name). :func:`guarded_run` also kills the WHOLE tree on a timeout, where
``subprocess.run`` kills the direct child only.

LIMITS, said once: Linux only. A test process that dies leaves its child alive for up to one poll (``POLL_S``). The
guard does not cover a child a test starts by another route (a shell script, ``os.system``). A process the test forks
(``multiprocessing`` with ``fork``) inherits the leash, which is why the watcher polls the parent pid as well.
"""
from __future__ import annotations

import os
import select
import subprocess
import sys
import time
from typing import Any, List, Optional, Sequence

from utils import procfs, procgroup
from utils.paths import src_root

#: SIGTERM -> SIGKILL, seconds. A test's tree is scratch, so there is nothing worth the 150 s a real run's stop gets
#: (``main.launcher.run.KILL_GRACE_SECONDS``): the launcher and the trainer still see a SIGTERM first.
GRACE_S = 5.0
#: How often the watcher looks at its parent pid and at the child's session (the leash itself wakes it at once).
POLL_S = 1.0
#: How often :func:`kill_session` re-reads the session while it waits for it to empty.
_SWEEP_S = 0.2

_WATCHER_CODE = ("import sys; sys.path.insert(0, sys.argv[1]); from utils.child_guard import _watch_main; "
                 "sys.exit(_watch_main(int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5])))")
#: ``_watch_main``'s argv, after the src dir: the child's session id (= its pid), the TEST process's pid, the leash fd, grace.
#: The watchers this process started: kept so each is reaped once it exits (a zombie per spawn otherwise).
_WATCHERS: List[subprocess.Popen] = []


def session_members(sid: int) -> List[int]:
    """The pids of every LIVE (non-zombie) process of session ``sid``."""
    return sorted(p.pid for p in procfs.snapshot().values() if p.session == sid and p.state not in ("Z", "X"))


def kill_session(sid: int, grace_s: float = GRACE_S) -> List[int]:
    """SIGTERM every process group of session ``sid``, then SIGKILL what is left ``grace_s`` later; the pids still
    alive afterwards (``[]`` once it is done). Never the caller's own session or init's."""
    if sid <= 1 or sid == os.getsid(0):
        return []

    def sweep(sig: int) -> None:
        for g in procgroup.session_groups(sid):
            procgroup.killpg_quiet(g, sig)

    sweep(15)
    end = time.monotonic() + grace_s
    while session_members(sid) and time.monotonic() < end:
        time.sleep(_SWEEP_S)
    for _ in range(100):                      # a process forked in the window is caught by the next sweep
        if not session_members(sid):
            break
        sweep(9)
        time.sleep(_SWEEP_S)
    return session_members(sid)


def _watch_main(sid: int, parent: int, leash_fd: int, grace_s: float) -> int:
    """The watcher: wait for the leash to be cut (or its parent, the test process, to change), then
    :func:`kill_session`."""
    while True:
        ready, _, _ = select.select([leash_fd], [], [], POLL_S)
        if ready:
            try:
                if os.read(leash_fd, 1):       # nobody writes; a byte would be a stray, not a cut
                    continue
            except OSError:
                pass
            break                              # EOF: the test process is gone
        if os.getppid() != parent:
            break
        if not session_members(sid):
            return 0                           # the child's tree is finished: nothing left to guard
    kill_session(sid, grace_s)
    return 0


def guarded_popen(argv: Sequence[str], *, grace_s: float = GRACE_S, **kwargs: Any) -> "subprocess.Popen[Any]":
    """``subprocess.Popen(argv, **kwargs)`` whose whole process tree dies with this process (module docstring).

    The child leads its own session (``start_new_session`` is the guard's, so the keyword is refused). The returned
    ``Popen`` is an ordinary one; keep it alive as long as the child should be: dropping it cuts the leash, which kills
    whatever is left of the child's tree."""
    for k in ("start_new_session", "process_group", "preexec_fn"):
        if k in kwargs:
            raise TypeError(f"guarded_popen owns {k!r}: the child's session is how its whole tree is found and killed")
    _WATCHERS[:] = [w for w in _WATCHERS if w.poll() is None]
    child = subprocess.Popen(list(argv), start_new_session=True, **kwargs)
    read_fd, write_fd = os.pipe()               # non-inheritable: only the watcher gets the read end, by pass_fds
    leash = os.fdopen(write_fd, "wb", buffering=0)
    try:
        watcher = subprocess.Popen(
            [sys.executable, "-I", "-c", _WATCHER_CODE, str(src_root()), str(child.pid), str(os.getpid()),
             str(read_fd), str(grace_s)],
            pass_fds=(read_fd,), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True, close_fds=True)
    except BaseException:
        leash.close()                           # never leave a child nobody watches
        kill_session(child.pid, 0.0)
        child.wait()
        raise
    finally:
        os.close(read_fd)
    _WATCHERS.append(watcher)
    child._child_guard_leash = leash            # type: ignore[attr-defined]  # closed when the Popen is dropped
    return child


def guarded_run(argv: Sequence[str], *, input: Optional[Any] = None, capture_output: bool = False,
                timeout: Optional[float] = None, check: bool = False, grace_s: float = GRACE_S,
                **kwargs: Any) -> "subprocess.CompletedProcess[Any]":
    """``subprocess.run`` through :func:`guarded_popen`. A timeout (or any exception) kills the child's whole TREE,
    where ``subprocess.run`` kills the direct child only and leaves a launcher's trainer running."""
    if capture_output:
        kwargs["stdout"] = kwargs["stderr"] = subprocess.PIPE
    if input is not None:
        kwargs["stdin"] = subprocess.PIPE
    with guarded_popen(argv, grace_s=grace_s, **kwargs) as process:
        try:
            stdout, stderr = process.communicate(input, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            kill_session(process.pid, grace_s)
            exc.stdout, exc.stderr = process.communicate()
            raise
        except BaseException:
            kill_session(process.pid, grace_s)
            raise
        code = process.poll()
    if check and code:
        raise subprocess.CalledProcessError(code, process.args, output=stdout, stderr=stderr)
    return subprocess.CompletedProcess(process.args, code, stdout, stderr)
