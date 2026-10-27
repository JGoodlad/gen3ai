"""ONE pytest session's OWN temp tree, and the sweep of the dead sessions' (``designs/ops/testing.md``, "Test TEMP DIRS live on the real disk").

Before this, every session on the box took the next free ``pytest-N`` under the SHARED ``<scratch>/pytest-of-<user>/``
and trusted pytest's own number rotation to leave the others' alone: a session's ``tmp_path`` survived another's start
and exit only while a ``.lock`` file in its numbered dir was younger than three days (``_pytest.pathlib.LOCK_TIMEOUT``),
which says nothing about whether the session is still alive, and its dead-session sweep ran at EVERY session's exit. The
box runs several gates at once (and a slow tier is hours), so the tree a session stands on must be one that no other
session's cleanup can reach by any rule but "its owner is dead":

* WHERE -- :func:`session_basetemp`: ``<scratch>/pytest-of-<user>/session-<pid>-<starttime>``, named for the process
  that owns it. ``(pid, starttime)`` is unique for the life of the box (``/proc/<pid>/stat`` field 22), so a REUSED pid
  is not mistaken for the owner. The root conftest hands it to pytest as the session's ``--basetemp`` (pytest then
  creates it on the first ``tmp_path`` and numbers nothing); every xdist worker is given ``<it>/popen-gwN`` by xdist.
  The ``pytest-of-<user>`` parent keeps its name, so ``scripts/ops/tmp_sweep.py``'s child-pruned entry still sees it.
* HOW LONG -- :func:`finish_session` removes the tree of a session that passed (what ``tmp_path_retention_policy =
  failed`` did to the numbered one); a FAILED session's tree stays for debugging.
* WHO CLEANS WHAT -- :func:`sweep_dead_sessions`, run by each new session: a tree whose owner is DEAD is removed unless
  it is among the newest ``keep`` (``tmp_path_retention_count``) and younger than ``retain_hours``. A tree whose owner
  is alive is never removed, at any age. :func:`sweep_dead_compile_caches` is the same rule for the per-process compile
  caches (``gen3ai_pytest_compile_<pid>_*``).

Liveness is ``utils.procfs`` (``/proc/<pid>/stat``), never ``kill(pid, 0)``: inside a sandboxed agent shell ``kill``
reads another agent's live process as dead (acf2193f), which for a sweep means deleting a live session's tree.

Loaded BY PATH by the root ``conftest.py`` (see ``utils.scratch``); it imports ``utils.procfs``, which is stdlib-only.
"""
from __future__ import annotations

import getpass
import os
import re
import shutil
import time
from typing import Callable, List, Optional, Tuple

from utils import procfs

SESSION_PREFIX = "session-"
#: A failed or killed session's tree is kept this long (and only the newest ``keep`` of them), then removed: it holds
#: the failing tests' ``tmp_path`` for debugging, and the 2026-09-30 inode exhaustion was retained trees.
RETAIN_HOURS = 24.0
#: The prefixes of the per-process compile caches the root conftest and ``agents.model.compile_cache`` create.
COMPILE_PREFIXES = ("gen3ai_pytest_compile_", "gen3ai_compile_")

_SESSION_RE = re.compile(rf"^{re.escape(SESSION_PREFIX)}(\d+)-(\d+)$")


def user_name() -> str:
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001 -- no passwd entry (a container): the uid, as tmp_sweep.py does
        return str(os.getuid())


def sessions_root(scratch_root: str) -> str:
    """``<scratch>/pytest-of-<user>``: the parent of every session's tree (and of the legacy ``pytest-N`` ones)."""
    return os.path.join(scratch_root, f"pytest-of-{user_name()}")


def owner_token(pid: Optional[int] = None) -> Tuple[int, int]:
    """``(pid, starttime)`` of ``pid`` (default: this process), from ``/proc``."""
    pid = os.getpid() if pid is None else pid
    st = procfs.read_stat(pid)
    if st is None:
        raise RuntimeError(f"cannot read /proc/{pid}/stat: the session's owner cannot be named")
    return pid, st.starttime


def session_name(pid: int, starttime: int) -> str:
    return f"{SESSION_PREFIX}{pid}-{starttime}"


def parse_session_name(name: str) -> Optional[Tuple[int, int]]:
    """``(pid, starttime)`` a session dir's name carries, or ``None`` when it is not one."""
    m = _SESSION_RE.match(name)
    return (int(m.group(1)), int(m.group(2))) if m else None


def session_basetemp(scratch_root: str, pid: Optional[int] = None) -> str:
    """The tree THIS process (or ``pid``) owns: ``<scratch>/pytest-of-<user>/session-<pid>-<starttime>``. Creates
    its parent only; the tree itself appears when pytest makes the basetemp, on the first ``tmp_path``."""
    root = sessions_root(scratch_root)
    os.makedirs(root, mode=0o700, exist_ok=True)
    return os.path.join(root, session_name(*owner_token(pid)))


def owner_alive(pid: int, starttime: int) -> bool:
    """Is the process that owned a session still running? A live pid with another start time is a REUSED pid: dead."""
    st = procfs.read_stat(pid)
    return st is not None and st.state not in ("Z", "X") and st.starttime == starttime


def _mine(path: str) -> bool:
    try:
        return os.lstat(path).st_uid == os.getuid()
    except OSError:
        return False


def sweep_dead_sessions(scratch_root: str, *, keep: int = 1, retain_hours: float = RETAIN_HOURS,
                        now: Optional[float] = None,
                        alive: Callable[[int, int], bool] = owner_alive) -> List[str]:
    """Remove the trees of DEAD sessions under ``<scratch>/pytest-of-<user>``; return what was removed.

    A tree is spared when its owner is alive (any age) or when it is among the ``keep`` newest dead ones and younger
    than ``retain_hours``. Names that are not a session's (the legacy ``pytest-N``, a symlink, another user's) are
    left to ``scripts/ops/tmp_sweep.py``."""
    root = sessions_root(scratch_root)
    now = time.time() if now is None else now
    try:
        names = os.listdir(root)
    except OSError:
        return []
    dead: List[Tuple[float, str]] = []
    for name in names:
        tok = parse_session_name(name)
        path = os.path.join(root, name)
        if tok is None or os.path.islink(path) or not _mine(path) or alive(*tok):
            continue
        try:
            dead.append((os.lstat(path).st_mtime, path))
        except OSError:
            continue
    dead.sort(reverse=True)                                    # newest first
    removed: List[str] = []
    for rank, (mtime, path) in enumerate(dead):
        if rank < keep and (now - mtime) / 3600.0 < retain_hours:
            continue
        shutil.rmtree(path, ignore_errors=True)
        removed.append(path)
    return removed


def finish_session(basetemp: Optional[str], exitstatus: int, policy: str) -> bool:
    """End of THIS session: remove its tree when ``policy`` says so (``failed``: the session passed; ``none``:
    always; ``all``: never). True when it was removed. Only a tree this call is handed, which the conftest takes from
    :func:`session_basetemp` for its own pid."""
    if not basetemp or not os.path.isdir(basetemp):
        return False
    if policy == "none" or (policy == "failed" and exitstatus == 0):
        shutil.rmtree(basetemp, ignore_errors=True)
        return True
    return False


def sweep_dead_compile_caches(scratch_root: str,
                              alive: Callable[[int], bool] = procfs.pid_alive) -> List[str]:
    """Remove this user's ``gen3ai_pytest_compile_<pid>_*`` / ``gen3ai_compile_<pid>_*`` dirs under ``scratch_root``
    whose creating process is dead (one that was SIGKILLed never reached its own cleanup); return what was removed.
    Mirrors ``agents.model.compile_cache.sweep_stale_temp_caches``: the liveness read is ``/proc``, so a live
    session in another agent's shell is never read as dead."""
    removed: List[str] = []
    try:
        names = os.listdir(scratch_root)
    except OSError:
        return removed
    for name in names:
        prefix = next((p for p in COMPILE_PREFIXES if name.startswith(p)), None)
        if prefix is None:
            continue
        head = name[len(prefix):].split("_", 1)[0]
        if not head.isdigit() or alive(int(head)):
            continue
        path = os.path.join(scratch_root, name)
        if _mine(path):
            shutil.rmtree(path, ignore_errors=True)
            removed.append(path)
    return removed
