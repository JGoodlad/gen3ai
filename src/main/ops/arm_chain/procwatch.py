"""``/proc`` watchers: process IDENTITY, bounded waits, and the identity-gated signal.

* **Never ``kill -0``.** Inside a sandboxed agent shell ``kill -0`` read a LIVE supervisor as dead (acf2193fa); a
  liveness read here is one ``/proc/<pid>/stat`` read (``utils.procfs``).
* **A pid alone is not an identity.** A :class:`ProcRef` is ``(pid, start ticks)``: a pid file that outlived its
  process and a pid the kernel re-used both read as GONE, never as the old process alive.
* **Every wait is bounded** (root rule 2); the caller says how long and what a timeout means.
* **Every signal is identity-gated** (SOP §3 rule 7, ``kill_if.sh``): the start ticks AND the cmdline must match.
"""
from __future__ import annotations

import os
import re
import signal as _signal
import time
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from utils import procfs


@dataclass(frozen=True)
class ProcRef:
    pid: int
    start_ticks: int

    def to_json(self) -> dict:
        return {"pid": self.pid, "start_ticks": self.start_ticks}

    @staticmethod
    def from_json(d: dict) -> "ProcRef":
        return ProcRef(int(d["pid"]), int(d["start_ticks"]))


def ref_of(pid: int, proc: str = "/proc") -> Optional[ProcRef]:
    st = procfs.read_stat(pid, proc)
    if st is None or st.state in ("Z", "X"):
        return None
    return ProcRef(pid, st.starttime)


def ref_from_pidfile(path: str, proc: str = "/proc") -> Optional[ProcRef]:
    """The live process a pid file names, or None (absent, unparsable, or the pid is gone)."""
    try:
        pid = int(open(path).read().strip())
    except (OSError, ValueError):
        return None
    return ref_of(pid, proc)


def alive(ref: ProcRef, proc: str = "/proc") -> bool:
    st = procfs.read_stat(ref.pid, proc)
    return st is not None and st.state not in ("Z", "X") and st.starttime == ref.start_ticks


def wait_gone(ref: ProcRef, timeout_s: float, poll_s: float = 30.0, proc: str = "/proc",
              sleep: Callable[[float], None] = time.sleep,
              monotonic: Callable[[], float] = time.monotonic) -> bool:
    """True once ``ref`` is gone; False if it is still alive after ``timeout_s``."""
    t0 = monotonic()
    while alive(ref, proc):
        left = timeout_s - (monotonic() - t0)
        if left <= 0:
            return False
        sleep(min(poll_s, left))
    return True


def signal_if(ref: ProcRef, argv_ere: str, sig: int = _signal.SIGTERM, proc: str = "/proc",
              kill: Callable[[int, int], None] = os.kill) -> Tuple[bool, str]:
    """Signal ``ref`` only if it is the same process (start ticks) AND its cmdline matches ``argv_ere``."""
    if not alive(ref, proc):
        return False, f"pid {ref.pid} (start {ref.start_ticks}) is gone or was re-used — NOT signalled"
    cmd = procfs.cmdline(ref.pid, proc)
    if not re.search(argv_ere, cmd):
        return False, f"pid {ref.pid} identity MISMATCH ({cmd[:100]!r} vs /{argv_ere}/) — NOT signalled"
    kill(ref.pid, sig)
    return True, f"pid {ref.pid} identity OK ({cmd[:100]!r}) — sent signal {sig}"
