"""The cf label PRODUCER's single-instance lock and parent binding — the one piece of the deleted cf
label SUPPLY (`cf_supply.py`, deletion pass L4) that the kept offline producer (`cf_producer.py`,
manifest D6) still needs.

The trainer no longer spawns or verifies a producer (`--cf-label-supply` and the whole cf training
half are deleted), so what is left is what a STANDALONE `python -m agents.training.cf_producer
<run_dir>` run needs against an old run's `cf_records/` ring: ONE producer per run directory
(`<run>/cf_producer.lock`, an `flock` the kernel releases when the holder dies by any path), and
the optional `--parent-pid` binding that makes a producer die with the process that started it.

Nothing here imports torch.
"""
from __future__ import annotations

import fcntl
import os
import signal
from typing import Optional

LOCK_NAME = "cf_producer.lock"
#: `cf_producer`'s exit code when another producer already holds the run's lock.
PRODUCER_EXIT_LOCK_HELD = 4


def lock_path(run_dir: str) -> str:
    return os.path.join(run_dir, LOCK_NAME)


def acquire_producer_lock(run_dir: str) -> Optional[int]:
    """Take the run's producer lock for this process's lifetime. Returns the fd, or None if held.

    `flock` is released by the kernel when the holder dies by ANY path (SIGKILL included), so a
    stale lock cannot outlive its producer. The holder's pid is written into the file for the
    message a second claimant prints."""
    fd = os.open(lock_path(run_dir), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    os.ftruncate(fd, 0)
    os.write(fd, f"{os.getpid()}\n".encode())
    os.fsync(fd)
    return fd


def live_producer_pid(run_dir: str) -> Optional[int]:
    """The pid of a producer HOLDING the run's lock right now, else None (probe, never keeps it)."""
    path = lock_path(run_dir)
    if not os.path.exists(path):
        return None
    fd = os.open(path, os.O_RDONLY)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except OSError:
            try:
                return int((os.pread(fd, 32, 0).decode().strip() or "0")) or -1
            except ValueError:
                return -1
        fcntl.flock(fd, fcntl.LOCK_UN)
        return None
    finally:
        os.close(fd)


def bind_to_parent(parent_pid: int) -> bool:
    """Die with the parent: PR_SET_PDEATHSIG(SIGTERM), then close the fork/prctl race by checking
    the parent is still the one we were told. Returns False when it already is not."""
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, int(signal.SIGTERM), 0, 0, 0)          # PR_SET_PDEATHSIG == 1
    except (OSError, AttributeError):                          # pragma: no cover - non-Linux
        pass
    return os.getppid() == int(parent_pid)
