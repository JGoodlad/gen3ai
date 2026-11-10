"""TEST SUPPORT: make ``utils.procfs``'s one reading of ``/proc/locks`` MISS a live lock.

The bug class (``utils.procfs``, 2026-11-09): the kernel hands the listing out one page per ``read()`` and resumes by
position, so a lock released between two reads makes it skip a live entry, and code that acted on "not listed" read a
held lock as free. Every site that asks who holds a lock has a test under such a listing, in two strengths:

* :func:`without`: a listing with the rows of some pids (or every row) REMOVED. Deterministic; the per-site tests.
* :class:`Torn`: the REAL tear, forced through the kernel: take :data:`FILL` locks ahead of the lock under test, read
  the listing's first pages, release them, read the rest. The reproduction of the incident, on a real listing.

A reader replaces ``procfs._read_listing`` (the one place the listing is opened): ``monkeypatch.setattr(procfs,
"_read_listing", reader)`` in a test's own process, :func:`install` in a child process a test starts. Stdlib only.
"""
from __future__ import annotations

import fcntl
import os
from pathlib import Path
from typing import Callable, List

from utils import procfs

Reader = Callable[[str], List[procfs.LockEntry]]

#: How many locks :class:`Torn` takes ahead of the lock under test: the listing skips that many rows, so the lock is
#: missed wherever it sits in a window this wide (about four pages of rows).
FILL = 300
_PAGE = 4096
LISTING = "/proc/locks"
_REAL: Reader = procfs._read_listing


def without(*pids: int) -> Reader:
    """A listing reader that omits every HELD row of ``pids`` (of every process when none is named): the listing as
    a torn read can return it. Blocked waiters stay (a tear that hides a waiter is another test's subject)."""
    def read(path: str) -> List[procfs.LockEntry]:
        return [e for e in _REAL(path) if e.blocked or (pids and e.pid not in pids)]
    return read


def listed(key: str) -> List[int]:
    """The pids the listing (as the reader in place hands it out) names holding ``key``: a test's PRECONDITION that
    its forced miss is in place. Never a verdict on who holds the lock."""
    return procfs.holders(key, procfs.read_locks(LISTING))


def install(reader: Reader) -> None:
    """Replace the listing reader for the life of THIS process (a child a test starts; a test's own process uses
    ``monkeypatch`` so the real reader comes back)."""
    procfs._read_listing = reader


class Torn:
    """A listing reader that really tears: each read hides the holder of the file ``key`` the way the kernel did.

    The lock table is one list per CPU, newest first, walked in CPU order, and a read resumes at the ROW NUMBER where
    the last one stopped. So, pinned to the lowest CPU this process may use: take :data:`FILL` fresh locks (they sit
    AHEAD of the lock under test, which is older or on a later CPU), read pages until the cursor is in the middle of
    those rows, release them all (every later row moves up by :data:`FILL`, past the cursor), read the rest: the lock
    under test is never returned. ``hid`` counts the reads that missed it, ``unforced`` the reads that could not be
    torn (the lock was not behind the fill: nothing is asserted about those, a test checks :meth:`hides` first)."""

    def __init__(self, scratch: Path, key: str, fill: int = FILL) -> None:
        self.key = key
        self.hid = 0
        self.unforced = 0
        self._files = [Path(scratch) / f"fill{i}" for i in range(fill)]
        for f in self._files:
            f.touch()
        self._fill_keys = {procfs.file_key(str(f)) for f in self._files}

    def _holds(self, line: str) -> bool:
        e = procfs._parse_listing([line])
        return bool(e) and e[0].file == self.key and not e[0].blocked

    def _is_fill(self, line: str) -> bool:
        e = procfs._parse_listing([line])
        return bool(e) and e[0].file in self._fill_keys

    def __call__(self, path: str) -> List[procfs.LockEntry]:
        allowed = os.sched_getaffinity(0)
        os.sched_setaffinity(0, {min(allowed)})
        fds: List[int] = []
        head = b""
        r = -1
        try:
            for f in self._files:
                fd = os.open(f, os.O_RDWR)
                fcntl.flock(fd, fcntl.LOCK_EX)
                fds.append(fd)
            with open(path) as f:
                whole = f.readlines()                    # where the lock under test sits while the fill is in place
            at = next((i for i, line in enumerate(whole) if self._holds(line)), None)
            ahead = sum(1 for line in whole[:at] if self._is_fill(line)) if at is not None else 0
            r = os.open(path, os.O_RDONLY)
            if at is not None and ahead >= len(self._files) // 2:
                aim = at - ahead // 2                    # the cursor stops here: past where the lock will be, before where it is
                while head.count(b"\n") < aim:
                    page = os.read(r, _PAGE)             # the kernel hands out whole rows, one page at most
                    if not page:
                        break
                    head += page
        finally:
            for fd in fds:
                os.close(fd)                             # the tear: these rows vanish between two reads
            os.sched_setaffinity(0, allowed)
        try:
            tail = b""
            while True:
                page = os.read(r, _PAGE)
                if not page:
                    break
                tail += page
        finally:
            os.close(r)
        lines = [line for line in (head + tail).decode().splitlines(keepends=True) if not self._is_fill(line)]
        if at is not None:
            if any(self._holds(line) for line in lines):
                self.unforced += 1
            else:
                self.hid += 1
        return procfs._parse_listing(lines)

    def hides(self, pid: int, tries: int = 3) -> bool:
        """Does a torn read miss ``pid``'s hold of the file while ``pid``'s own ``fdinfo`` shows it? The PRECONDITION
        of a forced-tear test: False means the tear could not be forced here (the test is then INCONCLUSIVE, never
        a pass by default)."""
        for _ in range(tries):
            listed = {e.pid for e in self(LISTING) if e.file == self.key and not e.blocked}
            if procfs.holds_lock(pid, self.key) and pid not in listed:
                return True
        return False
