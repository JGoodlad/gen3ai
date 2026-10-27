"""ONE non-blocking exclusive ``flock`` claim, held by an open file (``designs/tools/convergence_2026-10-26.md``).

The ops tools claim a thing (a port, a supervisor's state, a lane) by an exclusive ``flock`` on a file kept OPEN for the
claim's life: closing the file (or the holder dying) releases it, so a crash leaves no stale claim. Before this module
the open / ``LOCK_EX | LOCK_NB`` / close-on-refusal dance was copied into each tool; :func:`try_exclusive` is it, once.

STDLIB ONLY: it sits under the play path (``main.reads.playpath``) and the read supervisors.
"""
from __future__ import annotations

import fcntl
import os
from typing import IO, Optional


def try_exclusive(path: "str | os.PathLike[str]", mode: str = "w") -> Optional[IO[str]]:
    """The file at ``path`` opened with ``mode`` and holding an exclusive ``flock``, or ``None`` when another open file
    holds it (the file this call opened is closed again). ``mode`` is the caller's: ``"w"`` truncates the file on
    open (even when the claim is refused), ``"a+"`` keeps its content (a holder record the refused caller reads).
    The caller keeps the returned file open for the claim's life; closing it releases the claim. A flock is per open
    file, so two claims in ONE process also exclude each other."""
    fh = open(path, mode)  # noqa: SIM115 - held for the claim's life by the caller
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return fh
