"""ONE content digest of a file (``designs/tools/convergence_2026-10-26.md``): sha256, read in 1 MiB chunks.

The read and x35 tools each carried a byte-identical copy (``kakuna._sha``, ``reads.models.sha256_file``,
``x35.gc._file_sha``); they import this one. STDLIB ONLY: it sits under the play path (``main.reads.playpath``).
"""
from __future__ import annotations

import hashlib
import os

CHUNK = 1 << 20


def sha256_file(path: "str | os.PathLike[str]") -> str:
    """The lowercase hex sha256 of the file's bytes (``OSError`` when it cannot be read)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()
