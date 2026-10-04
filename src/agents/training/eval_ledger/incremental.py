"""INCREMENTAL READING of an append-only ledger stream, and the SQLite plumbing the two persisted indexes share
(``event_index`` for the requests stream, ``row_index`` for the row shards; F-ED-22).

The streams are APPEND-ONLY (a file only ever grows; a closed row shard is gzipped whole, once). So the index of a
stream is a CACHE: for each file it holds a :class:`Cursor` — how many (decompressed) bytes it has consumed, how
many lines, a digest of the last :data:`TAIL_BYTES` consumed bytes, and the file's size and mtime when it was last
looked at — and a catch-up reads only what lies beyond the cursor. A cursor that the file no longer continues (it
shrank, its tail changed, or it vanished) is a :class:`StaleCursor`, and the index is rebuilt from the streams, never
trusted.

Only COMPLETE lines advance a cursor. A final line without its newline is consumed only if it already parses as JSON
(a hand-written file, like the scans accept); one that does not is an unterminated line — a writer mid-append, or a
killed one — and stays unconsumed, reported as such.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import sqlite3
import struct
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agents.training.eval_ledger import store as ST

#: How many bytes before a cursor's offset its digest covers: enough to tell an append from a rewrite at the tail.
TAIL_BYTES = 64
#: How long a SQLite statement waits for another process's write transaction (the ledger lock is held for ms).
BUSY_S = 30.0


class StaleCursor(Exception):
    """The file is not the append-only continuation of what the index consumed (truncated, rewritten at its tail,
    or its representation changed in a way that lost bytes)."""


class IndexUnavailable(RuntimeError):
    """A persisted index cannot be used right now (locked by a long build, or a root it cannot write): the caller
    falls back to the full scan."""


@dataclass(frozen=True)
class Cursor:
    path: str          # the representative file (relative to the root) as last consumed: ``.jsonl`` or ``.jsonl.gz``
    off: int           # decompressed bytes consumed (complete lines)
    lines: int         # lines consumed, blanks included: the next line is number ``lines + 1``
    tail: str          # sha1 of the (up to TAIL_BYTES) bytes just before ``off``
    size: int          # st_size of the representative file when last looked at
    mtime_ns: int
    eol_owed: bool = False   # the last consumed line had no newline yet (hand-written file): the next byte is it


def stem_of(path: str) -> str:
    return path[:-3] if path.endswith(".gz") else path


def _digest(b: bytes) -> str:
    return hashlib.sha1(b).hexdigest()


def _gz_isize(path: str) -> int:
    """The decompressed length of a (single-member) gzip file, from its trailer."""
    with open(path, "rb") as f:
        f.seek(-4, os.SEEK_END)
        return int(struct.unpack("<I", f.read(4))[0])


def read_since(path: str, rel: str, cur: Optional[Cursor]) -> Tuple[bytes, int, int, "os.stat_result"]:
    """``(buf, prefix, start, st)``: ``buf[prefix:]`` is every byte beyond the cursor (``start`` = ``cur.off``, or 0)
    and ``buf[:prefix]`` the up-to-:data:`TAIL_BYTES` bytes before it. Nothing is read when the file is exactly as
    the cursor last saw it. Raises :class:`StaleCursor` when the file is not the cursor's continuation."""
    st = os.stat(path)
    if cur is None:
        data = _read_all(path)
        return data, 0, 0, st
    if cur.path == rel and cur.size == st.st_size and cur.mtime_ns == st.st_mtime_ns:
        return b"", 0, cur.off, st
    gz = path.endswith(".gz")
    if cur.path.endswith(".gz") and not gz:
        raise StaleCursor(f"{rel}: was a .gz when consumed and is plain now")
    if not gz:
        if st.st_size < cur.off:
            raise StaleCursor(f"{rel}: {st.st_size} bytes now, {cur.off} consumed (truncated)")
        lo = max(0, cur.off - TAIL_BYTES)
        with open(path, "rb") as f:
            f.seek(lo)
            buf = f.read()
        ST.IO.bytes_read += len(buf)
        prefix = cur.off - lo
        if _digest(buf[:prefix]) != cur.tail:
            raise StaleCursor(f"{rel}: the bytes before offset {cur.off} are not what was consumed (rewritten)")
        return buf, prefix, cur.off, st
    isize = _gz_isize(path)
    if isize < cur.off:
        raise StaleCursor(f"{rel}: {isize} decompressed bytes, {cur.off} consumed")
    if isize == cur.off:                 # the same content in another representation (a shard closed): nothing new
        return b"", 0, cur.off, st
    full = _read_all(path)
    lo = max(0, cur.off - TAIL_BYTES)
    prefix = cur.off - lo
    if _digest(full[lo:cur.off]) != cur.tail:
        raise StaleCursor(f"{rel}: the bytes before offset {cur.off} are not what was consumed (rewritten)")
    return full[lo:], prefix, cur.off, st


def _read_all(path: str) -> bytes:
    opener: Any = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        data = f.read()
    ST.IO.bytes_read += len(data)
    return data


@dataclass(frozen=True)
class Line:
    n: int                  # line number in the file
    off: int                # byte offset of the line's first byte (decompressed)
    length: int             # bytes, newline included when there is one
    obj: Any                # the parsed JSON (``None`` when ``err``)
    err: Optional[str]      # the JSON error text of a line that is not JSON
    final: bool             # the unterminated last line of the file


@dataclass(frozen=True)
class Consumed:
    lines: List[Line]
    off: int                # the advanced cursor
    n: int
    tail: str
    eol_owed: bool
    unterminated: bool      # the file ends in a non-blank, non-JSON line with no newline (left unconsumed)


def consume(buf: bytes, prefix: int, start: int, start_line: int, eol_owed: bool = False) -> Consumed:
    """Parse the complete lines of ``buf[prefix:]`` (see :class:`Consumed`). A complete line that is not JSON is a
    :class:`Line` with ``err`` (the caller decides: the events raise, the rows record a problem)."""
    data = buf[prefix:]
    pos = 0
    if eol_owed and data:
        if data[:1] != b"\n":
            raise StaleCursor("a line consumed without its newline was continued on the same line")
        pos = 1
    parts = data[pos:].split(b"\n")
    last = parts.pop()                    # what follows the final newline ("" when the data ends in one)
    out: List[Line] = []
    n = start_line
    for raw in parts:
        n += 1
        length = len(raw) + 1
        if raw.strip():
            ST.IO.lines_parsed += 1
            try:
                out.append(Line(n, start + pos, length, json.loads(raw), None, False))
            except ValueError as e:
                out.append(Line(n, start + pos, length, None, str(e), False))
        pos += length
    unterminated, owed = False, eol_owed and not data
    if last.strip():
        try:
            ST.IO.lines_parsed += 1
            obj = json.loads(last)
        except ValueError as e:
            unterminated = True
            out.append(Line(n + 1, start + pos, len(last), None, str(e), True))
        else:
            n += 1
            out.append(Line(n, start + pos, len(last), obj, None, True))
            pos += len(last)
            owed = True
    end = prefix + pos
    return Consumed(out, start + pos, n, _digest(buf[max(0, end - TAIL_BYTES):end]), owed, unterminated)


def next_cursor(rel: str, st: "os.stat_result", c: Consumed) -> Cursor:
    return Cursor(path=rel, off=c.off, lines=c.n, tail=c.tail, size=int(st.st_size), mtime_ns=int(st.st_mtime_ns),
                  eol_owed=c.eol_owed)


def refreshed(cur: Cursor, rel: str, st: "os.stat_result") -> Cursor:
    """The same consumed position, the file's current identity (a touch, or a shard closed into a ``.gz``)."""
    return Cursor(path=rel, off=cur.off, lines=cur.lines, tail=cur.tail, size=int(st.st_size),
                  mtime_ns=int(st.st_mtime_ns), eol_owed=cur.eol_owed)


# ------------------------------------------------------------------------------------------------ sqlite
#: One idle "keeper" connection per (process, database): while any connection is open, closing another does NOT
#: checkpoint the WAL — without it every operation's connection would be the last, and each close would fsync the
#: database (~10 ms on this box's NVMe: as much as the append itself). The WAL is checkpointed every 1,000 pages instead.
_KEEPERS: Dict[Tuple[int, str], Tuple[int, sqlite3.Connection]] = {}
_KEEPERS_GUARD = threading.Lock()


def connect(path: Path, *, create: bool) -> sqlite3.Connection:
    """A connection in autocommit mode (the caller opens its transactions) with WAL + ``synchronous=NORMAL``: the
    index is a cache, rebuildable from the streams, so a power loss costs a re-read, never correctness — a commit
    and the cursors it advances are ONE transaction, so the index is never ahead of or behind its own cursors."""
    if not create and not path.exists():
        raise IndexUnavailable(f"no index at {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=BUSY_S, isolation_level=None)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        key, ino = (os.getpid(), str(path)), os.stat(path).st_ino
        with _KEEPERS_GUARD:
            held = _KEEPERS.get(key)
            if held is None or held[0] != ino:                  # none yet, or the file was replaced under it
                if held is not None:
                    held[1].close()
                keeper = sqlite3.connect(str(path), timeout=BUSY_S, isolation_level=None, check_same_thread=False)
                keeper.execute("SELECT count(*) FROM sqlite_master").fetchall()    # open the file, take a shm slot
                _KEEPERS[key] = (ino, keeper)
    except sqlite3.DatabaseError:
        conn.close()
        raise
    return conn


def is_corruption(e: sqlite3.DatabaseError) -> bool:
    """A database error that means the FILE is bad (not a database, malformed, a table missing, a constraint the
    state breaks) — as opposed to another process holding a write transaction, or a bug of ours (a
    ``ProgrammingError``: a wrong statement is never cured by a rebuild, and must not be hidden by one)."""
    if isinstance(e, (sqlite3.ProgrammingError, sqlite3.NotSupportedError)):
        return False
    if isinstance(e, sqlite3.OperationalError):
        m = str(e).lower()
        return not ("locked" in m or "busy" in m)
    return True


def is_busy(e: sqlite3.DatabaseError) -> bool:
    """Another process holds the database's write lock (a long build): wait it out or fall back to the scan."""
    return isinstance(e, sqlite3.OperationalError) and ("locked" in str(e).lower() or "busy" in str(e).lower())


def close_keepers(path: Path) -> None:
    """Close this process's keeper on ``path``: the next close of the last connection checkpoints the WAL into the
    database file (a corrupt or replaced file is dealt with by :func:`discard`; tests use this to make the file hold
    everything before they damage it)."""
    with _KEEPERS_GUARD:
        held = _KEEPERS.pop((os.getpid(), str(path)), None)
    if held is not None:
        try:
            held[1].close()
        except sqlite3.Error:
            pass


def discard(path: Path) -> None:
    """Delete an index database and its WAL files (a corrupt one; the next open rebuilds)."""
    close_keepers(path)
    for suffix in ("", "-wal", "-shm", "-journal"):
        try:
            os.unlink(str(path) + suffix)
        except FileNotFoundError:
            pass


def resolve_shard(root: Path, rel: str) -> str:
    """The file that holds ``rel``'s bytes NOW: the recorded path, or its ``.gz`` twin (a shard closed since the
    index last looked — the decompressed bytes are the same)."""
    p = os.path.join(str(root), rel)
    if os.path.exists(p):
        return p
    alt = p[:-3] if p.endswith(".gz") else p + ".gz"
    return alt if os.path.exists(alt) else p
