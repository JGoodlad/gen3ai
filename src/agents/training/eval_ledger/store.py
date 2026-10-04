"""WHERE THE LEDGER LIVES, and the raw file operations under it: the archive root, the layout, the one CPU file
lock, the append-only JSONL files, the per-stream SCANS, and closing (gzipping) a shard.

**The root (§0b.3).** ``<archive>/_ledger/`` where ``<archive>`` = ``utils.paths.run_archive_dir()`` — archive
level, never per run, so a cross-run question needs no new run. ``$GEN3AI_MODELS_DIR`` is authoritative, and
pytest SEALS the archive (the ``run_archive`` fixture). A caller may name ANOTHER root (a measurement directory
outside ``models/``, a test's temp dir); under ``models/`` a writer may write ONLY to ``<archive>/_ledger/`` itself
(:func:`check_write_root` — ``models/`` stays read-only to everything else, and a worktree's own ``models/`` dies
with the worktree).

**The layout.** ``rows/<producer>/ledger.<writer id>.jsonl`` (gzipped once closed) · ``requests/events.<writer
id>.jsonl`` + ``requests/.lock`` · ``decisions/decisions.<writer id>.jsonl`` · ``references/references.<writer
id>.jsonl`` · ``README.md``. ONE WRITER PER FILE: ``writer id`` = UTC time + a per-process counter + host + pid + producer, and every
append is flushed and fsynced. A legacy FLAT directory of v1 shards (``<dir>/ledger.*.jsonl``, what ``main.h2h``
and the bot round robin wrote before v2) is still scanned, as producer ``legacy``.

The SCANS here (and the persisted indexes built on them, ``event_index`` / ``row_index`` — caches under
``<root>/.ledger_index/``, F-ED-22) are the only code that opens a ledger file for reading. Every consumer reads through
``eval_ledger.read`` with a declaration (``reader.py``); ``src/eval_ledger_reader_gate_test.py`` holds the line.
"""
from __future__ import annotations

import contextlib
import datetime as _dt
import fcntl
import glob
import itertools
import gzip
import json
import os
import re
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from agents.training.eval_ledger import schema as S

LEDGER_DIRNAME = "_ledger"
ROWS, REQUESTS, DECISIONS, REFERENCES = "rows", "requests", "decisions", "references"
SHARD_PREFIX, SHARD_SUFFIX = "ledger.", ".jsonl"
EVENTS_PREFIX, DECISIONS_PREFIX, REFERENCES_PREFIX = "events.", "decisions.", "references."
LOCK_NAME = ".lock"
LEGACY_PRODUCER = "legacy"
#: A closed shard's idle age before ``close-stale`` may gzip a DEAD writer's file (§0b.3).
STALE_IDLE_S = 24 * 3600
#: How long an append waits for the lock. The lock is held for milliseconds (a fold of the requests stream and
#: one or two appends), so a wait this long means a stuck holder, not contention.
LOCK_TIMEOUT_S = 60.0

_PRODUCER = re.compile(r"^[a-z0-9_]+$")

README = """# The eval COUNT ledger (`gen3_eval_count_row_v2`)

Append-only. One row per (batch x matchup). NEVER edit, move or delete a file here: a correction is a new row
that `supersedes` the old one. Spec: `designs/endstate/design_evaluation.md` §0b. Code:
`src/agents/training/eval_ledger/`. Check it: `python -m main.eval_ledger audit`.
(`.ledger_index/` is the one exception: a CACHE derived from the streams, safe to delete and rebuilt on demand.)
"""


class LedgerPathError(RuntimeError):
    """A ledger root this writer may not use."""


@dataclass
class IOStats:
    """Process-global counters of what the ledger's readers PARSE — the performance-SHAPE tests assert on these
    (F-ED-22): a claim or an append at any archive size parses the same handful of lines, and the full scans
    (:func:`iter_jsonl`) count here too, so a regression to a whole-archive fold is visible as a count that grows."""

    lines_parsed: int = 0
    bytes_read: int = 0

    def reset(self) -> None:
        self.lines_parsed = self.bytes_read = 0


IO = IOStats()


class LedgerLockTimeout(RuntimeError):
    """The ledger's file lock was not acquired within its bound."""


# ------------------------------------------------------------------------------------------------ the root
def archive_ledger_root() -> Path:
    """``<archive>/_ledger`` (the archive per ``utils.paths.run_archive_dir`` — raises its typed refusal when
    there is no archive, e.g. a sealed test without the ``run_archive`` fixture)."""
    from utils.paths import run_archive_dir

    return run_archive_dir() / LEDGER_DIRNAME


def _inside(p: str, root: str) -> bool:
    return p == root or p.startswith(root.rstrip(os.sep) + os.sep)


def check_write_root(path: "str | os.PathLike[str]") -> Path:
    """``path`` as a ledger root a WRITER may use, or :class:`LedgerPathError`. Outside every ``models/`` it is the
    caller's choice. Under one (main's archive, ``$GEN3AI_MODELS_DIR``, this checkout's own ``models/``) only the
    run archive's ``_ledger`` directory itself is allowed."""
    from utils.paths import main_models_dir, repo_root

    rp = os.path.realpath(os.fspath(path))
    roots = [repo_root() / "models"]
    md = main_models_dir()
    if md is not None:
        roots.append(md)
    env = os.environ.get("GEN3AI_MODELS_DIR")
    if env:
        roots.append(Path(env))
    allowed: Optional[str] = None
    try:
        allowed = os.path.realpath(archive_ledger_root())
    except Exception:                                               # noqa: BLE001 - no archive: nothing is allowed
        allowed = None
    for r in roots:
        if _inside(rp, os.path.realpath(r)) and rp != allowed:
            raise LedgerPathError(
                f"REFUSED: {path} is under {r} — models/ is read-only to a ledger writer except the run archive's "
                f"own ledger {allowed or '<archive>/_ledger'} (design_evaluation.md §0b.3); omit the root to write "
                "there, or name a directory outside models/")
    return Path(rp)


def resolve_root(root: "str | os.PathLike[str] | None") -> Path:
    """A READER's root: the named one, else the archive ledger."""
    return Path(root) if root is not None else archive_ledger_root()


def ensure_layout(root: Path) -> None:
    for sub in (ROWS, REQUESTS, DECISIONS, REFERENCES):
        (root / sub).mkdir(parents=True, exist_ok=True)
    readme = root / "README.md"
    if not readme.exists():
        tmp = root / f".README.md.{os.getpid()}.tmp"
        tmp.write_text(README)
        os.replace(tmp, readme)


# ------------------------------------------------------------------------------------------------ writer ids
#: A per-process counter: two writers one process opens within one second still get distinct ids.
_WRITER_SEQ = itertools.count()


def make_writer_id(producer: str, *, now: Optional[_dt.datetime] = None, host: Optional[str] = None,
                   pid: Optional[int] = None) -> str:
    if not _PRODUCER.match(producer):
        raise ValueError(f"producer {producer!r}: lowercase letters, digits and '_' only")
    t = (now or _dt.datetime.now(_dt.timezone.utc)).strftime("%Y%m%dT%H%M%SZ")
    h = re.sub(r"[^A-Za-z0-9.]", "_", host or socket.gethostname()) or "host"
    return f"{t}.{next(_WRITER_SEQ)}-{h}-{pid if pid is not None else os.getpid()}-{producer}"


@dataclass(frozen=True)
class WriterIdent:
    host: str
    pid: int
    producer: str


def parse_writer_id(writer_id: str) -> Optional[WriterIdent]:
    """``<time>.<n>-<host>-<pid>-<producer>`` (a v1 writer id ``<time>-<pid>`` has no host: ``None``)."""
    parts = writer_id.split("-")
    if len(parts) < 4 or not parts[-2].isdigit():
        return None
    return WriterIdent(host="-".join(parts[1:-2]), pid=int(parts[-2]), producer=parts[-1])


def this_host() -> str:
    return re.sub(r"[^A-Za-z0-9.]", "_", socket.gethostname()) or "host"


def pid_alive(pid: int) -> bool:
    """Is ``pid`` a live process on THIS host (signal 0; a process we may not signal is alive)."""
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ------------------------------------------------------------------------------------------------ the lock
_LOCKS: Dict[str, Tuple[int, int]] = {}
_LOCKS_GUARD = threading.Lock()


@contextlib.contextmanager
def locked(root: Path, timeout_s: float = LOCK_TIMEOUT_S) -> Iterator[None]:
    """The ledger's ONE exclusive lock: ``flock`` on ``<root>/requests/.lock`` (a CPU file lock; nothing to do with
    the GPU). Claims and claimed appends happen under it (§0b.4). Re-entrant within a process. Bounded: a holder
    that does not let go within ``timeout_s`` is :class:`LedgerLockTimeout`, never an unbounded wait."""
    key = os.path.realpath(root)
    with _LOCKS_GUARD:
        held = _LOCKS.get(key)
        if held is not None:
            _LOCKS[key] = (held[0], held[1] + 1)
    if held is not None:
        try:
            yield
        finally:
            with _LOCKS_GUARD:
                fd, depth = _LOCKS[key]
                _LOCKS[key] = (fd, depth - 1)
        return
    (root / REQUESTS).mkdir(parents=True, exist_ok=True)
    fd = os.open(root / REQUESTS / LOCK_NAME, os.O_RDWR | os.O_CREAT, 0o644)
    deadline = time.monotonic() + float(timeout_s)
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise LedgerLockTimeout(f"the ledger lock {root / REQUESTS / LOCK_NAME} was not acquired within "
                                            f"{timeout_s:.0f} s — a holder is stuck (it is held for milliseconds)")
                time.sleep(0.05)
        with _LOCKS_GUARD:
            _LOCKS[key] = (fd, 1)
        try:
            yield
        finally:
            with _LOCKS_GUARD:
                del _LOCKS[key]
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def lock_held(root: Path) -> bool:
    """Does THIS process hold the ledger lock on ``root`` (the persisted indexes are updated only under it)?"""
    with _LOCKS_GUARD:
        return os.path.realpath(root) in _LOCKS


# ------------------------------------------------------------------------------------------------ files
def append_line(path: Path, obj: Any) -> None:
    """Append one canonical JSON line, flushed and fsynced (a batch is minutes of play: it must survive a kill)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(S.canonical(obj) + "\n")
        f.flush()
        os.fsync(f.fileno())


def iter_jsonl(path: str) -> Iterator[Tuple[int, Any]]:
    """``(line number, object)`` for every non-blank line; a line that is not JSON is a typed refusal."""
    opener: Callable[..., Any] = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            IO.lines_parsed += 1
            IO.bytes_read += len(line)
            try:
                yield n, json.loads(line)
            except ValueError as e:
                raise S.LedgerSchemaError(f"{path}:{n}: not JSON ({e})") from None


def one_per_stem(paths: List[str]) -> List[str]:
    """One file per shard: when a ``.jsonl`` and its ``.jsonl.gz`` both exist (a close in flight: the ``.gz`` is
    complete before the ``.jsonl`` is unlinked), the ``.gz``."""
    gz = {p[:-3] for p in paths if p.endswith(".gz")}
    return sorted(p for p in paths if p.endswith(".gz") or p not in gz)


def _files(d: Path, prefix: str) -> List[str]:
    return one_per_stem(glob.glob(os.path.join(str(d), f"{prefix}*{SHARD_SUFFIX}"))
                        + glob.glob(os.path.join(str(d), f"{prefix}*{SHARD_SUFFIX}.gz")))


def event_files(root: Path) -> List[str]:
    """Every requests-stream file under ``root`` (one per stem, the ``.gz`` when both exist), in scan order."""
    return _files(root / REQUESTS, EVENTS_PREFIX)


def row_shards(root: Path) -> List[Tuple[str, str]]:
    """``(producer, path)`` of every row shard under ``root``: ``rows/<producer>/`` plus a legacy flat directory."""
    out = [(LEGACY_PRODUCER, p) for p in _files(root, SHARD_PREFIX)]
    rows = root / ROWS
    if rows.is_dir():
        for prod in sorted(os.listdir(rows)):
            if (rows / prod).is_dir():
                out.extend((prod, p) for p in _files(rows / prod, SHARD_PREFIX))
    return out


def shard_path(root: Path, producer: str, writer_id: str) -> Path:
    return root / ROWS / producer / f"{SHARD_PREFIX}{writer_id}{SHARD_SUFFIX}"


@dataclass(frozen=True)
class ScannedRow:
    row: Dict[str, Any]          # the v2 view (a v1 row upgraded)
    sha: str                     # content sha of the row AS STORED
    source: str                  # "<file>:<line>"
    producer: str


def scan_rows(root: Path, problems: Optional[List[str]] = None) -> List[ScannedRow]:
    """Every row under ``root``, each validated (v1 rows validated as v1 and upgraded). A malformed row is a typed
    :class:`~schema.LedgerSchemaError` naming its file and line — or, when ``problems`` is given (the audit), a
    line in it and the row is skipped. Order: producers, shard names, line order."""
    out: List[ScannedRow] = []
    for prod, p in row_shards(root):
        try:
            for n, obj in iter_jsonl(p):
                where = f"{os.path.relpath(p, root)}:{n}"
                try:
                    out.append(ScannedRow(S.as_v2(obj, where), S.content_sha(obj), where, prod))
                except S.LedgerSchemaError as e:
                    if problems is None:
                        raise
                    problems.append(str(e))
        except S.LedgerSchemaError as e:
            if problems is None:
                raise
            problems.append(str(e))
    return out


def _scan_stream(root: Path, sub: str, prefix: str, validate: Callable[[Any], List[str]],
                 problems: Optional[List[str]]) -> List[Tuple[Dict[str, Any], str]]:
    out: List[Tuple[Dict[str, Any], str]] = []
    for p in _files(root / sub, prefix):
        try:
            for n, obj in iter_jsonl(p):
                where = f"{os.path.relpath(p, root)}:{n}"
                errs = validate(obj)
                if errs:
                    msg = f"{where}: {len(errs)} schema problem(s):\n  " + "\n  ".join(errs)
                    if problems is None:
                        raise S.LedgerSchemaError(msg)
                    problems.append(msg)
                    continue
                out.append((obj, where))
        except S.LedgerSchemaError as e:
            if problems is None:
                raise
            problems.append(str(e))
    return out


def scan_events(root: Path, problems: Optional[List[str]] = None) -> List[Tuple[Dict[str, Any], str]]:
    return _scan_stream(root, REQUESTS, EVENTS_PREFIX, S.validate_event, problems)


def scan_decisions(root: Path, problems: Optional[List[str]] = None) -> List[Tuple[Dict[str, Any], str]]:
    return _scan_stream(root, DECISIONS, DECISIONS_PREFIX, S.validate_decision, problems)


def scan_references(root: Path, problems: Optional[List[str]] = None) -> List[Tuple[Dict[str, Any], str]]:
    return _scan_stream(root, REFERENCES, REFERENCES_PREFIX, S.validate_reference, problems)


def shard_has_unit(root: Path, producer: str, writer_id: str, unit: Tuple[Any, ...]) -> Optional[str]:
    """The ``row_id`` of a row for ``unit`` in ONE writer's shard, or ``None`` — the void rule's "no row exists"
    checked at the source, so a writer killed between appending its row and its ``row`` event is not replayed."""
    base = shard_path(root, producer, writer_id)
    for p in one_per_stem([str(x) for x in (base, Path(str(base) + ".gz")) if x.exists()]):
        for _n, obj in iter_jsonl(p):
            try:
                row = S.as_v2(obj)
            except S.LedgerSchemaError:
                continue
            k = S.batch_key(row)
            if k is not None and k[1:] == tuple(unit):
                return str(row["row_id"])
    return None


# ------------------------------------------------------------------------------------------------ closing
def gzip_shard(path: Path) -> Path:
    """Close a shard: write ``<path>.gz`` (via a temp file, fsynced, renamed), then unlink ``<path>``. A reader
    that globs in between sees both and reads the ``.gz`` (:func:`one_per_stem`)."""
    gz = Path(str(path) + ".gz")
    tmp = Path(str(path) + ".gz.tmp")
    with open(path, "rb") as src, gzip.open(tmp, "wb") as dst:
        dst.write(src.read())
    with open(tmp, "rb") as f:
        os.fsync(f.fileno())
    os.replace(tmp, gz)
    os.unlink(path)
    return gz


@dataclass
class StaleReport:
    closed: List[str]
    skipped: List[str]


def close_stale(root: Path, *, now: Optional[float] = None, alive: Callable[[int], bool] = pid_alive,
                host: Optional[str] = None, idle_s: float = STALE_IDLE_S, apply: bool = True) -> StaleReport:
    """Gzip every OPEN row shard whose writer is DEAD on this host AND whose file has been idle ``idle_s`` (§0b.3).
    A shard written on another host (its pid cannot be checked here) or by a live writer is skipped, and said.
    ``apply=False`` reports what WOULD be closed (``closed``) without touching a file."""
    now = time.time() if now is None else now
    host = host or this_host()
    rep = StaleReport([], [])
    for _prod, p in row_shards(root):
        if p.endswith(".gz"):
            continue
        wid = os.path.basename(p)[len(SHARD_PREFIX):-len(SHARD_SUFFIX)]
        ident = parse_writer_id(wid)
        if ident is None:
            rep.skipped.append(f"{p}: a writer id without host/pid (v1) — not closed")
        elif ident.host != host:
            rep.skipped.append(f"{p}: written on {ident.host}, not this host — its pid cannot be checked here")
        elif alive(ident.pid):
            rep.skipped.append(f"{p}: writer pid {ident.pid} is alive")
        elif now - os.path.getmtime(p) < idle_s:
            rep.skipped.append(f"{p}: idle {now - os.path.getmtime(p):.0f} s < {idle_s:.0f} s")
        else:
            if apply:
                gzip_shard(Path(p))
            rep.closed.append(p)
    return rep
