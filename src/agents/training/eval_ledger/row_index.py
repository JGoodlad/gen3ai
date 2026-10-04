"""THE ROW INDEX — a persisted, INCREMENTAL index of the row shards, so a read of ONE request (or one family) costs
that request's rows, not the archive's (F-ED-22; the reader half of the fix — ``event_index`` is the writer's).

Before this module every ``read`` scanned, parsed and validated every row of every shard under the root, and the
SPRT's own decision (its request's rows) paid for the whole archive. The shards are append-only, so this keeps in
``<root>/.ledger_index/rows.sqlite3``:

* **a cursor per shard** (``incremental.Cursor``), like the event index: a catch-up parses only the lines beyond it,
  validates each as the scan would (``schema.as_v2``), and records, per row, WHERE it is (shard, line, byte offset
  and length) and what a read selects on (``row_id``, ``ts``, the request, the family, ``supersedes``, the batch key,
  the seed block, the content sha);
* **what could make the index answer differently from a scan**, kept as it arrives: ``problems`` (a malformed row, a
  repeated ``row_id``) and ``shared_keys`` (a batch key or seed block held by more than one row).

**The fast path answers only when it can prove the scan would agree**, and otherwise returns ``None`` — the reader
then runs the full scan, which is correct for every state and raises its typed errors with their full messages:

* any ``problems`` or an unterminated shard line → scan (the scan raises);
* ``shared_keys`` → each group is evaluated exactly as the scan does, AS OF the read (a row counts only if its ``ts`` is
  not after ``as_of`` and no row with ``ts <= as_of`` supersedes it); a group with two live rows → scan (it raises
  ``DuplicateBatchError``); a group whose extra row was superseded (a correction) passes, as in the scan;
* a row body that is not what was indexed (its content sha differs) → the index is rebuilt once, then scan.

So the guarantees of ``reader`` — every returned row validated, a correction applied as of ``as_of``, NO DUPLICATE
anywhere in the ledger (by ``row_id``, batch key or seed block), ``as_of`` restricting rows and corrections — hold
unchanged; the global duplicate check is a table lookup instead of a scan.

Only ``own`` and ``family`` scopes use it (an ``any`` read consumes the whole archive: nothing to save). A root with
no ``.ledger_index/`` directory (a legacy flat directory, a measurement directory committed to git) is scanned and
never littered; the archive ledger builds its index on first read. ``GEN3AI_LEDGER_INDEX=0`` turns it off.
"""
from __future__ import annotations

import gzip
import json
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, TypeVar

from agents.training.eval_ledger import event_index as EI
from agents.training.eval_ledger import incremental as INC
from agents.training.eval_ledger import schema as S
from agents.training.eval_ledger import store as ST

FORMAT = "gen3_eval_row_index_v1"
T = TypeVar("T")

_SCHEMA_SQL = (
    "CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS files(stem TEXT PRIMARY KEY, path TEXT NOT NULL, producer TEXT NOT NULL, "
    "flat INTEGER NOT NULL, off INTEGER NOT NULL, lines INTEGER NOT NULL, tail TEXT NOT NULL, size INTEGER NOT NULL, "
    "mtime_ns INTEGER NOT NULL, eol_owed INTEGER NOT NULL, unterminated INTEGER NOT NULL)",
    "CREATE TABLE IF NOT EXISTS rows(id INTEGER PRIMARY KEY, stem TEXT NOT NULL, line INTEGER NOT NULL, "
    "off INTEGER NOT NULL, length INTEGER NOT NULL, row_id TEXT NOT NULL, ts_us INTEGER NOT NULL, request_id TEXT, "
    "family TEXT, supersedes TEXT, bkey TEXT, skey TEXT, sha TEXT NOT NULL)",
    "CREATE INDEX IF NOT EXISTS rows_row_id ON rows(row_id)",
    "CREATE INDEX IF NOT EXISTS rows_request ON rows(request_id) WHERE request_id IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS rows_family ON rows(family) WHERE family IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS rows_supersedes ON rows(supersedes) WHERE supersedes IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS rows_bkey ON rows(bkey) WHERE bkey IS NOT NULL",
    "CREATE INDEX IF NOT EXISTS rows_skey ON rows(skey) WHERE skey IS NOT NULL",
    "CREATE TABLE IF NOT EXISTS shared_keys(kind TEXT NOT NULL, k TEXT NOT NULL, PRIMARY KEY(kind, k))",
    "CREATE TABLE IF NOT EXISTS problems(id INTEGER PRIMARY KEY, msg TEXT NOT NULL)",
)
_TABLES = ("meta", "files", "rows", "shared_keys", "problems")
_CHUNK = 500


@dataclass
class Counters:
    syncs: int = 0
    full_builds: int = 0
    rows_indexed: int = 0
    fast_reads: int = 0
    fallbacks: int = 0
    rebuild_reasons: List[str] = field(default_factory=list)

    def reset(self) -> None:
        self.syncs = self.full_builds = self.rows_indexed = self.fast_reads = self.fallbacks = 0
        self.rebuild_reasons = []


COUNTERS = Counters()


class _Rebuild(Exception):
    pass


class _StaleBody(Exception):
    """A row's bytes are not what the index recorded."""


@dataclass(frozen=True)
class RowRec:
    stem: str
    line: int
    off: int
    length: int
    row_id: str
    ts_us: int
    request_id: Optional[str]
    family: Optional[str]
    supersedes: Optional[str]
    bkey: Optional[str]
    skey: Optional[str]
    sha: str


def _chunks(xs: Sequence[T], n: int = _CHUNK) -> List[Sequence[T]]:
    return [xs[i:i + n] for i in range(0, len(xs), n)]


def make_rec(stem: str, ln: INC.Line, row: Dict[str, Any], sha: str) -> RowRec:
    req = row.get("request") or None
    bk, sk = S.batch_key(row), S.seed_key(row)
    return RowRec(stem, ln.n, ln.off, ln.length, str(row["row_id"]), EI.ts_us(row["ts"]),
                  None if req is None else str(req["id"]), None if req is None else req.get("family"),
                  row["supersedes"], None if bk is None else S.canonical(list(bk)),
                  None if sk is None else S.canonical(list(sk)), sha)


class RowIndex:
    """The persisted index of one ledger root's row shards (module docstring)."""

    def __init__(self, root: "str | os.PathLike[str]") -> None:
        self.root = Path(root)
        self.path = self.root / EI.INDEX_DIRNAME / "rows.sqlite3"

    @staticmethod
    def usable(root: Path) -> bool:
        """May a read of ``root`` use (and update) a row index? Only where one already exists, or for the archive's
        own ledger — never a directory that merely holds rows (it would be littered with a cache)."""
        if not EI.enabled() or not root.is_dir():
            return False
        idx = root / EI.INDEX_DIRNAME
        if idx.is_dir():
            return os.access(idx, os.W_OK)
        try:
            arch = ST.archive_ledger_root()
        except Exception:                                           # noqa: BLE001 - no archive: nothing to index
            return False
        return os.path.realpath(root) == os.path.realpath(arch) and os.access(root, os.W_OK)

    # ------------------------------------------------------------------------------------------ public
    def rows_for(self, *, request_id: Optional[str] = None, family: Optional[str] = None,
                 as_of: Any = None) -> Optional[List[ST.ScannedRow]]:
        """The live rows of one request (or family) as of ``as_of`` — what ``reader.live_rows`` then the request /
        family filter would give, in the same order — or ``None`` when the scan must answer (module docstring)."""
        assert (request_id is None) != (family is None)
        t = None if as_of is None else EI._us(as_of)
        for attempt in (0, 1):
            try:
                got = self._recovering(lambda: self._rows_for(request_id, family, t))
            except _StaleBody as e:
                if attempt:
                    COUNTERS.fallbacks += 1
                    return None
                COUNTERS.rebuild_reasons.append(str(e))
                self.rebuild()
                continue
            if got is None:
                COUNTERS.fallbacks += 1
            else:
                COUNTERS.fast_reads += 1
            return got
        return None

    def rebuild(self) -> None:
        def go() -> None:
            conn = self._open()
            try:
                self._begin(conn)
                try:
                    self._sync_tx(conn, full=True, why="requested")
                    conn.execute("COMMIT")
                except BaseException:
                    self._rollback(conn)
                    raise
            finally:
                conn.close()

        self._recovering(go)

    def verify(self) -> List[str]:
        """Re-parse every shard UP TO ITS CURSOR and compare with the ``rows`` / ``shared_keys`` / ``problems`` tables."""
        out: List[str] = []
        if not self.path.exists():
            return out
        try:
            conn = INC.connect(self.path, create=False)
        except sqlite3.DatabaseError as e:
            return [f"row index: cannot open {self.path} ({e}); rebuild with `audit --rebuild-index`"]
        try:
            if (ic := conn.execute("PRAGMA integrity_check").fetchone()) is None or ic[0] != "ok":
                return [f"row index: sqlite integrity_check says {ic}"]
            meta = {k: json.loads(v) for k, v in conn.execute("SELECT k, v FROM meta")}
            if meta.get("format") != FORMAT:
                return [f"row index: format {meta.get('format')!r}, this code writes {FORMAT!r}"]
            want: Set[RowRec] = set()
            want_problems: List[str] = []
            keys: Dict[Tuple[str, str], int] = {}
            ids: Dict[str, int] = {}
            for stem, path, off, lines, tail in conn.execute(
                    "SELECT stem, path, off, lines, tail FROM files ORDER BY stem").fetchall():
                try:
                    data = INC._read_all(INC.resolve_shard(self.root, path))[:off]
                except OSError as e:
                    out.append(f"row index: tracked shard {path} cannot be read ({e})")
                    continue
                if len(data) != off or INC._digest(data[max(0, off - INC.TAIL_BYTES):off]) != tail:
                    out.append(f"row index: {path}: the first {off} bytes are not what the cursor's digest saw")
                    continue
                c = INC.consume(data, 0, 0, 0)
                if c.n != lines:
                    out.append(f"row index: {path}: {c.n} lines up to the cursor, the index says {lines}")
                for ln in c.lines:
                    rec, prob = self._parse(stem, path, ln)
                    if prob:
                        want_problems.append(prob)
                    if rec is not None:
                        want.add(rec)
                        ids[rec.row_id] = ids.get(rec.row_id, 0) + 1
                        for kind, k in (("b", rec.bkey), ("s", rec.skey)):
                            if k is not None:
                                keys[(kind, k)] = keys.get((kind, k), 0) + 1
            if out:
                return out
            have = {RowRec(*r) for r in conn.execute(
                "SELECT stem, line, off, length, row_id, ts_us, request_id, family, supersedes, bkey, skey, sha "
                "FROM rows")}
            if have != want:
                out.append(f"row index: the rows table holds {len(have)} rows, the shards fold to {len(want)} "
                           f"(only in shards: {len(want - have)}, only in index: {len(have - want)})")
            shared = {(str(a), str(b)) for a, b in conn.execute("SELECT kind, k FROM shared_keys")}
            if shared != {k for k, n in keys.items() if n > 1}:
                out.append("row index: shared_keys is not the set of keys held by more than one row")
            n_prob = conn.execute("SELECT count(*) FROM problems").fetchone()[0]
            n_dup = sum(n - 1 for n in ids.values() if n > 1)
            if n_prob != len(want_problems) + n_dup:
                out.append(f"row index: problems holds {n_prob}, the shards give {len(want_problems) + n_dup}")
            return out
        finally:
            conn.close()

    # ------------------------------------------------------------------------------------------ internals
    def _open(self) -> sqlite3.Connection:
        return INC.connect(self.path, create=True)

    def _recovering(self, fn: Callable[[], T]) -> T:
        try:
            return fn()
        except sqlite3.DatabaseError as e:
            if INC.is_busy(e):
                raise INC.IndexUnavailable(f"the row index {self.path} is busy: {e}") from e
            if not INC.is_corruption(e):
                raise
            COUNTERS.rebuild_reasons.append(f"corrupt index file: {e}")
            INC.discard(self.path)
            return fn()

    @staticmethod
    def _begin(conn: sqlite3.Connection) -> None:
        conn.execute("BEGIN IMMEDIATE")
        for sql in _SCHEMA_SQL:
            conn.execute(sql)

    @staticmethod
    def _rollback(conn: sqlite3.Connection) -> None:
        if conn.in_transaction:
            conn.execute("ROLLBACK")

    def _sync(self, conn: sqlite3.Connection) -> None:
        COUNTERS.syncs += 1
        try:
            self._begin(conn)
            try:
                self._sync_tx(conn, full=False)
                conn.execute("COMMIT")
                return
            except _Rebuild as why:
                self._rollback(conn)
                reason = str(why)
            except BaseException:
                self._rollback(conn)
                raise
            self._begin(conn)
            try:
                self._sync_tx(conn, full=True, why=reason)
                conn.execute("COMMIT")
            except BaseException:
                self._rollback(conn)
                raise
        except sqlite3.DatabaseError:
            self._rollback(conn)
            raise

    def _parse(self, stem: str, rel: str, ln: INC.Line) -> Tuple[Optional[RowRec], Optional[str]]:
        """``(record, None)`` for a valid row, ``(None, problem)`` for a line the scan would raise on."""
        if ln.final and ln.err is not None:
            return None, None                                       # unterminated: reported by the cursor
        where = f"{rel}:{ln.n}"
        if ln.err is not None:
            return None, f"{os.path.join(str(self.root), rel)}:{ln.n}: not JSON ({ln.err})"
        try:
            return make_rec(stem, ln, S.as_v2(ln.obj, where), S.content_sha(ln.obj)), None
        except S.LedgerSchemaError as e:
            return None, str(e)

    def _sync_tx(self, conn: sqlite3.Connection, *, full: bool, why: str = "") -> None:
        if full:
            for t in _TABLES:
                conn.execute(f"DELETE FROM {t}")
            COUNTERS.full_builds += 1
            COUNTERS.rebuild_reasons.append(why)
        else:
            meta = {k: json.loads(v) for k, v in conn.execute("SELECT k, v FROM meta")}
            if meta.get("format") != FORMAT:
                raise _Rebuild("no index yet" if not meta else f"format {meta.get('format')!r}")
        recorded = {r[0]: (INC.Cursor(path=r[1], off=r[4], lines=r[5], tail=r[6], size=r[7], mtime_ns=r[8],
                                      eol_owed=bool(r[9])), r[2], bool(r[3]), bool(r[10]))
                    for r in conn.execute("SELECT stem, path, producer, flat, off, lines, tail, size, mtime_ns, "
                                          "eol_owed, unterminated FROM files")}
        listed: Dict[str, Tuple[str, str, bool]] = {}
        for prod, p in ST.row_shards(self.root):
            listed[INC.stem_of(os.path.relpath(p, self.root))] = (p, prod, os.path.dirname(p) == str(self.root))
        gone = sorted(set(recorded) - set(listed))
        if gone:
            raise _Rebuild(f"tracked shard(s) vanished: {gone[:3]}")
        new_files: List[Tuple[Any, ...]] = []
        n_new = 0
        for stem, (path, prod, flat) in listed.items():
            rel = os.path.relpath(path, self.root)
            cur, _prod0, _flat0, unterminated0 = recorded.get(stem, (None, prod, flat, False))
            try:
                buf, prefix, start, st = INC.read_since(path, rel, cur)
                if len(buf) == prefix and cur is not None:
                    if (cur.path, cur.size, cur.mtime_ns) != (rel, st.st_size, st.st_mtime_ns):
                        cur = INC.refreshed(cur, rel, st)
                        new_files.append(self._file_row(stem, cur, prod, flat, unterminated0))
                    continue
                c = INC.consume(buf, prefix, start, 0 if cur is None else cur.lines,
                                False if cur is None else cur.eol_owed)
            except INC.StaleCursor as e:
                raise _Rebuild(str(e)) from None
            for ln in c.lines:
                rec, prob = self._parse(stem, rel, ln)
                if prob is not None:
                    conn.execute("INSERT INTO problems(msg) VALUES (?)", (prob,))
                if rec is not None:
                    self._insert(conn, rec)
                    n_new += 1
            new_files.append(self._file_row(stem, INC.next_cursor(rel, st, c), prod, flat, c.unterminated))
        conn.executemany("INSERT OR REPLACE INTO files(stem, path, producer, flat, off, lines, tail, size, mtime_ns, "
                         "eol_owed, unterminated) VALUES (?,?,?,?,?,?,?,?,?,?,?)", new_files)
        COUNTERS.rows_indexed += n_new
        conn.execute("INSERT OR REPLACE INTO meta(k, v) VALUES ('format', ?)", (json.dumps(FORMAT),))

    @staticmethod
    def _file_row(stem: str, c: INC.Cursor, prod: str, flat: bool, unterminated: bool) -> Tuple[Any, ...]:
        return (stem, c.path, prod, int(flat), c.off, c.lines, c.tail, c.size, c.mtime_ns, int(c.eol_owed),
                int(unterminated))

    @staticmethod
    def _insert(conn: sqlite3.Connection, r: RowRec) -> None:
        if conn.execute("SELECT 1 FROM rows WHERE row_id = ? LIMIT 1", (r.row_id,)).fetchone():
            conn.execute("INSERT INTO problems(msg) VALUES (?)",
                         (f"row_id {r.row_id} appears twice (a shard was copied)",))
        for kind, col, k in (("b", "bkey", r.bkey), ("s", "skey", r.skey)):
            if k is not None and conn.execute(f"SELECT 1 FROM rows WHERE {col} = ? LIMIT 1", (k,)).fetchone():
                conn.execute("INSERT OR IGNORE INTO shared_keys(kind, k) VALUES (?, ?)", (kind, k))
        conn.execute("INSERT INTO rows(stem, line, off, length, row_id, ts_us, request_id, family, supersedes, bkey, "
                     "skey, sha) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                     (r.stem, r.line, r.off, r.length, r.row_id, r.ts_us, r.request_id, r.family, r.supersedes,
                      r.bkey, r.skey, r.sha))

    # ------------------------------------------------------------------------------------------ the read
    def _dead(self, conn: sqlite3.Connection, ids: Sequence[str], t: Optional[int]) -> Set[str]:
        dead: Set[str] = set()
        for ch in _chunks(list(ids)):
            q = f"SELECT supersedes FROM rows WHERE supersedes IN ({','.join('?' * len(ch))})"
            args: List[Any] = list(ch)
            if t is not None:
                q += " AND ts_us <= ?"
                args.append(t)
            dead.update(str(r[0]) for r in conn.execute(q, args))
        return dead

    def _clean(self, conn: sqlite3.Connection, t: Optional[int]) -> bool:
        """Would a full scan of the ledger AS OF ``t`` raise? (``False`` = it might: the scan must answer.)"""
        if conn.execute("SELECT 1 FROM problems LIMIT 1").fetchone() or \
                conn.execute("SELECT 1 FROM files WHERE unterminated = 1 LIMIT 1").fetchone():
            return False
        for kind, k in conn.execute("SELECT kind, k FROM shared_keys").fetchall():
            col = "bkey" if kind == "b" else "skey"
            q = f"SELECT row_id FROM rows WHERE {col} = ?"
            args: List[Any] = [k]
            if t is not None:
                q += " AND ts_us <= ?"
                args.append(t)
            members = [str(r[0]) for r in conn.execute(q, args)]
            if len(members) - len(self._dead(conn, members, t) & set(members)) > 1:
                return False
        return True

    def _rows_for(self, request_id: Optional[str], family: Optional[str], t: Optional[int]
                  ) -> Optional[List[ST.ScannedRow]]:
        conn = self._open()
        try:
            self._sync(conn)
            if not self._clean(conn, t):
                return None
            col, val = ("request_id", request_id) if request_id is not None else ("family", family)
            q = ("SELECT stem, line, off, length, row_id, sha FROM rows WHERE " + col + " = ?"
                 + ("" if t is None else " AND ts_us <= ?"))
            cand = conn.execute(q, [val] if t is None else [val, t]).fetchall()
            dead = self._dead(conn, [str(r[4]) for r in cand], t)
            live = [r for r in cand if str(r[4]) not in dead]
            stems = sorted({str(r[0]) for r in live})
            files = {str(r[0]): (str(r[1]), str(r[2]), bool(r[3])) for r in conn.execute(
                "SELECT stem, path, producer, flat FROM files WHERE stem IN (%s)" % ",".join("?" * len(stems)),
                stems)} if stems else {}
        finally:
            conn.close()
        out: List[Tuple[Tuple[int, str, str, int], ST.ScannedRow]] = []
        by_stem: Dict[str, List[Tuple[Any, ...]]] = {}
        for r in live:
            by_stem.setdefault(str(r[0]), []).append(r)
        for stem, recs in by_stem.items():
            path, prod, flat = files[stem]
            full = os.path.join(str(self.root), path)
            gz = full.endswith(".gz")
            whole = gzip.open(full, "rb").read() if gz else None
            if whole is not None:
                ST.IO.bytes_read += len(whole)
            f = None if gz else open(full, "rb")
            try:
                for _stem, line, off, length, row_id, sha in recs:
                    if whole is not None:
                        raw = whole[off:off + length]
                    else:
                        assert f is not None
                        f.seek(off)
                        raw = f.read(length)
                        ST.IO.bytes_read += len(raw)
                    ST.IO.lines_parsed += 1
                    try:
                        obj = json.loads(raw)
                        ok = S.content_sha(obj) == sha
                    except ValueError:
                        ok = False
                    if not ok:
                        raise _StaleBody(f"{path}:{line}: the bytes are not what the index recorded for {row_id}")
                    where = f"{path}:{line}"
                    out.append(((0 if flat else 1, prod, path, int(line)),
                                ST.ScannedRow(S.as_v2(obj, where), sha, where, prod)))
            finally:
                if f is not None:
                    f.close()
        out.sort(key=lambda x: x[0])
        return [sr for _k, sr in out]
