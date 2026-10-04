"""THE REQUESTS-STREAM INDEX — a persisted, INCREMENTAL fold of ``requests/`` (F-ED-22).

Before this module, every claim and every claimed append folded the WHOLE archive's requests stream under the lock
(~7 µs per event, linear: 38 ms at 5,100 events, 0.5 s at 50,000). The stream is append-only and ``seq`` is assigned
under the lock, so a new event always folds AFTER every old one: the fold is incremental by nature. This module
keeps it in ``<root>/.ledger_index/events.sqlite3``:

* **A per-file cursor** (``incremental.Cursor``): how far each events file has been folded. A catch-up lists the
  files, ``stat``s them, and parses only the bytes beyond each cursor — O(new events), plus one ``stat`` per file.
* **The folded state, keyed so one request is one lookup**: ``requests`` (the open event, done / cancelled, the
  regimes its rows hold), ``claims`` (indexed by unit and by liveness), ``unit_rows`` (the ``row`` event of a unit),
  ``seed_keys`` (the seed-block uniqueness index, one primary key), ``families`` and ``family_events`` (every
  registration, for an ``as_of`` read), and ``meta`` (``max_seq``, the fold's ``problems``).
* **ONE implementation of what an event means**: a new event is applied by ``queue.apply_event`` — the function
  ``queue.fold`` is made of — on a ``QueueState`` loaded for just the keys that event touches (its request, its
  unit, its claim). The incremental fold and the from-scratch fold cannot disagree about semantics; what can differ
  is only whether the persisted state is what the streams fold to, which :meth:`EventIndex.verify` (``audit``) checks.

**It is a CACHE, never trusted blindly.** It is rebuilt from the streams, in one transaction (readers see the old
state or the new), whenever: it does not exist or is another format; the database is corrupt; a tracked file shrank,
was rewritten at its tail, or vanished; or a new event has ``seq`` <= the indexed maximum (an out-of-order or
duplicate ``seq`` — the from-scratch fold, which sorts, must place it). A malformed event is the same typed
:class:`~schema.LedgerSchemaError` the scan raised, naming file and line. What a catch-up cannot see is an edit
INSIDE an old, same-size region of a file (the streams are append-only: nothing does that); ``audit`` re-folds every
file up to its cursor and compares (``python -m main.eval_ledger audit``; ``--rebuild-index`` rebuilds).

**Locking.** Every method that writes (``state``, ``rebuild``, ``family_as_of``) runs under the ledger's ONE file
lock (``store.locked``) — the same lock that serialises events — so concurrent writers (one file each) see one
consistent index. ``GEN3AI_LEDGER_INDEX=0`` disables the index everywhere (the writer folds the whole stream, the
reader scans; the pre-F-ED-22 behaviour) — a debugging hatch and the tests' differential oracle.
"""
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, TypeVar

from agents.training.eval_ledger import incremental as INC
from agents.training.eval_ledger import queue as Q
from agents.training.eval_ledger import schema as S
from agents.training.eval_ledger import store as ST

INDEX_DIRNAME = ".ledger_index"
FORMAT = "gen3_eval_event_index_v1"
ENV_SWITCH = "GEN3AI_LEDGER_INDEX"
T = TypeVar("T")

_SCHEMA_SQL = (
    "CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS files(stem TEXT PRIMARY KEY, path TEXT NOT NULL, off INTEGER NOT NULL, "
    "lines INTEGER NOT NULL, tail TEXT NOT NULL, size INTEGER NOT NULL, mtime_ns INTEGER NOT NULL, "
    "eol_owed INTEGER NOT NULL)",
    "CREATE TABLE IF NOT EXISTS families(family_id TEXT PRIMARY KEY, ev TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS family_events(seq INTEGER PRIMARY KEY, family_id TEXT NOT NULL, ts_us INTEGER NOT NULL, "
    "ev TEXT NOT NULL)",
    "CREATE INDEX IF NOT EXISTS family_events_id ON family_events(family_id, seq)",
    "CREATE TABLE IF NOT EXISTS requests(request_id TEXT PRIMARY KEY, ev TEXT NOT NULL, done INTEGER NOT NULL, "
    "cancelled INTEGER NOT NULL, regimes TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS claims(seq INTEGER PRIMARY KEY, request_id TEXT NOT NULL, unit TEXT NOT NULL, "
    "writer_id TEXT NOT NULL, producer TEXT NOT NULL, host TEXT NOT NULL, pid INTEGER NOT NULL, "
    "expires_at TEXT NOT NULL, void_seq INTEGER, void_reason TEXT, row_id TEXT, live INTEGER NOT NULL)",
    "CREATE INDEX IF NOT EXISTS claims_unit ON claims(unit)",
    "CREATE INDEX IF NOT EXISTS claims_live ON claims(live) WHERE live = 1",
    "CREATE TABLE IF NOT EXISTS unit_rows(unit TEXT PRIMARY KEY, ev TEXT NOT NULL)",
    "CREATE TABLE IF NOT EXISTS seed_keys(k TEXT PRIMARY KEY, row_id TEXT NOT NULL)",
)
_TABLES = ("meta", "files", "families", "family_events", "requests", "claims", "unit_rows", "seed_keys")


@dataclass
class Counters:
    """What the index did (tests assert on it; ``store.IO`` counts what was parsed)."""

    syncs: int = 0
    full_builds: int = 0
    events_applied: int = 0
    corrupt_discards: int = 0
    rebuild_reasons: List[str] = field(default_factory=list)

    def reset(self) -> None:
        self.syncs = self.full_builds = self.events_applied = self.corrupt_discards = 0
        self.rebuild_reasons = []


COUNTERS = Counters()


def enabled() -> bool:
    """The persisted indexes are on unless ``GEN3AI_LEDGER_INDEX=0``."""
    return os.environ.get(ENV_SWITCH, "1") != "0"


class _Rebuild(Exception):
    """The index is not a prefix-fold of the streams any more: rebuild it from them."""


def _unit_key(u: Q.Unit) -> str:
    return S.canonical(list(u))


def _us(t: Any) -> int:
    """An aware timestamp as integer microseconds since the epoch (exact; a float epoch is not)."""
    import datetime as _dt

    return (t - _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc)) // _dt.timedelta(microseconds=1)


def ts_us(s: Any) -> int:
    t = S.parse_ts(s)
    assert t is not None, s
    return _us(t)


class _SeedTable(Dict[Tuple[Any, ...], str]):
    """``QueueState.seed_keys`` backed by the ``seed_keys`` table: membership, lookup and insert, each one primary-key
    probe. With a ``conn`` it works inside that connection's transaction (a catch-up); with only a ``path`` each probe
    opens its own short read connection (the writer's state, after the sync)."""

    def __init__(self, path: Path, conn: Optional[sqlite3.Connection] = None) -> None:
        super().__init__()
        self._path, self._conn = path, conn

    @staticmethod
    def _k(k: Tuple[Any, ...]) -> str:
        return S.canonical(list(k))

    def _probe(self, k: Tuple[Any, ...]) -> Optional[str]:
        if self._conn is not None:
            r = self._conn.execute("SELECT row_id FROM seed_keys WHERE k = ?", (self._k(k),)).fetchone()
        else:
            c = INC.connect(self._path, create=False)
            try:
                r = c.execute("SELECT row_id FROM seed_keys WHERE k = ?", (self._k(k),)).fetchone()
            finally:
                c.close()
        return None if r is None else str(r[0])

    def __contains__(self, k: object) -> bool:
        return self._probe(k) is not None  # type: ignore[arg-type]

    def __getitem__(self, k: Tuple[Any, ...]) -> str:
        v = self._probe(k)
        if v is None:
            raise KeyError(k)
        return v

    def get(self, k: Tuple[Any, ...], default: Any = None) -> Any:  # type: ignore[override]
        v = self._probe(k)
        return default if v is None else v

    def __setitem__(self, k: Tuple[Any, ...], v: str) -> None:
        assert self._conn is not None, "the seed-key table is written only inside a catch-up"
        self._conn.execute("INSERT INTO seed_keys(k, row_id) VALUES (?, ?)", (self._k(k), v))


class _Scope:
    """A ``QueueState`` loaded for only the keys an operation touches, from the persisted tables. Loads are
    idempotent and never replace an object already held (a claim mutated by ``apply_event`` and not yet flushed)."""

    def __init__(self, conn: sqlite3.Connection, st: Q.QueueState, *, empty: bool = False) -> None:
        self.conn, self.st, self.empty = conn, st, empty
        self._requests: Set[str] = set()
        self._units: Set[str] = set()

    def load_request(self, rid: str) -> None:
        if self.empty or rid in self._requests:
            return
        self._requests.add(rid)
        r = self.conn.execute("SELECT ev, done, cancelled, regimes FROM requests WHERE request_id = ?",
                              (rid,)).fetchone()
        if r is None:
            return
        self.st.requests.setdefault(rid, json.loads(r[0]))
        if r[1]:
            self.st.done.add(rid)
        if r[2]:
            self.st.cancelled.add(rid)
        regs = json.loads(r[3])
        if regs:
            self.st.request_regimes.setdefault(rid, set()).update(regs)

    def _claim(self, r: Tuple[Any, ...]) -> Q.Claim:
        exp = S.parse_ts(r[7])
        assert exp is not None
        return Q.Claim(seq=int(r[0]), writer_id=r[3], producer=r[4], host=r[5], pid=int(r[6]), expires_at=exp,
                       unit=tuple(json.loads(r[2])),  # type: ignore[arg-type]
                       void_seq=r[8], void_reason=r[9], row_id=r[10])

    _CLAIM_COLS = "seq, request_id, unit, writer_id, producer, host, pid, expires_at, void_seq, void_reason, row_id"

    def load_unit(self, u: Q.Unit) -> None:
        key = _unit_key(u)
        if self.empty or key in self._units:
            return
        self._units.add(key)
        seqs = []
        for r in self.conn.execute(f"SELECT {self._CLAIM_COLS} FROM claims WHERE unit = ? ORDER BY seq", (key,)):
            seqs.append(int(r[0]))
            if int(r[0]) not in self.st.claims:
                self.st.claims[int(r[0])] = self._claim(r)
        if seqs:
            self.st.by_unit[u] = seqs
        row = self.conn.execute("SELECT ev FROM unit_rows WHERE unit = ?", (key,)).fetchone()
        if row is not None:
            self.st.rows.setdefault(u, json.loads(row[0]))

    def load_claim(self, seq: int) -> None:
        if self.empty or seq in self.st.claims:
            return
        r = self.conn.execute(f"SELECT {self._CLAIM_COLS} FROM claims WHERE seq = ?", (seq,)).fetchone()
        if r is not None:
            self.st.claims[seq] = self._claim(r)

    def load_live(self) -> None:
        for (seq,) in self.conn.execute("SELECT seq FROM claims WHERE live = 1 ORDER BY seq").fetchall():
            self.load_claim(int(seq))

    def load_for(self, e: Dict[str, Any]) -> None:
        """Everything ``apply_event`` reads for ``e``."""
        kind = e["event"]
        if kind == "family":
            return
        self.load_request(e["request_id"])
        if kind in ("claim", "void", "row"):
            self.load_unit(Q.unit_of(e))
        if kind in ("void", "row"):
            self.load_claim(int(e["claim_seq"]))

    def flush(self) -> None:
        st, c = self.st, self.conn
        c.executemany(
            "INSERT OR REPLACE INTO claims(seq, request_id, unit, writer_id, producer, host, pid, expires_at, "
            "void_seq, void_reason, row_id, live) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            [(k.seq, k.unit[0], _unit_key(k.unit), k.writer_id, k.producer, k.host, k.pid, k.expires_at.isoformat(),
              k.void_seq, k.void_reason, k.row_id, int(k.live)) for k in st.claims.values()])
        c.executemany("INSERT OR REPLACE INTO unit_rows(unit, ev) VALUES (?, ?)",
                      [(_unit_key(u), json.dumps(ev, sort_keys=True)) for u, ev in st.rows.items()])
        c.executemany(
            "INSERT OR REPLACE INTO requests(request_id, ev, done, cancelled, regimes) VALUES (?,?,?,?,?)",
            [(rid, json.dumps(ev, sort_keys=True), int(rid in st.done), int(rid in st.cancelled),
              json.dumps(sorted(st.request_regimes.get(rid, ())))) for rid, ev in st.requests.items()])
        c.executemany("INSERT OR REPLACE INTO families(family_id, ev) VALUES (?, ?)",
                      [(fid, json.dumps(ev, sort_keys=True)) for fid, ev in st.families.items()])


def _put_meta(conn: sqlite3.Connection, **kv: Any) -> None:
    conn.executemany("INSERT OR REPLACE INTO meta(k, v) VALUES (?, ?)", [(k, json.dumps(v)) for k, v in kv.items()])


class EventIndex:
    """The persisted fold of one ledger root's requests stream (module docstring). Cheap to construct; every method
    opens its own short connection, so nothing is held across a ``fork`` or a thread."""

    def __init__(self, root: "str | os.PathLike[str]") -> None:
        self.root = Path(root)
        self.path = self.root / INDEX_DIRNAME / "events.sqlite3"

    # ------------------------------------------------------------------------------------------ public
    def state(self, *, request: Optional[str] = None, unit: Optional[Q.Unit] = None,
              claim_seq: Optional[int] = None, live: bool = False) -> Q.QueueState:
        """Catch up with the stream, then the ``QueueState`` the writer needs: the families, ``max_seq``, the fold's
        ``problems``, the seed-block table, and — loaded — only ``request``'s record, ``unit``'s claims and row,
        ``claim_seq``'s claim and, when ``live``, every live claim. Under the ledger lock."""
        self._need_lock()

        def go() -> Q.QueueState:
            conn = self._open()
            try:
                self._sync(conn)
                st = self._skeleton(conn)
                st.seed_keys = _SeedTable(self.path)
                sc = _Scope(conn, st)
                if request is not None:
                    sc.load_request(request)
                if unit is not None:
                    sc.load_unit(unit)
                if claim_seq is not None:
                    sc.load_claim(int(claim_seq))
                if live:
                    sc.load_live()
                return st
            finally:
                conn.close()

        return self._recovering(go)

    def family_as_of(self, family_id: str, as_of: Any = None) -> Optional[Dict[str, Any]]:
        """The registration ``fold(events, as_of)`` would hold for ``family_id``: the first registration in ``seq``
        order whose ``ts`` is not after ``as_of`` (``None`` = any). Takes the ledger lock itself."""
        t = None if as_of is None else _us(as_of)

        def go() -> Optional[Dict[str, Any]]:
            conn = self._open()
            try:
                self._sync(conn)
                if t is None:
                    r = conn.execute("SELECT ev FROM families WHERE family_id = ?", (family_id,)).fetchone()
                    return None if r is None else dict(json.loads(r[0]))
                for ts, ev in conn.execute("SELECT ts_us, ev FROM family_events WHERE family_id = ? ORDER BY seq",
                                           (family_id,)).fetchall():
                    if int(ts) <= t:
                        return dict(json.loads(ev))
                return None
            finally:
                conn.close()

        with ST.locked(self.root):
            return self._recovering(go)

    def rebuild(self) -> None:
        """Drop the persisted state and fold the streams from scratch (``audit --rebuild-index``). Under the lock."""
        self._need_lock()

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
        """Re-fold every events file UP TO ITS CURSOR from the streams and compare with what the index holds. Returns
        the problems (empty = the index is exactly what the streams fold to as far as it has read). Races nothing:
        it compares against the bytes the cursors cover, not the files' current size."""
        out: List[str] = []
        if not self.path.exists():
            return out
        try:
            conn = INC.connect(self.path, create=False)
        except sqlite3.DatabaseError as e:
            return [f"event index: cannot open {self.path} ({e}); rebuild with `audit --rebuild-index`"]
        try:
            if (ic := conn.execute("PRAGMA integrity_check").fetchone()) is None or ic[0] != "ok":
                return [f"event index: sqlite integrity_check says {ic}"]
            meta = {k: json.loads(v) for k, v in conn.execute("SELECT k, v FROM meta")}
            if meta.get("format") != FORMAT:
                return [f"event index: format {meta.get('format')!r}, this code writes {FORMAT!r}"]
            events: List[Dict[str, Any]] = []
            for stem, path, off, lines, tail, *_ in conn.execute(
                    "SELECT stem, path, off, lines, tail, size, mtime_ns FROM files ORDER BY stem").fetchall():
                full = INC.resolve_shard(self.root, path)
                try:
                    data = INC._read_all(full)[:off]
                except OSError as e:
                    out.append(f"event index: tracked file {path} cannot be read ({e})")
                    continue
                if len(data) != off or INC._digest(data[max(0, off - INC.TAIL_BYTES):off]) != tail:
                    out.append(f"event index: {path}: the first {off} bytes are not what the cursor's digest saw")
                    continue
                c = INC.consume(data, 0, 0, 0)
                if c.n != lines:
                    out.append(f"event index: {path}: {c.n} lines up to the cursor, the index says {lines}")
                for ln in c.lines:
                    if ln.err is not None or S.validate_event(ln.obj):
                        out.append(f"event index: {path}:{ln.n}: a line the index consumed does not validate now")
                    else:
                        events.append(ln.obj)
            if out:
                return out
            want = Q.fold(events)
            got = self._full_state(conn)
            out.extend(f"event index: {d}" for d in state_diff(want, got))
            want_fam = {int(e["seq"]) for e in events if e["event"] == "family"}
            have_fam = {int(r[0]) for r in conn.execute("SELECT seq FROM family_events")}
            if want_fam != have_fam:
                out.append(f"event index: family_events holds {len(have_fam)} registrations, the streams hold "
                           f"{len(want_fam)}")
            return out
        finally:
            conn.close()

    # ------------------------------------------------------------------------------------------ internals
    def _need_lock(self) -> None:
        if not ST.lock_held(self.root):
            raise RuntimeError("the event index is updated only under the ledger lock (store.locked)")

    def _open(self) -> sqlite3.Connection:
        return INC.connect(self.path, create=True)

    def _recovering(self, fn: Callable[[], T]) -> T:
        """Run ``fn``; a CORRUPT index file is discarded and ``fn`` runs once more on a fresh one (which rebuilds)."""
        try:
            return fn()
        except sqlite3.DatabaseError as e:
            if INC.is_busy(e):
                raise INC.IndexUnavailable(f"the event index {self.path} is busy: {e}") from e
            if not INC.is_corruption(e):
                raise
            COUNTERS.corrupt_discards += 1
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
        """Bring the index up to the end of the streams, in one transaction (rebuilding when it must)."""
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

    def _skeleton(self, conn: sqlite3.Connection) -> Q.QueueState:
        meta = {k: json.loads(v) for k, v in conn.execute("SELECT k, v FROM meta")}
        st = Q.QueueState()
        st.max_seq = int(meta.get("max_seq", 0))
        st.problems = list(meta.get("problems", []))
        for fid, ev in conn.execute("SELECT family_id, ev FROM families"):
            st.families[fid] = json.loads(ev)
        return st

    def _sync_tx(self, conn: sqlite3.Connection, *, full: bool, why: str = "") -> None:
        if full:
            for t in _TABLES:
                conn.execute(f"DELETE FROM {t}")
            COUNTERS.full_builds += 1
            COUNTERS.rebuild_reasons.append(why)
            meta: Dict[str, Any] = {}
        else:
            meta = {k: json.loads(v) for k, v in conn.execute("SELECT k, v FROM meta")}
            if meta.get("format") != FORMAT:
                raise _Rebuild("no index yet" if not meta else f"format {meta.get('format')!r}")
        recorded = {r[0]: INC.Cursor(path=r[1], off=r[2], lines=r[3], tail=r[4], size=r[5], mtime_ns=r[6],
                                     eol_owed=bool(r[7]))
                    for r in conn.execute("SELECT stem, path, off, lines, tail, size, mtime_ns, eol_owed FROM files")}
        listed = {INC.stem_of(os.path.relpath(p, self.root)): p for p in ST.event_files(self.root)}
        gone = sorted(set(recorded) - set(listed))
        if gone:
            raise _Rebuild(f"tracked file(s) vanished: {gone[:3]}")
        max_seq = int(meta.get("max_seq", 0))
        new: List[Dict[str, Any]] = []
        cursors: Dict[str, INC.Cursor] = {}
        for stem, path in listed.items():
            rel = os.path.relpath(path, self.root)
            cur = recorded.get(stem)
            try:
                buf, prefix, start, st = INC.read_since(path, rel, cur)
                if len(buf) == prefix and cur is not None:          # nothing beyond the cursor
                    if (cur.path, cur.size, cur.mtime_ns) != (rel, st.st_size, st.st_mtime_ns):
                        cursors[stem] = INC.refreshed(cur, rel, st)
                    continue
                c = INC.consume(buf, prefix, start, 0 if cur is None else cur.lines,
                                False if cur is None else cur.eol_owed)
            except INC.StaleCursor as e:
                raise _Rebuild(str(e)) from None
            for ln in c.lines:
                if ln.err is not None:
                    raise S.LedgerSchemaError(f"{path}:{ln.n}: not JSON ({ln.err})")
                errs = S.validate_event(ln.obj)
                if errs:
                    raise S.LedgerSchemaError(f"{rel}:{ln.n}: {len(errs)} schema problem(s):\n  " + "\n  ".join(errs))
                new.append(ln.obj)
            cursors[stem] = INC.next_cursor(rel, st, c)
        new.sort(key=lambda e: int(e["seq"]))                    # stable: ties keep the scan's file / line order
        if new and not full and int(new[0]["seq"]) <= max_seq:
            raise _Rebuild(f"an event with seq {new[0]['seq']} <= the indexed maximum {max_seq} (out of order)")
        st_ = self._skeleton(conn) if not full else Q.QueueState()
        st_.seed_keys = _SeedTable(self.path, conn)
        sc = _Scope(conn, st_, empty=full)
        seen: Set[int] = set()
        fam_rows: List[Tuple[int, str, int, str]] = []
        for e in new:
            if e["event"] == "family" and int(e["seq"]) not in seen:
                fam_rows.append((int(e["seq"]), e["family_id"], ts_us(e["ts"]), json.dumps(e, sort_keys=True)))
            sc.load_for(e)
            Q.apply_event(st_, e, seen)
        COUNTERS.events_applied += len(new)
        sc.flush()
        conn.executemany("INSERT OR REPLACE INTO family_events(seq, family_id, ts_us, ev) VALUES (?, ?, ?, ?)", fam_rows)
        conn.executemany("INSERT OR REPLACE INTO files(stem, path, off, lines, tail, size, mtime_ns, eol_owed) "
                         "VALUES (?,?,?,?,?,?,?,?)",
                         [(k, c.path, c.off, c.lines, c.tail, c.size, c.mtime_ns, int(c.eol_owed))
                          for k, c in cursors.items()])
        _put_meta(conn, format=FORMAT, max_seq=st_.max_seq, problems=st_.problems)

    def _full_state(self, conn: sqlite3.Connection) -> Q.QueueState:
        """The WHOLE persisted state as a ``QueueState`` (verification and tests; O(archive))."""
        st = self._skeleton(conn)
        sc = _Scope(conn, st)
        for (rid,) in conn.execute("SELECT request_id FROM requests").fetchall():
            sc.load_request(rid)
        for (seq,) in conn.execute("SELECT seq FROM claims ORDER BY seq").fetchall():
            sc.load_claim(int(seq))
        for k in st.claims.values():
            st.by_unit.setdefault(k.unit, []).append(k.seq)
        for unit, ev in conn.execute("SELECT unit, ev FROM unit_rows"):
            st.rows[tuple(json.loads(unit))] = json.loads(ev)  # type: ignore[index]
        for k, row_id in conn.execute("SELECT k, row_id FROM seed_keys"):
            dict.__setitem__(st.seed_keys, tuple(json.loads(k)), row_id)
        return st

    def full_state(self) -> Q.QueueState:
        """The whole persisted state (tests / ``audit``). Under the lock, after a catch-up."""
        self._need_lock()

        def go() -> Q.QueueState:
            conn = self._open()
            try:
                self._sync(conn)
                return self._full_state(conn)
            finally:
                conn.close()

        return self._recovering(go)


def state_diff(want: Q.QueueState, got: Q.QueueState) -> List[str]:
    """How two folded states differ, field by field (empty = equal)."""
    out: List[str] = []
    for name in ("requests", "families", "claims", "by_unit", "rows", "request_regimes"):
        a, b = getattr(want, name), getattr(got, name)
        if name == "request_regimes":
            a = {k: v for k, v in a.items() if v}
        if a != b:
            ka, kb = set(a), set(b)
            out.append(f"{name}: streams fold to {len(a)}, the index holds {len(b)} "
                       f"(only in streams: {sorted(map(str, ka - kb))[:3]}, only in index: {sorted(map(str, kb - ka))[:3]}, "
                       f"differing: {sorted(str(k) for k in ka & kb if a[k] != b[k])[:3]})")
    for name in ("done", "cancelled"):
        if getattr(want, name) != getattr(got, name):
            out.append(f"{name}: {sorted(getattr(want, name) ^ getattr(got, name))[:3]} differ")
    if dict(want.seed_keys) != dict(got.seed_keys):
        out.append(f"seed_keys: {len(want.seed_keys)} in the streams, {len(got.seed_keys)} in the index")
    if want.max_seq != got.max_seq:
        out.append(f"max_seq: {want.max_seq} in the streams, {got.max_seq} in the index")
    if want.problems != got.problems:
        out.append(f"problems: {want.problems[:2]} vs {got.problems[:2]}")
    return out
