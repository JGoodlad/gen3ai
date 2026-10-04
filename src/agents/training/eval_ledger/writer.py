"""THE WRITER — one process, one producer, one file per stream; every batch row written under a CLAIM (§0b.4).

    w = LedgerWriter(root, producer="h2h")                 # root None = <archive>/_ledger
    w.open_request(rid, kind="adhoc", purpose="audit", spec={...})
    c = w.claim(rid, batch=0, player=sha_a, opponent=sha_b, regime_id=reg, expected_wall_s=300)
    ... play the batch ...
    w.append_row(row, c)                                    # row["row_id"] == w.next_row_id()
    w.close()                                               # gzip this writer's row shard

The claim protocol (§0b.4, review M4), all under the ledger's one file lock:

1. ``claim`` folds the requests stream; refuses a unit that already has a row (:class:`AlreadyRecordedError`);
   voids every live claim on the unit that the deterministic void rule allows (``queue.void_reason``) and refuses
   one it does not (:class:`ClaimHeldError`); then appends ``claim`` with ``expires_at`` = now + max(10 min,
   4 x the expected batch wall).
2. ``append_row`` re-folds, and appends the row ONLY while its claim is still live (else
   :class:`ClaimVoidedError`: the batch is dropped — it is being replayed on the same seeds elsewhere). It refuses
   a duplicate batch key or seed block (:class:`DuplicateBatchError`), a row whose request block contradicts the
   request's ``open`` (kind / family / opened / purpose), a second regime inside one request, and a protocol other
   than the request's or its family's pinned one. Then it appends the row (fsync) and a ``row`` event (fsync).

A row WITHOUT a request (a backfill) is accepted only from a writer built with ``allow_unrequested=True``."""
from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from agents.training.eval_ledger import queue as Q
from agents.training.eval_ledger import schema as S
from agents.training.eval_ledger import store as ST

#: §0b.4: a claim's lifetime is 4 x the request's measured batch wall, with a floor of 10 minutes.
CLAIM_FLOOR_S = 600.0
CLAIM_WALL_FACTOR = 4.0


class LedgerClaimError(RuntimeError):
    """A claim or a claimed append the protocol refuses."""


class ClaimHeldError(LedgerClaimError):
    """Another writer holds a live claim on the unit, and the void rule does not allow voiding it."""


class ClaimVoidedError(LedgerClaimError):
    """This writer's claim was voided before its row was appended: the batch is DROPPED."""


class AlreadyRecordedError(LedgerClaimError):
    """The unit already has its row."""


class DuplicateBatchError(LedgerClaimError):
    """A row whose batch key or seed block is already in the ledger (§0b.4): double-counted games would make a
    decision look twice as certain as it is."""


class RequestSpecError(LedgerClaimError):
    """A request or family re-opened with different terms, or a row that contradicts its request."""


def _utc_now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _iso(t: _dt.datetime) -> str:
    return t.isoformat(timespec="seconds")


class LedgerWriter:
    """One writer process's handle on a ledger root (see the module docstring)."""

    def __init__(self, root: "str | os.PathLike[str] | None", producer: str, *, writer_id: Optional[str] = None,
                 clock: Callable[[], _dt.datetime] = _utc_now, alive: Callable[[int], bool] = ST.pid_alive,
                 host: Optional[str] = None, pid: Optional[int] = None, allow_unrequested: bool = False,
                 lock_timeout_s: float = ST.LOCK_TIMEOUT_S):
        self.root = ST.check_write_root(root if root is not None else ST.archive_ledger_root())
        ST.ensure_layout(self.root)
        self.producer = producer
        self.clock, self.alive = clock, alive
        self.host = host or ST.this_host()
        self.pid = int(pid if pid is not None else os.getpid())
        self.writer_id = writer_id or ST.make_writer_id(producer, now=clock(), host=self.host, pid=self.pid)
        self.allow_unrequested = allow_unrequested
        self.lock_timeout_s = lock_timeout_s
        self.rows_path = ST.shard_path(self.root, producer, self.writer_id)
        self.events_path = self.root / ST.REQUESTS / f"{ST.EVENTS_PREFIX}{self.writer_id}{ST.SHARD_SUFFIX}"
        self.decisions_path = self.root / ST.DECISIONS / f"{ST.DECISIONS_PREFIX}{self.writer_id}{ST.SHARD_SUFFIX}"
        self.references_path = (self.root / ST.REFERENCES
                                / f"{ST.REFERENCES_PREFIX}{self.writer_id}{ST.SHARD_SUFFIX}")
        if self.rows_path.exists() or Path(str(self.rows_path) + ".gz").exists():
            raise ST.LedgerPathError(f"{self.rows_path} already exists — one writer per file")
        self._row_seq = 0
        self._dec_seq = 0
        self._ref_seq = 0
        self.closed = False

    # ---------------------------------------------------------------------------------------- the stream
    def _lock(self) -> Any:
        return ST.locked(self.root, self.lock_timeout_s)

    def _state(self) -> Q.QueueState:
        st = Q.fold(e for e, _w in ST.scan_events(self.root))
        if st.problems:
            raise LedgerClaimError(f"the requests stream under {self.root} is inconsistent; run `python -m "
                                   "main.eval_ledger audit`:\n  " + "\n  ".join(st.problems[:10]))
        return st

    def _event(self, st: Q.QueueState, event: str, **fields: Any) -> Dict[str, Any]:
        e = {"schema": S.EVENT_SCHEMA, "event": event, "seq": st.max_seq + 1, "ts": _iso(self.clock()),
             "writer_id": self.writer_id, **fields}
        S.check(S.validate_event(e), f"{event} event")
        ST.append_line(self.events_path, e)
        st.max_seq += 1
        return e

    def register_family(self, family_id: str, *, decision_kind: str, rule: str, protocol: str,
                        commit: Optional[str] = None) -> Dict[str, Any]:
        """Register a request FAMILY (idempotent; different terms are :class:`RequestSpecError`). Its ``protocol``
        is PINNED: the ledger refuses a row into the family at another one (§0c rule 6)."""
        with self._lock():
            st = self._state()
            new = {"family_id": family_id, "decision_kind": decision_kind, "rule": rule, "protocol": protocol,
                   "commit": commit}
            old = st.families.get(family_id)
            if old is not None:
                if Q._family_ident(old) != Q._family_ident(new):
                    raise RequestSpecError(f"family {family_id!r} is registered with other terms: "
                                           f"{ {k: old[k] for k in new} }")
                return old
            return self._event(st, "family", **new)

    def open_request(self, request_id: str, *, kind: str, purpose: str, family: Optional[str] = None,
                     regime_id: Optional[str] = None, protocol: Optional[str] = None,
                     spec: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
        """Open a request (idempotent: a re-open with the same terms returns the original ``open``; different
        terms are :class:`RequestSpecError`, naming the fields). A family must be registered first; a request in a
        family inherits the family's pinned protocol, and may not name another."""
        with self._lock():
            st = self._state()
            if family is not None:
                fam = st.families.get(family)
                if fam is None:
                    raise RequestSpecError(f"family {family!r} is not registered — register_family first")
                if protocol is not None and protocol != fam["protocol"]:
                    raise RequestSpecError(f"family {family!r} pins protocol {fam['protocol']!r}, not {protocol!r}")
                protocol = fam["protocol"]
            new = {"request_id": request_id, "kind": kind, "purpose": purpose, "family": family,
                   "regime_id": regime_id, "protocol": protocol, "spec": dict(spec or {})}
            old = st.requests.get(request_id)
            if old is not None:
                if Q.open_ident(old) != Q.open_ident(new):
                    diff = [k for k in ("kind", "purpose", "family", "regime_id", "protocol", "spec")
                            if S.canonical(old[k]) != S.canonical(new[k])]
                    raise RequestSpecError(f"request {request_id!r} is open with other terms ({diff}): "
                                           f"{ {k: old[k] for k in diff} } vs { {k: new[k] for k in diff} }")
                return old
            return self._event(st, "open", **new)

    def finish_request(self, request_id: str, *, cancel_reason: Optional[str] = None) -> Dict[str, Any]:
        """``done`` (or ``cancel`` with a reason) for an open request."""
        with self._lock():
            st = self._state()
            if request_id not in st.requests:
                raise RequestSpecError(f"request {request_id!r} was never opened")
            if cancel_reason is not None:
                return self._event(st, "cancel", request_id=request_id, reason=cancel_reason)
            return self._event(st, "done", request_id=request_id)

    def _row_in_shard(self, c: Q.Claim) -> Optional[str]:
        return ST.shard_has_unit(self.root, c.producer, c.writer_id, c.unit)

    def void_dead(self, now: Optional[_dt.datetime] = None) -> List[Dict[str, Any]]:
        """Apply the void rule to EVERY live claim (the scheduler's sweep); returns the ``void`` events written."""
        with self._lock():
            st = self._state()
            out = []
            for c in sorted(st.claims.values(), key=lambda c: c.seq):
                why = Q.void_reason(c, now=now or self.clock(), host=self.host, alive=self.alive,
                                    row_in_shard=self._row_in_shard)
                if why:
                    out.append(self._void(st, c, why))
            return out

    def _void(self, st: Q.QueueState, c: Q.Claim, reason: str) -> Dict[str, Any]:
        e = self._event(st, "void", **self._unit_fields(c.unit), claim_seq=c.seq, reason=reason)
        c.void_seq, c.void_reason = e["seq"], reason
        return e

    @staticmethod
    def _unit_fields(u: Q.Unit) -> Dict[str, Any]:
        return {"request_id": u[0], "batch": u[1], "player": u[2], "opponent": u[3], "regime_id": u[4]}

    def claim(self, request_id: str, *, batch: int, player: str, opponent: str, regime_id: str,
              expected_wall_s: float) -> Q.Claim:
        """Claim one unit (module docstring, step 1). ``player`` / ``opponent`` are the sha256s (``id:<id>`` for a
        sha-less side) exactly as the row's :func:`schema.batch_key` will spell them."""
        u: Q.Unit = (str(request_id), int(batch), str(player), str(opponent), str(regime_id))
        with self._lock():
            st = self._state()
            req = st.requests.get(request_id)
            if req is None:
                raise RequestSpecError(f"request {request_id!r} is not open")
            if request_id in st.done or request_id in st.cancelled:
                raise RequestSpecError(f"request {request_id!r} is closed")
            if req["regime_id"] is not None and req["regime_id"] != regime_id:
                raise RequestSpecError(f"request {request_id!r} pins regime {req['regime_id']}, not {regime_id}")
            if u in st.rows:
                raise AlreadyRecordedError(f"unit {u} already has row {st.rows[u]['row_id']}")
            now = self.clock()
            for c in st.live_claims(u):
                rid = self._row_in_shard(c)
                if rid is not None:                     # a writer killed between its row and its `row` event
                    self._event(st, "row", **self._unit_fields(u), claim_seq=c.seq, row_id=rid,
                                seed_key=self._seed_key_of(c, rid))
                    raise AlreadyRecordedError(f"unit {u} already has row {rid} (its row event was repaired)")
                why = Q.void_reason(c, now=now, host=self.host, alive=self.alive, row_in_shard=self._row_in_shard)
                if why is None:
                    raise ClaimHeldError(f"unit {u} is claimed by {c.writer_id} (seq {c.seq}) until "
                                         f"{_iso(c.expires_at)}, and its writer is not known dead")
                self._void(st, c, why)
            life = max(CLAIM_FLOOR_S, CLAIM_WALL_FACTOR * float(expected_wall_s))
            e = self._event(st, "claim", **self._unit_fields(u), producer=self.producer, host=self.host,
                            pid=self.pid, expires_at=_iso(now + _dt.timedelta(seconds=life)))
            return Q.Claim(seq=e["seq"], writer_id=self.writer_id, producer=self.producer, host=self.host,
                           pid=self.pid, expires_at=S.parse_ts(e["expires_at"]), unit=u)  # type: ignore[arg-type]

    def _seed_key_of(self, c: Q.Claim, row_id: str) -> Optional[List[Any]]:
        base = ST.shard_path(self.root, c.producer, c.writer_id)
        for p in ST.one_per_stem([str(x) for x in (base, Path(str(base) + ".gz")) if x.exists()]):
            for _n, obj in ST.iter_jsonl(p):
                if obj.get("row_id") == row_id:
                    return S.key_list(S.seed_key(S.as_v2(obj)))
        return None

    # ---------------------------------------------------------------------------------------- rows
    def next_row_id(self) -> str:
        return f"{self.writer_id}:{self._row_seq}"

    def append_row(self, row: Dict[str, Any], claim: Optional[Q.Claim] = None) -> str:
        """Validate and append ``row`` under its live ``claim`` (module docstring, step 2); returns its ``row_id``."""
        if self.closed:
            raise LedgerClaimError("this writer is closed")
        if row.get("row_id") != self.next_row_id():
            raise S.LedgerSchemaError(f"row_id {row.get('row_id')!r} is not this writer's next id "
                                      f"{self.next_row_id()!r}")
        S.check_row(row, f"{self.rows_path.name} row {self._row_seq}")
        if row["request"] is None:
            if not self.allow_unrequested:
                raise LedgerClaimError("a v2 producer writes every batch FOR a request (row.request is null); only "
                                       "a backfill writer (allow_unrequested=True) may write an unrequested row")
            ST.append_line(self.rows_path, row)
            self._row_seq += 1
            return str(row["row_id"])
        if claim is None:
            raise LedgerClaimError("a requested row is appended only under its claim")
        u = Q.row_unit(row)
        if u != claim.unit:
            raise LedgerClaimError(f"the row's unit {u} is not the claim's {claim.unit}")
        with self._lock():
            st = self._state()
            c = st.claims.get(claim.seq)
            if c is None or c.writer_id != self.writer_id or c.unit != u:
                raise LedgerClaimError(f"claim seq {claim.seq} is not this writer's claim on {u}")
            if c.void_seq is not None:
                raise ClaimVoidedError(f"claim seq {c.seq} on {u} was voided ({c.void_reason}, seq {c.void_seq}) — "
                                       "the batch is DROPPED; it is replayed on the same seeds by whoever re-claimed")
            if u in st.rows:
                raise DuplicateBatchError(f"unit {u} already has row {st.rows[u]['row_id']}")
            sk = S.seed_key(row)
            if sk is not None and sk in st.seed_keys:
                raise DuplicateBatchError(f"the seed block {sk} is already recorded as {st.seed_keys[sk]}")
            self._check_against_request(st, row)
            ST.append_line(self.rows_path, row)
            self._row_seq += 1
            self._event(st, "row", **self._unit_fields(u), claim_seq=c.seq, row_id=row["row_id"],
                        seed_key=S.key_list(sk))
        return str(row["row_id"])

    @staticmethod
    def _check_against_request(st: Q.QueueState, row: Mapping[str, Any]) -> None:
        req = row["request"]
        op = st.requests.get(req["id"])
        if op is None:
            raise RequestSpecError(f"request {req['id']!r} is not open")
        want = {"kind": op["kind"], "family": op["family"], "opened": op["ts"]}
        bad = {k: (req[k], v) for k, v in want.items() if req[k] != v}
        if bad:
            raise RequestSpecError(f"the row's request block contradicts request {req['id']!r}'s open: "
                                   f"{ {k: f'{a!r} != {b!r}' for k, (a, b) in bad.items()} }")
        if row["purpose"] != op["purpose"]:
            raise RequestSpecError(f"the row's purpose {row['purpose']!r} is not request {req['id']!r}'s "
                                   f"{op['purpose']!r}")
        rid = row["regime"]["regime_id"]
        regs = st.request_regimes.get(req["id"], set()) | ({op["regime_id"]} if op["regime_id"] else set())
        if regs and rid not in regs:
            raise RequestSpecError(f"request {req['id']!r} already holds regime {sorted(regs)}; one regime per "
                                   f"request, this row is {rid}")
        proto = row["regime"]["protocol"]
        if op["protocol"] is not None and proto != op["protocol"]:
            raise RequestSpecError(f"request {req['id']!r} pins protocol {op['protocol']!r}; this row is {proto!r}")
        if op["family"] is not None:
            fam = st.families[op["family"]]
            if proto != fam["protocol"]:
                raise RequestSpecError(f"family {op['family']!r} pins protocol {fam['protocol']!r} (§0c rule 6); "
                                       f"this row is {proto!r}")

    # ---------------------------------------------------------------------------------------- decisions, references
    def append_decision(self, *, kind: str, subject: str, consumed: Any, verdict: str, rule: str, rule_version: str,
                        request_id: Optional[str] = None, family: Optional[str] = None) -> Dict[str, Any]:
        """Append a DECISION row: what it decided, the rows it consumed (``consumed`` is an ``eval_ledger.read``
        result: their ids, count and digest are recorded), its ``as_of`` (the read's, else now), the rule and the
        verdict. Never edited afterwards."""
        ids = sorted(r["row_id"] for r in consumed.rows)
        d = {"schema": S.DECISION_SCHEMA, "decision_id": f"{self.writer_id}:d{self._dec_seq}", "kind": kind,
             "subject": subject, "request_id": request_id, "family": family,
             "consumed": {"digest": consumed.digest, "count": len(ids), "row_ids": ids},
             "as_of": _iso(consumed.as_of) if consumed.as_of is not None else _iso(self.clock()), "rule": rule,
             "rule_version": rule_version, "verdict": verdict, "ts": _iso(self.clock()), "writer_id": self.writer_id}
        S.check(S.validate_decision(d), "decision")
        ST.append_line(self.decisions_path, d)
        self._dec_seq += 1
        return d

    def append_reference(self, *, members: Sequence[Mapping[str, str]], weights: Sequence[float], solver: str,
                         solver_version: str, draws: int, seed: int, rows_digest: str) -> Dict[str, Any]:
        """Append an IMMUTABLE frozen reference mixture (§2.3)."""
        r = {"schema": S.REFERENCE_SCHEMA, "reference_id": f"{self.writer_id}:r{self._ref_seq}",
             "members": [dict(m) for m in members], "weights": [float(w) for w in weights], "solver": solver,
             "solver_version": solver_version, "draws": int(draws), "seed": int(seed), "rows_digest": rows_digest,
             "created_at": _iso(self.clock()), "writer_id": self.writer_id}
        S.check(S.validate_reference(r), "reference")
        ST.append_line(self.references_path, r)
        self._ref_seq += 1
        return r

    def close(self) -> None:
        """A clean exit: gzip this writer's row shard (§0b.3). Idempotent."""
        if not self.closed and self.rows_path.exists():
            ST.gzip_shard(self.rows_path)
        self.closed = True

    def __enter__(self) -> "LedgerWriter":
        return self

    def __exit__(self, exc_type: Any, *_exc: Any) -> None:
        if exc_type is None:                         # a CLEAN exit closes the shard; a crash leaves it to close-stale
            self.close()
