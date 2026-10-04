"""THE READER — the ONE way to consume ledger rows, and every consumer DECLARES what it reads (§0b.7).

    DECL = ReaderDecl(
        name="sprt_promotion",
        purposes=frozenset({"promotion"}),
        regime=RegimeFilter(protocol="gen3_eval_protocol_v1_h2h", play="greedy", opponent_play="greedy",
                            mirrored=True),
        requests="own",            # "own" = the caller's request; "family" = its family (group-sequential
                                   #   decisions only, with decision_kind); "any" = an estimate
        selection="exclude",       # drop the rows a SELECTING decision consumed (the games that selected a node do
                                   #   not rate it); "include" keeps them
        flags_ok=frozenset(),      # the legacy flags this reader accepts (schema.FLAGS)
        inference="conditional",   # "conditional" (meter noise only) or "across_runs" (the run term too)
    )
    got = eval_ledger.read(DECL, request_id=..., players=..., as_of=...)

What a read GUARANTEES, or refuses with a typed error:

* every row validated (a v1 row upgraded); a correction applied (a superseded row dropped) — both AS OF the read;
* NO DUPLICATES anywhere in the ledger, by batch key or by seed block (:class:`~writer.DuplicateBatchError`), so a
  duplicate can never reach an estimator whether or not ``audit`` has run;
* the declared purposes, flags and regime filter, the declared request scope (``family`` only for a registered
  group-sequential decision kind whose family names its rule), and ``selection``;
* ONE regime per call (:class:`MixedRegimeError`) — and, in a family, the family's pinned protocol;
* ``as_of`` (an aware timestamp) restricts rows, corrections, family registrations AND decisions to those that
  existed then: the time travel a back-test or an eviction needs.

``src/eval_ledger_reader_gate_test.py`` fails any module that reads the ledger without a declaration, and checks
that each declaration is spelled out (every field, the purposes and the scope as literals)."""
from __future__ import annotations

import datetime as _dt
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

from agents.training.eval_ledger import queue as Q
from agents.training.eval_ledger import schema as S
from agents.training.eval_ledger import store as ST
from agents.training.eval_ledger.writer import DuplicateBatchError

REQUEST_SCOPES = ("own", "family", "any")
SELECTIONS = ("exclude", "include")
INFERENCES = ("conditional", "across_runs")
#: "every purpose" — a reader that legitimately reads all of them (a per-request resume, an audit view) says so.
ALL_PURPOSES: FrozenSet[str] = frozenset(S.PURPOSES)


class ReaderDeclError(ValueError):
    """A declaration that is not a declaration (an unknown purpose, scope, flag, or a family read without a
    group-sequential decision kind), or a call outside its declaration."""


class MixedRegimeError(ValueError):
    """Rows of more than one regime (or, in a pinned family, another protocol) offered to one read."""


@dataclass(frozen=True)
class RegimeFilter:
    """The regime fields a reader pins (``None`` = not pinned). Whatever is pinned, a read returns ONE regime."""

    protocol: Optional[str] = None
    play: Optional[str] = None
    opponent_play: Optional[str] = None
    mirrored: Optional[bool] = None
    seat_rule: Optional[str] = None
    eval_core: Optional[str] = None

    def problems(self) -> List[str]:
        out = []
        if self.protocol is not None and self.protocol not in S.PROTOCOLS:
            out.append(f"regime.protocol {self.protocol!r} is not a known protocol {S.PROTOCOLS}")
        for k in ("play", "opponent_play"):
            v = getattr(self, k)
            if v is not None and v not in S.PLAYS:
                out.append(f"regime.{k} {v!r} not in {S.PLAYS}")
        if self.seat_rule is not None and self.seat_rule not in S.SEAT_RULES:
            out.append(f"regime.seat_rule {self.seat_rule!r} not in {S.SEAT_RULES}")
        return out

    def matches(self, regime: Mapping[str, Any]) -> bool:
        return all(getattr(self, k) is None or regime.get(k) == getattr(self, k)
                   for k in ("protocol", "play", "opponent_play", "mirrored", "seat_rule", "eval_core"))


@dataclass(frozen=True)
class ReaderDecl:
    """What a reader consumes. Every field is REQUIRED (no defaults): a declaration that relied on one would not be
    a declaration. ``decision_kind`` is required exactly when ``requests == "family"``."""

    name: str
    purposes: FrozenSet[str]
    regime: RegimeFilter
    requests: str
    selection: str
    flags_ok: FrozenSet[str]
    inference: str
    decision_kind: Optional[str] = None

    def __post_init__(self) -> None:
        bad: List[str] = []
        if not (isinstance(self.name, str) and self.name):
            bad.append("name: a non-empty string")
        if not (isinstance(self.purposes, frozenset) and self.purposes and self.purposes <= ALL_PURPOSES):
            bad.append(f"purposes: a non-empty frozenset of {S.PURPOSES} (got {self.purposes!r})")
        if not isinstance(self.regime, RegimeFilter):
            bad.append("regime: a RegimeFilter")
        else:
            bad.extend(self.regime.problems())
        if self.requests not in REQUEST_SCOPES:
            bad.append(f"requests: one of {REQUEST_SCOPES}")
        if self.selection not in SELECTIONS:
            bad.append(f"selection: one of {SELECTIONS}")
        if not (isinstance(self.flags_ok, frozenset) and self.flags_ok <= frozenset(S.FLAGS)):
            bad.append(f"flags_ok: a frozenset of {S.FLAGS}")
        if self.inference not in INFERENCES:
            bad.append(f"inference: one of {INFERENCES}")
        if self.requests == "family":
            if self.decision_kind not in S.GROUP_SEQUENTIAL_KINDS:
                bad.append(f"decision_kind: a family read is legal only for a group-sequential decision kind "
                           f"{S.GROUP_SEQUENTIAL_KINDS} (got {self.decision_kind!r})")
        elif self.decision_kind is not None:
            bad.append("decision_kind: only a family read names one")
        if bad:
            raise ReaderDeclError(f"ReaderDecl {self.name!r}: " + "; ".join(bad))


@dataclass(frozen=True)
class LedgerRead:
    """A read's result: the rows (v2 views, in ledger order), their single regime, the digest a decision records,
    and what was asked."""

    decl: ReaderDecl
    rows: Tuple[Dict[str, Any], ...]
    regime_id: Optional[str]
    digest: str
    as_of: Optional[_dt.datetime]
    root: Path
    request_id: Optional[str] = None
    family: Optional[str] = None
    excluded: Tuple[str, ...] = field(default=())

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self.rows)

    def __len__(self) -> int:
        return len(self.rows)


def _as_of(x: Any) -> Optional[_dt.datetime]:
    if x is None:
        return None
    t = S.parse_ts(x)
    if t is None:
        raise ReaderDeclError(f"as_of {x!r}: an aware ISO-8601 timestamp or datetime (a run-step as_of is not built "
                              "yet — design_evaluation.md §10, the U1 hand-off)")
    return t


def _side(p: Mapping[str, Any]) -> str:
    return S._sha_or_id(p)


def _check_duplicates(rows: Sequence[ST.ScannedRow]) -> None:
    for keyfn, what in ((S.batch_key, "batch key (request, batch, player, opponent, regime)"),
                        (S.seed_key, "seed block (the same games)")):
        seen: Dict[Tuple[Any, ...], str] = {}
        dups: List[str] = []
        for sr in rows:
            k = keyfn(sr.row)
            if k is None:
                continue
            if k in seen:
                dups.append(f"{sr.row['row_id']} ({sr.source}) repeats {seen[k]} on {what}")
            else:
                seen[k] = f"{sr.row['row_id']} ({sr.source})"
        if dups:
            raise DuplicateBatchError(f"{len(dups)} duplicate batch(es) in the ledger — double-counted games make a "
                                      "decision look more certain than it is; run `python -m main.eval_ledger "
                                      "audit`:\n  " + "\n  ".join(dups[:10]))


def live_rows(root: Path, as_of: Optional[_dt.datetime] = None) -> List[ST.ScannedRow]:
    """Every row AS OF ``as_of``, validated, corrections applied, and checked for duplicates (row ids, batch keys,
    seed blocks). The shared front half of :func:`read` and the audit's views."""
    scanned = ST.scan_rows(root) if root.exists() else []
    if as_of is not None:
        scanned = [sr for sr in scanned if S.parse_ts(sr.row["ts"]) <= as_of]  # type: ignore[operator]
    ids: Dict[str, str] = {}
    for sr in scanned:
        rid = sr.row["row_id"]
        if rid in ids:
            raise DuplicateBatchError(f"row_id {rid} appears twice ({ids[rid]}, {sr.source}) — a shard was copied")
        ids[rid] = sr.source
    dead = {sr.row["supersedes"] for sr in scanned if sr.row["supersedes"]}
    live = [sr for sr in scanned if sr.row["row_id"] not in dead]
    _check_duplicates(live)
    return live


def _select(decl: ReaderDecl, root: "str | os.PathLike[str] | None", request_id: Optional[str],
            family: Optional[str], players: Optional[Iterable[str]], opponents: Optional[Iterable[str]],
            purposes: Optional[Iterable[str]], regime_id: Optional[str], as_of: Any
            ) -> Tuple[List[ST.ScannedRow], Optional[Dict[str, Any]], Optional[_dt.datetime], Path, List[str]]:
    if not isinstance(decl, ReaderDecl):
        raise ReaderDeclError(f"read() takes a ReaderDecl, not {type(decl).__name__}")
    if decl.requests == "own" and (request_id is None or family is not None):
        raise ReaderDeclError(f"{decl.name}: requests='own' reads ONE request: pass request_id (and no family)")
    if decl.requests == "family" and (family is None or request_id is not None):
        raise ReaderDeclError(f"{decl.name}: requests='family' reads ONE family: pass family (and no request_id)")
    if decl.requests == "any" and (request_id is not None or family is not None):
        raise ReaderDeclError(f"{decl.name}: requests='any' is an estimate over every eligible row; a request or "
                              "family read must DECLARE requests='own' / 'family'")
    want_p = frozenset(purposes) if purposes is not None else decl.purposes
    if not want_p <= decl.purposes:
        raise ReaderDeclError(f"{decl.name}: purposes {sorted(want_p - decl.purposes)} are outside the declaration")
    t = _as_of(as_of)
    path = ST.resolve_root(root)
    live = live_rows(path, t)

    fam: Optional[Dict[str, Any]] = None
    if decl.requests == "family":
        st = Q.fold((e for e, _w in ST.scan_events(path)), as_of=t) if path.exists() else Q.QueueState()
        fam = st.families.get(family)  # type: ignore[arg-type]
        if fam is None:
            raise ReaderDeclError(f"{decl.name}: family {family!r} is not registered (as of {t}) — a family read "
                                  "needs the family to name its decision kind, rule and protocol")
        if fam["decision_kind"] != decl.decision_kind:
            raise ReaderDeclError(f"{decl.name}: family {family!r} is registered for {fam['decision_kind']!r}, not "
                                  f"{decl.decision_kind!r}")

    pl = set(players) if players is not None else None
    op = set(opponents) if opponents is not None else None
    out: List[ST.ScannedRow] = []
    for sr in live:
        r = sr.row
        if r["purpose"] not in want_p or not set(r["flags"]) <= decl.flags_ok:
            continue
        if not decl.regime.matches(r["regime"]) or (regime_id is not None and r["regime"]["regime_id"] != regime_id):
            continue
        req = r["request"] or {}
        if decl.requests == "own" and req.get("id") != request_id:
            continue
        if decl.requests == "family" and req.get("family") != family:
            continue
        if pl is not None and _side(r["player"]) not in pl:
            continue
        if op is not None and _side(r["opponent"]) not in op:
            continue
        out.append(sr)

    excluded: List[str] = []
    if decl.selection == "exclude" and out:
        subjects = (pl or set()) | (op or set())
        drop = set()
        for d, _w in (ST.scan_decisions(path) if path.exists() else []):
            if d["kind"] not in S.SELECTING_DECISION_KINDS:
                continue
            if t is not None and S.parse_ts(d["ts"]) > t:  # type: ignore[operator]
                continue
            if subjects and d["subject"] not in subjects:
                continue
            drop.update(d["consumed"]["row_ids"])
        excluded = sorted(sr.row["row_id"] for sr in out if sr.row["row_id"] in drop)
        out = [sr for sr in out if sr.row["row_id"] not in drop]
    if fam is not None:
        protos = sorted({sr.row["regime"]["protocol"] for sr in out})
        if protos and protos != [fam["protocol"]]:
            raise MixedRegimeError(f"{decl.name}: family {family!r} pins protocol {fam['protocol']!r}; its rows are "
                                   f"{protos} (§0c rule 6)")
    return out, fam, t, path, excluded


def _result(decl: ReaderDecl, out: List[ST.ScannedRow], t: Optional[_dt.datetime], path: Path,
            request_id: Optional[str], family: Optional[str], excluded: List[str]) -> LedgerRead:
    regimes = sorted({sr.row["regime"]["regime_id"] for sr in out})
    if len(regimes) > 1:
        raise MixedRegimeError(f"{decl.name}: the read holds {len(regimes)} regimes {regimes}; one regime per call — "
                               "narrow the declaration's RegimeFilter or pass regime_id (rows of different regimes "
                               "are never pooled, §0c rule 2)")
    return LedgerRead(decl=decl, rows=tuple(sr.row for sr in out), regime_id=regimes[0] if regimes else None,
                      digest=S.rows_digest((sr.row["row_id"], sr.sha) for sr in out), as_of=t, root=path,
                      request_id=request_id, family=family, excluded=tuple(excluded))


def read(decl: ReaderDecl, *, root: "str | os.PathLike[str] | None" = None, request_id: Optional[str] = None,
         family: Optional[str] = None, players: Optional[Iterable[str]] = None,
         opponents: Optional[Iterable[str]] = None, purposes: Optional[Iterable[str]] = None,
         regime_id: Optional[str] = None, as_of: Any = None) -> LedgerRead:
    """The rows ``decl`` consumes (module docstring) — ONE regime, or :class:`MixedRegimeError`. ``players`` /
    ``opponents`` are sha256s (``id:<id>`` for a sha-less side); ``purposes`` and ``regime_id`` narrow the
    declaration at call time (a purpose outside the declaration is refused)."""
    out, _fam, t, path, excluded = _select(decl, root, request_id, family, players, opponents, purposes, regime_id,
                                           as_of)
    return _result(decl, out, t, path, request_id, family, excluded)


def read_by_regime(decl: ReaderDecl, *, root: "str | os.PathLike[str] | None" = None,
                   request_id: Optional[str] = None, family: Optional[str] = None,
                   players: Optional[Iterable[str]] = None, opponents: Optional[Iterable[str]] = None,
                   purposes: Optional[Iterable[str]] = None, as_of: Any = None) -> Dict[str, LedgerRead]:
    """``{regime_id: read}`` — the same declared read split into one read PER regime (a listing that must not pool
    regimes, e.g. ``main.h2h read`` over a directory of several edges)."""
    out, _fam, t, path, excluded = _select(decl, root, request_id, family, players, opponents, purposes, None, as_of)
    by: Dict[str, List[ST.ScannedRow]] = {}
    for sr in out:
        by.setdefault(sr.row["regime"]["regime_id"], []).append(sr)
    return {rid: _result(decl, rs, t, path, request_id, family, [x for x in excluded]) for rid, rs in sorted(by.items())}
