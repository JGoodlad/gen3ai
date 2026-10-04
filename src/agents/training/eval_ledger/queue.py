"""THE REQUEST QUEUE as a deterministic FOLD of the requests stream (§0b.4), and the rule that voids a dead
writer's claim.

The stream holds seven event kinds (``schema.EVENT_KINDS``), each with a global ``seq`` assigned under the ledger
lock (so the fold's order is the order the lock granted, across every writer's file):

* ``open`` — a request exists: its kind, purpose, family, and (when the opener knew them) its pinned regime and
  protocol, plus a free ``spec`` a re-open must repeat exactly;
* ``family`` — a request FAMILY is registered: its group-sequential decision kind, its rule, its pinned protocol
  and (optionally) its pinned commit (§0c rule 6);
* ``claim`` — one writer holds one UNIT ``(request, batch, player, opponent, regime)`` until ``expires_at``;
* ``void`` — a claim is void (its writer is dead, or the claim expired, with no row written);
* ``row`` — the claimed unit's row is on disk (written under the lock, after the row itself);
* ``done`` / ``cancel`` — the request is closed.

**The void rule (deterministic).** A claim is VOIDABLE iff it has no ``row`` event, no row exists for its unit in
its writer's own shard, AND (its writer's pid is dead on this host, OR now > ``expires_at``). Voiding writes a
``void`` event; the unit is then re-claimed and replayed on the SAME seed block, so the replay is the same games.
A writer whose claim was voided drops its batch (``writer.ClaimVoidedError``)."""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Set, Tuple

from agents.training.eval_ledger import schema as S

Unit = Tuple[str, int, str, str, str]


def unit_of(e: Mapping[str, Any]) -> Unit:
    return (str(e["request_id"]), int(e["batch"]), str(e["player"]), str(e["opponent"]), str(e["regime_id"]))


def row_unit(row: Mapping[str, Any]) -> Optional[Unit]:
    k = S.batch_key(row)
    return None if k is None else (k[1], k[2], k[3], k[4], k[5])


@dataclass
class Claim:
    seq: int
    writer_id: str
    producer: str
    host: str
    pid: int
    expires_at: _dt.datetime
    unit: Unit
    void_seq: Optional[int] = None
    void_reason: Optional[str] = None
    row_id: Optional[str] = None

    @property
    def live(self) -> bool:
        return self.void_seq is None and self.row_id is None


@dataclass
class QueueState:
    requests: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    done: Set[str] = field(default_factory=set)
    cancelled: Set[str] = field(default_factory=set)
    families: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    claims: Dict[int, Claim] = field(default_factory=dict)
    by_unit: Dict[Unit, List[int]] = field(default_factory=dict)
    rows: Dict[Unit, Dict[str, Any]] = field(default_factory=dict)
    seed_keys: Dict[Tuple[Any, ...], str] = field(default_factory=dict)
    request_regimes: Dict[str, Set[str]] = field(default_factory=dict)
    max_seq: int = 0
    problems: List[str] = field(default_factory=list)

    def live_claims(self, unit: Unit) -> List[Claim]:
        return [self.claims[s] for s in self.by_unit.get(unit, []) if self.claims[s].live]


def fold(events: Iterable[Mapping[str, Any]], as_of: Optional[_dt.datetime] = None) -> QueueState:
    """The queue state after every event (with ``ts <= as_of`` when given), applied in ``seq`` order. An event that
    contradicts the state (a second ``open`` with another spec, a claim on a recorded unit, a ``void`` or ``row``
    naming no claim, a duplicate ``seq``) is recorded in ``problems`` — the audit reports them; the writer refuses
    to act on a state that has any."""
    st = QueueState()
    seen: Set[int] = set()
    for e in sorted(events, key=lambda e: int(e["seq"])):
        apply_event(st, e, seen, as_of)
    return st


def apply_event(st: QueueState, e: Mapping[str, Any], seen: Set[int], as_of: Optional[_dt.datetime] = None) -> None:
    """ONE event's effect on ``st`` — the body of :func:`fold`'s loop, and the ONLY implementation of the fold's
    semantics: the persisted index (``event_index``) applies each new event through it, on a state loaded for just
    the keys the event touches (its request, its unit, its claim), so an incremental fold and a from-scratch fold
    cannot disagree about what an event means. Events must arrive in ``seq`` order; ``seen`` holds every ``seq``
    applied so far."""
    seq = int(e["seq"])
    if seq in seen:
        st.problems.append(f"seq {seq} appears twice in the requests stream")
        return
    seen.add(seq)
    st.max_seq = max(st.max_seq, seq)
    if as_of is not None and S.parse_ts(e["ts"]) > as_of:  # type: ignore[operator]
        return
    kind = e["event"]
    if kind == "family":
        old = st.families.get(e["family_id"])
        if old is not None and _family_ident(old) != _family_ident(e):
            st.problems.append(f"family {e['family_id']!r} registered twice with different terms (seq {seq})")
        elif old is None:
            st.families[e["family_id"]] = dict(e)
        return
    rid = e["request_id"]
    if kind == "open":
        old = st.requests.get(rid)
        if old is not None and open_ident(old) != open_ident(e):
            st.problems.append(f"request {rid!r} opened twice with different terms (seq {seq})")
        elif old is None:
            if e["family"] is not None and e["family"] not in st.families:
                st.problems.append(f"request {rid!r} names family {e['family']!r}, not registered before it")
            st.requests[rid] = dict(e)
        return
    if rid not in st.requests:
        st.problems.append(f"{kind} event seq {seq} names request {rid!r}, which was never opened")
        return
    if kind == "done":
        st.done.add(rid)
    elif kind == "cancel":
        st.cancelled.add(rid)
    elif kind == "claim":
        u = unit_of(e)
        if u in st.rows:
            st.problems.append(f"claim seq {seq} on unit {u}, which already has a row")
            return
        live = st.live_claims(u)
        if live:
            st.problems.append(f"claim seq {seq} on unit {u} while claim seq {live[0].seq} is live")
            return
        exp = S.parse_ts(e["expires_at"])
        assert exp is not None
        st.claims[seq] = Claim(seq=seq, writer_id=e["writer_id"], producer=e["producer"], host=e["host"],
                               pid=int(e["pid"]), expires_at=exp, unit=u)
        st.by_unit.setdefault(u, []).append(seq)
    else:
        c = st.claims.get(int(e["claim_seq"]))
        if c is None or c.unit != unit_of(e):
            st.problems.append(f"{kind} event seq {seq} names claim seq {e['claim_seq']}, which is not a claim "
                               "on its unit")
            return
        if not c.live:
            st.problems.append(f"{kind} event seq {seq} on claim seq {c.seq}, which is no longer live")
            return
        if kind == "void":
            c.void_seq, c.void_reason = seq, e["reason"]
        else:
            c.row_id = e["row_id"]
            st.rows[c.unit] = dict(e)
            st.request_regimes.setdefault(rid, set()).add(c.unit[4])
            sk = e["seed_key"]
            if sk is not None:
                t = tuple(sk)
                if t in st.seed_keys:
                    st.problems.append(f"row event seq {seq}: its seed block was already recorded as "
                                       f"{st.seed_keys[t]}")
                else:
                    st.seed_keys[t] = e["row_id"]


def open_ident(e: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (e["kind"], e["purpose"], e["family"], e["regime_id"], e["protocol"], S.canonical(e["spec"]))


def _family_ident(e: Mapping[str, Any]) -> Tuple[Any, ...]:
    return (e["decision_kind"], e["rule"], e["protocol"], e["commit"])


def void_reason(claim: Claim, *, now: _dt.datetime, host: str, alive: Callable[[int], bool],
                row_in_shard: Callable[[Claim], Optional[str]]) -> Optional[str]:
    """Why ``claim`` may be voided now (``pid_dead`` / ``expired``), or ``None``. Deterministic in its inputs: the
    clock, the host, the liveness oracle and the claimant's own shard."""
    if not claim.live or row_in_shard(claim) is not None:
        return None
    if claim.host == host and not alive(claim.pid):
        return "pid_dead"
    if now > claim.expires_at:
        return "expired"
    return None
