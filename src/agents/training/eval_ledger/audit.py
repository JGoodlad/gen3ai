"""THE AUDIT — the whole ledger validated, plus every CROSS-record invariant (§0b.4 item 5). The scheduler runs it
at startup and refuses to start on a failure; ``python -m main.eval_ledger audit`` runs it by hand.

The invariants (each a line in :attr:`AuditReport.problems` when broken):

* every row, event, decision and reference validates (v1 rows as v1, then upgraded);
* ``row_id`` is unique; among live rows the batch key and the seed block are unique;
* every ``supersedes`` resolves, to a row with the same batch key, and no row is superseded twice;
* every row's ``request`` resolves to an ``open`` whose kind / family / opened / purpose it repeats, and its
  ``family`` to a registration; one regime per request; one protocol per pinned family, equal to the pin;
* every requested row has its writer's claim, not voided, and a ``row`` event naming it; every ``row`` event
  names a row on disk;
* the requests stream folds without contradiction (``queue.fold``);
* every decision's consumed rows exist (live) and their digest matches; its request / family resolves;
* reference ids are unique;
* the persisted INDEXES (``.ledger_index/``, F-ED-22 — caches, never the record) are exactly what the streams fold to
  as far as their cursors have read (``event_index.verify`` / ``row_index.verify``: every file re-read up to its
  cursor and compared). ``rebuild_index=True`` (``audit --rebuild-index``) drops and rebuilds them first.

``verify`` re-checks one decision: its rows and digest. Re-deriving its VERDICT needs the decision's rule
registered in :data:`RULES`: the plateau's TIER-1 GSPRT is (``agents.training.plateau_t1``, U9a, 2026-10-06); the
SPRT and A/B rules land with U4 / X5. A registered function returns ``None`` for a decision whose ``rule`` it does not
own (a future two-tier plateau decision shares the kind)."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from agents.training.eval_ledger import event_index as EI
from agents.training.eval_ledger import queue as Q
from agents.training.eval_ledger import row_index as RI
from agents.training.eval_ledger import schema as S
from agents.training.eval_ledger import store as ST

def _plateau_rule(d: Mapping[str, Any], rows: List[Dict[str, Any]]) -> Optional[str]:
    from agents.training import plateau_t1 as T1   # lazy: the rule imports sprt, which the ledger does not need

    return T1.rederive(d, rows)


#: decision kind -> a function re-deriving its verdict from (decision, rows), or ``None`` when the decision's
#: ``rule`` is not one it owns.
RULES: Dict[str, Callable[[Mapping[str, Any], List[Dict[str, Any]]], Optional[str]]] = {"plateau": _plateau_rule}


@dataclass
class AuditReport:
    root: str
    problems: List[str] = field(default_factory=list)
    counts: Dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.problems


def check_indexes(root: Path, *, rebuild: bool = False) -> List[str]:
    """The persisted indexes' problems (empty = each is what the streams fold to up to its cursors, or absent).
    ``rebuild`` drops and rebuilds both from the streams first (under the ledger lock). A root that cannot be
    written, or a stream that does not parse, is reported — never raised."""
    from agents.training.eval_ledger import incremental as INC
    import sqlite3

    out: List[str] = []
    if rebuild:
        try:
            with ST.locked(root):
                EI.EventIndex(root).rebuild()
            RI.RowIndex(root).rebuild()
        except (OSError, sqlite3.Error, INC.IndexUnavailable, S.LedgerSchemaError) as e:
            out.append(f"index rebuild failed: {type(e).__name__}: {e}")
            return out
    out.extend(EI.EventIndex(root).verify())
    out.extend(RI.RowIndex(root).verify())
    return out


def audit(root: Path, *, rebuild_index: bool = False) -> AuditReport:
    rep = AuditReport(root=str(root))
    if not root.exists():
        rep.counts = {"rows": 0}
        return rep
    P = rep.problems
    scanned = ST.scan_rows(root, problems=P)
    events = [e for e, _w in ST.scan_events(root, problems=P)]
    decisions = ST.scan_decisions(root, problems=P)
    references = ST.scan_references(root, problems=P)
    st = Q.fold(events)
    P.extend(f"requests stream: {p}" for p in st.problems)

    by_id: Dict[str, ST.ScannedRow] = {}
    for sr in scanned:
        rid = sr.row["row_id"]
        if rid in by_id:
            P.append(f"row_id {rid} appears twice ({by_id[rid].source}, {sr.source})")
        by_id[rid] = sr
    sup = Counter(sr.row["supersedes"] for sr in scanned if sr.row["supersedes"])
    for target, n in sup.items():
        if n > 1:
            P.append(f"row {target} is superseded {n} times")
    for sr in scanned:
        t = sr.row["supersedes"]
        if t is None:
            continue
        old = by_id.get(t)
        if old is None:
            P.append(f"{sr.source}: supersedes {t}, which does not exist")
        elif S.batch_key(old.row) != S.batch_key(sr.row) or S.seed_key(old.row) != S.seed_key(sr.row):
            P.append(f"{sr.source}: supersedes {t} but names another batch / seed block")
    live = [sr for sr in scanned if sr.row["row_id"] not in sup]
    for keyfn, what in ((S.batch_key, "batch key"), (S.seed_key, "seed block")):
        seen: Dict[Any, str] = {}
        for sr in live:
            bk = keyfn(sr.row)
            if bk is None:
                continue
            if bk in seen:
                P.append(f"DUPLICATE {what}: {sr.row['row_id']} ({sr.source}) repeats {seen[bk]}")
            else:
                seen[bk] = sr.row["row_id"]

    regimes: Dict[str, set] = {}
    for sr in live:
        r, req = sr.row, sr.row["request"]
        if req is None:
            continue
        op = st.requests.get(req["id"])
        if op is None:
            P.append(f"{sr.source}: request {req['id']!r} was never opened")
            continue
        for k, v in (("kind", op["kind"]), ("family", op["family"]), ("opened", op["ts"])):
            if req[k] != v:
                P.append(f"{sr.source}: request.{k} {req[k]!r} != the open's {v!r}")
        if r["purpose"] != op["purpose"]:
            P.append(f"{sr.source}: purpose {r['purpose']!r} != request {req['id']!r}'s {op['purpose']!r}")
        if req["family"] is not None:
            fam = st.families.get(req["family"])
            if fam is None:
                P.append(f"{sr.source}: family {req['family']!r} is not registered")
            elif r["regime"]["protocol"] != fam["protocol"]:
                P.append(f"{sr.source}: protocol {r['regime']['protocol']!r} in family {req['family']!r} pinned to "
                         f"{fam['protocol']!r}")
        if op["protocol"] is not None and r["regime"]["protocol"] != op["protocol"]:
            P.append(f"{sr.source}: protocol {r['regime']['protocol']!r} in request pinned to {op['protocol']!r}")
        regimes.setdefault(req["id"], set()).add(r["regime"]["regime_id"])
        u = Q.row_unit(r)
        ev = st.rows.get(u)  # type: ignore[arg-type]
        if ev is None or ev["row_id"] != r["row_id"]:
            P.append(f"{sr.source}: requested row {r['row_id']} has no `row` event (claim unrecorded, or a writer "
                     "died between its row and its event)")
            continue
        c = st.claims.get(int(ev["claim_seq"]))
        writer = r["row_id"].rsplit(":", 1)[0]
        if c is None or c.writer_id != writer or c.void_seq is not None:
            P.append(f"{sr.source}: row {r['row_id']} is not under a live claim of its own writer")
    for rid, regs in regimes.items():
        if len(regs) > 1:
            P.append(f"request {rid!r} holds {len(regs)} regimes {sorted(regs)} (one regime per request)")
    for u, ev in st.rows.items():
        if ev["row_id"] not in by_id:
            P.append(f"`row` event seq {ev['seq']} names row {ev['row_id']}, which is not on disk")

    live_ids = {sr.row["row_id"]: sr for sr in live}
    for d, where in decisions:
        ids = d["consumed"]["row_ids"]
        missing = [i for i in ids if i not in live_ids]
        if missing:
            P.append(f"{where}: decision {d['decision_id']} consumed rows not live in the ledger: {missing[:4]}")
        elif S.rows_digest((i, live_ids[i].sha) for i in ids) != d["consumed"]["digest"]:
            P.append(f"{where}: decision {d['decision_id']}'s consumed digest does not match its rows")
        if d["request_id"] is not None and d["request_id"] not in st.requests:
            P.append(f"{where}: decision {d['decision_id']} names request {d['request_id']!r}, never opened")
        if d["family"] is not None and d["family"] not in st.families:
            P.append(f"{where}: decision {d['decision_id']} names family {d['family']!r}, not registered")
    ref_ids = Counter(r["reference_id"] for r, _w in references)
    P.extend(f"reference {k} appears {n} times" for k, n in ref_ids.items() if n > 1)
    P.extend(check_indexes(root, rebuild=rebuild_index))
    rep.counts = {"rows": len(scanned), "live_rows": len(live), "requests": len(st.requests),
                  "families": len(st.families), "claims": len(st.claims),
                  "live_claims": sum(1 for c in st.claims.values() if c.live), "decisions": len(decisions),
                  "references": len(references)}
    return rep


@dataclass
class VerifyReport:
    decision_id: str
    problems: List[str]
    verdict_rederived: Optional[str]
    note: str

    @property
    def ok(self) -> bool:
        return not self.problems


def verify(root: Path, decision_id: str) -> VerifyReport:
    """Re-check ONE decision against the ledger (module docstring)."""
    decs = {d["decision_id"]: d for d, _w in ST.scan_decisions(root)}
    d = decs.get(decision_id)
    if d is None:
        return VerifyReport(decision_id, [f"no decision {decision_id!r} under {root}"], None, "")
    from agents.training.eval_ledger.reader import live_rows

    live = {sr.row["row_id"]: sr for sr in live_rows(root, S.parse_ts(d["as_of"]))}
    ids = d["consumed"]["row_ids"]
    probs = [f"consumed row {i} is not live as of {d['as_of']}" for i in ids if i not in live]
    if not probs and S.rows_digest((i, live[i].sha) for i in ids) != d["consumed"]["digest"]:
        probs.append("the consumed rows' digest does not match the decision's")
    rule = RULES.get(d["kind"])
    got = None if rule is None else rule(d, [live[i].row for i in ids if i in live])
    if got is None:
        return VerifyReport(decision_id, probs, None, f"no rule registered for {d['kind']!r} / {d['rule']!r}: rows "
                            "and digest checked, the verdict is not re-derived")
    if got != d["verdict"]:
        probs.append(f"the rule re-derives {got!r}, the decision says {d['verdict']!r}")
    return VerifyReport(decision_id, probs, got, "verdict re-derived")


def show(root: Path) -> Dict[str, Any]:
    """A summary: rows by producer / purpose / regime, the queue, the streams."""
    scanned = ST.scan_rows(root) if root.exists() else []
    st = Q.fold(e for e, _w in ST.scan_events(root)) if root.exists() else Q.QueueState()
    return {
        "root": str(root), "rows": len(scanned),
        "by_producer": dict(Counter(sr.producer for sr in scanned)),
        "by_purpose": dict(Counter(sr.row["purpose"] for sr in scanned)),
        "by_regime": dict(Counter(f"{sr.row['regime']['regime_id']} ({sr.row['regime']['protocol']})"
                                  for sr in scanned)),
        "by_schema_origin": dict(Counter("v1 (upgraded)" if "v1_id" in sr.row["regime"] else "v2" for sr in scanned)),
        "requests_open": sorted(set(st.requests) - st.done - st.cancelled),
        "requests_closed": sorted(st.done | st.cancelled), "families": sorted(st.families),
        "live_claims": sum(1 for c in st.claims.values() if c.live),
        "decisions": len(ST.scan_decisions(root)) if root.exists() else 0,
        "references": len(ST.scan_references(root)) if root.exists() else 0,
    }
