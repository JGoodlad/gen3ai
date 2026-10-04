"""``python -m main.eval_ledger`` — the eval COUNT ledger's operator CLI (design_evaluation.md §0b; eval unit U1).

    python -m main.eval_ledger audit [--root DIR] [--json] [--rebuild-index]
                                                                     # validate everything + every cross invariant
                                                                     #   (+ the persisted indexes; --rebuild-index
                                                                     #   drops and rebuilds them from the streams)
    python -m main.eval_ledger show [--root DIR]                     # rows by producer / purpose / regime; the queue
    python -m main.eval_ledger verify <decision_id> [--root DIR]     # one decision's rows + digest (+ verdict)
    python -m main.eval_ledger void-dead [--root DIR]                # the void rule over every live claim
    python -m main.eval_ledger close-stale [--root DIR] [--apply]    # gzip a DEAD writer's idle shard (24 h)
    python -m main.eval_ledger family-register --id F --decision-kind ab_verdict --rule R --protocol P [--commit C]
    python -m main.eval_ledger request-close <request_id> [--cancel REASON]

``--root`` defaults to the run archive's ledger, ``<archive>/_ledger`` (``utils.paths.run_archive_dir``;
``$GEN3AI_MODELS_DIR`` is authoritative). ``audit`` exits 1 on any problem; the scheduler runs it at startup and
refuses to start on a failure. The write commands write ONLY the requests stream (and ``close-stale --apply``
gzips a shard it proves dead); nothing here edits a row.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from agents.training.eval_ledger import audit as AU
from agents.training.eval_ledger import schema as S
from agents.training.eval_ledger import store as ST
from agents.training.eval_ledger.writer import LedgerWriter


def _root(a: argparse.Namespace) -> Path:
    return ST.resolve_root(a.root)


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m main.eval_ledger", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("audit", "show", "void-dead", "close-stale"):
        p = sub.add_parser(name)
        p.add_argument("--root", default=None)
        if name in ("audit", "show"):
            p.add_argument("--json", action="store_true")
        if name == "audit":
            p.add_argument("--rebuild-index", action="store_true",
                           help="drop and rebuild the persisted indexes (<root>/.ledger_index/, caches) from the "
                                "streams, then verify them (writes only that directory)")
        if name == "close-stale":
            p.add_argument("--apply", action="store_true", help="gzip (default: report what would be closed)")
    v = sub.add_parser("verify")
    v.add_argument("decision_id")
    v.add_argument("--root", default=None)
    f = sub.add_parser("family-register")
    f.add_argument("--root", default=None)
    f.add_argument("--id", required=True)
    f.add_argument("--decision-kind", required=True, choices=S.GROUP_SEQUENTIAL_KINDS)
    f.add_argument("--rule", required=True, help="the registered rule (a doc anchor + version)")
    f.add_argument("--protocol", required=True, choices=S.PROTOCOLS)
    f.add_argument("--commit", default=None, help="the commit the family's cells are played at")
    c = sub.add_parser("request-close")
    c.add_argument("request_id")
    c.add_argument("--root", default=None)
    c.add_argument("--cancel", default=None, metavar="REASON")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    a = _parser().parse_args(argv)
    root = _root(a)
    if a.cmd == "audit":
        rep = AU.audit(root, rebuild_index=a.rebuild_index)
        if a.json:
            print(json.dumps({"root": rep.root, "ok": rep.ok, "problems": rep.problems, "counts": rep.counts},
                             indent=1, sort_keys=True))
        else:
            for p in rep.problems:
                print(f"[audit] PROBLEM: {p}")
            print(f"[audit] {root}: {rep.counts} — {'OK' if rep.ok else f'{len(rep.problems)} problem(s)'}")
        return 0 if rep.ok else 1
    if a.cmd == "show":
        s = AU.show(root)
        print(json.dumps(s, indent=1, sort_keys=True))
        return 0
    if a.cmd == "verify":
        v = AU.verify(root, a.decision_id)
        for p in v.problems:
            print(f"[verify] PROBLEM: {p}")
        print(f"[verify] {a.decision_id}: {'OK' if v.ok else 'FAILED'} — {v.note}")
        return 0 if v.ok else 1
    if a.cmd == "close-stale":
        stale = ST.close_stale(root, apply=a.apply)
        verb = "closed" if a.apply else "would close (dry run; --apply to gzip)"
        for p in stale.closed:
            print(f"[close-stale] {verb} {p}")
        for why in stale.skipped:
            print(f"[close-stale] skipped {why}")
        return 0
    w = LedgerWriter(root, producer="ledger_cli")
    if a.cmd == "void-dead":
        for e in w.void_dead():
            print(f"[void-dead] voided claim seq {e['claim_seq']} on {e['request_id']} batch {e['batch']} "
                  f"({e['reason']})")
        return 0
    if a.cmd == "family-register":
        e = w.register_family(a.id, decision_kind=a.decision_kind, rule=a.rule, protocol=a.protocol, commit=a.commit)
        print(json.dumps(e, sort_keys=True))
        return 0
    if a.cmd == "request-close":
        e = w.finish_request(a.request_id, cancel_reason=a.cancel)
        print(json.dumps(e, sort_keys=True))
        return 0
    raise AssertionError(a.cmd)


if __name__ == "__main__":
    sys.exit(main())
