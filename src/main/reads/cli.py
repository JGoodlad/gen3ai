"""``python -m main.reads`` -- the verbs of a registered read (``designs/tools/reads.md``).

    python -m main.reads check-spec --spec read_spec.json              # validate a spec; a pin must name a commit
                                 [--pin-pending]                       #   review a spec BEFORE its pin exists (it has none)
    python -m main.reads plan    --spec read_spec.json --state DIR     # write the state; copy settled models
    python -m main.reads run     --state DIR --lane anchor|s --hours H # work one lane (detached; resumable)
                                 [--concurrency K]                     #   anchor: K units at once (default 1)
                                 [--max-units N]                       #   anchor: start N units, finish, stop (stage 0)
    python -m main.reads resume  --state DIR --lane anchor|s --hours H # = run, after printing what is left; a HELD lane
                                 [--by WHO] [--why TEXT]               #   is released first (--by required then)
    python -m main.reads hold    --state DIR --lane anchor|s --by WHO --why TEXT   # no new unit; the units in flight
                                                                                   # finish; the supervisor exits 0
    python -m main.reads pid     --state DIR --lane anchor|s [--wait S]  # the lane's LIVE supervisor pid (it records itself)
    python -m main.reads status  --state DIR                           # + supervisor_pids (the pid to watch), holds, each
                                                                       #   label's `exhausted` units (retries spent)
    python -m main.reads repin   --state DIR --to COMMIT --by WHO --why TEXT  # accept another playing commit IF the play
                                                                              # path is byte-identical (else refuse)
    python -m main.reads assemble --state DIR                          # -> cells.json (regime ids ALWAYS; an anchor group
                                                                       #    names its unplayed and EXHAUSTED units)
    python -m main.reads decide  --state DIR                           # -> inputs.json, decision.json
    python -m main.reads record  --state DIR [--out PATH]              # -> record.json (schema-checked; INCOMPLETE_BY_STOP
                                                                       #    only for a spec that registered a reduced read)
    python -m main.reads check-record PATH
    python -m main.reads search-cost --state DIR | --cells DIR...      # S-U stage 0: the searched games' cost, G-PREFIX

Exit 0 on success, 2 on a refusal (named on stderr), 3 when R moved under an S lane.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m main.reads", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check-spec")
    c.add_argument("--spec", required=True)
    c.add_argument("--pin-pending", action="store_true",
                   help="review a spec whose pin is not decided yet: it must carry none; its sha256 is not final")
    p = sub.add_parser("plan")
    p.add_argument("--spec", required=True)
    p.add_argument("--state", required=True)
    for n in ("run", "resume"):
        r = sub.add_parser(n)
        r.add_argument("--state", required=True)
        r.add_argument("--lane", required=True, choices=("anchor", "s"))
        stop = r.add_mutually_exclusive_group(required=True)
        stop.add_argument("--hours", type=float, help="the registered stop, from now")
        stop.add_argument("--stop-at-epoch", type=int, help="the registered stop, an absolute epoch")
        r.add_argument("--workers", type=int, default=2, help="S engines per heavy-slot unit")
        r.add_argument("--threads", type=int, default=2, help="the Rust core's worker threads per S engine")
        r.add_argument("--unit-s", type=int, default=2400, help="an S unit's time box (one heavy-slot hold), s")
        r.add_argument("--concurrency", type=int, default=1,
                       help="ANCHOR lane only: units at once, each its own short-lane job and port (default 1; "
                            "capped at the box's heavy-slot count)")
        r.add_argument("--max-units", type=int, default=None,
                       help="ANCHOR lane only: start at most N units in this supervisor's life, let them finish, stop "
                            "(S-U's stage 0 = the queue's first units; not a registered stop)")
        if n == "resume":
            r.add_argument("--by", default=None, help="who releases a HELD lane (required only when it is held)")
            r.add_argument("--why", default=None, help="the reason, recorded with the release (default: resume)")
    h = sub.add_parser("hold", help="stop launching units on a lane; the units in flight finish; the supervisor exits")
    h.add_argument("--state", required=True)
    h.add_argument("--lane", required=True, choices=("anchor", "s"))
    h.add_argument("--by", required=True, help="who is holding the lane")
    h.add_argument("--why", required=True, help="the reason, recorded with the hold")
    pd = sub.add_parser("pid", help="print the pid of the lane's live supervisor (as it recorded itself)")
    pd.add_argument("--state", required=True)
    pd.add_argument("--lane", required=True, choices=("anchor", "s"))
    pd.add_argument("--wait", type=float, default=0.0, help="wait up to this many seconds (max 600) for it to start")
    rp = sub.add_parser("repin", help="accept another playing commit when the play path is byte-identical")
    rp.add_argument("--state", required=True)
    rp.add_argument("--to", required=True, help="the commit (sha, tag or branch) the read will play from")
    rp.add_argument("--by", required=True, help="who is repinning")
    rp.add_argument("--why", required=True, help="the reason, recorded with the repin")
    for n in ("status", "assemble", "decide"):
        sub.add_parser(n).add_argument("--state", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("--state", required=True)
    rec.add_argument("--out", default=None, help="also write the record here (e.g. <stage>/reads/<arm>_look<k>.json)")
    ck = sub.add_parser("check-record")
    ck.add_argument("path")
    sc = sub.add_parser("search-cost", help="S-U stage 0: the searched games' per-decision cost and G-PREFIX")
    where = sc.add_mutually_exclusive_group(required=True)
    where.add_argument("--state", help="a read's state: every searched unit, + G-PREFIX per searched group")
    where.add_argument("--cells", nargs="+", help="main.anchors --out dirs holding search_decisions.jsonl")
    u = sub.add_parser("s-unit", help="(internal) one S unit inside its heavy slot")
    u.add_argument("--state", required=True)
    u.add_argument("--group", required=True)
    u.add_argument("--box-s", type=int, required=True)
    u.add_argument("--deadline-epoch", type=int, default=None, help="the registered stop (absolute)")
    u.add_argument("--threads", type=int, default=2)
    u.add_argument("--engine", action="append", required=True, help="<model label>:<member>[,<member>...]")
    return ap


def _dump(doc: object) -> None:
    json.dump(doc, sys.stdout, indent=1, sort_keys=True, default=str)
    print()


def main(argv: Optional[List[str]] = None) -> int:
    from main.reads import api

    a = _parser().parse_args(argv)
    try:
        if a.cmd == "check-spec":
            _dump(api.check_spec(Path(a.spec), pin_pending=a.pin_pending))
            return 0
        if a.cmd == "plan":
            _dump(api.plan(Path(a.spec), Path(a.state)))
            return 0
        if a.cmd in ("run", "resume"):
            stop = a.stop_at_epoch if a.stop_at_epoch is not None else time.time() + 3600.0 * a.hours
            if a.cmd == "resume":
                rel = api.release(Path(a.state), a.lane, by=a.by, why=a.why)     # a held lane is released first
                st = api.status(Path(a.state))
                if rel["released"]:
                    st["released_hold"] = rel
                _dump(st)
            return api.run(Path(a.state), a.lane, stop, workers=a.workers, threads=a.threads, unit_s=a.unit_s,
                           concurrency=a.concurrency, max_units=a.max_units)
        if a.cmd == "hold":
            doc = api.hold(Path(a.state), a.lane, by=a.by, why=a.why)
            for w in doc["warnings"]:
                print(f"[main.reads] WARNING: {w}", file=sys.stderr)
            _dump(doc)
            return 0
        if a.cmd == "pid":
            print(api.pid(Path(a.state), a.lane, wait_s=a.wait))
            return 0
        if a.cmd == "repin":
            _dump(api.repin(Path(a.state), a.to, by=a.by, why=a.why))
            return 0
        if a.cmd == "status":
            _dump(api.status(Path(a.state)))
            return 0
        if a.cmd == "assemble":
            cells = api.assemble(Path(a.state))
            _dump({g: dict({"regime_id": grp["regime_id"], "inconclusive": grp["inconclusive"]},
                           # an anchor group's unplayed units, and which of them EXHAUSTED their retries (reads.md §6)
                           **({k: grp[k] for k in ("unplayed", "exhausted_units", "unplayed_exhausted_units")}
                              if lane == "anchor" else {}))
                   for lane in ("anchor", "s") for g, grp in cells[lane].items()})
            return 0
        if a.cmd == "decide":
            _dump(api.decide(Path(a.state)))
            return 0
        if a.cmd == "record":
            rec = api.record(Path(a.state), out=Path(a.out) if a.out else None)
            print(api.summary_line(rec))
            return 0
        if a.cmd == "check-record":
            api.load_record(Path(a.path))
            print("OK")
            return 0
        if a.cmd == "search-cost":
            from main.reads import search_cost as SC

            _dump(SC.state_report(Path(a.state)) if a.state else SC.report([Path(c) for c in a.cells]))
            return 0
        from main.reads import runner

        return runner.s_unit(Path(a.state), a.group, a.engine, a.box_s, deadline_epoch=a.deadline_epoch,
                             threads=a.threads)
    except api.REFUSALS as exc:
        print(f"[main.reads] REFUSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
