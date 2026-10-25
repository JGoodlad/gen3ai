"""``python -m main.reads`` -- the verbs of a registered read (``designs/tools/reads.md``).

    python -m main.reads check-spec --spec read_spec.json              # validate a spec (no file but it is read)
    python -m main.reads plan    --spec read_spec.json --state DIR     # write the state; copy settled models
    python -m main.reads run     --state DIR --lane anchor|s --hours H # work one lane (detached; resumable)
    python -m main.reads resume  --state DIR --lane anchor|s --hours H # = run, after printing what is left
    python -m main.reads status  --state DIR
    python -m main.reads assemble --state DIR                          # -> cells.json (regime ids ALWAYS)
    python -m main.reads decide  --state DIR                           # -> inputs.json, decision.json
    python -m main.reads record  --state DIR [--out PATH]              # -> record.json (schema-checked)
    python -m main.reads check-record PATH

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
    for n in ("status", "assemble", "decide"):
        sub.add_parser(n).add_argument("--state", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("--state", required=True)
    rec.add_argument("--out", default=None, help="also write the record here (e.g. <stage>/reads/<arm>_look<k>.json)")
    ck = sub.add_parser("check-record")
    ck.add_argument("path")
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
            s = api.load_spec(Path(a.spec))
            _dump({"read": s.read, "rule": s.rule, "models": sorted(s.models), "anchor": sorted(s.anchor),
                   "s": sorted(s.s), "sha256": s.sha256})
            return 0
        if a.cmd == "plan":
            _dump(api.plan(Path(a.spec), Path(a.state)))
            return 0
        if a.cmd in ("run", "resume"):
            stop = a.stop_at_epoch if a.stop_at_epoch is not None else time.time() + 3600.0 * a.hours
            if a.cmd == "resume":
                _dump(api.status(Path(a.state)))
            return api.run(Path(a.state), a.lane, stop, workers=a.workers, threads=a.threads, unit_s=a.unit_s)
        if a.cmd == "status":
            _dump(api.status(Path(a.state)))
            return 0
        if a.cmd == "assemble":
            cells = api.assemble(Path(a.state))
            _dump({g: {"regime_id": grp["regime_id"], "inconclusive": grp["inconclusive"]}
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
        from main.reads import runner

        return runner.s_unit(Path(a.state), a.group, a.engine, a.box_s, deadline_epoch=a.deadline_epoch,
                             threads=a.threads)
    except api.REFUSALS as exc:
        print(f"[main.reads] REFUSED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
