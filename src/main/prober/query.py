"""JSON CLI over the probing infrastructure — for agents and scripts.

Prints JSON to stdout (and JSON ``{"error": ...}`` to stdout with exit code 1 on
failure, so an agent always gets parseable output). Mirrors ``ProbeSession``.

A typical investigation:
    python -m main.prober.query summary  <run_dir>                         # orient
    python -m main.prober.query list     <run_dir> --outcome loss          # pick battles
    python -m main.prober.query scan     <run_dir> --outcome loss --opponent X  # worst turn per battle, ranked
    python -m main.prober.query overview <summary.json>                    # model-free digest
    python -m main.prober.query find     <summary.json> value_drop --limit 5
    python -m main.prober.query find     <summary.json> disagree           # loads the model
    python -m main.prober.query analyze  <summary.json> <inv> [--tier nearest]

``<summary.json>`` is a battle id from list/summary output (the ``id`` field).
"""

from __future__ import annotations

import argparse
import json
import sys

_FIND_CRITERIA = ["switch", "uncertain", "faint", "disagree",
                  "value_drop", "low_value", "high_value"]

_EXAMPLES = """\
examples:
  # 1. orient on a run (steps, opponents, win/loss, model identity, checkpoints)
  python -m main.prober.query summary models/run_X

  # 2. list the losses at a step, grab a battle id
  python -m main.prober.query list models/run_X --outcome loss --step 8000000

  # 2b. cross-battle: the worst turn in EVERY loss vs an opponent, ranked (model-free)
  python -m main.prober.query scan models/run_X --outcome loss --opponent aggressive_v2 --limit 10
  python -m main.prober.query scan models/run_X --outcome loss --metric td_residual

  # 3. model-free per-decision digest of that battle (V(s), ΔV, TD, flags, `notable`)
  python -m main.prober.query overview <id>

  # 4. rank decisions by where the critic's value cratered, or where the model disagrees
  python -m main.prober.query find <id> value_drop --limit 5
  python -m main.prober.query find <id> disagree

  # 5. full forensic analysis of one decision (loads exact→nearest→recent model)
  python -m main.prober.query analyze <id> 7 --tier nearest
"""


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m main.prober.query",
        description="JSON probing CLI for agents (summary/list/overview/find/analyze).",
        epilog=_EXAMPLES, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("summary", help="orient on a run: steps/opponents/win-loss/identity")
    ps.add_argument("root", help="run dir / eval_traces dir")

    pl = sub.add_parser("list", help="list battles in a run dir")
    pl.add_argument("root", help="run dir / eval_traces dir")
    pl.add_argument("--outcome", choices=["win", "loss"])
    pl.add_argument("--opponent")
    pl.add_argument("--step", type=int)

    psc = sub.add_parser(
        "scan", help="cross-battle turning-point scan (model-free): worst ΔV/TD per battle, ranked")
    psc.add_argument("root", help="run dir / eval_traces dir")
    psc.add_argument("--outcome", choices=["win", "loss"])
    psc.add_argument("--opponent")
    psc.add_argument("--step", type=int)
    psc.add_argument("--limit", type=int, default=None, help="cap the number of battles returned")
    psc.add_argument("--metric", default="value_drop", choices=["value_drop", "td_residual"],
                     help="rank by most-negative ΔV (default) or critic TD surprise")

    for name, helptext in (("overview", "model-free per-decision digest"),
                           ("find", "rank/list invocations matching a criterion"),
                           ("analyze", "full analysis of one decision")):
        sp = sub.add_parser(name, help=helptext)
        sp.add_argument("battle", help="a battle id (the *_summary.json path from list/summary)")
        if name == "find":
            sp.add_argument("criterion", choices=_FIND_CRITERIA)
            sp.add_argument("--limit", type=int, default=None, help="cap the number of hits")
        if name == "analyze":
            sp.add_argument("inv", type=int, help="invocation index")
        sp.add_argument("--ckpt", default=None, help="checkpoint override (else exact→nearest→recent)")
        sp.add_argument("--tier", default="auto", choices=["auto", "nearest", "recent"])
    return p


def _run(args) -> object:
    from main.prober.session import ProbeSession

    if args.cmd == "summary":
        return ProbeSession(args.root).run_summary()
    if args.cmd == "list":
        return ProbeSession(args.root).battles(
            outcome=args.outcome, opponent=args.opponent, step=args.step)
    if args.cmd == "scan":
        return ProbeSession(args.root).scan(
            outcome=args.outcome, opponent=args.opponent, step=args.step,
            limit=args.limit, metric=args.metric)
    sess = ProbeSession(args.battle, ckpt_override=args.ckpt, tier=args.tier)
    if args.cmd == "overview":
        return sess.battle_overview(args.battle)
    if args.cmd == "find":
        return sess.find(args.battle, args.criterion, limit=args.limit)
    return sess.analyze(args.battle, args.inv)  # analyze


def main() -> None:
    args = _build_parser().parse_args()
    try:
        out = _run(args)
    except Exception as e:  # noqa: BLE001 — agents want a JSON error, not a traceback
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}, indent=2))
        sys.exit(1)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
