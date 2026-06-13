"""JSON CLI over the probing infrastructure — for agents and scripts.

Prints JSON to stdout (and JSON ``{"error": ...}`` to stdout with exit code 1 on
failure, so an agent always gets parseable output). Mirrors ``ProbeSession``.

A typical investigation:
    python -m main.prober.query triage   <run_dir>                         # RANK failure LEVERS (start here)
    python -m main.prober.query probe    <run_dir> is_faster               # is a derived fact already in the rep?
    python -m main.prober.query summary  <run_dir>                         # orient
    python -m main.prober.query list     <run_dir> --outcome loss          # pick battles
    python -m main.prober.query scan     <run_dir> --outcome loss --opponent X  # worst turn per battle, ranked
    python -m main.prober.query overview <summary.json>                    # model-free digest
    python -m main.prober.query find     <summary.json> value_drop --limit 5
    python -m main.prober.query find     <summary.json> disagree           # loads the model
    python -m main.prober.query analyze  <summary.json> <inv> [--tier nearest]
    python -m main.prober.query falsify  <summary.json> [--inv N] [--seeds 40]  # luck vs mistake (re-rolls)
    python -m main.prober.query falsify-scan <run_dir> [--opponent X]            # RUN-LEVEL reducible-vs-aleatoric
    python -m main.prober.query calibration  <run_dir> [--step N]                # split `unattributed`: critic vs lost

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
  # 0. RANK the failure levers across a whole run (which obs/reward/critic axis recovers the most)
  python -m main.prober.query triage models/run_X
  python -m main.prober.query triage models/run_X --opponent heuristic2 --step 46000019

  # 0b. PROBE whether a derived fact is already in the model's representation (or we should add it)
  python -m main.prober.query probe models/run_X is_faster      # rep vs the provided active_outspeed
  python -m main.prober.query probe models/run_X damage_taken   # rep vs active_exp — is the SPREAD there?

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

  # 6. dice attribution: were the worst decisions of this loss LUCK or a MISTAKE?
  #    (re-rolls the real turns via the reconstruction layer — bridge-eval traces only)
  python -m main.prober.query falsify <id>                       # worst 3 decisions by δ
  python -m main.prober.query falsify <id> --inv 7 --seeds 60    # one chosen decision

  # 7. RUN-LEVEL: aggregate luck-vs-mistake over every loss → a crater-fraction BRACKET
  #    (aleatoric [LUCK] / unattributed [NEUTRAL residual] / proven policy_reducible [MISTAKE];
  #    critic_headroom_upper_bound = LUCK+NEUTRAL is an UPPER BOUND — read its caveats)
  python -m main.prober.query falsify-scan models/run_X                       # all losses
  python -m main.prober.query falsify-scan models/run_X --opponent aggressive_v2 --concurrency 4

  # 8. CALIBRATION: resolve falsify-scan's `unattributed` craters into critic_overvalued
  #    (epistemic) vs lost_position, via recorded V(s) vs realized return G(s) — selection-aware
  python -m main.prober.query calibration models/run_X --step 30000000 --concurrency 8
"""


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m main.prober.query",
        description="JSON probing CLI for agents (triage/probe/summary/list/scan/history-saliency/"
                    "overview/find/analyze/falsify/falsify-scan/calibration).",
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

    pt = sub.add_parser(
        "triage", help="loss attribution: rank failure CATEGORIES by recoverable win-rate (the lever order)")
    pt.add_argument("root", help="run dir / eval_traces dir")
    pt.add_argument("--step", type=int, help="step to triage (default: latest with loss traces)")
    pt.add_argument("--opponent", help="restrict to one opponent")

    pp = sub.add_parser(
        "probe", help="linear-probe the model's activations for a derived quantity (is it already in the rep?)")
    pp.add_argument("root", help="run dir / eval_traces dir")
    pp.add_argument("target", choices=["is_faster", "damage_taken", "faint_soon",
                                       "faint_healthy", "big_hit_incoming", "opp_switches",
                                       "opp_status_move"])
    pp.add_argument("--step", type=int, help="step to probe (default: latest with traces)")
    pp.add_argument("--opponent", help="restrict to one opponent")
    pp.add_argument("--which", default="vf", choices=["vf", "pi"],
                    help="probe the value-head (vf, default) or policy-head (pi) features")
    pp.add_argument("--max-decisions", type=int, default=1500, help="cap decisions sampled (default 1500)")

    ph = sub.add_parser(
        "history-saliency", help="per-turn-slot saliency of the turn-history block (can we shorten it?)")
    ph.add_argument("root", help="run dir / eval_traces dir")
    ph.add_argument("--step", type=int, help="step to measure (default: latest with traces)")
    ph.add_argument("--opponent", help="restrict to one opponent")
    ph.add_argument("--max-decisions", type=int, default=400, help="cap decisions sampled (default 400)")

    svi = sub.add_parser(
        "switch-vs-info",
        help="MODEL-FREE: does our policy switch more when it knows LESS about the opponent?")
    svi.add_argument("root", help="run dir / eval_traces dir")
    svi.add_argument("--step", type=int, help="step (default: latest with traces)")
    svi.add_argument("--opponent", help="restrict to one opponent")
    svi.add_argument("--outcome", choices=["win", "loss"], help="restrict to win/loss battles")
    svi.add_argument("--max-battles", type=int, default=400, help="cap battles scanned (default 400)")

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

    pfa = sub.add_parser(
        "falsify", help="dice attribution: luck-vs-mistake for a battle's worst decisions "
                        "(re-rolls the real turns; bridge-eval traces with *_reconstruction.json only)")
    pfa.add_argument("battle", help="a battle id (the *_summary.json path from list/summary)")
    pfa.add_argument("--inv", type=int, action="append",
                     help="falsify this invocation (repeatable; default: --worst by δ)")
    pfa.add_argument("--worst", type=int, default=3,
                     help="anchor the N most-negative-δ move decisions (default 3)")
    pfa.add_argument("--seeds", type=int, default=40, help="re-roll seeds per arm (default 40)")
    pfa.add_argument("--alts", type=int, default=3, help="alternative actions to test (default 3)")
    pfa.add_argument("--followup", default="random", choices=["random", "default"],
                     help="policy for NEW mid-turn decisions in re-rolled timelines")

    pfs = sub.add_parser(
        "falsify-scan", help="RUN-LEVEL luck-vs-mistake attribution: falsify every loss's worst "
                             "craters into a crater-fraction BRACKET (aleatoric/unattributed/"
                             "proven-policy; critic_headroom_upper_bound is an UPPER BOUND — read "
                             "caveats). Input to the distributional-critic decision; bridge-eval only")
    pfs.add_argument("root", help="run dir / eval_traces dir")
    pfs.add_argument("--outcome", default="loss", choices=["win", "loss"],
                     help="which battles to falsify (default loss — craters live in losses)")
    pfs.add_argument("--opponent")
    pfs.add_argument("--step", type=int)
    pfs.add_argument("--limit", type=int, default=20, help="max battles to falsify (default 20)")
    pfs.add_argument("--worst", type=int, default=2,
                     help="worst-δ decisions falsified per battle (default 2)")
    pfs.add_argument("--seeds", type=int, default=32, help="re-roll seeds per arm (default 32)")
    pfs.add_argument("--alts", type=int, default=2, help="alternative actions per decision (default 2)")
    pfs.add_argument("--followup", default="random", choices=["random", "default"])
    pfs.add_argument("--concurrency", type=int, default=1,
                     help="battles falsified in parallel (default 1). Each re-roll spawns Node, so "
                          "raise this only on an IDLE box — it contends with a live training run.")

    pca = sub.add_parser(
        "calibration", help="critic CALIBRATION: split falsify_scan's `unattributed` craters into "
                            "critic_overvalued (epistemic) vs lost_position via recorded V(s) vs "
                            "realized return G(s) — a selection-aware reliability curve. Model-free")
    pca.add_argument("root", help="run dir / eval_traces dir")
    pca.add_argument("--outcome", default="loss", choices=["win", "loss"])
    pca.add_argument("--opponent")
    pca.add_argument("--step", type=int)
    pca.add_argument("--limit", type=int, default=20, help="max loss battles to falsify (default 20)")
    pca.add_argument("--worst", type=int, default=2)
    pca.add_argument("--seeds", type=int, default=32)
    pca.add_argument("--alts", type=int, default=2)
    pca.add_argument("--followup", default="random", choices=["random", "default"])
    pca.add_argument("--concurrency", type=int, default=8,
                     help="re-roll parallelism for the falsify pass (default 8; lower on a busy box)")
    pca.add_argument("--bins", type=int, default=10, help="reliability-curve V-bins (default 10)")
    pca.add_argument("--overvalue-tau", type=float, default=5.0,
                     help="reliability-gap (return units) above which a crater is critic_overvalued")

    pdt = sub.add_parser(
        "decision-table", help="per-decision forensic export (move-cat / policy conf / reward / "
                               "critic dV / incoming-KO belief) over a run — the consolidated plumbing")
    pdt.add_argument("root", help="run dir / eval_traces dir / summary.json")
    pdt.add_argument("--step", type=int, action="append", help="filter to this step (repeatable)")
    pdt.add_argument("--opponent", action="append", help="filter to this opponent (repeatable)")
    pdt.add_argument("--outcome", choices=["win", "loss"], action="append", help="filter (repeatable)")
    pdt.add_argument("--cat", action="append",
                     help="filter to a move category (selfko/recovery/setup/stall/status/switch/attack_or_other)")
    pdt.add_argument("--max-battles", type=int, default=None, help="cap battles scanned")
    pdt.add_argument("--out", default=None, help="write the FULL table as JSONL to this path")
    pdt.add_argument("--limit", type=int, default=None,
                     help="print the first N full rows (default: a per-category digest)")
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
    if args.cmd == "triage":
        return ProbeSession(args.root).triage(step=args.step, opponent=args.opponent)
    if args.cmd == "probe":
        return ProbeSession(args.root).probe(
            args.target, step=args.step, opponent=args.opponent,
            which=args.which, max_decisions=args.max_decisions)
    if args.cmd == "history-saliency":
        return ProbeSession(args.root).history_saliency(
            step=args.step, opponent=args.opponent, max_decisions=args.max_decisions)
    if args.cmd == "switch-vs-info":
        return ProbeSession(args.root).switch_vs_info(
            step=args.step, opponent=args.opponent, outcome=args.outcome,
            max_battles=args.max_battles)
    if args.cmd == "falsify":
        return ProbeSession(args.battle).falsify(
            args.battle, invs=args.inv, worst=args.worst,
            n_seeds=args.seeds, n_alts=args.alts, followup=args.followup)
    if args.cmd == "falsify-scan":
        return ProbeSession(args.root).falsify_scan(
            outcome=args.outcome, opponent=args.opponent, step=args.step,
            limit=args.limit, worst=args.worst, n_seeds=args.seeds,
            n_alts=args.alts, followup=args.followup, concurrency=args.concurrency)
    if args.cmd == "calibration":
        return ProbeSession(args.root).calibration(
            outcome=args.outcome, opponent=args.opponent, step=args.step,
            limit=args.limit, worst=args.worst, n_seeds=args.seeds, n_alts=args.alts,
            followup=args.followup, concurrency=args.concurrency,
            n_bins=args.bins, overvalue_tau=args.overvalue_tau)
    if args.cmd == "decision-table":
        from main.prober import forensics
        rows = ProbeSession(args.root).decision_table(
            steps=args.step, opponents=args.opponent, outcomes=args.outcome,
            categories=args.cat, max_battles=args.max_battles)
        if args.out:
            with open(args.out, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            return {"wrote": args.out, "n_rows": len(rows),
                    "digest": forensics.decision_table_digest(rows)}
        if args.limit is not None:
            return rows[:args.limit]
        return forensics.decision_table_digest(rows)
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
