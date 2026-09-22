"""THE BEST-RESPONSE GAP — ``python -m main.best_response_gap``.

The POPULATION loop's meter, offline: no training, no server, nothing written under ``models/``.
Given one or more ``--exploiter`` runs it reports, per ROUND and per team ARCHETYPE, how much a
fresh best responder wins against the generalist it was trained against — and the round-over-round
change in that gap, paired on archetype. **The loop is working iff the gap falls.** The engine, the
full rationale and every refusal: ``src/agents/training/best_response_gap.py``.

```bash
export PYTHONPATH=$PYTHONPATH:src

# the archive read — era-1 (target arm W) against era-2 (target the plateau parent)
python -m main.best_response_gap \
    ai_v13_05_exploit_big5starmie ai_v13_06_exploit_ddtar_spikes ai_v13_10_exploit_stall \
    ai_v13_13_exploit5_offense ai_v13_14_exploit5_balance ai_v13_15_exploit5_stall \
    --json /tmp/brgap.json --md /tmp/brgap.md

# ... which REFUSES on dose (3.815e-08 vs 8.392e-09, 4.55x apart). Print it anyway:
python -m main.best_response_gap <the six runs> --allow-unmatched

# one round alone, on the endpoint convention the banked numbers use
python -m main.best_response_gap ai_v13_13_exploit5_offense --stat endpoint

# + N FRESH head-to-head games on the bridge (CPU, niced, seed 0, concurrency 1)
python -m main.best_response_gap ai_v13_13_exploit5_offense --play 20 [--greedy]
```
(in a linked worktree, first: ``export PYTHONPATH=$PYTHONPATH:src``)

🚨 **THE TWO RATES ARE DIFFERENT POPULATIONS.** The training-time series is GREEDY-vs-GREEDY (the
eval regime; ``eval_sentinel_greedy`` does not govern it). ``--play`` defaults to the TRAINING
regime, stochastic@1 on both sides. Every printed rate states its regime and the gap table is
built from the series alone — a played rate is reported BESIDE it, never folded into it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

from agents.training import best_response_gap as engine
from agents.training.best_response_gap import BestResponseGapError

DEFAULT_JSON = "best_response_gap.json"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m main.best_response_gap",
        description="Best-response gap: how much a fresh exploiter beats the generalist it was "
                    "trained against, by round and archetype, with the round-over-round delta.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="The gap FALLING round over round is the population loop working. A comparison at "
               "unmatched budget / dose / regime is REFUSED — that is how the 2026-09-21 "
               "era-2/era-1 read was confounded (a 4.5x dose gap nobody registered).")
    p.add_argument("runs", nargs="+", metavar="RUN",
                   help="exploiter run dirs, or bare run names resolved against the MAIN "
                        "checkout's models/.")
    p.add_argument("--stat", choices=("pooled", "endpoint"), default="pooled",
                   help="which rate drives the gap: every post-fork cycle pooled (default, the "
                        "higher-power read), or the LAST cycle (the banked convention). Both are "
                        "always printed.")
    p.add_argument("--rounds", nargs="+", default=[], metavar="NAME=ROUND",
                   help="pin a round. NAME is an exploiter run name or a target run name. "
                        "Default: inferred from the target's step, ascending.")
    p.add_argument("--teamsets", default=None, metavar="PATH",
                   help=f"archetype -> teamset map (default: {engine.DEFAULT_TEAMSETS}).")
    p.add_argument("--allow-unmatched", action="store_true",
                   help="print a comparison whose exploiters differ on budget / dose / regime. "
                        "The mismatch is carried in the header and in the JSON.")
    p.add_argument("--budget-tol", type=float, default=engine.DEFAULT_BUDGET_TOL)
    p.add_argument("--dose-tol", type=float, default=engine.DEFAULT_DOSE_TOL)
    p.add_argument("--play", type=int, default=0, metavar="N",
                   help="ALSO play N fresh exploiter-vs-target battles per run on the bridge. "
                        "Reported beside the series, never folded into it.")
    p.add_argument("--greedy", action="store_true",
                   help="--play in the EVAL regime (argmax both sides), which is the regime the "
                        "training-time series was measured in. Default is the TRAINING regime.")
    p.add_argument("--play-seed", type=int, default=0)
    p.add_argument("--impl", choices=("rust", "node"), default="rust")
    p.add_argument("--concurrency", type=int, default=1,
                   help="battles in flight. Above 1 is REFUSED (unreproducible).")
    p.add_argument("--draws", type=int, default=engine.DEFAULT_DRAWS)
    p.add_argument("--bootstrap-seed", type=int, default=engine.DEFAULT_BOOTSTRAP_SEED)
    p.add_argument("--json", dest="json_out", default=DEFAULT_JSON, metavar="PATH",
                   help=f"write the report JSON here (default: ./{DEFAULT_JSON}).")
    p.add_argument("--no-json", action="store_true", help="do not write the JSON.")
    p.add_argument("--md", dest="md_out", default=None, metavar="PATH")
    p.add_argument("--check", action="store_true",
                   help="read every run and run the matched gate; print nothing else, play "
                        "nothing, write nothing. Exit non-zero on any refusal.")
    p.add_argument("--quiet", action="store_true")
    return p


def parse_rounds(items: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise BestResponseGapError(f"--rounds {item!r}: expected NAME=ROUND")
        name, _, value = item.partition("=")
        try:
            out[name.strip()] = int(value)
        except ValueError as exc:
            raise BestResponseGapError(f"--rounds {item!r}: {value!r} is not an integer") from exc
    return out


# --------------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------------

def _pct(v: Optional[float], width: int = 7) -> str:
    return "—".rjust(width) if v is None else f"{v * 100:+.2f}".rjust(width)


def _rate(v: Optional[float]) -> str:
    return "—" if v is None else f"{v:.4f}"


def _table(header: Sequence[str], body: Sequence[Sequence[str]], indent: str = "  ") -> List[str]:
    if not body:
        return [indent + "(no rows)"]
    widths = [max(len(header[i]), *(len(b[i]) for b in body)) for i in range(len(header))]
    lines = [indent + "  ".join(h.ljust(w) for h, w in zip(header, widths)),
             indent + "  ".join("-" * w for w in widths)]
    lines += [indent + "  ".join(c.ljust(w) for c, w in zip(b, widths)) for b in body]
    return lines


def render_text(doc: Dict[str, Any]) -> str:
    out: List[str] = []
    out.append("BEST-RESPONSE GAP — how much a fresh exploiter beats the generalist it trained on")
    out.append(f"  gap = win rate − {doc['null_rate']}; the population loop is working iff the "
               "gap FALLS round over round")
    out.append(f"  stat: {doc['stat']}   regime: {doc['rate_regime']}")
    if doc.get("unmatched"):
        out.append("")
        out.append("🚨 UNMATCHED — this comparison was printed under --allow-unmatched. Every "
                   "number below carries these confounds:")
        for m in doc["mismatches"]:
            out.append(f"     {m['message']}")
    for c in doc.get("caveats", []):
        out.append(f"  ⚠ CAVEAT: {c}")
    out.append("")

    for blk in doc["rounds"]:
        target = f"{blk['target_run']}"
        if blk.get("target_step") is not None:
            target += f" @{blk['target_step']:,}"
        budget = f"{blk['budget']:,}" if blk.get("budget") is not None else "—"
        dose = f"{blk['dose_rate']:.4g}" if blk.get("dose_rate") is not None else "—"
        out.append(f"ROUND {blk['round']} · target {target} · exploiter budget {budget} steps · "
                   f"dose {dose}")
        body = [[r["archetype"] or "UNASSIGNED", r["membership"], r["run"],
                 _rate(r["endpoint_rate"]), _rate(r["pooled_rate"]), str(r["games"]),
                 _pct(r["gap"]), f"[{_pct(r['gap_lo'])}, {_pct(r['gap_hi'])}]"]
                for r in blk["rows"]]
        out += _table(["archetype", "teams", "run", "endpoint", "pooled", "n", "gap pp",
                       "95% CI pp"], body)
        for u in blk.get("unassigned", []):
            out.append(f"    ⚠ {u['run']}: archetype UNASSIGNED ({u['why']}) — it cannot be "
                       "paired across rounds.")
        out.append("")

    if not doc["deltas"]:
        out.append("ROUND-OVER-ROUND Δ: one round only — a gap needs two rounds to fall.")
    for d in doc["deltas"]:
        out.append(f"ROUND-OVER-ROUND Δ (round {d['later_round']} − round {d['earlier_round']}), "
                   "paired on ARCHETYPE")
        body = [[p["archetype"], _pct(p["gap_earlier"]), _pct(p["gap_later"]), _pct(p["delta"]),
                 f"[{_pct(p['lo'])}, {_pct(p['hi'])}]"] for p in d["per_archetype"]]
        out += _table(["archetype", f"gap r{d['earlier_round']}", f"gap r{d['later_round']}",
                       "Δ pp", "95% CI pp (Newcombe)"], body)
        if d["unpaired"]:
            out.append(f"    ⚠ UNPAIRED, excluded from the mean: {', '.join(d['unpaired'])}")
        if d["mean_delta"] is None:
            out.append(f"    MEAN Δ: — ({d['verdict']})")
        else:
            out.append(f"    MEAN Δ over {d['n_pairs']} archetype(s): {_pct(d['mean_delta'])} pp  "
                       f"[{_pct(d['lo'])}, {_pct(d['hi'])}]  "
                       f"(two-level bootstrap, {d['n_pairs']} pairing units)")
        out.append(f"    VERDICT: {d['verdict']}")
        out.append("")

    played = doc.get("played") or {}
    if played:
        out.append("FRESH HEAD-TO-HEAD GAMES (played now, NOT the series — different population)")
        body = []
        for name, p in played.items():
            body.append([name, p["regime"].split(" (")[0],
                         f"{p['wins']}/{p['finished']}", _rate(p["win_rate"]),
                         f"[{_rate((p['ci'] or [None])[0])}, {_rate((p['ci'] or [None, None])[1])}]"
                         if p.get("ci") else "—",
                         str(p["timeouts"]),
                         "INCONCLUSIVE" if p["inconclusive"] else ""])
        out += _table(["run", "regime", "wins/finished", "rate", "95% CI", "timeouts", ""], body)
        out.append("    A timeout is never scored as a loss; >25% of attempted battles timing "
                   "out makes the cell INCONCLUSIVE, not a result.")
        out.append("")

    out.append("PROVENANCE — every fact below is READ from what the run wrote down")
    for r in doc["runs"]:
        tgt = r["target"]
        step = (f" @{tgt['resolved_num_timesteps']:,}"
                if isinstance(tgt.get("resolved_num_timesteps"), int) else "")
        note = "  ⚠ lineage DERIVED from original_command" if r["lineage_derived"] else ""
        budget = f"{r['budget']:,}" if isinstance(r.get("budget"), int) else "—"
        dose = f"{r['dose_rate']:.4g}" if r.get("dose_rate") is not None else "—"
        n_post = sum(1 for p in r["series"] if p["post_fork"])
        out.append(f"  {r['run']}  round {r['round']}  archetype "
                   f"{r['archetype'] or 'UNASSIGNED'} ({r['membership']})")
        out.append(f"      target {tgt['run']}{step} -> {tgt['resolved_file']}{note}")
        out.append(f"      budget {budget} steps · dose {dose} · {n_post}/{len(r['series'])} "
                   f"post-fork cycles · teams "
                   f"{', '.join(os.path.basename(t) for t in r['teams']) or '(none)'}")
        out.append(f"      target pilots {'its OWN pinned team(s)' if tgt['pins_own_teams'] else 'the shared POOL' if tgt['pins_own_teams'] is False else 'an UNKNOWN team source'}")
    return "\n".join(out)


def render_markdown(doc: Dict[str, Any]) -> str:
    out = ["# Best-response gap", "",
           f"`gap = win rate − {doc['null_rate']}`; the population loop is working iff the gap "
           "FALLS round over round.", "",
           f"* **stat**: `{doc['stat']}`", f"* **regime**: {doc['rate_regime']}", ""]
    if doc.get("unmatched"):
        out += ["> 🚨 **UNMATCHED** — printed under `--allow-unmatched`. Confounds:", ">"]
        out += [f"> * {m['message']}" for m in doc["mismatches"]]
        out.append("")
    for blk in doc["rounds"]:
        step = f" @{blk['target_step']:,}" if blk.get("target_step") is not None else ""
        out += [f"## Round {blk['round']} — target `{blk['target_run']}`{step}", "",
                (f"budget {blk['budget']:,} steps · dose "
                 f"{blk['dose_rate']:.4g}" if blk.get("budget") and blk.get("dose_rate")
                 else "_budget/dose not recorded_"), "",
                "| archetype | teams | run | endpoint | pooled | n | gap pp | 95% CI pp |",
                "|---|---|---|---:|---:|---:|---:|---|"]
        for r in blk["rows"]:
            out.append(f"| {r['archetype'] or 'UNASSIGNED'} | {r['membership']} | `{r['run']}` | "
                       f"{_rate(r['endpoint_rate'])} | {_rate(r['pooled_rate'])} | {r['games']} | "
                       f"**{_pct(r['gap']).strip()}** | [{_pct(r['gap_lo']).strip()}, "
                       f"{_pct(r['gap_hi']).strip()}] |")
        out.append("")
    for c in doc.get("caveats", []):
        out += [f"> ⚠ **CAVEAT:** {c}", ""]
    for d in doc["deltas"]:
        out += [f"## Δ round {d['later_round']} − round {d['earlier_round']} (paired on archetype)",
                "",
                f"| archetype | gap r{d['earlier_round']} | gap r{d['later_round']} | Δ pp | "
                "95% CI pp |", "|---|---:|---:|---:|---|"]
        for p in d["per_archetype"]:
            out.append(f"| {p['archetype']} | {_pct(p['gap_earlier']).strip()} | "
                       f"{_pct(p['gap_later']).strip()} | **{_pct(p['delta']).strip()}** | "
                       f"[{_pct(p['lo']).strip()}, {_pct(p['hi']).strip()}] |")
        mean = ("—" if d["mean_delta"] is None else
                f"**{_pct(d['mean_delta']).strip()} pp** [{_pct(d['lo']).strip()}, "
                f"{_pct(d['hi']).strip()}]")
        out += ["", f"**MEAN Δ over {d['n_pairs']} archetype(s):** {mean}", "",
                f"**VERDICT: {d['verdict']}**", ""]
    return "\n".join(out)


# --------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    log = (lambda *a: None) if args.quiet else (lambda *a: print(*a, file=sys.stderr))

    try:
        teamsets = engine.load_teamsets(args.teamsets)
        runs = [engine.read_exploiter(ref, teamsets) for ref in args.runs]
        override = parse_rounds(args.rounds)
        rounds = engine.assign_rounds(runs, override)
        mismatches = engine.check_matched(
            runs, budget_tol=args.budget_tol, dose_tol=args.dose_tol,
            allow_unmatched=args.allow_unmatched)
    except BestResponseGapError as exc:
        print(f"\n[best_response_gap] REFUSAL ({type(exc).cause}) — {exc}\n", file=sys.stderr)
        return 2

    if args.check:
        for r in runs:
            log(f"  {r.run}: round {rounds.get(engine.target_key(r))} · target {r.target_run} "
                f"· budget {r.budget} · dose {r.dose_rate} · archetype {r.archetype}")
        log(f"  {len(mismatches)} mismatch(es); {len(runs)} run(s) read.")
        return 1 if mismatches else 0

    played: Dict[str, Any] = {}
    if args.play:
        for r in runs:
            log(f"  [play] {r.run} vs {r.target_run}: {args.play} games "
                f"({'greedy' if args.greedy else 'stochastic'}, impl={args.impl}) …")
            try:
                played[r.run] = engine.play_head_to_head(
                    r, games=args.play, seed=args.play_seed, impl=args.impl,
                    greedy=args.greedy, concurrency=args.concurrency)
            except BestResponseGapError as exc:
                print(f"\n[best_response_gap] REFUSAL ({type(exc).cause}) — {exc}\n",
                      file=sys.stderr)
                return 2

    doc = engine.build_report(runs, stat=args.stat, rounds=rounds, mismatches=mismatches,
                              draws=args.draws, bootstrap_seed=args.bootstrap_seed,
                              played=played or None)
    doc["_meta"] = {
        "tool": "main.best_response_gap",
        "argv": list(argv if argv is not None else sys.argv[1:]),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cwd": os.getcwd(),
        "teamsets": str(args.teamsets or engine.DEFAULT_TEAMSETS),
        "play": {"games": args.play, "greedy": args.greedy, "seed": args.play_seed,
                 "impl": args.impl, "concurrency": args.concurrency} if args.play else None,
    }

    if not args.no_json and args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(doc, fh, indent=1)
        log(f"  wrote {args.json_out}")
    if args.md_out:
        with open(args.md_out, "w") as fh:
            fh.write(render_markdown(doc))
        log(f"  wrote {args.md_out}")
    if not args.quiet:
        print()
        print(render_text(doc))
    if any(p.get("inconclusive") for p in played.values()):
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
