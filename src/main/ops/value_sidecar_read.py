"""THE TRAINING-SIDE CALIBRATION READ — `python -m main.ops.value_sidecar_read <run>`.

The counterpart to `main.ops.critic_read`, and the distinction between them is the whole point:

* **`critic_read` reads EVAL battles** — a greedy trainee against a fixed roster, sampled under a
  quota that prefers losses. It answers *"is this critic calibrated on the eval distribution?"*
* **this reads the TRAINING buffer** — the stochastic policy against the live self-play curriculum,
  scored against `win_target`, the label the BCE actually minimises. It answers *"is this critic
  calibrated on the distribution it is being fit to?"*

A critic can pass one and fail the other, and until `gen3_value_sidecar_v1` only the first was
measurable. Neither supersedes the other; a disagreement between them is a finding about
generalisation, not a defect in either instrument.

**WHAT IT REPORTS.** Mean V against mean target, the Murphy decomposition of the Brier score
(reliability / resolution / uncertainty) and skill, sliced four ways — by TURN bucket, by OPPONENT
CLASS, by episode OUTCOME and by training STEP — each with a battle-clustered bootstrap CI.

🚨 **`critic_resolution` IS THE METER, `critic_reliability` IS NOT** — the same rule the training
leaf states for the live TB family. A base-rate forecaster scores a perfect 0 reliability and a
useless 0 resolution, so a change that improves reliability while resolution stays flat has moved
the number that was never the disease. The report prints resolution first and says so.

🚨 **THE BOOTSTRAP CLUSTERS BY EPISODE, NEVER BY ROW.** Decisions inside one battle share a board,
a team matchup and a dice stream; the sidecar samples uniformly over buffer cells, so a long
episode contributes proportionally more rows and a row-level interval would understate its width by
exactly that correlation. `agents.training.stats.cluster_bootstrap_ci` is imported, never
re-implemented.

🚨 **SLICING BY OUTCOME IS NOT A CALIBRATION CHECK AND THE REPORT SAYS SO.** Conditioning on the
outcome and then asking whether V predicted it is circular — within the won slice the target is 1.0
by construction, so "mean V" there is a statement about how confident the critic was on states that
happened to win, which is a resolution component, not an error. It is reported because the
asymmetry between the two slices is informative (an optimistic critic reads high in BOTH), and
labelled so nobody quotes it as a bias.

**REFUSALS OVER SILENCE.** A run with no sidecar, a sidecar with no header (so the critic mode —
and therefore whether `v` is a probability at all — is unknown), a `shaped` sidecar read as if it
were calibration, a slice below the cell floor: each refuses or is marked, never averaged into a
confident number.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone

from agents.model.critic_mode import is_winprob
from agents.training.stats import MIN_CELL_N, cluster_bootstrap_ci
from agents.training.value_sidecar import read_sidecar, sidecar_path
from agents.training.wrappers import MaskableAgentWrapper as _W
from main.ops.run_ref import refuse, resolve_run_dir

TOOL = "value_sidecar_read"
TOOL_VERSION = 1

#: The opponent-class codes the env tags rows with, by name. Imported from the wrapper that
#: DEFINES them rather than re-listed, so a new class cannot leave this reader silently mislabelling.
OPP_CLASS_NAMES = {
    _W.OPP_CLASS_BOT: "bot",
    _W.OPP_CLASS_POOL: "pool",
    _W.OPP_CLASS_STABLE: "stable",
    _W.OPP_CLASS_EXPLOITER: "exploiter",
}

#: Turn buckets. Coarse and fixed rather than quantile-derived: a quantile bucketing moves with the
#: run's own episode-length distribution, so two runs' "early game" would not be the same states —
#: and episode length is itself a primary endpoint on a win-prob arm.
TURN_BUCKETS = ((0, 5), (5, 10), (10, 20), (20, 40), (40, 80), (80, 250))

HEADLINE = (
    "critic_resolution is the meter (HIGHER is better); reliability alone is not — "
    "a base-rate forecaster scores a perfect 0 reliability and a useless 0 resolution."
)


# ── the statistics ────────────────────────────────────────────────────────────────────────────
def murphy(preds, targets):
    """The Brier score and its Murphy decomposition: BS = reliability - resolution + uncertainty.

    Binned on the forecast into 10 equal-width bins, which is the same binning
    `scaffolding.reliability_table` uses for the live `win_prob/*` family — so a sidecar number and
    a TB number are the same statistic and not two neighbouring ones.

    Returns a dict, or ``None`` when there is nothing to decompose. **`None`, never zeros**: a
    zero reliability and an unmeasurable one look identical in a table and mean opposite things.
    """
    n = len(preds)
    if n < MIN_CELL_N:
        return None
    base = sum(targets) / n
    brier = sum((p - y) ** 2 for p, y in zip(preds, targets)) / n
    bins = defaultdict(list)
    for p, y in zip(preds, targets):
        bins[min(9, int(p * 10))].append((p, y))
    reliability = resolution = 0.0
    for cell in bins.values():
        k = len(cell)
        p_bar = sum(p for p, _ in cell) / k
        y_bar = sum(y for _, y in cell) / k
        reliability += k * (p_bar - y_bar) ** 2
        resolution += k * (y_bar - base) ** 2
    reliability /= n
    resolution /= n
    uncertainty = base * (1.0 - base)
    return {
        "n": n,
        "base_rate": base,
        "mean_v": sum(preds) / n,
        "mean_target": base,
        "brier": brier,
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        # Murphy's identity. It must close to ~0; a residual that does not is an arithmetic bug in
        # this function, not a property of the critic, and it is printed rather than asserted away.
        "decomp_residual": brier - (reliability - resolution + uncertainty),
        # Brier skill against the base rate: 1 = perfect, 0 = no better than always forecasting the
        # base rate, NEGATIVE = worse than that. `None` when there is no uncertainty to beat (a
        # slice that was won or lost every time), because the ratio is undefined, not infinite.
        "skill": (1.0 - brier / uncertainty) if uncertainty > 0 else None,
    }


def _cell(rows, *, draws, seed):
    """One slice's statistics plus a battle-clustered CI on the mean signed error (V - target)."""
    usable = [r for r in rows if r.get("target_known")]
    if len(usable) < MIN_CELL_N:
        return {"n": len(usable), "under_floor": True, "floor": MIN_CELL_N}
    preds = [float(r["v"]) for r in usable]
    targets = [float(r["target"]) for r in usable]
    out = murphy(preds, targets) or {"n": len(usable), "under_floor": True}
    errs = [p - y for p, y in zip(preds, targets)]
    clusters = [str(r["episode"]) for r in usable]
    lo, hi = cluster_bootstrap_ci(errs, clusters, draws=draws, seed=seed)
    out["mean_error"] = sum(errs) / len(errs)
    out["mean_error_ci"] = [lo, hi]
    # The sign convention, stated once so the report never has to hedge: POSITIVE = the critic is
    # OPTIMISTIC on this slice (it forecast more win probability than the outcomes delivered).
    out["direction"] = (
        "UNRESOLVED (the CI straddles 0)" if lo is None or hi is None or (lo <= 0.0 <= hi)
        else ("OPTIMISTIC" if out["mean_error"] > 0 else "PESSIMISTIC"))
    return out


def _slices(rows, *, draws, seed):
    """The four registered slicings. A slice with no rows is ABSENT, never an empty cell."""
    by_turn, by_opp, by_outcome, by_step = {}, {}, {}, {}

    for lo, hi in TURN_BUCKETS:
        sel = [r for r in rows if lo <= float(r.get("turn", -1)) < hi]
        if sel:
            by_turn[f"{lo}-{hi}"] = _cell(sel, draws=draws, seed=seed)

    for code, name in OPP_CLASS_NAMES.items():
        sel = [r for r in rows if r.get("opp_class") == code]
        if sel:
            by_opp[name] = _cell(sel, draws=draws, seed=seed)

    for label, want in (("won", 1.0), ("lost", 0.0)):
        sel = [r for r in rows if r.get("target_known") and float(r["target"]) == want]
        if sel:
            by_outcome[label] = _cell(sel, draws=draws, seed=seed)

    # Training step, bucketed per 1M — the same bucketing every other ops instrument uses, so a
    # sidecar trend lines up with a TB trend row for row.
    buckets = defaultdict(list)
    for r in rows:
        buckets[int(r.get("step", 0)) // 1_000_000].append(r)
    for m, sel in sorted(buckets.items()):
        by_step[f"{m}M-{m + 1}M"] = _cell(sel, draws=draws, seed=seed)

    return {"by_turn": by_turn, "by_opponent_class": by_opp,
            "by_outcome": by_outcome, "by_step": by_step}


# ── the report ────────────────────────────────────────────────────────────────────────────────
def _f(x, nd=4):
    return "     —" if x is None else f"{x:.{nd}f}"


def _ci(pair):
    if not pair or pair[0] is None or pair[1] is None:
        return "[      —,       —]"
    return f"[{pair[0]:+.4f}, {pair[1]:+.4f}]"


def _table(title, cells, note=""):
    if not cells:
        return [f"### {title}", "", "_(no rows in any cell)_", ""]
    out = [f"### {title}", ""]
    if note:
        out += [note, ""]
    out += ["| slice | n | mean V | mean target | resolution | reliability | brier | skill "
            "| mean error | 95% CI (episode-clustered) | reading |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for name, c in cells.items():
        if c.get("under_floor"):
            out.append(f"| `{name}` | {c['n']} | — | — | — | — | — | — | — | — | "
                       f"UNDER THE CELL FLOOR (n < {c.get('floor', MIN_CELL_N)}) — not reported |")
            continue
        out.append(
            f"| `{name}` | {c['n']} | {_f(c['mean_v'])} | {_f(c['mean_target'])} | "
            f"**{_f(c['resolution'])}** | {_f(c['reliability'])} | {_f(c['brier'])} | "
            f"{_f(c['skill'])} | {c['mean_error']:+.4f} | {_ci(c['mean_error_ci'])} | "
            f"{c['direction']} |")
    return out + [""]


def render_md(doc) -> str:
    h, o = doc["header"], doc["overall"]
    L = [f"# Training-side value sidecar — `{doc['run']}`", "",
         f"_{doc['tool']} v{doc['tool_version']}, generated {doc['generated_at']}_", "",
         f"`{doc['invocation']}`", "",
         "## What this is", "",
         "The critic scored against **its own training target** — the `win_target` label the BCE "
         "minimises, on the states the rollout buffer actually held. This is NOT "
         "`main.ops.critic_read`, which reads EVAL battles (a greedy trainee, a fixed roster, a "
         "loss-preferring quota). A critic can pass one and fail the other; a disagreement is a "
         "finding about generalisation, not a defect in either.", "",
         f"🚨 {HEADLINE}", "",
         "## The sample", "",
         f"| rows | {doc['n_rows']:,} |", "|---|---|",
         f"| labelled (`target_known`) | {doc['n_labelled']:,} |",
         f"| episodes | {doc['n_episodes']:,} |",
         f"| critic mode | `{h.get('critic_mode')}` "
         f"({'V is a PROBABILITY' if h.get('v_is_probability') else 'V is a SHAPED RETURN'}) |",
         f"| sampling fraction | {h.get('fraction')} (seed {h.get('seed')}) |",
         f"| step range | {doc['step_lo']:,} … {doc['step_hi']:,} |",
         f"| completed episodes | {doc['n_complete']:,} |",
         f"| timeouts (clock reached the cap) | {doc['n_timeout']:,} |", ""]

    if doc.get("notes"):
        L += ["## Notes", ""] + [f"- {n}" for n in doc["notes"]] + [""]

    L += ["## Overall", ""]
    L += _table("pooled", {"all": o})
    s = doc["slices"]
    L += _table("By TURN bucket", s["by_turn"],
                "Game phase. ⚠️ A battle turn is not a decision index — one turn can carry several "
                "decisions, so consecutive rows can share a turn.")
    L += _table("By OPPONENT CLASS", s["by_opponent_class"],
                "⚠️ The env tags the CLASS (bot / pool / stable / exploiter), never the archetype "
                "name or the snapshot step — those are not reachable per-step today.")
    L += _table("By OUTCOME", s["by_outcome"],
                "🚨 **NOT a calibration check.** Conditioning on the outcome makes the target "
                "constant within each slice by construction, so `mean error` here is a resolution "
                "component, not a bias. Read the ASYMMETRY between the two rows — an optimistic "
                "critic reads high in both.")
    L += _table("By TRAINING STEP (1M buckets)", s["by_step"],
                "The trend. Compare against the run's own `win_prob/critic_resolution` series — "
                "same statistic, different distribution (training vs eval).")
    return "\n".join(L) + "\n"


# ── the CLI ───────────────────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m main.ops.value_sidecar_read",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Calibration of the critic against its OWN TRAINING TARGET, from a run's "
                    "value sidecar (gen3_value_sidecar_v1).",
        epilog=(
            "Refusals over silence: a run with no sidecar, a sidecar with no header (the critic "
            "mode — and so whether V is a probability at all — would be unknown), a sidecar in "
            "which nothing is labelled, and a slice under the cell floor are each refused or "
            "marked, never averaged into a confident number.\n\n"
            "This is NOT main.ops.critic_read. That one reads EVAL battles; this reads the "
            "TRAINING buffer. Neither supersedes the other."))
    p.add_argument("run", help="a run NAME under models/ or a run DIRECTORY")
    p.add_argument("--out", default=None,
                   help="directory for value_sidecar_read.{json,md} (default: print only)")
    p.add_argument("--bootstrap-draws", type=int, default=2000,
                   help="episode-clustered bootstrap resamples (default 2000)")
    p.add_argument("--seed", type=int, default=7, help="bootstrap seed (default 7)")
    p.add_argument("--allow-shaped", action="store_true",
                   help="read a --critic shaped sidecar anyway. REFUSED by default: under shaped "
                        "V is a PopArt-normalised shaped return whose scale moves over the run, "
                        "so a Brier decomposition of it is a category error, not a loose reading.")
    p.add_argument("--quiet", action="store_true", help="suppress progress lines")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    say = (lambda _m: None) if args.quiet else (lambda m: print(f"[{TOOL}] {m}", flush=True))

    run_dir = resolve_run_dir(args.run)
    try:
        header, rows = read_sidecar(str(run_dir))
    except FileNotFoundError:
        refuse(f"REFUSING: no value sidecar at {sidecar_path(str(run_dir))}",
               "  This run was trained without one (--value-sidecar off, or before",
               "  gen3_value_sidecar_v1 landed). It cannot be reconstructed after the fact — the",
               "  rollout buffer it read is long gone. Nothing is read and nothing is concluded.")
    except ValueError as e:
        refuse(f"REFUSING: {e}")

    mode = header.get("critic_mode")
    if not is_winprob(mode) and not args.allow_shaped:
        refuse(f"REFUSING: this sidecar was written under --critic {mode!r}, so `v` is a "
               "PopArt-normalised SHAPED RETURN,",
               "  not a probability. A Brier decomposition of it is a category error rather than "
               "a loose reading:",
               "  its scale moves over the run, so the same number means different things at 2M "
               "and 20M.",
               "  Pass --allow-shaped if you want the raw pairs anyway. Nothing is concluded.")

    labelled = [r for r in rows if r.get("target_known")]
    if not labelled:
        refuse(f"REFUSING: {len(rows):,} sidecar rows and NONE of them is labelled.",
               "  Every row's episode was still in progress at its rollout's end, or the sidecar",
               "  ran before WinProbLabelCallback's back-fill (a callback-ORDER defect — the",
               "  sidecar must be registered after it). Nothing is read and nothing is concluded.")
    say(f"{len(rows):,} rows ({len(labelled):,} labelled) from {sidecar_path(str(run_dir))}")

    notes = []
    if any(r.get("opp_class") is None for r in rows):
        notes.append("Some rows carry NO opponent class — the buffer had no `opp_class` obs key. "
                     "Since `gen3_value_sidecar_v1` every win-prob run emits it, so this means a "
                     "PRE-WIDENING checkpoint was resumed under its own saved observation space. "
                     "Those rows are ABSENT from the by-class table, never pooled into `bot` "
                     "(0 is a real class, so a default would invent a curriculum).")
    notes.append("The opponent is identified by CLASS only. The bot's archetype name and the pool "
                 "snapshot's step are chosen per episode and never reach the observation; an "
                 "opponent LADDER RATING does not exist at training time at all (a snapshot's Elo "
                 "is derived post-hoc by `main.elo` from the finished run's ladder). Look a rating "
                 "up by snapshot afterwards — it was never a per-step fact.")
    incomplete = sum(1 for r in rows if not r.get("ep_complete"))
    if incomplete:
        notes.append(f"{incomplete:,} row{'' if incomplete == 1 else 's'} "
                     f"belong{'s' if incomplete == 1 else ''} to an episode that straddles a "
                     "rollout boundary; its `ep_len` is the length WITHIN the buffer, not the "
                     "episode's. Every length statistic filters on `ep_complete`.")

    steps = [int(r.get("step", 0)) for r in rows]
    doc = {
        "tool": TOOL, "tool_version": TOOL_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "invocation": "python -m main.ops.value_sidecar_read " + " ".join(sys.argv[1:]),
        "run": run_dir.name, "run_dir": str(run_dir), "header": header,
        "n_rows": len(rows), "n_labelled": len(labelled),
        "n_episodes": len({r.get("episode") for r in labelled}),
        "n_complete": sum(1 for r in rows if r.get("ep_complete")),
        "n_timeout": sum(1 for r in rows if r.get("timeout")),
        "step_lo": min(steps) if steps else 0, "step_hi": max(steps) if steps else 0,
        "headline": HEADLINE, "notes": notes,
        "overall": _cell(rows, draws=args.bootstrap_draws, seed=args.seed),
        "slices": _slices(rows, draws=args.bootstrap_draws, seed=args.seed),
    }

    md = render_md(doc)
    print(md)
    if args.out:
        import os
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "value_sidecar_read.json"), "w") as f:
            json.dump(doc, f, indent=1, default=float)
        with open(os.path.join(args.out, "value_sidecar_read.md"), "w") as f:
            f.write(md)
        say(f"wrote {args.out}/value_sidecar_read.{{json,md}}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
