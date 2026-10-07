"""X5 A/B LOOK 3: the read, at the registered n = 8 seeds per arm (design_x5_belief_tokens.md §7.4, §7.9). For each
of the two registered strength reads it reads the family's look-1, look-2 AND look-3 requests through
``main.h2h.cross``'s declared family read, builds the FULL 8 × 8 matrix (look 1's 9 cells + look 2's 16 + look 3's
39), and decides at look 3 (boundary 1.874 on df 14; NOT DETECTED below it, labelled INFERIOR iff the upper one-sided
95 % bound < −δ; rule 8) → ``result.json`` + ``result.md`` beside this file. The SENSITIVITY line (fixed_mass 13M ×
blob 15M, all 64 cells under ``x5ab_look3_wall13``) gets the same statistic, REPORTED and never a verdict (Decision
record 2026-10-06, 11d27574).

    PYTHONPATH=<706fa536 checkout>/src python read_look3.py [--root <ledger root>] [--progress]

THE PIN: the estimator and the ledger reader are imported from the checkout on PYTHONPATH, and this script REFUSES
(exit 5) unless that checkout's ``git rev-parse HEAD`` is 706fa536 (Decision record 2026-10-06, bef16d61: HEAD code
is not used for any X5 read). ``main.h2h.cross`` there is byte-identical to the family commit ``bcb0296c``.

``--progress`` prints per-request cell counts only (an interim read is PROGRESS, never a verdict). A missing / short /
aborted cell, a cell recorded under two looks, an earlier look's request not holding exactly its registered block, a
regime other than looks 1-2's, or a broken precondition makes that read INCONCLUSIVE (never interpreted).

DRY RUN ONLY: ``--models``, ``--seeds``, ``--look-seeds``, ``--min-pairs``, ``--no-pin-check`` and ``--out-dir``
point the same code at a scratch ledger; a read off the registered seeds or below 1,000 pairs is INCONCLUSIVE by
construction (the statistic is still printed, to prove the path)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
from agents.training import eval_ledger as L  # noqa: E402
from main.h2h import cross as X  # noqa: E402
import plan_look3 as PLAN  # noqa: E402

HERE = Path(__file__).resolve().parent
LOOK = 3
#: looks 1 and 2's regime (their READ entries; `../x5ab_look2_2026-10-06/README.md`)
REGIME = "adfdbee824c9eb0a"
#: the owner's PRE-COMMITTED adoption rule (Decision record 2026-10-06, 079dee3e)
OWNER_RULE = ("X5 (fixed_mass) is ADOPTED unless look 3's MATCHED-STEPS read is INFERIOR (the upper one-sided 95 % "
              "bound of Δ < −δ = −3.5 pp); NON-INFERIOR adopts; NOT DETECTED without INFERIOR adopts")


def code_head() -> str:
    """``git rev-parse HEAD`` of the checkout ``main.h2h.cross`` was imported from."""
    src = Path(X.__file__).resolve().parents[2]
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=src.parent, capture_output=True, text=True,
                          check=True).stdout.strip()


def _fmt_stat(st: Any) -> str:
    if not st:
        return "INCONCLUSIVE (no statistic)"
    lo, hi = st["ci95_pp"]
    return (f"Δ̂ {st['delta_hat_pp']:+.2f} pp [{lo:+.2f}, {hi:+.2f}], t {st['t']:.3f} (= (Δ̂ + δ) / SE) on {st['df']} "
            f"df, SE {st['se_pp']:.2f}, upper one-sided 95 % {st['upper95_one_sided_pp']:+.2f}")


def _matrix(st: Any, rows: List[str], cols: List[str], n_prev: int) -> List[str]:
    out = ["| row \\ col | " + " | ".join(cols) + " | row mean |", "|---" * (len(cols) + 2) + "|"]
    for i, (r, line, rm) in enumerate(zip(rows, st["h_pp"], st["row_means_pp"])):
        cells = [f"{x:.2f}" if (i < n_prev and j < n_prev) else f"**{x:.2f}**" for j, x in enumerate(line)]
        out.append(f"| {r} | " + " | ".join(cells) + f" | {rm:.2f} |")
    out.append("| col mean | " + " | ".join(f"{x:.2f}" for x in st["col_means_pp"]) + " | |")
    return out


def family_regime(root: Path, fam: str) -> Optional[str]:
    """The family's one regime id (``eval_ledger.read`` REFUSES a read spanning two regimes)."""
    return L.read(X.FAMILY_READ_OFF, root=root, family=fam).regime_id


def read_one(root: Path, read: str, got: Dict[str, Dict[int, Path]], seeds: Sequence[int],
             look_seeds: Sequence[int], fails: List[str], min_pairs: int) -> Dict[str, Any]:
    R = PLAN.READS[read]
    reqs = list(R.earlier) + [R.request]
    per = {rq: PLAN.look_cells(root, R.family, rq) for rq in reqs}
    rows_p, cols_p = PLAN.rows_cols(got, read)
    rs = [PLAN.sha(rows_p[s]) for s in seeds if s in rows_p]
    cs = [PLAN.sha(cols_p[s]) for s in seeds if s in cols_p]
    reasons = list(fails)
    # the uniqueness guard, read side: each earlier look holds EXACTLY its registered block, look 3 none of them
    earlier_blocks, new = PLAN.blocks(seeds, look_seeds, len(R.earlier))
    key = {(r, c): (PLAN.sha(rows_p[r]), PLAN.sha(cols_p[c])) for r in seeds for c in seeds
           if r in rows_p and c in cols_p}
    held: set = set()
    for rq, blk in zip(R.earlier, earlier_blocks):
        have = {(c.player, c.opponent) for c in per[rq]}
        if have != {key[rc] for rc in blk if rc in key}:
            reasons.append(f"{rq} holds {len(have)} cell(s), not exactly its registered {len(blk)}")
        held |= have
    dup = [f"{c.player_id} vs {c.opponent_id}" for c in per[R.request] if (c.player, c.opponent) in held]
    if dup:
        reasons.append(f"{R.request} replays earlier-look cell(s): {dup}")
    try:
        regime = family_regime(root, R.family)
    except Exception as e:                    # a read spanning two regimes is refused by the ledger reader
        regime = None
        reasons.append(f"regime: {e}")
    if regime is not None and regime != REGIME:
        reasons.append(f"regime {regime} is not looks 1-2's {REGIME} (the A/B's protocol is FROZEN, Amendment 1)")
    m = X.matrix_from_cells([c for rq in reqs for c in per[rq]], rs, cs, min_pairs=min_pairs)
    st = None
    if m.h is not None:                       # the statistic, even when an outer reason makes the read INCONCLUSIVE
        try:
            st = X.cross_stat(m.h).as_dict()
        except X.CrossInputError as e:       # a matrix the statistic is not defined on is an invalid input
            reasons.append(f"cross statistic undefined: {e}")
    d = X.summarize(LOOK, m, reasons)
    return {"family": R.family, "registered": R.registered, "requests": reqs,
            "n_cells": {rq: len(per[rq]) for rq in reqs}, "regime_id": regime,
            "decision": d.as_dict(), "verdict": X.verdict_text(d), "stat_shown": st, "cells": list(m.cells),
            "rows": [PLAN.label(rows_p[s]) for s in seeds if s in rows_p],
            "cols": [PLAN.label(cols_p[s]) for s in seeds if s in cols_p],
            "row_sha256": rs, "col_sha256": cs}


def implication(steps: Optional[Dict[str, Any]]) -> str:
    if steps is None:
        return "no matched-steps read"
    d = steps["decision"]
    if d["outcome"] == X.INCONCLUSIVE:
        return "matched steps INCONCLUSIVE: the rule has no input until the look is repaired and re-read"
    if d["label"] == X.INFERIOR:
        return "matched steps INFERIOR: under the owner's pre-committed rule, X5 is NOT adopted"
    return (f"matched steps {steps['verdict']} (not INFERIOR): under the owner's pre-committed rule, X5 is "
            f"adopted — the owner's decision to take; purpose metric (1) is "
            f"{'DUE (strength NON-INFERIOR)' if d['outcome'] == X.NON_INFERIOR else 'NOT tested (it follows a NON-INFERIOR matched-steps verdict only), reported as not run'}")


def render_md(out: Dict[str, Any]) -> str:
    seeds = out["seeds"]
    n_prev = out["look_seeds"][-1] if out["look_seeds"] else 0
    b = X.LOOK_BOUNDARIES[LOOK]
    L_: List[str] = ["# X5 A/B look 3 — the read (generated by read_look3.py)", "",
                     f"Estimator and ledger reader imported from the checkout at `{out['code_head']}` "
                     f"(registered pin 706fa536).", ""]
    if out["dry_run"]:
        L_ += [f"**DRY RUN** (min pairs {out['min_pairs']}, seeds {seeds}): not a read of the experiment.", ""]
    L_ += [f"## Strength (registered, design_x5_belief_tokens.md §7.4 / §7.9; look {LOOK}, n = {len(seeds)}, "
           f"df {2 * (len(seeds) - 1)}, NON-INFERIOR iff t ≥ {b}; else NOT DETECTED, INFERIOR iff the upper one-sided "
           f"95 % bound < −3.5 pp; rule 8)", ""]
    for name, rows, title in (("matched_steps", "fm 15M", "matched steps (fixed_mass 15M × blob 15M)"),
                              ("matched_wall_time", "fm 12M",
                               "matched wall time, END-TO-END (fixed_mass 12M × blob 15M)"),
                              ("sensitivity_wall_13M", "fm 13M",
                               "SENSITIVITY line, steady-state wall (fixed_mass 13M × blob 15M) — never a verdict")):
        e = out.get(name)
        if e is None:
            continue
        d = e["decision"]
        head = (f"the outcome table would read **{e['verdict']}** (reported only)" if not e["registered"]
                else f"**{e['verdict']}**")
        L_ += [f"### {title}: {head}", "",
               f"- {_fmt_stat(d['stat'] or e['stat_shown'])}; boundary {d['boundary']}",
               "- cells: " + ", ".join(f"{n} from {rq}" for rq, n in e["n_cells"].items())
               + f"; regime {e['regime_id']}",
               f"- reasons: {d['reasons'] or 'none'}; near a boundary: {d['near_boundary'] or 'none'}", ""]
        st = d["stat"] or e["stat_shown"]
        if st:
            k = n_prev if e["registered"] else 0
            L_ += [f"h_ij = {rows} seed i's score (pp) vs blob 15M seed j (look-3 cells in bold):", ""]
            L_ += _matrix(st, [f"fm s{s}" for s in seeds], [f"blob s{s}" for s in seeds], k) + [""]
    L_ += ["## What the owner's pre-committed rule implies (079dee3e; no decision is taken here)", "",
           f"- rule: {OWNER_RULE}", f"- implies: {out['owner_rule_implies']}", "",
           "## Preconditions", "", f"- failures: {out['preconditions'] or 'none'}"]
    return "\n".join(L_) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--models", default=str(PLAN.ARCHIVE_MODELS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(PLAN.SEEDS))
    ap.add_argument("--look-seeds", type=int, nargs="*", default=list(PLAN.LOOK_SEEDS))
    ap.add_argument("--min-pairs", type=int, default=X.MIN_PAIRS)
    ap.add_argument("--no-pin-check", action="store_true", help="DRY RUN: skip the runs' TRAINING-pin check")
    ap.add_argument("--out-dir", default=str(HERE))
    ap.add_argument("--reads", nargs="+", default=["steps", "wall", "wall13"], choices=sorted(PLAN.READS))
    a = ap.parse_args()
    head = code_head()
    print(f"[read] estimator checkout HEAD {head}", file=sys.stderr)
    if head != PLAN.PIN:
        print(f"REFUSED: the estimator is imported from {head}, not the registered pin {PLAN.PIN}", file=sys.stderr)
        return 5
    root = Path(a.root) if a.root else L.archive_ledger_root()
    dry = (a.min_pairs != X.MIN_PAIRS or list(a.seeds) != list(PLAN.SEEDS) or a.no_pin_check
           or list(a.look_seeds) != list(PLAN.LOOK_SEEDS) or Path(a.models) != PLAN.ARCHIVE_MODELS)
    if dry and Path(a.out_dir).resolve() == HERE:
        print("REFUSED: a dry-run read never writes result.* beside this file (pass --out-dir)", file=sys.stderr)
        return 3
    if a.progress:
        for read in a.reads:
            R = PLAN.READS[read]
            for rq in list(R.earlier) + [R.request]:
                cs = PLAN.look_cells(root, R.family, rq)
                done = sum(1 for c in cs if c.n_pairs >= a.min_pairs)
                print(f"{read} {rq}: {len(cs)} cells, {done} at >= {a.min_pairs} pairs, "
                      f"{sum(c.n_pairs for c in cs)} pairs, aborted {sum(c.aborted for c in cs)}")
        return 0
    got, fails = PLAN.resolve(Path(a.models), a.seeds, not a.no_pin_check)
    out: Dict[str, Any] = {"look": LOOK, "dry_run": dry, "min_pairs": a.min_pairs, "seeds": a.seeds,
                           "look_seeds": a.look_seeds, "preconditions": fails, "code_head": head,
                           "run_names": {f"{arm}_s{s}": PLAN.run_dir(Path(a.models), arm, s).name
                                         for arm in ("fm", "blob") for s in a.seeds},
                           "cited": ["blob-path identity at 708dcb0a vs e5e660dd (ledger 2026-10-06 BANK)",
                                     "look-3 seeds' learner goldens identical across e5e660dd / 708dcb0a / 706fa536 "
                                     "(the training agent, 2026-10-07)",
                                     "restarts / crashes / init_num_threads: the runs' ledger BANK entries"]}
    for read, name in (("steps", "matched_steps"), ("wall", "matched_wall_time"), ("wall13", "sensitivity_wall_13M")):
        if read in a.reads:
            out[name] = read_one(root, read, got, a.seeds, a.look_seeds, fails, a.min_pairs)
    out["owner_rule"] = OWNER_RULE
    out["owner_rule_implies"] = implication(out.get("matched_steps"))
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    (od / "result.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    (od / "result.md").write_text(render_md(out))
    print(json.dumps({k: out[k]["verdict"] for k in ("matched_steps", "matched_wall_time", "sensitivity_wall_13M")
                      if k in out}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
