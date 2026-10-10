"""Static-token screen LOOK 3 (the FINAL look): the registered read (design_static_tokens.md §8.1 + §8.2), at P_st.

Reads look 1's request ``st_look1_steps`` (9 cells) and look 2's ``st_look2_steps`` (16 cells), both REUSED, and look
3's ``st_look3_steps`` (39 new cells) of family ``st_screen_strength_steps`` from the eval COUNT ledger through
``main.h2h.cross``'s declared family read, builds the FULL 8 × 8 matrix (rows = static seeds 1001-1008, columns =
legacy seeds 1001-1008, keyed by each final's sha256), and writes ``result.json`` + ``result.md`` beside this file.

    cd <6c6d2e09 checkout>/src && python -c "import runpy,sys; sys.argv=['read_look3.py']; \
        runpy.run_path('<this file>', run_name='__main__')"          # the pin's src/ is sys.path[0]

THE PIN: the estimator and the ledger reader are imported from the 6c6d2e09 checkout; this script REFUSES (exit 5)
unless that checkout's ``git rev-parse HEAD`` is 6c6d2e09 (§8.2: every cross and read at P_st, never at HEAD).

THE STATISTIC (§8.1, X5 §7.4's, at n = 8): ``main.h2h.cross.cross_stat`` — Δ̂ = mean(h) − 50, V̂ = (s²_R + s²_C)/8,
df = 2(n − 1) = 14; t_NI = (Δ̂ + 3.5)/√V̂ (cross_stat's ``t``) and t_SUP = Δ̂/√V̂.

THE OUTCOME TABLE AT LOOK 3, THE FINAL LOOK (§8.1's table with §8.2's look-3 boundary b = 1.874 in place of 2.132;
rule 8 on every comparison; INCONCLUSIVE on any invalid input) — looks 1 and 2's declared table
(``read_look1.outcome``), unchanged but for b, DECLARED HERE BEFORE THE READ:
  BETTER        t_SUP ≥ b                                                         ADOPT static (subject to the speed rule)
  EQUIVALENT    the look's interval Δ̂ ± b·√V̂ lies inside (−δ, +δ)                ADOPT
  NON-INFERIOR  t_NI ≥ b, neither of the above                                     ADOPT (reported as non-inferior)
  INFERIOR      Δ̂ + b·√V̂ < −δ                                                     legacy stays; split per §7
  NOT DETECTED  none of the above (the interval straddles −δ)                     legacy stays; back to the owner
§8.2: "A look that crosses no boundary CONTINUES to the next look rather than reading NOT DETECTED, except at look 3
(8 seeds), where the table applies as written" — so look 3 has NO CONTINUE, and X5's futility rule (looks 1-2 only,
``cross.decide``) has no row here: a Δ̂ ≤ −δ that is not INFERIOR reads NOT DETECTED.
The OWNER RULING of 2026-10-09 (Decision record): NON-INFERIOR adopts even if t_SUP shows static detectably worse
inside δ; "worse but inside δ" is NOT a row. The table above is that rule.

REPORTED beside the outcome, never the outcome: the fixed-sample 90 % interval Δ̂ ± t_{0.95,14}·√V̂ (t = 1.761), the
95 % two-sided interval, X5's ``cross.decide`` verdict (whose INFERIOR label uses t_{0.95,14}), and — DESCRIPTIVE
ONLY, NOT THE DECISION — the owner's FUTURE closing-test rule (registration pending): pass iff the 90 % two-sided
interval for Δ includes 0 or lies above it (its upper end ≥ 0) AND its lower end > −2.0 pp, read here on the
fixed-sample 90 % interval with rule 8."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent / "static_screen_look1_2026-10-08"))
from agents.training import eval_ledger as L  # noqa: E402
from main.h2h import cross as X  # noqa: E402
import plan_look3 as PLAN  # noqa: E402
import read_look1 as R1  # noqa: E402  (look 1's declared outcome table + rule-8 comparison)

LOOK = 3
DELTA = R1.DELTA
CLOSING_FLOOR_PP = -2.0


def code_head() -> str:
    src = Path(X.__file__).resolve().parents[2]
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=src.parent, capture_output=True, text=True,
                          check=True).stdout.strip()


def final_outcome(dhat: float, se: float, b: float):
    """``read_look1.outcome`` at b, with §8.2's final-look rule: no CONTINUE and no futility row ⇒ NOT DETECTED."""
    o, prim = R1.outcome(dhat, se, b)
    if o not in ("BETTER", "EQUIVALENT", "NON-INFERIOR", "INFERIOR"):
        o = "NOT DETECTED"
    return o, prim


def closing_rule(dhat: float, se: float, df: int) -> Dict[str, Any]:
    """DESCRIPTIVE ONLY: the owner's future closing-test rule on the fixed-sample 90 % two-sided interval."""
    q = X.t_ppf(0.95, df)
    lo, hi = dhat - q * se, dhat + q * se
    near: List[str] = []
    upper_ok = not R1._cmp(hi, 0.0, False, "90 % upper end vs 0", near)        # hi ≥ 0 ⇔ not (hi < 0)
    lower_ok = R1._cmp(lo, CLOSING_FLOOR_PP, True, "90 % lower end vs -2.0", near)
    if any("upper end" in n for n in near):
        upper_ok = False                                                          # rule 8: touching is not passing
    return {"label": "DESCRIPTIVE ONLY — NOT THE DECISION (the owner's future closing-test rule, unregistered)",
            "t_0.95": q, "interval90": [lo, hi], "includes_zero_or_above": upper_ok,
            "lower_above_minus_2": lower_ok, "would_pass": upper_ok and lower_ok, "near_boundary": near}


def _matrix_md(st: Dict[str, Any], rows: List[str], cols: List[str]) -> List[str]:
    n1, n2 = len(PLAN.LOOK1_SEEDS), len(PLAN.LOOK2_SEEDS)
    out = ["| static \\ legacy | " + " | ".join(cols) + " | row mean R_i |", "|---" * (len(cols) + 2) + "|"]
    for i, (r, line, rm) in enumerate(zip(rows, st["h_pp"], st["row_means_pp"])):
        cells = []
        for j, x in enumerate(line):
            if i < n1 and j < n1:
                cells.append(f"{x:.2f}")
            elif i < n2 and j < n2:
                cells.append(f"_{x:.2f}_")
            else:
                cells.append(f"**{x:.2f}**")
        out.append(f"| {r} | " + " | ".join(cells) + f" | {rm:.2f} |")
    out.append("| column mean C_j | " + " | ".join(f"{x:.2f}" for x in st["col_means_pp"]) + " | |")
    return out


def render_md(out: Dict[str, Any]) -> str:
    b = X.LOOK_BOUNDARIES[LOOK]
    L_ = ["# Static-token screen look 3 (FINAL) — the read (generated by read_look3.py)", "",
          f"Estimator and ledger reader imported from the checkout at `{out['code_head']}` (P_st 6c6d2e09). "
          f"Family `{PLAN.FAMILY}`, requests `{PLAN.REQ1}` ({out['n_cells'][PLAN.REQ1]} cells, reused) + "
          f"`{PLAN.REQ2}` ({out['n_cells'][PLAN.REQ2]} cells, reused) + `{PLAN.REQ3}` "
          f"({out['n_cells'][PLAN.REQ3]} cells, new), regime `{out['regime_id']}`.", ""]
    st = out["stat"]
    if out["inconclusive"]:
        L_ += [f"## INCONCLUSIVE: {out['inconclusive']}", ""]
    if st:
        p, f, c = out["primary"], out["fixed_sample"], out["closing_rule_descriptive"]
        L_ += [f"## Outcome at look 3, the FINAL look (b = {b}): **{out['outcome']}**", "",
               f"- Δ̂ = **{st['delta_hat_pp']:+.2f} pp**; V̂ = {st['v_hat']:.4f} (s²_R {st['s2_rows']:.4f}, "
               f"s²_C {st['s2_cols']:.4f}), √V̂ = {st['se_pp']:.3f} pp, df {st['df']}",
               f"- t_NI = (Δ̂ + 3.5)/√V̂ = **{p['t_NI']:.3f}**, t_SUP = Δ̂/√V̂ = **{p['t_SUP']:.3f}**, both vs **{b}**",
               f"- look-3 interval (Δ̂ ± {b}·√V̂, the boundary, §8.2 — the registered reading): "
               f"**[{p['interval'][0]:+.2f}, {p['interval'][1]:+.2f}] pp**",
               f"- 90 % interval, fixed-sample (Δ̂ ± {out['t_exact_95']:.3f}·√V̂; REPORTED only): "
               f"[{f['interval'][0]:+.2f}, {f['interval'][1]:+.2f}] pp",
               f"- 95 % two-sided (cross.py's reported interval): [{st['ci95_pp'][0]:+.2f}, {st['ci95_pp'][1]:+.2f}] pp",
               f"- the fixed-sample reading's outcome would be: {out['fixed_sample_outcome']} "
               f"({'same' if out['readings_agree'] else 'DIFFERENT — reported, not the outcome'})",
               f"- X5's estimator (`main.h2h.cross.decide`, look 3): {out['x5_verdict']}",
               f"- rule 8 near-boundary: {p['near_boundary'] + f['near_boundary'] or 'none'}",
               f"- **DESCRIPTIVE ONLY, NOT THE DECISION** — the owner's future closing-test rule (90 % interval "
               f"[{c['interval90'][0]:+.2f}, {c['interval90'][1]:+.2f}] includes 0 or lies above it: "
               f"{c['includes_zero_or_above']}; lower end > −2.0: {c['lower_above_minus_2']}): would "
               f"{'PASS' if c['would_pass'] else 'FAIL'}", "",
               "h_ij = static seed i's score (pp, a draw = ½) against legacy seed j, 1,000 mirrored pairs per cell "
               "(look-1 cells plain, look-2 cells italic, look-3 cells bold):", ""]
        L_ += _matrix_md(st, [f"static s{s}" for s in PLAN.SEEDS], [f"legacy s{s}" for s in PLAN.SEEDS]) + [""]
        L_ += ["| cell | pairs | W | L | D | aborted | score (pp) |", "|---|---|---|---|---|---|---|"]
        for cc in out["cells"]:
            L_.append(f"| {cc['player']} vs {cc['opponent']} | {cc['n_pairs']} | {cc['w']} | {cc['l']} | {cc['d']} | "
                      f"{cc['aborted']} | {cc['score_pp']:.2f} |")
        L_.append("")
    L_ += ["## Preconditions", "", f"- failures: {out['preconditions'] or 'none'}",
           f"- S3 amendment-2 check: {out['s3_check']}"]
    return "\n".join(L_) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--progress", action="store_true")
    ap.add_argument("--out-dir", default=str(HERE))
    a = ap.parse_args()
    head = code_head()
    print(f"[read] estimator checkout HEAD {head}", file=sys.stderr)
    if head != PLAN.PIN:
        print(f"REFUSED: the estimator is imported from {head}, not P_st {PLAN.PIN}", file=sys.stderr)
        return 5
    root = Path(a.root) if a.root else L.archive_ledger_root()
    reqs = (PLAN.REQ1, PLAN.REQ2, PLAN.REQ3)
    got_cells = {r: PLAN.look_cells(root, r) for r in reqs}
    if a.progress:
        for req, cs in got_cells.items():
            done = sum(1 for c in cs if c.n_pairs >= X.MIN_PAIRS)
            print(f"{req}: {len(cs)} cells, {done} at >= {X.MIN_PAIRS} pairs, {sum(c.n_pairs for c in cs)} pairs, "
                  f"aborted {sum(c.aborted for c in cs)}")
        return 0
    got, fails, facts = PLAN.resolve(PLAN.ARCHIVE_MODELS)
    ok, s3line = PLAN.P1.s3_valid()
    if not ok:
        fails.append(f"S3 amendment-2 check not VALID: {s3line}")
    rs = [PLAN.P1.sha(got["static"][s]) for s in PLAN.SEEDS if s in got["static"]]
    cs = [PLAN.P1.sha(got["legacy"][s]) for s in PLAN.SEEDS if s in got["legacy"]]
    reasons: List[str] = list(fails)
    # the uniqueness guard, read side: each request holds exactly its registered cells and no cell is in two
    n1, n2 = len(PLAN.LOOK1_SEEDS), len(PLAN.LOOK2_SEEDS)
    want = {PLAN.REQ1: {(rs[i], cs[j]) for i in range(n1) for j in range(n1)}}
    want[PLAN.REQ2] = {(rs[i], cs[j]) for i in range(n2) for j in range(n2)} - want[PLAN.REQ1]
    want[PLAN.REQ3] = {(rs[i], cs[j]) for i in range(len(rs)) for j in range(len(cs))} - want[PLAN.REQ1] - want[PLAN.REQ2]
    for req in reqs:
        have = {(c.player, c.opponent) for c in got_cells[req]}
        if have != want[req]:
            reasons.append(f"{req} holds {len(have)} cell(s), {len(have & want[req])} of its registered "
                           f"{len(want[req])} and {len(have - want[req])} other(s)")
    regime = L.read(X.FAMILY_READ_OFF, root=root, family=PLAN.FAMILY).regime_id
    m = X.matrix_from_cells([c for r in reqs for c in got_cells[r]], rs, cs)
    reasons = list(m.inconclusive) + reasons
    d = X.summarize(LOOK, m, reasons)
    out: Dict[str, Any] = {"look": LOOK, "final_look": True, "family": PLAN.FAMILY, "requests": list(reqs),
                           "n_cells": {r: len(got_cells[r]) for r in reqs}, "regime_id": regime,
                           "code_head": head, "preconditions": fails, "s3_check": s3line, "seed_facts": facts,
                           "row_sha256": rs, "col_sha256": cs, "cells": list(m.cells), "inconclusive": reasons,
                           "x5_decision": d.as_dict(), "x5_verdict": X.verdict_text(d), "stat": None}
    if m.h is not None and not reasons:
        st = X.cross_stat(m.h)
        out["stat"] = st.as_dict()
        b = X.LOOK_BOUNDARIES[LOOK]
        t95 = X.t_ppf(0.95, st.df)
        o, prim = final_outcome(st.delta_hat, st.se, b)
        fo, fixed = final_outcome(st.delta_hat, st.se, t95)
        out.update({"outcome": o, "primary": prim, "fixed_sample": fixed, "fixed_sample_outcome": fo,
                    "readings_agree": fo == o, "t_exact_95": t95, "t_exact_975": X.t_ppf(0.975, st.df),
                    "closing_rule_descriptive": closing_rule(st.delta_hat, st.se, st.df)})
    else:
        out["outcome"] = "INCONCLUSIVE"
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    (od / "result.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n")
    (od / "result.md").write_text(render_md(out))
    print(json.dumps({"outcome": out["outcome"], "x5_verdict": out["x5_verdict"],
                      "delta_hat": (out["stat"] or {}).get("delta_hat_pp")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
