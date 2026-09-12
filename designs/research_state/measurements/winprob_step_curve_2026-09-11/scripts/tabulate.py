#!/usr/bin/env python3
"""THE STEP CURVE — six `critic_read` pair reports -> the registered table.

Six reads: {draw1, draw2} x {20M, 40M, 73M} vs the run's OWN 10M of the SAME draw.

🚨 THE CURVE IS BUILT FROM DELTAS, NOT LEVELS, and that is forced by the tool rather than
preferred. `critic_read` QUOTA-MATCHES each pair — it subsamples both sides to the elementwise
minimum of their realized per-opponent W/L/D profiles — so the 10M control is re-estimated on a
DIFFERENT matched frame in every pair and its level is pair-dependent (measured: 0.797573 in the
20M pair, 0.795735 in the 40M pair, a 0.0019 wobble on the decision row). A curve of levels would
silently mix that wobble into its gaps. Each pair's DELTA, by contrast, is computed on one matched
frame with its own battle-clustered CI, and is anchored at 0 by construction.

So: `D(s, d)` is the delta of checkpoint `s` against 10M on draw `d`, and the within-checkpoint
eval-draw spread is `W(s) = |D(s,1) - D(s,2)|` — a spread of DELTAS, which is also exactly the
quantity the registered bar names ("the delta between checkpoints against that spread").

🚨 THE LADDER'S hp400 FLOOR DECIDES NOTHING HERE. It was measured on a different run, a different
sentinel panel, and the other side of the 2026-09-07 opponent-regime boundary; rule 20 forbids one
frame type's controls barring another's. It is printed as a reference and never as a bar.

Usage:  tabulate.py <reads_dir> [--json OUT.json]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ARM_STEPS = [18_660_864, 40_935_168, 73_121_280]
LABEL = {9_969_408: "10M", 18_660_864: "20M", 40_935_168: "40M", 73_121_280: "73M"}
DRAWS = ["draw1", "draw2"]

#: (key, role). DECISION rows decide; the rest are printed and never decide.
ROWS = [
    ("cond.opp_class_auc.t4_10", "DECISION"),
    ("gate.resolution.bot", "DECISION"),
    ("gate.resolution.all", "DECISION"),
    ("cond.opp_class_auc.t1_3", "DECISION"),
    ("identity.resolution_cap_share", "REPORTED — resolution as a SHARE of its base-rate cap"),
    ("gate.skill.bot", "REPORTED"),
    ("cond.own_team_r2.t1", "PROVISIONAL (rule 18)"),
    ("cond.spread_ratio.t1_3", "REPORTED (run-level floor)"),
    ("cond.spread_ratio.t4_10", "REPORTED (run-level floor)"),
    ("identity.bias.ALL", "REPORTED (run-level, rule 19)"),
    ("identity.bias.late (turn>=25)", "REPORTED (run-level, rule 19)"),
    ("gate.ece.all", "REPORTED"),
    ("cond.calibration_slope.all", "REPORTED (run-level)"),
]


def load(reads: Path) -> tuple[dict, list[str]]:
    recs: dict = {}
    missing: list[str] = []
    for draw in DRAWS:
        recs[draw] = {}
        for step in ARM_STEPS:
            p = reads / f"{draw}_{step}_vs_10M" / "critic_read.json"
            if not p.exists():
                missing.append(f"{draw}/{LABEL[step]}")
                continue
            d = json.loads(p.read_text())
            recs[draw][step] = {r["key"]: r for r in d["deltas"]}
    return recs, missing


def fmt(x, nd=4):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:+.{nd}f}"


def render(recs: dict, missing: list[str]) -> tuple[str, dict]:
    L: list[str] = []
    payload: dict = {"rows": {}, "missing": missing}
    if missing:
        L.append(f"> ⚠️ **INCOMPLETE — no report for: {', '.join(missing)}.** "
                 "Every verdict below is provisional on those.\n")
    for key, role in ROWS:
        D = {d: {s: recs[d].get(s, {}).get(key, {}).get("delta") for s in ARM_STEPS}
             for d in DRAWS}
        CI = {d: {s: recs[d].get(s, {}).get(key, {}).get("ci") for s in ARM_STEPS}
              for d in DRAWS}
        ctl = {d: {s: recs[d].get(s, {}).get(key, {}).get("control") for s in ARM_STEPS}
               for d in DRAWS}
        if all(D[d][s] is None for d in DRAWS for s in ARM_STEPS):
            L.append(f"\n### `{key}` — {role}\n\nNO DATA.")
            continue
        W = {}
        for s in ARM_STEPS:
            a, b = D["draw1"][s], D["draw2"][s]
            W[s] = abs(a - b) if (a is not None and b is not None) else None
        maxW = max([w for w in W.values() if w is not None], default=None)

        L.append(f"\n### `{key}` — {role}\n")
        L.append("| vs 10M | Δ draw1 | CI draw1 | Δ draw2 | CI draw2 | "
                 "W(s) = |Δ₁−Δ₂| |")
        L.append("|---|---|---|---|---|---|")
        for s in ARM_STEPS:
            c1, c2 = CI["draw1"][s], CI["draw2"][s]
            L.append(
                f"| {LABEL[s]} | {fmt(D['draw1'][s])} | "
                f"{'—' if not c1 else f'[{c1[0]:+.4f}, {c1[1]:+.4f}]'} | "
                f"{fmt(D['draw2'][s])} | "
                f"{'—' if not c2 else f'[{c2[0]:+.4f}, {c2[1]:+.4f}]'} | "
                f"{'—' if W[s] is None else f'{W[s]:.4f}'} |")

        # consecutive gaps, in DELTA space (10M is the origin, Δ=0 by construction)
        L.append("")
        L.append("| consecutive gap | draw1 | draw2 | bar = max(W) over the pair | "
                 "clears bar (|Δ| > bar) on BOTH draws? |")
        L.append("|---|---|---|---|---|")
        seq = [(9_969_408, None)] + [(s, s) for s in ARM_STEPS]
        gaps = []
        rising = True
        for (s_from, k_from), (s_to, k_to) in zip(seq, seq[1:]):
            g = {}
            for d in DRAWS:
                a = 0.0 if k_from is None else D[d][k_from]
                b = D[d][k_to]
                g[d] = None if (a is None or b is None) else b - a
            wf = 0.0 if k_from is None else W.get(k_from)
            wt = W.get(k_to)
            bar = max(w for w in (wf, wt) if w is not None) if (
                wf is not None and wt is not None) else None
            # 🚨 MAGNITUDE, then direction. The bar asks "is this gap bigger than one re-draw
            # of the same checkpoint" — a question about SIZE. A rise-only test (`g > bar`)
            # silently marks every downward move as "no", which would have hidden the one row
            # that moves monotonically past its bar in this whole measurement (`gate.ece.all`,
            # where DOWN is BETTER). Direction is reported beside the magnitude, never folded
            # into it; `rising` — which (i) needs — still requires the gaps to be UP.
            clears = (bar is not None
                      and all(g[d] is not None and abs(g[d]) > bar for d in DRAWS))
            up = all(g[d] is not None and g[d] > 0 for d in DRAWS)
            down = all(g[d] is not None and g[d] < 0 for d in DRAWS)
            rising = rising and clears and up
            gaps.append({"from": LABEL[s_from], "to": LABEL[s_to],
                         "draw1": g["draw1"], "draw2": g["draw2"],
                         "bar": bar, "clears": clears, "up": up, "down": down})
            note = ("**YES — UP on both**" if clears and up else
                    "**YES — DOWN on both**" if clears and down else
                    "clears, but signs disagree" if clears else "no")
            L.append(f"| {LABEL[s_from]} → {LABEL[s_to]} | {fmt(g['draw1'])} | "
                     f"{fmt(g['draw2'])} | {'—' if bar is None else f'{bar:.4f}'} | "
                     f"{note} |")

        e2e = {d: D[d][73_121_280] for d in DRAWS}
        e2e_ci = {d: CI[d][73_121_280] for d in DRAWS}
        have_e2e = all(e2e[d] is not None for d in DRAWS)
        ci_covers_zero = all(
            c is not None and c[0] <= 0 <= c[1] for c in e2e_ci.values()) if have_e2e else False
        within = (have_e2e and maxW is not None
                  and all(abs(e2e[d]) <= maxW for d in DRAWS))
        if rising and have_e2e and all(e2e[d] > 0 for d in DRAWS):
            verdict = "**(i) SUPPORTED** — monotone rise, every gap above the draw spread"
        elif within and ci_covers_zero:
            verdict = ("**(ii) SUPPORTED** — flat: |Δ(10M→73M)| within max W on both draws "
                       "and both CIs cover zero")
        else:
            verdict = "**UNDECIDED**"
        L.append(f"\n{key} ⇒ {verdict}  (max W = "
                 f"{'—' if maxW is None else f'{maxW:.4f}'}; "
                 f"Δ(10M→73M) = {fmt(e2e['draw1'])} / {fmt(e2e['draw2'])})")
        # the quota-match wobble in the control's own level, as a diagnostic
        wob = []
        for d in DRAWS:
            vals = [v for v in ctl[d].values() if v is not None]
            if len(vals) > 1:
                wob.append(f"{d} {max(vals) - min(vals):.4f}")
        if wob:
            L.append(f"  · control-level wobble across pairs (quota match): {', '.join(wob)}")
        payload["rows"][key] = {
            "role": role, "deltas": {d: {str(s): D[d][s] for s in ARM_STEPS} for d in DRAWS},
            "ci": {d: {str(s): CI[d][s] for s in ARM_STEPS} for d in DRAWS},
            "W": {str(s): W[s] for s in ARM_STEPS}, "max_W": maxW,
            "gaps": gaps, "end_to_end": e2e, "verdict": verdict}
    return "\n".join(L), payload


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reads")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    recs, missing = load(Path(a.reads))
    text, payload = render(recs, missing)
    print(text)
    if a.json:
        Path(a.json).write_text(json.dumps(payload, indent=1))
        print(f"\nwrote {a.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
