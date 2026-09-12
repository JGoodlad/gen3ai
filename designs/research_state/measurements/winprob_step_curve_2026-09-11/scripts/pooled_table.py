#!/usr/bin/env python3
"""THE STRENGTH-ROBUST ROW — `frame_check.json` -> the three readings on IDENTICAL states.

`cond.opp_class_auc.t4_10` is the AUC of **V**, and V's ability to separate pool from bot must
fall as the two classes' OUTCOMES converge — which is exactly what the trainee getting stronger
against a FIXED panel does (see `strength_drift.py`). So that row cannot answer the question the
step curve exists to ask, which is about the REPRESENTATION.

This row can. `value_pooled -> opponent class` decodes the class from the win head's literal
input, out-of-fold and battle-grouped, and never touches outcomes — so the closing bot-pool
win-rate gap cannot produce a move in it.

Three readings, same states, same procedure:
  leak    own-team -> class     the ARTEFACT that made the probe read's 0.846 spurious; must be
                                at chance on a full-capture frame, and is reported so a reader
                                sees that rather than takes it on trust
  pooled  value_pooled -> class the REPRESENTATION's class information  <- the primary reading
  V       V -> class            the scalar OUTPUT's class information (outcome-convergent)

Usage:  pooled_table.py <frame_check.json> [--json OUT.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ORDER = ["10M", "20M", "40M", "73M"]
DRAWS = ["draw1", "draw2"]
BUCKET = "t4_10"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frame_check")
    ap.add_argument("--bucket", default=BUCKET)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    raw = json.loads(Path(a.frame_check).read_text())
    rows = raw if isinstance(raw, list) else list(raw.values())
    by: dict = {}
    for r in rows:
        tag = Path(r["dir"]).name
        ob = r.get("observability", {}).get(a.bucket, {})
        by[tag] = {
            "leak": r.get("own_team_leak", {}).get("own_team_LOO_to_opp_class_AUC"),
            "pooled": ob.get("pooled_to_opp_class_AUC"),
            "V": ob.get("V_to_opp_class_AUC"),
            "n_states": ob.get("n_states"),
        }

    print(f"### The three readings at turns {a.bucket}, on identical states\n")
    print("| checkpoint | draw | own-team leak (chance = 0.50) | "
          "**pooled → class** | V → class |")
    print("|---|---|---|---|---|")
    for lab in ORDER:
        for d in DRAWS:
            e = by.get(f"{lab}_{d}")
            if not e:
                continue
            print(f"| {lab} | {d} | {e['leak']:.4f} | **{e['pooled']:.4f}** | {e['V']:.4f} |")

    print(f"\n### The registered bar on `pooled → class` @ {a.bucket}\n")
    P = {lab: {d: (by.get(f"{lab}_{d}") or {}).get("pooled") for d in DRAWS} for lab in ORDER}
    W = {}
    for lab in ORDER:
        x, y = P[lab]["draw1"], P[lab]["draw2"]
        W[lab] = abs(x - y) if (x is not None and y is not None) else None
    print("| checkpoint | draw1 | draw2 | W(s) = draw spread |")
    print("|---|---|---|---|")
    for lab in ORDER:
        x, y = P[lab]["draw1"], P[lab]["draw2"]
        print(f"| {lab} | {'—' if x is None else f'{x:.4f}'} | "
              f"{'—' if y is None else f'{y:.4f}'} | "
              f"{'—' if W[lab] is None else f'{W[lab]:.4f}'} |")
    print("\n| gap | draw1 | draw2 | bar = max(W) | clears the bar on BOTH draws? |")
    print("|---|---|---|---|---|")
    verdicts = []
    for f, t in zip(ORDER, ORDER[1:]):
        g = {}
        for d in DRAWS:
            x, y = P[f][d], P[t][d]
            g[d] = None if (x is None or y is None) else y - x
        bar = (max(W[f], W[t]) if (W[f] is not None and W[t] is not None) else None)
        ok = bar is not None and all(v is not None and abs(v) > bar for v in g.values())
        same = all(v is not None and v < 0 for v in g.values()) or \
            all(v is not None and v > 0 for v in g.values())
        verdicts.append({"gap": f"{f}->{t}", **g, "bar": bar,
                         "clears": ok, "signs_agree": same})
        note = ("**YES — and DOWN on both**" if ok and same and g["draw1"] < 0 else
                "**YES — and UP on both**" if ok and same else
                "no" if not ok else "clears, but signs disagree")
        g1 = "—" if g["draw1"] is None else f"{g['draw1']:+.4f}"
        g2 = "—" if g["draw2"] is None else f"{g['draw2']:+.4f}"
        bs = "—" if bar is None else f"{bar:.4f}"
        print(f"| {f} → {t} | {g1} | {g2} | {bs} | {note} |")
    e2e = {d: (P["73M"][d] - P["10M"][d]) if (P["73M"][d] is not None
                                             and P["10M"][d] is not None) else None
           for d in DRAWS}
    maxW = max([w for w in W.values() if w is not None], default=None)
    e1 = "—" if e2e["draw1"] is None else f"{e2e['draw1']:+.4f}"
    e2 = "—" if e2e["draw2"] is None else f"{e2e['draw2']:+.4f}"
    mw = "—" if maxW is None else f"{maxW:.4f}"
    print(f"\n**10M → 73M:** {e1} / {e2}  (max W = {mw})")
    rise = all(v["clears"] and v["signs_agree"] and (v["draw1"] or 0) > 0 for v in verdicts)
    flat = (maxW is not None and all(v is not None and abs(v) <= maxW for v in e2e.values()))
    print("\n**Verdict:** " + (
        "(i) SUPPORTED — monotone rise past the bar" if rise else
        "(ii) SUPPORTED — flat within the draw spread" if flat else
        "**UNDECIDED between (i) and (ii); (i) REFUTED directionally** — no gap is "
        "up-and-past-the-bar, and |10M→73M| exceeds max W"))
    if a.json:
        Path(a.json).write_text(json.dumps(
            {"bucket": a.bucket, "points": by, "W": W, "gaps": verdicts,
             "end_to_end": e2e, "max_W": maxW}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
