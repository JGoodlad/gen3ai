#!/usr/bin/env python3
"""THE 75M RUN-LEVEL FLOOR — Metamon `SmallRL` HOME cells. W_b's half-cells, and the floor.

Same code path as arm S's ``metamon_cells.py`` and the pair read's ``pair_metamon_cells.py``, run
on W_b's half-cells with the SAME harness (``run_cell.sh``, unchanged) and the SAME team seeds
(``BASE_OURS 20260914`` / ``BASE_THEIRS 20270914``), so all three arms drew the identical team pairs.

Only the GREEDY regime is run: anchors-SOP rule 2 — the recurring read is greedy-vs-greedy, verified
per decision, and "temperature 1.0" is not a fixed yardstick across policies. The MIXED cell is the
one whose Metamon half has now died three times with the same ``RecursionError`` and it buys nothing
a floor read needs.

Comparators (greedy, home pool, 100 games each):
  armW  ai_v13_02_flywheel_winprob @75,005,952 — 0.650, the FLOOR's other half (pair read)
  armS  ai_v13_01_flywheel_shaped  @75,005,952 — 0.630 (flywheel_armS_reads_2026-09-14)
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

Z = 1.959963985

COMPARATORS = {
    "greedy": {
        "armW": {"k": 65, "n": 100,
                 "label": "arm W (ai_v13_02_flywheel_winprob) @75,005,952, greedy-vs-greedy, home "
                          "pool — the FLOOR's other half",
                 "source": "designs/research_state/measurements/flywheel_pair_read_2026-09-15"},
        "armS": {"k": 63, "n": 100,
                 "label": "arm S (ai_v13_01_flywheel_shaped) @75,005,952, greedy-vs-greedy, home pool",
                 "source": "designs/research_state/measurements/flywheel_armS_reads_2026-09-14"},
        "winprob75M": {"k": 52, "n": 100,
                       "label": "ai_v12_02_winprob_critic@75M, greedy-vs-greedy, home pool",
                       "source": "designs/research_state/measurements/metamon_matched_regime_2026-09-14"},
    },
}


def wilson(k: int, n: int):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def newcombe(k1, n1, k2, n2):
    p1, l1, u1 = wilson(k1, n1)
    p2, l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return (d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2),
            d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))


def main() -> int:
    root = Path(sys.argv[1])
    rows = [json.loads(ln) for ln in (root / "games.jsonl").read_text().splitlines() if ln.strip()]
    summary = json.loads((root / "summary.json").read_text())

    by = defaultdict(list)
    for r in rows:
        by[(r["regime"], r["challenger"])].append(r)
        by[(r["regime"], "POOLED")].append(r)

    out = {"n_games_total": len(rows),
           "model": "ai_v13_04_flywheel_winprob_b/final_model.zip @75,005,952 (W_b, seed 1002)",
           "opponent": "Metamon SmallRL (ckpt 40, 13.9M), VanillaAttention, CPU",
           "team_set": "home — our 719-team gen3ou pool on both sides (the de-risk's nickname-free export)",
           "cells": {}, "regime_rows": {}}

    for (regime, who), rs in sorted(by.items()):
        w = sum(1 for r in rs if r["our_result"] == "win")
        loss = sum(1 for r in rs if r["our_result"] == "loss")
        tie = len(rs) - w - loss
        p, lo, hi = wilson(w, len(rs))
        rec = {"n": len(rs), "W": w, "L": loss, "T": tie,
               "win_rate": round(p, 4), "wilson95": [round(lo, 4), round(hi, 4)],
               "mean_turns": round(sum(r["turns"] for r in rs) / len(rs), 1),
               "max_turns": max(r["turns"] for r in rs),
               "hit_forfeit_limit": sum(1 for r in rs if r["hit_forfeit_limit"]),
               "sides_disagree": sum(1 for r in rs if not r["sides_agree"]),
               "metamon_row_missing": sum(1 for r in rs if not r["metamon_row_found"])}
        key = f"{regime}|{who}"
        (out["regime_rows"] if who == "POOLED" else out["cells"])[key] = rec

    for regime, comps in COMPARATORS.items():
        row = out["regime_rows"].get(f"{regime}|POOLED")
        if not row:
            continue
        row["comparators"] = {}
        for tag, comp in comps.items():
            # SIGN CONVENTION: comparator MINUS W_b, so a positive number means the comparator
            # is ahead. armW MINUS W_b IS THE 75M RUN-LEVEL FLOOR for this cell.
            d, lo, hi = newcombe(comp["k"], comp["n"], row["W"], row["n"])
            cp, clo, chi = wilson(comp["k"], comp["n"])
            row["comparators"][tag] = {
                "label": comp["label"], "source": comp["source"],
                "win_rate": round(cp, 4), "wilson95": [round(clo, 4), round(chi, 4)],
                "n": comp["n"],
                "comparator_minus_armWb": round(d, 4), "newcombe95": [round(lo, 4), round(hi, 4)],
                "verdict": ("NOT DETECTED — the difference interval covers zero (rule 6: this is "
                            "never 'equal'; equivalence needs the delta's own CI inside a bar, and "
                            "no bar exists for this row)" if lo <= 0 <= hi else
                            "DETECTED at 95% — the difference interval excludes zero"),
            }

    ver = []
    for hc in summary.get("halfcells", []):
        ot, mt = hc.get("our_timing") or {}, hc.get("metamon_timing") or {}
        ver.append({"cell": f"{hc['regime']}|{hc['challenger']}",
                    "our_stochastic_kwarg": ot.get("stochastic_kwarg_values"),
                    "our_temperature_flag": ot.get("temperature_flag"),
                    "metamon_sample_kwarg": mt.get("sample_kwarg_values"),
                    "metamon_argmax_match_rate": mt.get("argmax_match_rate"),
                    "metamon_regime_declared": hc.get("metamon_regime"),
                    "metamon_rc": hc.get("metamon_rc"),
                    "metamon_record": ("present" if mt else
                                       "MISSING — the Metamon process died after its last game; "
                                       "our side's 50 results are complete and the regime was set "
                                       "at process start, but the per-decision argmax check for "
                                       "this half-cell is n/a"),
                    "failures": hc.get("failures")})
    out["regime_verification"] = ver
    if "armW" in (out["regime_rows"].get("greedy|POOLED", {}).get("comparators") or {}):
        c = out["regime_rows"]["greedy|POOLED"]["comparators"]
        fl = abs(c["armW"]["comparator_minus_armWb"])
        dd = abs(c["armS"]["win_rate"] - c["armW"]["win_rate"])
        d_sw, lo_sw, hi_sw = newcombe(COMPARATORS["greedy"]["armS"]["k"], 100,
                                      COMPARATORS["greedy"]["armW"]["k"], 100)
        inside = (lo_sw <= fl <= hi_sw) or (lo_sw <= -fl <= hi_sw)
        out["floor_verdict"] = {
            "THE_FLOOR_armW_minus_armWb": round(fl, 4),
            "THE_FINDING_armS_minus_armW": round(d_sw, 4),
            "finding_ci95": [round(lo_sw, 4), round(hi_sw, 4)],
            "clause_a_abs_delta_gt_floor": bool(abs(d_sw) > fl),
            "clause_b_ci_excludes_floor_point": bool(not inside),
            "verdict": ("OUTSIDE THE 75M FLOOR" if (abs(d_sw) > fl and not inside)
                        else "WITHIN FLOOR at n = 2"),
            "caveats": ["one replicate pair BOUNDS a floor; no CI attaches to it (rules 19/22)",
                        "WITHIN FLOOR is NEVER 'equivalent' (rule 6)",
                        "100 games resolves ~+/-10 pp at best (anchors SOP sec 6)"]}
    (root / "armWb_cells.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["regime_rows"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
