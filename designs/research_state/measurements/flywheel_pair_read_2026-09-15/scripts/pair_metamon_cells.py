#!/usr/bin/env python3
"""THE PAIR READ — Metamon `SmallRL`. Arm W's half-cells, and the three-way table.

Same code path as arm S's ``metamon_cells.py``; this adds the two comparators the pair read needs
and keeps each one IN ITS OWN REGIME (SOP rule 1: a number never leaves without its regime).

Regimes (identical cells, identical team seeds, to arm S's read):
  greedy  BOTH sides greedy, our 719-team HOME pool
  mixed   OUR side greedy, Metamon at its OWN eval default (temperature 1.0), HOME pool

Comparators, per regime:
  armS         ai_v13_01_flywheel_shaped @75,005,952 — THE PAIR'S OTHER ARM, same harness, same
               seeds, same cells, run 2026-09-14 (flywheel_armS_reads_2026-09-14)
  winprob75M   ai_v12_02_winprob_critic @75M — the free third leg (pin f971caf2, --ent-coef 0.02)
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
        "armS": {"k": 63, "n": 100,
                 "label": "arm S (ai_v13_01_flywheel_shaped) @75,005,952, greedy-vs-greedy, home pool",
                 "source": "designs/research_state/measurements/flywheel_armS_reads_2026-09-14"},
        "winprob75M": {"k": 52, "n": 100,
                       "label": "ai_v12_02_winprob_critic@75M, greedy-vs-greedy, home pool",
                       "source": "designs/research_state/measurements/metamon_matched_regime_2026-09-14"},
    },
    "mixed": {
        "armS": {"k": 84, "n": 100,
                 "label": "arm S @75,005,952, ours greedy vs SmallRL t1.0, home pool",
                 "source": "designs/research_state/measurements/flywheel_armS_reads_2026-09-14"},
        "winprob75M": {"k": 89, "n": 120,
                       "label": "ai_v12_02_winprob_critic@75M, ours greedy vs SmallRL t1.0, home pool",
                       "source": "designs/research_state/measurements/metamon_derisk_2026-09-14"},
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
           "model": "ai_v13_02_flywheel_winprob/final_model.zip @75,005,952 (arm W)",
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
            # SIGN CONVENTION: comparator MINUS arm W, so a positive number means the comparator
            # is ahead — the same orientation as the pair's Delta(S - W) on every other row.
            d, lo, hi = newcombe(comp["k"], comp["n"], row["W"], row["n"])
            cp, clo, chi = wilson(comp["k"], comp["n"])
            row["comparators"][tag] = {
                "label": comp["label"], "source": comp["source"],
                "win_rate": round(cp, 4), "wilson95": [round(clo, 4), round(chi, 4)],
                "n": comp["n"],
                "comparator_minus_armW": round(d, 4), "newcombe95": [round(lo, 4), round(hi, 4)],
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
    (root / "armW_cells.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["regime_rows"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
