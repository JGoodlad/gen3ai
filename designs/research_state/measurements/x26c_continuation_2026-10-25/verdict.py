"""X26c read: the REGISTERED verdict (PREREG §5), written before any 480M game was played.

    python verdict.py --kakuna-state ~/gen3ai_archive/x26c_read/kakuna --s-results s_results.json \
        --k9b k9b.json --out verdict.json

Kakuna: two ``main.x35.kakuna`` states, one per final (labels ``pstar`` and ``x26c_480m``), panels H / F / P72,
unit seeds 0-19. U = D1's H statistic (``main.x35.d1.strat_score``: the equal-weight panel mean, Kish n_eff, Wilson);
Δ = ``newcombe_strat(480M, P*)``. S1 = ``x26c_480m_pool`` from ``s_read.py``, against the drift read's 380M node.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PANELS = ("heldout72_hpfix", "forum72", "p72")
ON, OFF = "x26c_480m", "pstar"
UNITS_PLANNED = 20 * len(PANELS)              # per final
MIN_UNIT_FRAC = 0.90
#: the X26 drift read's 380M node on the registered R (``x26_drift_preread/results.json``; dbc49cc1)
S380_POOL_PP, S380_POOL_SE = 58.23, 0.55
EPS = 1e-9                                    # rule 8: a bound within EPS of its boundary does not clear it
Z = 1.959963985


def clears(lb: float, bar: float) -> bool:
    return lb > bar + EPS


def kakuna_block(states: list[Path]) -> dict[str, Any]:
    from main.x35 import d1 as D
    from main.x35 import kakuna as KA

    units, pooled, where = [], {}, {}
    for state in states:
        us = KA.load_plan(state)
        units += us
        pooled.update(KA.pool(state, us))
        where.update({u.name: state for u in us})
    out: dict[str, Any] = {}
    for lab in (ON, OFF):
        cells = pooled[lab]
        counts = {k: (cells[k]["wins"], cells[k]["n"]) for k in PANELS}
        ss = D.strat_score(counts, PANELS)
        out[lab] = dict(ss.as_dict(), units={k: cells[k]["units"] for k in PANELS},
                        skipped={k: cells[k]["skipped"] for k in PANELS},
                        ties={k: cells[k]["ties"] for k in PANELS}, units_counted=sum(cells[k]["units"] for k in PANELS))
    d, lo, hi = D.newcombe_strat(D.strat_score({k: (pooled[ON][k]["wins"], pooled[ON][k]["n"]) for k in PANELS}),
                                 D.strat_score({k: (pooled[OFF][k]["wins"], pooled[OFF][k]["n"]) for k in PANELS}))
    out["delta"] = {"d": d, "lo": lo, "hi": hi}
    # descriptive: unit-paired differences (same panel, same unit seed), panel-equal-weight
    per = {}
    for u in units:
        s = KA.read_summary(where[u.name], u.name)
        if s is not None and KA.counts(s, u.games):
            per[(u.label, u.teamset, u.unit_seed)] = int(s["wins"]) / int(s["n"])
    diffs = []
    for k in PANELS:
        for seed in range(20):
            a, b = per.get((ON, k, seed)), per.get((OFF, k, seed))
            if a is not None and b is not None:
                diffs.append((k, a - b))
    by = {k: [x for kk, x in diffs if kk == k] for k in PANELS}
    if all(len(v) > 1 for v in by.values()):
        m = sum(sum(v) / len(v) for v in by.values()) / len(PANELS)
        var = sum((sum((x - sum(v) / len(v)) ** 2 for x in v) / (len(v) - 1)) / len(v) for v in by.values()) / 9
        out["paired_units"] = {"d": m, "lo": m - Z * math.sqrt(var), "hi": m + Z * math.sqrt(var),
                               "pairs": {k: len(v) for k, v in by.items()}}
    # the D1 seeds' subset (0-6) beside D1 look 2's C0 (0.471), descriptive
    sub = {}
    for lab in (ON, OFF):
        w = {k: [0, 0] for k in PANELS}
        for u in units:
            if u.label == lab and u.unit_seed < 7:
                s = KA.read_summary(where[u.name], u.name)
                if s is not None and KA.counts(s, u.games):
                    w[u.teamset][0] += int(s["wins"])
                    w[u.teamset][1] += int(s["n"])
        if all(n for _x, n in w.values()):
            sub[lab] = D.strat_score({k: tuple(v) for k, v in w.items()}, PANELS).as_dict()
    out["seeds_0_6"] = sub
    return out


def decide(k: dict[str, Any], s1: dict[str, Any] | None, k9b_verdict: str) -> dict[str, Any]:
    reasons = []
    if k9b_verdict != "PASS":
        reasons.append(f"K9(b) {k9b_verdict}")
    for lab in (ON, OFF):
        if k[lab]["units_counted"] < MIN_UNIT_FRAC * UNITS_PLANNED - EPS:
            reasons.append(f"{lab}: {k[lab]['units_counted']} / {UNITS_PLANNED} units counted (< 90 %)")
    if reasons:
        return {"verdict": "VOID", "reasons": reasons}
    goal_b = clears(k[ON]["wilson95"][0], 0.50)
    climbing = clears(k["delta"]["lo"], 0.0)
    out: dict[str, Any] = {"goal_b_met": goal_b, "trajectory": "CLIMBING" if climbing else "FLAT"}
    if s1 is not None and s1.get("se_pp") is not None:
        d = s1["s_pp"] - S380_POOL_PP
        se = math.sqrt(s1["se_pp"] ** 2 + S380_POOL_SE ** 2)
        rising = clears(d - Z * se, 0.0)
        out["s1"] = {"s_480m": s1["s_pp"], "se": s1["se_pp"], "d_vs_380m": d, "lo": d - Z * se, "hi": d + Z * se,
                     "rising": rising}
        if not climbing:
            out["d3_revisit"] = rising
    else:
        out["s1"] = None
        if not climbing:
            out["d3_revisit"] = "UNREAD (S1 missing)"
    out["verdict"] = ("GOAL (b) MET; " if goal_b else "goal (b) not met; ") + out["trajectory"]
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--kakuna-state", action="append", required=True, help="one per final's state (P*, 480M)")
    ap.add_argument("--s-results", default=None)
    ap.add_argument("--k9b", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    k = kakuna_block([Path(x).expanduser() for x in a.kakuna_state])
    s1 = None
    if a.s_results:
        nodes = {n["label"]: n for n in json.loads(Path(a.s_results).read_text())["nodes"]}
        s1 = nodes.get("x26c_480m_pool")
        if s1 is not None and s1.get("inconclusive"):
            s1 = None
    k9 = json.loads(Path(a.k9b).read_text())
    doc = {"kakuna": k, "decision": decide(k, s1, k9["verdict"]), "k9b": k9}
    Path(a.out).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps(doc["decision"], indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
