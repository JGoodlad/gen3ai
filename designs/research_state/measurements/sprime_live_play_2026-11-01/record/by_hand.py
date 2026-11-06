"""S′-LIVE by hand: recompute `searched_play`'s estimates and class from the record's own inputs, with a loop that shares
no code with `main.reads.rules_searched_play` (PREREG §5). Frozen record script, never a tool.

    python3 by_hand.py ~/gen3ai_archive/sprime_live_play_2026-11-01/record.json > by_hand.json
"""
import json
import sys
from fractions import Fraction

import numpy as np

PANELS = ("heldout72_hpfix", "forum72", "p72")       # registered order: H, F, P72
SEED, B, EPS, MARGIN = 20261101, 2000, 0.0005, 0.03


def y(unit):
    return Fraction(2 * unit["wins"] + unit["ties"], 2 * unit["n"])


def main(path):
    rec = json.load(open(path))
    units = {u["unit"]: u for u in rec["inputs"]["units"]}
    per = {kind: {p: {} for p in PANELS} for kind in ("paired", "go_imputed", "null_imputed")}
    counts = {p: {"pairs": 0, "searched_missing": 0, "raw_missing": 0, "neither": 0} for p in PANELS}
    for d in rec["inputs"]["deals"]:
        p, s = d["panel"], d["unit_seed"]
        rs, ss = d["raw"]["status"], d["searched"]["status"]
        ok_r, ok_s = rs == "counted", ss == "counted"
        assert rs in ("counted", "exhausted") and ss in ("counted", "exhausted"), d
        yr = y(units[d["raw"]["unit"]]) if ok_r else None
        ys = y(units[d["searched"]["unit"]]) if ok_s else None
        if ok_r and ok_s:
            counts[p]["pairs"] += 1
            for kind in per:
                per[kind][p].setdefault(s, []).append(ys - yr)
            continue
        if ok_r:
            counts[p]["searched_missing"] += 1
            go, null = -yr, 1 - yr
        elif ok_s:
            counts[p]["raw_missing"] += 1
            go, null = ys - 1, ys
        else:
            counts[p]["neither"] += 1
            go, null = Fraction(-1), Fraction(1)
        per["go_imputed"][p].setdefault(s, []).append(go)
        per["null_imputed"][p].setdefault(s, []).append(null)
    out = {"counts": counts}
    for kind, by_panel in per.items():
        rng = np.random.default_rng(SEED)
        reps, G = [], {}
        for p in PANELS:
            stats = [sum(v, Fraction(0)) / len(v) for _, v in sorted(by_panel[p].items())]
            G[p] = sum(stats, Fraction(0)) / len(stats)
            x = np.array([float(t) for t in stats])
            idx = rng.integers(0, len(x), size=(B, len(x)))
            reps.append(x[idx].mean(axis=1))
        gu_reps = np.mean(reps, axis=0)
        lo, hi = np.percentile(gu_reps, [2.5, 97.5])
        out[kind] = {"G": {p: str(G[p]) for p in PANELS}, "G_U": str(sum(G.values(), Fraction(0)) / 3),
                     "G_U_float": float(sum(G.values(), Fraction(0)) / 3), "ci95": [float(lo), float(hi)],
                     "panel_ci95": {p: [float(v) for v in np.percentile(reps[i], [2.5, 97.5])]
                                    for i, p in enumerate(PANELS)}}
    lb, ub = out["go_imputed"]["ci95"][0], out["null_imputed"]["ci95"][1]
    near = abs(lb) <= EPS or abs(ub - MARGIN) <= EPS
    go = lb > EPS
    null = (not go) and ub < MARGIN - EPS
    out["class"] = "STOP" if near and not (go or null) else ("GO-LEAKY" if go else ("NULL" if null else "STOP"))
    out["boundary"] = near
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
