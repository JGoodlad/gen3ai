"""S′-HONEST by hand: recompute `searched_play`'s estimates, class and GOAL-B point from the record's own inputs, with
a loop that shares no code with `main.reads.rules_searched_play` (PREREG §5), plus PREREG §12's REPORTED quantities
the record does not print: the achieved bootstrap SE and the power it implies, the identity sample's breakdown, and the
cross-read difference against S′-LIVE (UNPAIRED, SELECTED). Frozen record script, never a tool.

    python3 by_hand.py ~/gen3ai_archive/sprime_honest_play_2026-11-06/record.json \
        ~/gen3ai_archive/sprime_honest_play_2026-11-06/identity/rescore.json > by_hand.json
"""
import json
import math
import sys
from collections import Counter
from fractions import Fraction

import numpy as np

PANELS = ("heldout72_hpfix", "forum72", "p72")       # registered order: H, F, P72
SEED, B, EPS, MARGIN = 20261106, 2000, 0.0005, 0.03
SLIVE_G_U, SLIVE_SE = 0.04599415204678363, 0.00974  # S′-LIVE's paired G_U (its record) and realised SE (PREREG §9)


def y(unit):
    return Fraction(2 * unit["wins"] + unit["ties"], 2 * unit["n"])


def phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def bucket(turn):
    return "10-19" if turn < 20 else "20-29" if turn < 30 else "30-63" if turn < 64 else "64+"


def main(path, rescore_path):
    rec = json.load(open(path))
    units = {u["unit"]: u for u in rec["inputs"]["units"]}
    per = {kind: {p: {} for p in PANELS} for kind in ("paired", "go_imputed", "null_imputed")}
    counts = {p: {"pairs": 0, "searched_missing": 0, "raw_missing": 0, "neither": 0} for p in PANELS}
    goal_b = {p: [0, 0] for p in PANELS}                # searched wins (a missing searched game a loss), deals
    for d in rec["inputs"]["deals"]:
        p, s = d["panel"], d["unit_seed"]
        rs, ss = d["raw"]["status"], d["searched"]["status"]
        ok_r, ok_s = rs == "counted", ss == "counted"
        assert rs in ("counted", "exhausted") and ss in ("counted", "exhausted"), d
        yr = y(units[d["raw"]["unit"]]) if ok_r else None
        ys = y(units[d["searched"]["unit"]]) if ok_s else None
        goal_b[p][1] += 1
        goal_b[p][0] += units[d["searched"]["unit"]]["wins"] if ok_s else 0
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
                     "se_bootstrap": float(np.std(gu_reps, ddof=1)),
                     "panel_ci95": {p: [float(v) for v in np.percentile(reps[i], [2.5, 97.5])]
                                    for i, p in enumerate(PANELS)}}
    lb, ub = out["go_imputed"]["ci95"][0], out["null_imputed"]["ci95"][1]
    near = abs(lb) <= EPS or abs(ub - MARGIN) <= EPS
    go = lb > EPS
    null = (not go) and ub < MARGIN - EPS
    out["class"] = "STOP" if near and not (go or null) else ("GO-HONEST" if go else ("NULL" if null else "STOP"))
    out["boundary"] = near
    score = Fraction(sum(Fraction(w, n) for w, n in goal_b.values()), 3)
    out["goal_b_score"] = {"counts": goal_b, "exact": str(score), "float": float(score)}
    # §12: the achieved SE, the iid SE at the achieved discordance, and the power each implies at +2.30 pp
    se = out["paired"]["se_bootstrap"]
    disc = rec["decision"]["discordance"]
    se_iid = math.sqrt(disc["share"] / disc["pairs"])
    out["power"] = {"se_realised": se, "se_iid": se_iid, "discordance": disc["share"],
                    "p_go_at_2.30pp_realised": phi(0.023 / se - 1.96), "p_go_at_2.30pp_iid": phi(0.023 / se_iid - 1.96)}
    # §12: the cross-read difference, S′-LIVE − this read, UNPAIRED (different deals) and SELECTED (§2)
    diff = SLIVE_G_U - out["paired"]["G_U_float"]
    se_x = math.sqrt(SLIVE_SE ** 2 + se ** 2)
    out["cross_read"] = {"slive_G_U": SLIVE_G_U, "this_G_U": out["paired"]["G_U_float"], "difference": diff,
                         "se_quadrature": se_x, "ci95": [diff - 1.96 * se_x, diff + 1.96 * se_x],
                         "label": "UNPAIRED, SELECTED: REPORTED, never decided"}
    # §12: the identity guard's sample by panel, turn bucket and unit-seed range
    rs = json.load(open(rescore_path))
    by = {"panel": Counter(), "turn": Counter(), "unit_seed": Counter()}
    for r in rs["rescored"]:
        _, panel, seed, _ = r["cell"].split("__")
        turn = int(r["key"].split("|")[2][1:])
        by["panel"][panel] += 1
        by["turn"][bucket(turn)] += 1
        lo_s = 700 + 17 * ((int(seed) - 700) // 17)
        by["unit_seed"][f"{lo_s}-{lo_s + 16}"] += 1
    out["identity"] = {"n_rescored": rs["n_rescored"], "n_sampled": rs["n_sampled"], "mismatches": len(rs["mismatches"]),
                       "rebuilt": rs["rebuilt"], "torn_tails": rs["torn_tails"], "cpu_s": rs["cpu_s"],
                       "by": {k: dict(sorted(v.items())) for k, v in by.items()}}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
