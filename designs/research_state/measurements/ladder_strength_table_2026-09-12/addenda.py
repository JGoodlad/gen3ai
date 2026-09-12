#!/usr/bin/env python3
"""Addenda to the strength table — the three POST-HOC observations, labelled as post hoc.

Reads `ladder_strength.json` (produced by `read_ladder_strength.py`); touches nothing under
`models/`. None of these three rows was pre-registered; each is reported as an OBSERVATION with
its alternative left open, never as a detection.

  (A) is `win_rate_vs_bots` an independent DESCRIPTOR of the headline, or an INPUT to it?
  (B) the WITHIN-LEVER spread, for the two levers that have a seed replicate.
  (C) the floor's widest pair crosses a PIN boundary; the same-pin control pair does not.
"""
from __future__ import annotations

import json
import math
import sys

Z = 1.959963984540054


def rank(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    r = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def corr(x, y):
    mx, my = sum(x) / len(x), sum(y) / len(y)
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    den = math.sqrt(sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y))
    return num / den if den else float("nan")


def main() -> int:
    path = sys.argv[1]
    doc = json.load(open(path))
    arms = doc["arms"]
    out = {"note": "POST-HOC observations, not pre-registered rows"}

    # ── (A) the bot row is an INPUT to the headline, not an independent descriptor ─────────
    elo = [a["headline"]["elo"] for a in arms]
    wr = [a["desc"]["win_rate_vs_bots"] for a in arms]
    out["A_bot_row_is_an_input"] = {
        "pearson_elo_vs_win_rate_vs_bots": round(corr(elo, wr), 4),
        "spearman_elo_vs_win_rate_vs_bots": round(corr(rank(elo), rank(wr)), 4),
        "win_rate_vs_bots_range": [min(wr), max(wr)],
        "mechanism": ("snapshot_ladder.fit_ladder folds each snapshot's HISTORICAL BOT EDGES into "
                      "the fit as source (2) — they are what pins the absolute scale. So a run's "
                      "bot performance is not an independent descriptor of its ladder rating; it "
                      "is one of the two inputs the rating is solved from (the other being the "
                      "dense frozen-vs-frozen matrix). A high correlation here is partly "
                      "MECHANICAL and must never be read as two instruments agreeing."),
    }

    # ── (B) within-lever spread, the two levers with a seed replicate ──────────────────────
    by = {a["label"]: a for a in arms}
    floor = doc["floors"]["headline"]["floor"]
    pairs = [("vf15", "vf15_b"), ("strata", "strata_b"), ("lambda09", "lambda09_b"),
             ("ctrl10M_b", "ctrl10M_c")]
    rows = []
    for x, y in pairs:
        if x not in by or y not in by:
            continue
        a, b = by[x]["headline"], by[y]["headline"]
        d = a["elo"] - b["elo"]
        se = math.sqrt(a["se"] ** 2 + b["se"] ** 2)
        rows.append({"pair": f"{x} - {y}", "delta": round(d, 1), "abs": round(abs(d), 1),
                     "x_over_floor": round(abs(d) / floor, 2), "se": round(se, 1),
                     "ci95": [round(d - Z * se, 1), round(d + Z * se, 1)],
                     "same_pin": by[x]["desc"]["pin_short"] == by[y]["desc"]["pin_short"],
                     "pins": [by[x]["desc"]["pin_short"], by[y]["desc"]["pin_short"]],
                     "seeds": [by[x]["desc"]["seed"], by[y]["desc"]["seed"]]})
    out["B_within_lever_spread"] = {"floor": floor, "pairs": rows}

    # ── (C) the floor's widest pair crosses a PIN boundary ────────────────────────────────
    ctrl = ["ctrl10M", "ctrl10M_b", "ctrl10M_c"]
    out["C_control_pairs_by_pin"] = [
        {"pair": p["pair"], "abs": p["abs"], "se": p["se"],
         "same_pin": by[p["pair"].split(" - ")[0]]["desc"]["pin_short"]
                     == by[p["pair"].split(" - ")[1]]["desc"]["pin_short"],
         "pins": [by[p["pair"].split(" - ")[0]]["desc"]["pin_short"],
                  by[p["pair"].split(" - ")[1]]["desc"]["pin_short"]],
         "seeds": [by[p["pair"].split(" - ")[0]]["desc"]["seed"],
                   by[p["pair"].split(" - ")[1]]["desc"]["seed"]]}
        for p in doc["floors"]["headline"]["pairs"]]
    out["C_note"] = ("The two control pairs that CROSS the f3502568 -> 377a5aa1 pin boundary read "
                     "35.3 and 45.0; the one SAME-PIN pair (ctrl10M_b vs ctrl10M_c, seeds "
                     "1001/1002) reads 9.7. With se(Delta) = 22.9 on every pair, 9.7 and 45.0 are "
                     "NOT distinguishable, so this is an OBSERVATION with the seed lottery fully "
                     "open, never a pin effect. It does say the 45.0 floor bounds run-to-run "
                     "variance INCLUDING a code increment, which makes it conservative for a "
                     "same-pin comparison - and rule 3 (max pairwise) wants exactly that bar.")
    for a in arms:
        out.setdefault("pins_by_arm", {})[a["label"]] = {
            "pin": a["desc"]["pin_short"], "seed": a["desc"]["seed"],
            "seed_source": a["desc"]["seed_source"]}

    print(json.dumps(out, indent=1))
    with open(path.replace("ladder_strength.json", "addenda.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
