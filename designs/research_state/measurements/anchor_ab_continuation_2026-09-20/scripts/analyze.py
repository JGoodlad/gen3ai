#!/usr/bin/env python3
"""THE A/B CONTINUATION ANCHOR READ — pool the sub-cells, contrast the arms, check the instruments.

Nothing here re-derives a win rate from a battle log: every sub-cell's W/L/T comes from the tool of
record's own ``summary.json``.  This script (1) pools the four 100-game sub-cells per cell, (2) puts
the registered CI on each of the three paired contrasts, (3) records the integrity counters the
registration named, and (4) runs the post-hoc alignment check that licenses the PAIRED descriptor.

    python analyze.py <out-dir> <dest.json>

🚨 THE REGISTERED BAR IS THE CI ON THE DIFFERENCE EXCLUDING ZERO.  A CI that straddles is NOT
DETECTED -- never "equivalent" (rule 6).  The 0.090 run-level floor is reported BESIDE it as the
stricter secondary, with the caveat registered in PREDICTION.md sec 3: it is a BETWEEN-SEED quantity
and C - W is a WITHIN-LINEAGE contrast, so it is the wrong null in kind for that row.
"""
from __future__ import annotations

import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

Z = 1.959963984540054
FLOOR = 0.090            # |arm W - W_b| on the SmallRL away cell, flywheel_wb_floor_read sec 6.1
ARMS = ("W", "C", "F")
ARM_LABEL = {"W": "arm W (frozen parent, 75,005,952)",
             "C": "continuation ai_v13_09_wcont (87,097,344)",
             "F": "fold path ai_v13_08_fold1_cont (87,097,344)"}
CONTRASTS = (("C", "W"), ("C", "F"), ("F", "W"))


def wilson(k: int, n: int):
    if n == 0:
        return 0.0, 0.0, 1.0
    p = k / n
    d = 1 + Z * Z / n
    c = p + Z * Z / (2 * n)
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return p, (c - h) / d, (c + h) / d


def newcombe(k1, n1, k2, n2):
    """Newcombe's hybrid-score interval on the difference of two INDEPENDENT proportions."""
    p1, l1, u1 = wilson(k1, n1)
    p2, l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def paired_bootstrap(pairs, draws=20000, seed=20260919):
    """Bootstrap over MATCHED game units (same team seed, half and index -> the same draw).

    Registered as a DESCRIPTOR only.  It is the tighter interval when the pairing is real, and the
    registration keeps the unpaired Newcombe as primary precisely so the headline does not depend on
    an alignment this function has to verify first.
    """
    if not pairs:
        return None
    rng = random.Random(seed)
    n = len(pairs)
    d0 = sum(a - b for a, b in pairs) / n
    out = []
    for _ in range(draws):
        s = 0
        for _ in range(n):
            a, b = pairs[rng.randrange(n)]
            s += a - b
        out.append(s / n)
    out.sort()
    return {"delta": round(d0, 4), "boot95": [round(out[int(0.025 * draws)], 4),
                                              round(out[int(0.975 * draws) - 1], 4)],
            "n_pairs": n}


def load(out_dir: Path):
    cells = defaultdict(list)
    for d in sorted(out_dir.iterdir()):
        if not (d / "summary.json").exists():
            continue
        arm, opp, teamset, seed = d.name.split("_")
        s = json.loads((d / "summary.json").read_text())
        rec = {"tag": d.name, "arm": arm, "opponent": opp, "teamset": teamset,
               "team_seed": int(seed[1:]), "dir": str(d), "summary": s,
               "games": [json.loads(x) for x in (d / "games.jsonl").read_text().splitlines() if x]}
        rec["halves"] = {}
        for half, fn in (("ours_challenge", "peer_acceptor_report.json"),
                         ("peer_challenge", "peer_challenger_report.json")):
            f = d / half / fn
            if f.exists():
                r = json.loads(f.read_text())
                rec["halves"][half] = {
                    "n_decisions": r.get("n_decisions"),
                    "argmax_checked": r.get("argmax_checked"),
                    "argmax_matched": r.get("argmax_matched"),
                    "argmax_match_rate": r.get("argmax_match_rate"),
                    "sample_kwarg_values": r.get("sample_kwarg_values") or r.get("sample_kwargs"),
                    "regime_check_ok": r.get("regime_check_ok"),
                    "peer_error": r.get("error"),
                    "peer_results_present": bool(r.get("peer_results"))}
        cells[(opp, teamset, arm)].append(rec)
    return cells


def pool(recs):
    """Pool the sub-cells of ONE (opponent, team set, arm) cell.

    🚨 A sub-cell whose ``status`` is not OK is EXCLUDED from the pooled n and reported separately:
    a partial n must never be mistaken for a full one (SOP sec 5.1).
    """
    ok = [r for r in recs if r["summary"]["status"] == "OK"]
    bad = [r for r in recs if r["summary"]["status"] != "OK"]
    wins = sum(r["summary"]["wins"] for r in ok)
    n = sum(r["summary"]["n"] for r in ok)
    p, lo, hi = wilson(wins, n) if n else (None, None, None)
    per_decision = []
    composite = []
    for r in ok:
        composite.append(r["summary"]["provenance"]["peer_report"]["regime_verified"])
        for h in r["halves"].values():
            if h["argmax_match_rate"] is not None:
                per_decision.append(h["argmax_match_rate"])
    return {
        "n": n, "wins": wins, "losses": sum(r["summary"]["losses"] for r in ok),
        "ties": sum(r["summary"]["ties"] for r in ok),
        "win_rate": round(p, 4) if n else None,
        "wilson95": [round(lo, 4), round(hi, 4)] if n else None,
        "sub_cells": [{"seed": r["team_seed"], "n": r["summary"]["n"],
                       "wins": r["summary"]["wins"], "win_rate": r["summary"]["win_rate"],
                       "status": r["summary"]["status"],
                       "hit_forfeit_limit": r["summary"]["hit_forfeit_limit"],
                       "mean_turns": r["summary"]["mean_turns"],
                       "distinct_our_teams": r["summary"]["distinct_our_teams"],
                       "team_source_asymmetry": r["summary"]["team_source_asymmetry"],
                       "model_step": r["summary"]["cell"]["model_step"],
                       "model_rung": r["summary"]["cell"]["model_rung"],
                       "model_loader": r["summary"]["cell"]["model_loader"]} for r in recs],
        "excluded_sub_cells": [{"seed": r["team_seed"], "status": r["summary"]["status"],
                                "n": r["summary"]["n"], "wins": r["summary"]["wins"]} for r in bad],
        "by_half": {h: {"wins": sum(r["summary"]["by_half"][h]["wins"] for r in ok),
                        "n": sum(r["summary"]["by_half"][h]["n"] for r in ok)}
                    for h in ("ours_challenge", "peer_challenge")} if ok else {},
        "integrity": {
            # 🚨 THE TWO FLAGS ARE KEPT APART (hazard W-J). The composite is an AND over both
            # halves and goes false if a peer raises ANYWHERE -- including after its last decision.
            "composite_regime_verified": composite,
            "per_decision_argmax_rates": sorted(set(round(x, 4) for x in per_decision)),
            "per_decision_all_1_0000": all(abs(x - 1.0) < 1e-9 for x in per_decision) and bool(per_decision),
            "hit_forfeit_limit_total": sum(r["summary"]["hit_forfeit_limit"] for r in ok),
            "n_defaults_total": sum(r["summary"]["n_defaults"] for r in ok),
            "n_redecides_total": sum(r["summary"]["n_redecides"] for r in ok),
            "mean_turns": round(sum(r["summary"]["mean_turns"] * r["summary"]["n"] for r in ok) / n, 2) if n else None,
            "team_source_asymmetry_any": any(r["summary"]["team_source_asymmetry"] for r in ok),
            "peer_errors": [(r["team_seed"], h, v["peer_error"])
                            for r in recs for h, v in r["halves"].items() if v["peer_error"]],
            "timeout_rate": 0.0,
            "timeout_note": ("main.anchors has no timeout outcome: a stalled series is a NAMED "
                             "failure (no_progress / short_series) that voids the sub-cell, so the "
                             "registered >25% INCONCLUSIVE rule reads on excluded_sub_cells."),
        },
    }


def match_pairs(recs_a, recs_b):
    """Align games across two arms by (team seed, half, index) and verify the draw actually matches.

    The alignment is only usable if the two arms drew the SAME team at the same slot; that is the
    property this checks rather than assumes, and the count of verified slots is reported.
    """
    def key(recs):
        out = {}
        for r in recs:
            if r["summary"]["status"] != "OK":
                continue
            for g in r["games"]:
                out[(r["team_seed"], g["half"], g["index"])] = (g["won"], g.get("our_team"))
        return out
    ka, kb = key(recs_a), key(recs_b)
    shared = sorted(set(ka) & set(kb))
    same_team = sum(1 for k in shared if ka[k][1] == kb[k][1])
    pairs = [(1.0 if ka[k][0] else 0.0, 1.0 if kb[k][0] else 0.0) for k in shared]
    return pairs, {"n_shared_slots": len(shared), "same_our_team": same_team,
                   "alignment_verified": bool(shared) and same_team == len(shared)}


def main() -> int:
    out_dir, dest = Path(sys.argv[1]), Path(sys.argv[2])
    cells = load(out_dir)
    keys = sorted({(o, t) for (o, t, _) in cells})
    doc = {"what": "A/B continuation vs its frozen parent vs the fold path, external anchor",
           "bar": ("PRIMARY: the 95% CI on the DIFFERENCE excludes zero (Newcombe). A CI that "
                   "straddles zero is NOT DETECTED, never 'equivalent'. SECONDARY (stricter): the "
                   "difference also clears the 0.090 run-level floor by both clauses."),
           "arms": ARM_LABEL, "floor": FLOOR, "cells": {}}
    for opp, ts in keys:
        cell = {"arms": {}, "contrasts": {}}
        for a in ARMS:
            if (opp, ts, a) in cells:
                cell["arms"][a] = pool(cells[(opp, ts, a)])
        for hi, lo in CONTRASTS:
            if hi not in cell["arms"] or lo not in cell["arms"]:
                continue
            A, B = cell["arms"][hi], cell["arms"][lo]
            if not A["n"] or not B["n"]:
                continue
            d, l, u = newcombe(A["wins"], A["n"], B["wins"], B["n"])
            pairs, align = match_pairs(cells[(opp, ts, hi)], cells[(opp, ts, lo)])
            boot = paired_bootstrap(pairs) if align["alignment_verified"] else None
            inside_floor = (l <= FLOOR <= u) or (l <= -FLOOR <= u)
            cell["contrasts"][f"{hi}-{lo}"] = {
                "delta": round(d, 4), "newcombe95": [round(l, 4), round(u, 4)],
                "n_hi": A["n"], "n_lo": B["n"],
                "PRIMARY_ci_excludes_zero": bool(l > 0 or u < 0),
                "primary_verdict": "DETECTED" if (l > 0 or u < 0) else "NOT DETECTED",
                "secondary_floor_clause_a_abs_gt_floor": bool(abs(d) > FLOOR),
                "secondary_floor_clause_b_ci_excludes_floor_point": bool(not inside_floor),
                "secondary_verdict": ("OUTSIDE THE FLOOR" if (abs(d) > FLOOR and not inside_floor)
                                      else "WITHIN FLOOR"),
                "paired_descriptor": boot, "pair_alignment": align}
        doc["cells"][f"{opp}|{ts}"] = cell
    dest.write_text(json.dumps(doc, indent=1))
    print(json.dumps(doc, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
