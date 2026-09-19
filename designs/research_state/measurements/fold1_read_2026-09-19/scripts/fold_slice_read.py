#!/usr/bin/env python3
"""ROW 2 — PER-SLICE PILOTING, the MATCHED-EXTRACTION ROW.

Every ref pilots the SAME pinned taught team against the SAME fixed opponent
(``untaught_meter_opponent`` = ``ai_v9_29_rev1_0823@24,000,000``) drawing the SAME 800-long
pool-team sequence under the SAME dice, so a ref-vs-ref difference is a PILOTING difference on the
same games and not a head-to-head.

**MATCHED-EXTRACTION DISCIPLINE** (`feedback-matched-extraction-row`, the era standard):

* both arms on ONE harness — never a run's own eval win rate against an h2h number;
* the baseline is MATCHED and ZERO-HEAD-START — **arm W itself**, same role, same opponent, same
  pinned team;
* the draws are **PAIRED** by construction (CRN on the battle index);
* **per-cell first, with Wilson intervals**; a pooled number is secondary and labelled;
* 🚨 **seniority is a SEPARATE term from extraction.** The fold carries 6M steps arm W does not and
  each teacher carries 8M, so ``fold − W = seniority + extraction`` and this row cannot split them.
  The nearest bound available on the seniority term is the untaught row — the same 6M of ordinary
  progress read OFF the slice.

⚠️ **THE INTERVAL ON A DIFFERENCE IS NEWCOMBE's, WHICH IS CONSERVATIVE HERE.** The games are paired
under CRN but ``main.untaught_meter`` retains only per-cell counts (no per-battle outcome vector),
so the pairing cannot be exploited and the two-proportion interval is wider than the paired one
would be. A cluster bootstrap over TWO teams is not a CI and none is printed.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

Z = 1.959963984540054


def wilson(k: int, n: int):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + Z * Z / n
    c = p + Z * Z / (2 * n)
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return p, (c - h) / d, (c + h) / d


def newcombe(k1: int, n1: int, k2: int, n2: int):
    """Newcombe's method 10 for a difference of two independent proportions (CONSERVATIVE here)."""
    p1, l1, u1 = wilson(k1, n1)
    p2, l2, u2 = wilson(k2, n2)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi


TEAM_LABEL = {
    "U_f6229d2c": "Big-5 (BALANCE) — t1 `ai_v13_05_exploit_big5starmie`'s pinned team "
                  "data/teams/sample/f6229d2c867e21d6.txt",
    "U_9eb3abdc": "DDTar/Spikes (OFFENSE) — t2 `ai_v13_06_exploit_ddtar_spikes`'s pinned team "
                  "data/teams/sample/9eb3abdc52876a63.txt",
}
TEACHER_OF = {"U_f6229d2c": "t1_big5", "U_9eb3abdc": "t2_ddtar"}


def main() -> int:
    shards = [Path(p) for p in sys.argv[1:-1]]
    dest = Path(sys.argv[-1])
    cells: dict = {}
    meta: dict = {}
    for p in shards:
        raw = json.loads(p.read_text())
        d = raw["result"]
        meta.setdefault("teams", d["teams"])
        meta.setdefault("teams_manifest", raw["_meta"].get("teams_manifest"))
        meta.setdefault("games_per_team", raw["_meta"].get("games_per_team"))
        meta.setdefault("team_pins", {t["key"]: t["pin_sha"] for t in raw["_meta"]["teams"]})
        meta.setdefault("opponent_resolved", raw["_meta"]["opponent"]["resolved_file"])
        for ref, lv in d["levels"].items():
            for t, c in lv["per_team"].items():
                cells[(ref, t)] = c
        meta.setdefault("timeouts", []).append({str(p.name): d.get("timeouts")})
    teams = meta["teams"]

    out: dict = {
        "what": "registered row 2 — per-slice piloting on the two TAUGHT teams (matched extraction)",
        "opponent": ("registry name `untaught_meter_opponent` = ai_v9_29_rev1_0823@24,000,000 — ONE "
                     "frozen era-external checkpoint, the same for every ref. NOT arm W's pool "
                     "sentinels: a sentinel is the trainee's OWN snapshot, so arm W's and W_b's "
                     "differ (floor read hazard F-G) and the seed floor would then face different "
                     "opponents from the treatment contrast."),
        "games_per_cell": meta["games_per_team"], "teams": teams,
        "team_pin_sha": meta["team_pins"], "opponent_resolved": meta["opponent_resolved"],
        "team_labels": TEAM_LABEL, "shard_timeouts": meta["timeouts"],
        "interval_note": ("Wilson per cell; Newcombe on every difference. CONSERVATIVE under CRN — "
                          "the games are paired and the tool retains no per-battle outcomes."),
        "seniority_caveat": ("fold − W = seniority + extraction and this row cannot split them: the "
                             "fold has +6,094,848 steps arm W does not, each teacher +8,060,928."),
        "levels": {}, "per_team": {},
    }

    for t in teams:
        rows = {}
        for ref in sorted({r for (r, tt) in cells if tt == t}):
            c = cells[(ref, t)]
            p, lo, hi = wilson(c["wins"], c["finished"])
            rows[ref] = {"wins": c["wins"], "finished": c["finished"],
                         "attempted": c["attempted"], "timeouts": c["timeouts"],
                         "win_rate": round(p, 4), "wilson95": [round(lo, 4), round(hi, 4)]}
        out["levels"][t] = rows

        rec: dict = {"label": TEAM_LABEL.get(t, t), "contrasts": {}}
        def diff(a, b, label):
            if (a, t) not in cells or (b, t) not in cells:
                return None
            ca, cb = cells[(a, t)], cells[(b, t)]
            d, lo, hi = newcombe(ca["wins"], ca["finished"], cb["wins"], cb["finished"])
            return {"label": label, "delta": round(d, 4),
                    "newcombe95": [round(lo, 4), round(hi, 4)]}

        rec["contrasts"]["fold_p6M_minus_armW"] = diff(
            "fold_p6M", "armW", "THE FINDING at +6M — fold minus the zero-head-start PARENT")
        rec["contrasts"]["fold_p3M_minus_armW"] = diff(
            "fold_p3M", "armW", "THE FINDING at +3M")
        rec["contrasts"]["THE_FLOOR_armW_minus_armWb"] = diff(
            "armW", "armWb", "🚨 THE SEED FLOOR ON THIS CELL — arm W minus W_b, same team, same "
                             "opponent, same games")
        teach = TEACHER_OF[t]
        rec["contrasts"]["teacher_minus_fold_p6M"] = diff(
            teach, "fold_p6M", f"THE TEACHER'S CEILING — {teach} on its OWN team minus the fold")
        rec["contrasts"]["teacher_minus_armW"] = diff(
            teach, "armW", f"{teach} minus the parent it was forked from — the whole gap the fold "
                           f"was asked to close")
        other = "t2_ddtar" if teach == "t1_big5" else "t1_big5"
        rec["contrasts"]["other_teacher_minus_armW"] = diff(
            other, "armW", f"DESCRIPTOR — the OTHER teacher ({other}) on a team it never trained on")

        fl = rec["contrasts"]["THE_FLOOR_armW_minus_armWb"]
        for key in ("fold_p6M_minus_armW", "fold_p3M_minus_armW"):
            f, g = fl, rec["contrasts"][key]
            if not f or not g:
                continue
            floor = abs(f["delta"])
            lo, hi = g["newcombe95"]
            inside = (lo <= floor <= hi) or (lo <= -floor <= hi)
            rec.setdefault("verdicts", {})[key] = {
                "delta": g["delta"], "abs_delta": round(abs(g["delta"]), 4),
                "newcombe95": [lo, hi], "floor_on_this_cell": round(floor, 4),
                "clause_a_abs_delta_gt_floor": bool(abs(g["delta"]) > floor),
                "clause_b_ci_excludes_floor_point": bool(not inside),
                "verdict": ("OUTSIDE THE FLOOR" if (abs(g["delta"]) > floor and not inside)
                            else "WITHIN FLOOR at n = 2"),
                "direction": "fold ABOVE parent" if g["delta"] > 0 else "fold BELOW parent"}
        out["per_team"][t] = rec

    # --- THE DIVERGENCE: did the Big-5 slice gain more than the DDTar slice? -------------------
    a, b = "U_f6229d2c", "U_9eb3abdc"
    ga = out["per_team"][a]["contrasts"]["fold_p6M_minus_armW"]
    gb = out["per_team"][b]["contrasts"]["fold_p6M_minus_armW"]
    if ga and gb:
        out["the_divergence"] = {
            "question": ("the completion entry recorded the two teachers' gated shares diverging "
                         "2.5x (t1 Big-5 0.226 -> 0.305, t2 DDTar 0.210 -> 0.122). Registered: does "
                         "the BIG-5 slice gain more than the DDTAR slice?"),
            "big5_gain": ga["delta"], "ddtar_gain": gb["delta"],
            "big5_minus_ddtar": round(ga["delta"] - gb["delta"], 4),
            "big5_gained_more": bool(ga["delta"] > gb["delta"]),
            "note": ("a difference of two independent differences on 800 games each; no interval is "
                     "printed for it because the four cells are not a factorial design and an "
                     "interval here would invite a verdict the design cannot carry. DESCRIPTOR.")}

    # --- pooled, SECONDARY and labelled --------------------------------------------------------
    pooled = {}
    for ref in sorted({r for (r, _t) in cells}):
        w = sum(cells[(ref, t)]["wins"] for t in teams if (ref, t) in cells)
        n = sum(cells[(ref, t)]["finished"] for t in teams if (ref, t) in cells)
        if n:
            p, lo, hi = wilson(w, n)
            pooled[ref] = {"wins": w, "finished": n, "win_rate": round(p, 4),
                           "wilson95": [round(lo, 4), round(hi, 4)]}
    out["pooled_SECONDARY"] = {
        "levels": pooled,
        "warning": ("POOLED ACROSS TEAMS — secondary by the matched-extraction rule and by rule 10. "
                    "A pooled rate and a team-clustered one have disagreed in SIGN on this "
                    "programme's own data. Read the per-team rows above.")}

    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
