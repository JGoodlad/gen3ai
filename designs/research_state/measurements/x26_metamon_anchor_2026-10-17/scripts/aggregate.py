"""Pool the done units of xq.py into rows.jsonl + cells.json (next to this script's parent).

A unit counts iff status OK, n == unit games, regime_verified_decisions true and
team_source_asymmetry false (the X22 / N0-read rule). Every other unit is listed, not counted.
History comparisons are independent-sample Newcombe intervals (`main.anchors.results.newcombe`).

    python aggregate.py        (run from the worktree's src/ so main.anchors imports)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "src"))
from main.anchors.results import newcombe, wilson  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import xq  # noqa: E402

OUT = Path(__file__).resolve().parents[1]

# (cell, label, wins, n, source) — earlier models of ours against the SAME opponent / regime / team set.
HISTORY = {
    "K_greedy_h72": [("N0 ai_v14_01_base @75,005,952", 92, 200, "kakuna_goal_read_2026-09-28 (b)")],
    "K_t10_h72": [("N0 ai_v14_01_base @75,005,952", 76, 200, "kakuna_goal_read_2026-09-28 (c)")],
    "S_greedy_away": [
        ("K2 ai_v14_07_g0p_k2 @91,127,808 (G0', the v14 plateau)", 375, 600, "learner_battery_2026-09-26 ga_levels"),
        ("C_fix ai_v14_06_lbat_ctrl_fix", 395, 600, "learner_battery_2026-09-26 ga_levels"),
        ("ai_v12_02_winprob_critic @75,005,952", 65, 100, "metamon_matched_regime_2026-09-14"),
    ],
    "S_greedy_home": [
        ("K2 ai_v14_07_g0p_k2 @91,127,808 (G0', the v14 plateau)", 453, 600, "learner_battery_2026-09-26 ga_levels"),
        ("C_fix ai_v14_06_lbat_ctrl_fix", 430, 600, "learner_battery_2026-09-26 ga_levels"),
        ("ai_v12_02_winprob_critic @75,005,952", 52, 100, "metamon_matched_regime_2026-09-14"),
    ],
    "S_t1_away": [("ai_v12_02_winprob_critic @75,005,952", 56, 100, "metamon_matched_regime_2026-09-14")],
    "K_greedy_away": [],
}


def main():
    rows, cells, skipped = [], {}, []
    for name, cell, seed in xq.units():
        d = xq.STATE / "cells" / name
        sp = d / "summary.json"
        if not sp.exists():
            continue
        s = json.loads(sp.read_text())
        ok = (s.get("status") == "OK" and s.get("n") == cell[5]
              and s.get("regime_verified_decisions") is True
              and s.get("team_source_asymmetry") is False)
        if not ok:
            skipped.append({"unit": name, "status": s.get("status"), "n": s.get("n")})
            continue
        c = cells.setdefault(cell[0], {"units": 0, "wins": 0, "losses": 0, "ties": 0, "n": 0,
                                       "hit_forfeit_limit": 0, "their_argmax": [],
                                       "peer_dirty_units": [], "turns": [], "teams": set()})
        c["units"] += 1
        for k in ("wins", "losses", "ties", "n", "hit_forfeit_limit"):
            c[k] += s[k]
        c["their_argmax"] += s.get("their_argmax_match_rates") or []
        if not s.get("peer_clean"):
            c["peer_dirty_units"].append({"unit": name, "notes": s.get("peer_exit_notes")})
        for line in (d / "games.jsonl").read_text().splitlines():
            r = json.loads(line)
            r.update({"x26_cell": cell[0], "unit": name, "unit_seed": seed})
            rows.append(r)
            c["turns"].append(r["turns"])
            c["teams"].add(r["our_team"])
    out = {}
    for k, c in cells.items():
        p, lo, hi = wilson(c["wins"], c["n"])
        t = sorted(c["turns"])
        hist = []
        for label, w, n, src in HISTORY.get(k, []):
            d, dlo, dhi = newcombe(c["wins"], c["n"], w, n)
            hist.append({"model": label, "wins": w, "n": n, "rate": w / n, "source": src,
                         "x26_minus_it": [d, dlo, dhi],
                         "verdict": "X26 BETTER" if dlo > 0 else ("X26 WORSE" if dhi < 0 else "NOT DETECTED")})
        out[k] = {
            "units": c["units"], "n": c["n"], "wins": c["wins"], "losses": c["losses"],
            "ties": c["ties"], "win_rate": p, "wilson95": [lo, hi],
            "vs_0.50": "BETTER" if lo > 0.5 else ("WORSE" if hi < 0.5 else "NOT DETECTED"),
            "hit_forfeit_limit": c["hit_forfeit_limit"],
            "their_argmax_min": min(c["their_argmax"]) if c["their_argmax"] else None,
            "their_argmax_max": max(c["their_argmax"]) if c["their_argmax"] else None,
            "peer_dirty_units": c["peer_dirty_units"], "median_turns": t[len(t) // 2],
            "max_turns": t[-1], "distinct_our_teams": len(c["teams"]), "history": hist,
        }
    with open(OUT / "rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (OUT / "cells.json").write_text(json.dumps({"cells": out, "skipped_units": skipped}, indent=1) + "\n")
    for k, v in out.items():
        print(f"{k:15s} {v['wins']}-{v['losses']}-{v['ties']} n={v['n']} {v['win_rate']:.3f} "
              f"[{v['wilson95'][0]:.3f}, {v['wilson95'][1]:.3f}] {v['vs_0.50']}")
        for h in v["history"]:
            d = h["x26_minus_it"]
            print(f"      vs {h['model']}: {h['rate']:.3f} -> {d[0]:+.3f} [{d[1]:+.3f}, {d[2]:+.3f}] {h['verdict']}")
    print("skipped:", skipped)


if __name__ == "__main__":
    main()
