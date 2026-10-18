"""Pool kq2.py's done units (+ the X26 read's banked cells) into rows.jsonl + cells.json.

A unit COUNTS iff status OK, n == 20, regime_verified_decisions true, team_source_asymmetry
false AND peer_clean true (the brief's rule; stricter than the X26 read's, which only listed
dirty peers). Every other unit is listed under `skipped_units`, never counted.

Cells:
  K2_greedy_h72    K2 vs Kakuna greedy, home72, seeds 0-11 (this read)
  K2_greedy_away   K2 vs Kakuna greedy, away, seeds 0-19 (this read)
  X26_greedy_away  X26 vs Kakuna greedy, away: seeds 0-4 from x26_metamon_anchor_2026-10-17
                   (K_greedy_away_00..04) + seeds 5-19 from this read, pooled ONLY if every row's
                   regime stamp (STAMP below) is identical across the two reads.
  X26_greedy_h72   the X26 read's K_greedy_h72 (seeds 0-11), unchanged, for the K2 comparison.

Differences X26 - K2 per team set are independent-sample Newcombe intervals
(`main.anchors.results.newcombe`), as the SOP's history columns are.

    PYTHONPATH=<worktree>/src python aggregate.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[4] / "src"))
from main.anchors.results import newcombe, wilson  # noqa: E402

sys.path.insert(0, str(HERE))
import kq2  # noqa: E402

OUT = HERE.parent
X26_STATE = Path("/home/goodlad/dev/gen3ai-reads/x26_metamon_anchor_2026-10-17")
GAMES = 20
# The regime stamp: every field that names WHAT was played, by whom, under which protocol.
STAMP = ("opponent", "opponent_version", "opponent_commit", "our_regime", "their_regime",
         "regime_matched", "teamset", "our_team_source", "our_team_count", "their_team_count",
         "battle_format", "model_zip", "model_step", "model_loader", "forfeit_turn_limit",
         "showdown_pin", "server_impl", "server_version", "our_side", "our_transport",
         "mirrored_pairs")

SOURCES = []  # (cell, unit, seed, unit dir, read)
for unit, cell, seed in kq2.units():
    SOURCES.append((cell, unit, seed, kq2.STATE / "cells" / unit, "k2_x26_kakuna_both_teamsets_2026-10-17"))
for i in range(5):
    SOURCES.append(("X26_greedy_away", f"K_greedy_away_{i:02d}", i, X26_STATE / "cells" / f"K_greedy_away_{i:02d}",
                    "x26_metamon_anchor_2026-10-17"))
for i in range(12):
    SOURCES.append(("X26_greedy_h72", f"K_greedy_h72_{i:02d}", i, X26_STATE / "cells" / f"K_greedy_h72_{i:02d}",
                    "x26_metamon_anchor_2026-10-17"))


def verdict(lo, hi):
    return "BETTER" if lo > 0.5 else ("WORSE" if hi < 0.5 else "NOT DETECTED")


def main():
    rows, cells, skipped = [], {}, []
    for cell, unit, seed, d, read in SOURCES:
        sp = d / "summary.json"
        if not sp.exists():
            continue
        s = json.loads(sp.read_text())
        ok = (s.get("status") == "OK" and s.get("n") == GAMES
              and s.get("regime_verified_decisions") is True
              and s.get("team_source_asymmetry") is False
              and s.get("peer_clean") is True)
        if not ok:
            skipped.append({"cell": cell, "unit": unit, "read": read, "status": s.get("status"),
                            "n": s.get("n"), "regime_verified_decisions": s.get("regime_verified_decisions"),
                            "team_source_asymmetry": s.get("team_source_asymmetry"),
                            "peer_clean": s.get("peer_clean"), "peer_exit_notes": s.get("peer_exit_notes")})
            continue
        c = cells.setdefault(cell, {"units": [], "seeds": [], "wins": 0, "losses": 0, "ties": 0, "n": 0,
                                    "hit_forfeit_limit": 0, "their_argmax": [], "turns": [], "teams": set(),
                                    "stamps": {}, "reads": set(), "by_half": {}})
        c["units"].append(unit)
        c["seeds"].append(seed)
        c["reads"].add(read)
        for k in ("wins", "losses", "ties", "n", "hit_forfeit_limit"):
            c[k] += s[k]
        c["their_argmax"] += s.get("their_argmax_match_rates") or []
        for line in (d / "games.jsonl").read_text().splitlines():
            r = json.loads(line)
            r.update({"read_cell": cell, "unit": unit, "unit_seed": seed, "source_read": read})
            rows.append(r)
            c["turns"].append(r["turns"])
            c["teams"].add(r["our_team"])
            h = c["by_half"].setdefault(r["half"], [0, 0])
            h[0] += bool(r["won"])
            h[1] += 1
            st = json.dumps({k: r.get(k) for k in STAMP}, sort_keys=True)
            c["stamps"][st] = c["stamps"].get(st, 0) + 1
    out = {}
    for k, c in cells.items():
        p, lo, hi = wilson(c["wins"], c["n"])
        t = sorted(c["turns"])
        out[k] = {
            "units": len(c["units"]), "seeds": sorted(c["seeds"]), "reads": sorted(c["reads"]),
            "n": c["n"], "wins": c["wins"], "losses": c["losses"], "ties": c["ties"],
            "win_rate": p, "wilson95": [lo, hi], "vs_0.50": verdict(lo, hi),
            "by_half": {h: {"wins": w, "n": n, "rate": w / n} for h, (w, n) in sorted(c["by_half"].items())},
            "hit_forfeit_limit": c["hit_forfeit_limit"],
            "their_argmax_min": min(c["their_argmax"]) if c["their_argmax"] else None,
            "their_argmax_max": max(c["their_argmax"]) if c["their_argmax"] else None,
            "median_turns": t[len(t) // 2] if t else None, "max_turns": t[-1] if t else None,
            "distinct_our_teams": len(c["teams"]),
            "regime_stamps": [{"rows": n, **json.loads(st)} for st, n in c["stamps"].items()],
            "stamp_identical": len(c["stamps"]) == 1,
        }
    diffs = {}
    for ts, (x, k) in {"home72": ("X26_greedy_h72", "K2_greedy_h72"),
                       "away": ("X26_greedy_away", "K2_greedy_away")}.items():
        if x in out and k in out:
            d, dlo, dhi = newcombe(out[x]["wins"], out[x]["n"], out[k]["wins"], out[k]["n"])
            diffs[ts] = {"x26": [out[x]["wins"], out[x]["n"]], "k2": [out[k]["wins"], out[k]["n"]],
                         "x26_minus_k2": [d, dlo, dhi],
                         "verdict": "X26 BETTER" if dlo > 0 else ("X26 WORSE" if dhi < 0 else "NOT DETECTED")}
    (OUT / "cells.json").write_text(json.dumps({"cells": out, "x26_minus_k2": diffs,
                                                "skipped_units": skipped}, indent=1) + "\n")
    with open(OUT / "rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    for k, v in out.items():
        print(f"{k:16s} units {v['units']:2d} n {v['n']:3d} W-L-T {v['wins']}-{v['losses']}-{v['ties']} "
              f"{v['win_rate']:.3f} [{v['wilson95'][0]:.3f}, {v['wilson95'][1]:.3f}] {v['vs_0.50']} "
              f"stamp_identical={v['stamp_identical']}")
    for ts, v in diffs.items():
        d, lo, hi = v["x26_minus_k2"]
        print(f"X26 - K2 {ts:7s} {d:+.3f} [{lo:+.3f}, {hi:+.3f}] {v['verdict']}")
    print(f"skipped units: {len(skipped)}")
    for s in skipped:
        print("  ", s)


if __name__ == "__main__":
    main()
