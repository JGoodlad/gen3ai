#!/usr/bin/env python3
"""G-A aggregate — ROUND 2's external-anchor KILL guard (round-2 registration §4.3), REGISTERED n only.

Round 1's aggregate with the guard moved to B2 − C2 (B2 = ai_v13_27_popr2_loop, the round-2 loop
generalist; C2 = ai_v13_28_popr2_ctrl, the round-2 no-exploiter control) and the descriptors B2 − B,
C2 − C, B2 − G0, C2 − G0 read from ROUND 1's BANKED units (same tree 7c511161, same seeds):
``../../../population_loop_r1_2026-09-23/read/ga_rows``. Usage: ga_aggregate.py <ga_rows> <out.json>
[--progress]. Round 1's docstring follows (read B as B2 and C as C2 for the guard).

Pools each model's 12 × 100-game units (6 team seeds × {away, home}) = 1,200 greedy-vs-greedy games
against metamon:SmallRL, and applies the registered rule to B − C (B = ai_v13_22_popr1_loop, the loop
generalist; C = ai_v13_23_popr1_ctrl, the no-exploiter control): **KILL iff B − C (pooled 1,200,
Newcombe 95 %) is OUTSIDE and BELOW the 0.110 floor** (|Δ| > 0.110 AND the CI excludes −0.110).
Every unit must be status OK, n = 100, and regime_verified_decisions true, or the cell is VOID.
Descriptors: C − G0, B − G0 (G0 = ai_v13_12_plateau), and each teamset separately (never folded
into the verdict).

Run with PYTHONPATH containing src (Newcombe/Wilson are the anchors tool's own).
Usage: ga_aggregate.py <ga_rows dir> <out.json> [--progress]
"""
import json
import sys
from pathlib import Path

from main.anchors.results import newcombe, wilson

MODELS = ["B2_loop", "C2_ctrl"]
R1_MODELS = ["B_loop", "C_ctrl", "G0_plateau"]
R1_ROWS = Path(__file__).resolve().parents[3] / "population_loop_r1_2026-09-23" / "read" / "ga_rows"
SEEDS = [0, 10, 20, 30, 40, 50]
TEAMSETS = ["away", "home"]
FLOOR = 0.110


def main():
    rows, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    progress = "--progress" in sys.argv
    units, problems, missing = {}, [], []
    for base, m in [(rows, m) for m in MODELS] + [(R1_ROWS, m) for m in R1_MODELS]:
        for ts in TEAMSETS:
            for s in SEEDS:
                u = f"{m}_{ts}_s{s}"
                f = base / u / "summary.json"
                if not f.exists():
                    missing.append(u)
                    continue
                d = json.loads(f.read_text())
                pr = d.get("provenance", {}).get("peer_report", {})
                rec = {"n": d["n"], "wins": d["wins"], "losses": d["losses"], "ties": d["ties"],
                       "status": d["status"], "hit_forfeit_limit": d.get("hit_forfeit_limit"),
                       "regime_verified_decisions": pr.get("series_regime_verified_decisions",
                                                           pr.get("regime_verified_decisions")),
                       "argmax_match_rate": pr.get("argmax_match_rate"),
                       "peer_clean": pr.get("series_peer_clean", pr.get("peer_clean")),
                       "model_loader": pr.get("model_loader"),
                       "team_source_asymmetry": d.get("team_source_asymmetry")}
                units[u] = rec
                if rec["status"] != "OK" or rec["n"] != 100 or rec["regime_verified_decisions"] is not True \
                        or rec["team_source_asymmetry"]:
                    problems.append(u)
    if progress:
        print(f"units on disk {len(units)}/60 (24 new + 36 banked), missing {len(missing)}, problems {problems} — PROGRESS ONLY")
        return 0
    if missing:
        print(f"REFUSING: {len(missing)} unit(s) missing — the verdict is read at the registered n only")
        return 2

    def pool(m, ts=None):
        w = n = 0
        for u, r in units.items():
            if u.startswith(m + "_") and (ts is None or f"_{ts}_" in u):
                w += r["wins"]
                n += r["n"]
        return w, n

    levels = {}
    for m in MODELS + R1_MODELS:
        w, n = pool(m)
        lv = wilson(w, n)
        levels[m] = {"wins": w, "n": n, "rate": w / n, "wilson95": [lv[1], lv[2]],
                     "by_teamset": {ts: dict(zip(("wins", "n"), pool(m, ts))) for ts in TEAMSETS}}

    def contrast(a, b, ts=None):
        wa, na = pool(a, ts)
        wb, nb = pool(b, ts)
        d, lo, hi = newcombe(wa, na, wb, nb)
        return {"delta": d, "ci95": [lo, hi], "n": [na, nb]}

    bc = contrast("B2_loop", "C2_ctrl")
    d, (_lo, hi) = bc["delta"], bc["ci95"]
    bc.update({"floor": FLOOR, "clause_a_abs_gt_floor": abs(d) > FLOOR,
               "ci_excludes_minus_floor": hi < -FLOOR,
               "OUTSIDE_BELOW": abs(d) > FLOOR and d < 0 and hi < -FLOOR})
    doc = {"what": "G-A, round 2's external-anchor KILL guard (round-2 registration §4.3)",
           "regime": "greedy-vs-greedy, metamon:SmallRL (ckpt 40), --server rust, 100-game units",
           "tree": "7c511161", "units": units, "problems": problems, "levels": levels,
           "B2_minus_C2": bc,
           "B2_minus_C2_by_teamset (descriptor)": {ts: contrast("B2_loop", "C2_ctrl", ts) for ts in TEAMSETS},
           "B2_minus_B (descriptor, banked B)": contrast("B2_loop", "B_loop"),
           "C2_minus_C (descriptor, banked C)": contrast("C2_ctrl", "C_ctrl"),
           "B2_minus_G0 (descriptor, banked G0)": contrast("B2_loop", "G0_plateau"),
           "C2_minus_G0 (descriptor, banked G0)": contrast("C2_ctrl", "G0_plateau"),
           "VOID": bool(problems), "KILL": bool(not problems and bc["OUTSIDE_BELOW"])}
    out_path.write_text(json.dumps(doc, indent=1))
    print(json.dumps({k: v for k, v in doc.items() if k != "units"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
