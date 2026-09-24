#!/usr/bin/env python3
"""G-A aggregate — the external-anchor KILL guard (registration §4.3), read at the REGISTERED n only.

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

MODELS = ["B_loop", "C_ctrl", "G0_plateau"]
SEEDS = [0, 10, 20, 30, 40, 50]
TEAMSETS = ["away", "home"]
FLOOR = 0.110


def main():
    rows, out_path = Path(sys.argv[1]), Path(sys.argv[2])
    progress = "--progress" in sys.argv
    units, problems, missing = {}, [], []
    for m in MODELS:
        for ts in TEAMSETS:
            for s in SEEDS:
                u = f"{m}_{ts}_s{s}"
                f = rows / u / "summary.json"
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
        print(f"units on disk {len(units)}/36, missing {len(missing)}, problems {problems} — PROGRESS ONLY")
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
    for m in MODELS:
        w, n = pool(m)
        lv = wilson(w, n)
        levels[m] = {"wins": w, "n": n, "rate": w / n, "wilson95": [lv[1], lv[2]],
                     "by_teamset": {ts: dict(zip(("wins", "n"), pool(m, ts))) for ts in TEAMSETS}}

    def contrast(a, b, ts=None):
        wa, na = pool(a, ts)
        wb, nb = pool(b, ts)
        d, lo, hi = newcombe(wa, na, wb, nb)
        return {"delta": d, "ci95": [lo, hi], "n": [na, nb]}

    bc = contrast("B_loop", "C_ctrl")
    d, (lo, hi) = bc["delta"], bc["ci95"]
    bc.update({"floor": FLOOR, "clause_a_abs_gt_floor": abs(d) > FLOOR,
               "ci_excludes_minus_floor": hi < -FLOOR,
               "OUTSIDE_BELOW": abs(d) > FLOOR and d < 0 and hi < -FLOOR})
    doc = {"what": "G-A, the external-anchor KILL guard (registration §4.3)",
           "regime": "greedy-vs-greedy, metamon:SmallRL (ckpt 40), --server rust, 100-game units",
           "tree": "7c511161", "units": units, "problems": problems, "levels": levels,
           "B_minus_C": bc,
           "B_minus_C_by_teamset (descriptor)": {ts: contrast("B_loop", "C_ctrl", ts) for ts in TEAMSETS},
           "C_minus_G0 (descriptor)": contrast("C_ctrl", "G0_plateau"),
           "B_minus_G0 (descriptor)": contrast("B_loop", "G0_plateau"),
           "VOID": bool(problems), "KILL": bool(not problems and bc["OUTSIDE_BELOW"])}
    out_path.write_text(json.dumps(doc, indent=1))
    print(json.dumps({k: v for k, v in doc.items() if k != "units"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
