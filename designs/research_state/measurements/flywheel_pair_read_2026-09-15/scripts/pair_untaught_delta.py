#!/usr/bin/env python3
"""THE PAIR READ — the UNTAUGHT METER, paired over TEAMS.

Same tool, same registry OPPONENT (`untaught_meter_opponent` = `ai_v9_29_rev1_0823@24,000,000`),
same seed 0 and concurrency 1 as arm S's 2026-09-14 read; both arms measured in ONE invocation so
the two refs see the identical 8 teams and the identical 200 games each.

``main.untaught_meter`` publishes a level and a CI per ref, and a delta only against a
``--baseline`` (which neither of these is), so the paired contrast is taken here off the meter's own
per-team rows on ONE resampled team index set — rule 10, the team is the unit, and the paired draw
removes the team-difficulty component the two refs share.

The 0.02 leg joins in as a THIRD leg from arm S's banked run of the same deterministic meter
(seed 0, concurrency 1, same 8 teams, same opponent). Arm S appears in BOTH files, so its level is
its own reproduction check: if the meter is deterministic the two readings agree to the last win.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

FLOORS_PP = {
    "frozen_dose_folds_six_draws": 1.66,
    "controller_live_folds": 4.27,
    "TC_replicate_pairs_END_depth": 1.19,
    "TC_replicate_pairs_p1M_depth": 4.00,
    "G5_three_continuation_arms": 1.00,
}
HERE = Path("/home/goodlad/.claude/jobs/9ab51de6/tmp/pair_read/untaught")
BANKED = Path("/home/goodlad/dev/gen3ai/designs/research_state/measurements/"
              "flywheel_armS_reads_2026-09-14/out/untaught")


def per_team(d: dict) -> tuple[list[str], dict[str, np.ndarray], dict[str, np.ndarray], dict]:
    res = d["result"]
    teams = res["teams"]
    lv = res["levels"]
    per = {r: np.array([lv[r]["per_team"][t]["win_rate"] for t in teams], float) for r in lv}
    fin = {r: np.array([lv[r]["per_team"][t]["finished"] for t in teams], float) for r in lv}
    return teams, per, fin, res


def paired(per_a: np.ndarray, per_b: np.ndarray, seed: int = 20260915) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(per_a)
    idx = rng.integers(0, n, size=(20000, n))           # ONE index set, shared by both refs
    dd = per_a[idx].mean(axis=1) - per_b[idx].mean(axis=1)
    point = float(per_a.mean() - per_b.mean())
    lo, hi = (float(x) for x in np.percentile(dd, [2.5, 97.5]))
    return point, lo, hi


def verdict(point_pp: float, lo_pp: float, hi_pp: float) -> str:
    ci_clear = not (lo_pp <= 0 <= hi_pp)
    clears = abs(point_pp) > FLOORS_PP["TC_replicate_pairs_END_depth"]
    if clears and ci_clear:
        return ("DETECTED at 95% and past the 1.19pp END-depth replicate floor — but the meter's "
                "axes and floors are established at ~1M fold depths and NEITHER arm is a fold, so "
                "this is a DESCRIPTOR, not an endpoint (registration sec 8.5); at n=1 per arm rule "
                "22 makes it a run-level CANDIDATE")
    if not ci_clear:
        return "the CI covers zero — NOT DETECTED (never 'equal': rule 6)"
    return ("the CI clears zero but the magnitude is inside the 1.19pp END-depth floor — "
            "NOT DETECTED")


def main() -> int:
    out: dict = {"floors_pp_in_evidence": FLOORS_PP,
                 "floor_note": ("a floor is a property of the DEPTH and of the REGIME "
                                "(UNDERSTANDING sec 3.3); every floor here is a FOLD floor at ~1M "
                                "depth and neither arm is a fold — they are quoted, not applied "
                                "as a bar"),
                 "configs": {}}
    for tag in ("registry", "auto"):
        p = HERE / f"untaught_{tag}.json"
        if not p.exists():
            out["configs"][tag] = {"error": f"no {p}"}
            continue
        teams, per, fin, res = per_team(json.loads(p.read_text()))
        rec: dict = {
            "config": ("the REGISTRY `untaught_meter_config` (ai_v9_29_rev1_0823's "
                       "snapshots/model_config.json)" if tag == "registry"
                       else "--config auto (each model's OWN model_config.json)"),
            "opponent": "registry name `untaught_meter_opponent` = ai_v9_29_rev1_0823@24,000,000",
            "levels_pp": {r: round(100 * float(per[r].mean()), 2) for r in per},
            "n_finished": {r: int(fin[r].sum()) for r in fin},
            "per_team_pp": {r: {t: round(100 * v, 2) for t, v in zip(teams, per[r])} for r in per},
            "n_teams": len(teams),
            "timeouts": res.get("timeouts"),
            "contrasts": {},
        }
        if "armS" in per and "armW" in per:
            pt, lo, hi = paired(per["armS"], per["armW"])
            rec["contrasts"]["armS_minus_armW"] = {
                "label": "THE PAIR — arm S (shaped) minus arm W (win-prob), both measured in ONE "
                         "invocation on the identical teams and games",
                "delta_pp": round(100 * pt, 2), "ci95_pp": [round(100 * lo, 2), round(100 * hi, 2)],
                "teams_where_armS_higher": int((per["armS"] > per["armW"]).sum()),
                "verdict": verdict(100 * pt, 100 * lo, 100 * hi)}

        # The free third leg, joined from arm S's banked run of the same deterministic meter.
        b = BANKED / f"untaught_{tag}.json"
        if b.exists() and "armW" in per:
            bteams, bper, _bfin, _ = per_team(json.loads(b.read_text()))
            if bteams == teams and "winprob75M" in bper:
                rec["third_leg"] = {
                    "source": str(b.relative_to(Path('/home/goodlad/dev/gen3ai'))),
                    "join": ("the SAME deterministic meter (seed 0, concurrency 1, identical 8 "
                             "teams, identical registry opponent) run on 2026-09-14; joined here "
                             "team by team"),
                    "winprob75M_level_pp": round(100 * float(bper["winprob75M"].mean()), 2),
                    "armS_level_pp_banked": round(100 * float(bper["armS"].mean()), 2),
                    "armS_level_pp_here": round(100 * float(per["armS"].mean()), 2),
                    "armS_reproduction_delta_pp": round(
                        100 * float(per["armS"].mean() - bper["armS"].mean()), 4),
                }
                pt, lo, hi = paired(bper["winprob75M"], per["armW"])
                rec["contrasts"]["winprob75M_minus_armW"] = {
                    "label": "the 0.02 leg (ai_v12_02_winprob_critic@75M) minus arm W — SAME critic "
                             "objective, so this row is the ENTROPY-COEFFICIENT axis; confounded by "
                             "pin (f971caf2 vs 6eb9c776) and by --ent-coef (0.02 vs 0.05)",
                    "delta_pp": round(100 * pt, 2),
                    "ci95_pp": [round(100 * lo, 2), round(100 * hi, 2)],
                    "verdict": verdict(100 * pt, 100 * lo, 100 * hi)}
                pt, lo, hi = paired(bper["armS"], bper["winprob75M"])
                rec["contrasts"]["armS_minus_winprob75M_banked"] = {
                    "label": "arm S minus the 0.02 leg — REPRODUCED from the banked file for the "
                             "three-way table (arm S's own 2026-09-14 read reported -3.75 pp)",
                    "delta_pp": round(100 * pt, 2),
                    "ci95_pp": [round(100 * lo, 2), round(100 * hi, 2)],
                    "verdict": verdict(100 * pt, 100 * lo, 100 * hi)}
        out["configs"][tag] = rec

    reg, aut = out["configs"].get("registry"), out["configs"].get("auto")
    if reg and aut and "error" not in reg and "error" not in aut:
        out["both_resolutions_agree"] = {
            "levels_identical": reg["levels_pp"] == aut["levels_pp"],
            "note": ("registration sec 8.5: a level that only exists under ONE config resolution is "
                     "not a level. Neither arm is architecture-identical to the meter's registry "
                     "config, so both resolutions are read and both are reported.")}
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("pair_untaught_delta.json")
    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in (v or {}).items() if kk != "per_team_pp"}
                      for k, v in out["configs"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
