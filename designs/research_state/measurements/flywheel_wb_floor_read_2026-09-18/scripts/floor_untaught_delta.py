#!/usr/bin/env python3
"""THE 75M RUN-LEVEL FLOOR — the UNTAUGHT METER, paired over TEAMS.

Same tool, same registry OPPONENT (``untaught_meter_opponent`` = ``ai_v9_29_rev1_0823@24,000,000``),
same registry config, same seed 0 and concurrency 1 as arm S's 2026-09-14 read and the pair read's
2026-09-16 one; **W_b and arm W were measured in ONE invocation**, so the two refs saw the identical
8 teams and the identical 200 games each -- exactly as arm S and arm W were.

``main.untaught_meter`` publishes a level and a CI per ref and a delta only against a ``--baseline``
(which none of these is), so every paired contrast is taken here off the meter's own per-team rows
on ONE resampled team index set -- rule 10, the team is the unit, and the paired draw removes the
team-difficulty component the refs share. The procedure and the seed are copied VERBATIM from
``pair_untaught_delta.py``.

🚨 **THE FLOOR IS |arm W - W_b|.** Until now the only floors this meter had were FOLD floors at
~1M depth (1.19 / 1.66 / 4.27 pp), quoted rather than applied because neither arm is a fold. This
is the meter's first RUN-LEVEL floor at ANY depth, and it is at 75M. One pair bounds it and does
not estimate it (rules 19/22).

Arm W appears in BOTH files, so its level here is its own reproduction check against the pair
read's: a deterministic meter at seed 0 / concurrency 1 must agree to the last win.
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
HERE = Path("/home/goodlad/.claude/jobs/9ab51de6/tmp/wb_read/untaught")
PAIR = Path("/home/goodlad/dev/gen3ai/designs/research_state/measurements/"
            "flywheel_pair_read_2026-09-15/out/untaught")


def per_team(d: dict):
    res = d["result"]
    teams = res["teams"]
    lv = res["levels"]
    per = {r: np.array([lv[r]["per_team"][t]["win_rate"] for t in teams], float) for r in lv}
    fin = {r: np.array([lv[r]["per_team"][t]["finished"] for t in teams], float) for r in lv}
    return teams, per, fin, res


def paired(per_a: np.ndarray, per_b: np.ndarray, seed: int = 20260915):
    """VERBATIM from pair_untaught_delta.py, same seed, same 20,000 draws."""
    rng = np.random.default_rng(seed)
    n = len(per_a)
    idx = rng.integers(0, n, size=(20000, n))           # ONE index set, shared by both refs
    dd = per_a[idx].mean(axis=1) - per_b[idx].mean(axis=1)
    point = float(per_a.mean() - per_b.mean())
    lo, hi = (float(x) for x in np.percentile(dd, [2.5, 97.5]))
    return point, lo, hi


def main() -> int:
    out: dict = {"floors_pp_fold_depth_quoted_not_applied": FLOORS_PP,
                 "floor_note": ("a floor is a property of the DEPTH and of the REGIME "
                                "(UNDERSTANDING sec 3.3); every floor above is a FOLD floor at ~1M "
                                "depth and none of these arms is a fold. The RUN-LEVEL 75M floor "
                                "measured here is |arm W - W_b|."),
                 "configs": {}}
    for tag in ("registry", "auto"):
        p, q = HERE / f"untaught_{tag}.json", PAIR / f"untaught_{tag}.json"
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
            "n_teams": len(teams), "timeouts": res.get("timeouts"), "contrasts": {},
        }
        if "armW" in per and "armWb" in per:
            pt, lo, hi = paired(per["armW"], per["armWb"])
            rec["contrasts"]["THE_FLOOR_armW_minus_armWb"] = {
                "label": "THE 75M RUN-LEVEL FLOOR -- arm W (seed 1001) minus W_b (seed 1002), both "
                         "measured in ONE invocation on the identical teams and games",
                "delta_pp": round(100 * pt, 2),
                "abs_delta_pp": round(abs(100 * pt), 2),
                "ci95_pp": [round(100 * lo, 2), round(100 * hi, 2)],
                "teams_where_armW_higher": int((per["armW"] > per["armWb"]).sum()),
                "note": ("this |delta| IS the floor. The CI printed is the CI of the FLOOR PAIR's "
                         "own delta, not a CI on the floor as a quantity -- one pair bounds a "
                         "floor and does not estimate one (rules 19/22)."),
            }
        # join the pair read's banked file for the FINDING and the reproduction check
        if q.exists():
            pteams, pper, _pfin, _ = per_team(json.loads(q.read_text()))
            rec["pair_read_join"] = {"source": str(q.relative_to(Path("/home/goodlad/dev/gen3ai"))),
                                     "same_teams": bool(pteams == teams)}
            if pteams == teams and "armS" in pper and "armW" in pper:
                rec["pair_read_join"]["armW_level_pp_banked"] = round(100 * float(pper["armW"].mean()), 2)
                rec["pair_read_join"]["armW_level_pp_here"] = round(100 * float(per["armW"].mean()), 2)
                rec["pair_read_join"]["armW_reproduction_delta_pp"] = round(
                    100 * float(per["armW"].mean() - pper["armW"].mean()), 4)
                rec["pair_read_join"]["armS_level_pp_banked"] = round(100 * float(pper["armS"].mean()), 2)
                pt, lo, hi = paired(pper["armS"], pper["armW"])
                rec["contrasts"]["THE_FINDING_armS_minus_armW"] = {
                    "label": "THE FINDING -- arm S minus arm W, recomputed from the pair read's own "
                             "banked per-team rows by this same paired procedure and seed",
                    "delta_pp": round(100 * pt, 2),
                    "ci95_pp": [round(100 * lo, 2), round(100 * hi, 2)],
                    "teams_where_armS_higher": int((pper["armS"] > pper["armW"]).sum())}
                if "armWb" in per:
                    pt2, lo2, hi2 = paired(pper["armS"], per["armWb"])
                    rec["contrasts"]["armS_minus_armWb"] = {
                        "label": "arm S minus W_b -- the treatment contrast taken against the OTHER "
                                 "seed. 🚨 arm S and W_b were measured in DIFFERENT invocations of "
                                 "the meter (2026-09-16 and 2026-09-18); the meter is deterministic "
                                 "at seed 0 / concurrency 1 and the arm-W reproduction check above "
                                 "is what licenses the join.",
                        "delta_pp": round(100 * pt2, 2),
                        "ci95_pp": [round(100 * lo2, 2), round(100 * hi2, 2)],
                        "teams_where_armS_higher": int((pper["armS"] > per["armWb"]).sum())}
        f = rec["contrasts"].get("THE_FLOOR_armW_minus_armWb")
        g = rec["contrasts"].get("THE_FINDING_armS_minus_armW")
        if f and g:
            fl = f["abs_delta_pp"]
            dd = abs(g["delta_pp"]); lo, hi = g["ci95_pp"]
            inside = (lo <= fl <= hi) or (lo <= -fl <= hi)
            rec["floor_verdict"] = {
                "finding_abs_pp": dd, "finding_ci95_pp": [lo, hi], "floor_abs_pp": fl,
                "clause_a_abs_delta_gt_floor": bool(dd > fl),
                "clause_b_ci_excludes_floor_point": bool(not inside),
                "verdict": ("OUTSIDE THE 75M FLOOR" if (dd > fl and not inside)
                            else "WITHIN FLOOR at n = 2"),
                "caveats": [
                    "the meter is a DESCRIPTOR, not an endpoint (registration sec 8.5) -- clearing a "
                    "floor does not promote it to one",
                    "one replicate pair BOUNDS a floor; no CI attaches to it (rules 19/22)",
                    "WITHIN FLOOR is NEVER 'equivalent' (rule 6)",
                    "the 1.44x realized-dose gap (H-F) confounds arm S and is not measured here",
                ]}
        out["configs"][tag] = rec

    reg, aut = out["configs"].get("registry"), out["configs"].get("auto")
    if reg and aut and "error" not in reg and "error" not in aut:
        out["both_resolutions_agree"] = {
            "levels_identical": reg["levels_pp"] == aut["levels_pp"],
            "note": ("registration sec 8.5: a level that only exists under ONE config resolution is "
                     "not a level. Neither arm is architecture-identical to the meter's registry "
                     "config, so both resolutions are read and both are reported.")}
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("floor_untaught_delta.json")
    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in (v or {}).items() if kk != "per_team_pp"}
                      for k, v in out["configs"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
