#!/usr/bin/env python3
"""The PAIRED arm-S-minus-comparator delta on the untaught meter, cluster-bootstrapped over TEAMS.

``main.untaught_meter`` publishes a level and a CI per ref, and a delta only against a ``--baseline``
(which is not what is wanted here — neither of these two is the other's parent).  The registered
read is a descriptor of the two 75M arms side by side, so this reads the meter's own per-team rows
back out of its JSON and takes the paired contrast on ONE resampled team index set — rule 10, the
team is the unit, and a paired draw is what removes the team-difficulty component the two refs share.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# The meter's floors are a property of the DEPTH and of the REGIME, and are quoted with both
# (UNDERSTANDING sec 3.3).  Neither of these is a fold, so the fold floors do not apply; the closest
# thing in evidence is the END-depth replicate pair scatter.
FLOORS_PP = {
    "frozen_dose_folds_six_draws": 1.66,
    "controller_live_folds": 4.27,
    "TC_replicate_pairs_END_depth": 1.19,
    "TC_replicate_pairs_p1M_depth": 4.00,
    "G5_three_continuation_arms": 1.00,
}


def main() -> int:
    out = {}
    for tag in ("registry", "auto"):
        p = Path(f"/home/goodlad/.claude/jobs/9ab51de6/tmp/armS_reads/untaught/untaught_{tag}.json")
        if not p.exists():
            out[tag] = {"error": f"no {p}"}
            continue
        d = json.loads(p.read_text())
        res = d["result"]
        teams = res["teams"]
        lv = res["levels"]
        refs = list(lv)
        per = {r: np.array([lv[r]["per_team"][t]["win_rate"] for t in teams], float) for r in refs}
        fin = {r: np.array([lv[r]["per_team"][t]["finished"] for t in teams], float) for r in refs}
        rng = np.random.default_rng(20260914)
        n = len(teams)
        # ONE index set per draw, shared by both refs — the paired convention the meter itself uses.
        idx = rng.integers(0, n, size=(20000, n))
        means = {r: per[r][idx].mean(axis=1) for r in refs}
        a, b = "armS", "winprob75M"
        dd = means[a] - means[b]
        point = float(per[a].mean() - per[b].mean())
        lo, hi = (float(x) for x in np.percentile(dd, [2.5, 97.5]))
        out[tag] = {
            "config": ("the REGISTRY `untaught_meter_config` (ai_v9_29_rev1_0823's "
                       "snapshots/model_config.json)" if tag == "registry"
                       else "--config auto (each model's OWN model_config.json)"),
            "opponent": "registry name `untaught_meter_opponent` = ai_v9_29_rev1_0823@24,000,000",
            "levels_pp": {r: round(100 * float(per[r].mean()), 2) for r in refs},
            "n_finished": {r: int(fin[r].sum()) for r in refs},
            "per_team_pp": {r: {t: round(100 * v, 2) for t, v in zip(teams, per[r])} for r in refs},
            "paired_delta_armS_minus_winprob75M_pp": round(100 * point, 2),
            "ci95_pp": [round(100 * lo, 2), round(100 * hi, 2)],
            "teams_where_armS_higher": int((per[a] > per[b]).sum()),
            "n_teams": n,
            "timeouts": res.get("timeouts"),
            "floors_pp_in_evidence": FLOORS_PP,
            "verdict": None,
        }
        clears = abs(100 * point) > FLOORS_PP["TC_replicate_pairs_END_depth"]
        ci_clear = not (lo <= 0 <= hi)
        out[tag]["verdict"] = (
            "DETECTED at 95% and past the 1.19pp END-depth replicate floor" if (clears and ci_clear)
            else "the CI covers zero — NOT DETECTED (never 'equal': rule 6)" if not ci_clear
            else "the CI clears zero but the magnitude is inside the 1.19pp END-depth floor — "
                 "NOT DETECTED")
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("untaught_delta.json")
    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per_team_pp"}
                      for k, v in out.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
