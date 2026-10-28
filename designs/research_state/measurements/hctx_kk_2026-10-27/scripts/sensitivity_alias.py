"""POST-HOC sensitivity of hctx_kk_2026-10-27 (README FINDING 1; not registered, read by no verdict).

rc72b_05.txt writes Gengar's move as ``- wisp``, a Showdown alias that the server packs as Will-O-Wisp; the meter's
strict team identity keeps the raw id, so the six units that drew rc72b_05 on side A were refused (two games each).
Here a side-A capture that matches nothing is retried with ``Will-O-Wisp`` written back as ``wisp``; every other
check is the registered one, and the primary statistic is recomputed over all units that then count.

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=<checkout>/src python .../scripts/sensitivity_alias.py --state DIR --out OUT.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import record as R  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from main.team_strength import kakuna_lane as KL

    orig = KL.match_team
    retried = []

    def match(text, ids, mon_keys):
        tid, tier = orig(text, ids, mon_keys)
        if tid is None and "Will-O-Wisp" in text:
            tid, tier2 = orig(text.replace("Will-O-Wisp", "wisp"), ids, mon_keys)
            if tid is not None:
                retried.append(tid)
                tier = f"alias:{tier2}"
        return tid, tier

    KL.match_team = match
    L = R.load(Path(a.state).expanduser())
    games, verdicts = L["games"], L["verdicts"]
    prim = R.unit_boot(R.by_unit(games))
    out = {"schema": "hctx_kk_sensitivity_alias_v1", "post_hoc": True,
           "units_counted": sum(v["counts"] for v in verdicts), "alias_matches": len(retried),
           "alias_teams": sorted(set(retried)), "primary": prim, "class_if_registered": R.classify(prim["ci"]),
           "vs_home72": R.unit_boot(R.by_unit(games, lambda g: g["control"] == "home72"), R.SEED + 1),
           "vs_forum72": R.unit_boot(R.by_unit(games, lambda g: g["control"] == "forum72"), R.SEED + 2),
           "D2_team_resampled": R.team_boot(games, R.SEED + 4)}
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: out[k] for k in ("units_counted", "alias_matches", "primary", "class_if_registered")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
