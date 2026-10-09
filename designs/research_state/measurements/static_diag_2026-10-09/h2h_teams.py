"""The look-1 + look-2 cross games broken down by TEAM ARCHETYPE (DESCRIPTIVE, never a verdict).

Reads the screen family's rows ONLY through the eval ledger's declared family read (`main.h2h.cross.FAMILY_READ_OFF`,
the same declaration the registered read uses), at the pin. Each row carries, per team id, the player's and the
opponent's [games, wins]; a mirrored pair plays both teams from both seats, so for each team T we get static's
score when STATIC HOLDS T and when STATIC FACES T. Teams are mapped to the pool's archetype classifier
(`agents.training.team_archetypes.classify_team`) through the h2h team table's own packing (`Gen3Teambuilder`).
Game LENGTH is not on a row (only the outcome digest), so no length breakdown is possible from the ledger.

Run: run_at_pin.sh h2h_teams.py   -> h2h_teams.json beside this file.
"""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODELS = Path("/home/goodlad/dev/gen3ai/models")
FAMILY = "st_screen_strength_steps"
REQS = ("st_look1_steps", "st_look2_steps")


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    from agents.training import eval_ledger as L
    from agents.training.team_archetypes import classify_team
    from main.h2h import cross as X
    from utils.team_loader import TeamLoader
    from utils.teambuilder import Gen3Teambuilder

    arm_of = {}
    for arm in ("legacy", "static"):
        for s in range(1001, 1006):
            arm_of[sha(MODELS / f"rb_st_{arm}_s{s}" / "final_model.zip")] = (arm, s)
    got = L.read(X.FAMILY_READ_OFF, root=L.archive_ledger_root(), family=FAMILY)
    loader = TeamLoader()
    texts = list(loader.get_all_teams()) + list(loader.get_sample_teams())
    b = Gen3Teambuilder(texts)
    assert len(b.packed_teams) == len(texts)
    team_info = {}
    for t, p in zip(texts, b.packed_teams):
        rec = classify_team(t)
        team_info[L.team_id(p)] = {"archetype": rec["archetype"], "tags": rec["tags"], "species": rec["species"]}
    # per team: static's wins/games holding it, and facing it
    hold = defaultdict(lambda: [0, 0])
    face = defaultdict(lambda: [0, 0])
    n_rows = 0
    per_cell = defaultdict(lambda: defaultdict(lambda: [0, 0]))     # (static seed, legacy seed) -> arch -> [w, g]
    per_cell_l = defaultdict(lambda: defaultdict(lambda: [0, 0]))   # the same, legacy holding the team
    first_keys = None
    for row in got.rows:
        if (row.get("request") or {}).get("id") not in REQS:
            continue
        first_keys = first_keys or sorted(row)
        ps, os_ = row["player"].get("sha256"), row["opponent"].get("sha256")
        if ps not in arm_of or os_ not in arm_of:
            continue
        pa, oa = arm_of[ps][0], arm_of[os_][0]
        if pa == oa:
            continue
        n_rows += 1
        cell = (arm_of[ps][1], arm_of[os_][1]) if pa == "static" else (arm_of[os_][1], arm_of[ps][1])
        for tid, v in row["teams"].items():
            (pg, pw), (og, ow) = v["p"], v["o"]
            arch = team_info[tid]["archetype"] if tid in team_info else None
            if arch is not None:
                hw, hg = (pw, pg) if pa == "static" else (ow, og)
                per_cell[cell][arch][0] += hw; per_cell[cell][arch][1] += hg
                lw, lg = (ow, og) if pa == "static" else (pw, pg)          # LEGACY holding this team
                per_cell_l[cell][arch][0] += lw; per_cell_l[cell][arch][1] += lg
            if pa == "static":          # static is the player: holds T on p games, faces T on o games
                hold[tid][0] += pw; hold[tid][1] += pg
                face[tid][0] += og - ow; face[tid][1] += og       # draws count as non-wins here
            else:
                hold[tid][0] += ow; hold[tid][1] += og
                face[tid][0] += pg - pw; face[tid][1] += pg
    by_arch = defaultdict(lambda: {"hold": [0, 0], "face": [0, 0], "teams": 0})
    unknown = 0
    for tid in set(hold) | set(face):
        info = team_info.get(tid)
        if info is None:
            unknown += 1
            continue
        a = by_arch[info["archetype"]]
        a["teams"] += 1
        for k, d in (("hold", hold), ("face", face)):
            a[k][0] += d[tid][0]; a[k][1] += d[tid][1]
    tags = defaultdict(lambda: {"hold": [0, 0], "face": [0, 0], "teams": 0})
    for tid in set(hold) | set(face):
        info = team_info.get(tid)
        if info is None:
            continue
        for tg in info["tags"]:
            a = tags[tg]
            a["teams"] += 1
            for k, d in (("hold", hold), ("face", face)):
                a[k][0] += d[tid][0]; a[k][1] += d[tid][1]

    def fmt(dd):
        out = {}
        for k, a in sorted(dd.items()):
            h, f = a["hold"], a["face"]
            out[k] = {"teams": a["teams"], "games_hold": h[1], "static_win_rate_holding": round(h[0] / h[1], 4) if h[1] else None,
                      "games_face": f[1], "static_win_rate_facing": round(f[0] / f[1], 4) if f[1] else None}
        return out
    res = {"schema": "static_diag_h2h_teams_v1", "tag": "DESCRIPTIVE", "family": FAMILY, "requests": list(REQS),
           "regime_id": got.regime_id, "rows_used": n_rows, "row_keys": first_keys, "teams_seen": len(set(hold) | set(face)),
           "teams_unmapped": unknown,
           "note": "HOLDING rate = static wins / games (a draw is a non-win); FACING rate = games legacy did not win / games (a draw counts FOR static: the row keeps only per-team wins). Draws are ~0.65 % of games.",
           "overall": {"static_wins": int(sum(v[0] for v in hold.values())), "games": int(sum(v[1] for v in hold.values()))},
           "by_archetype": fmt(by_arch), "by_tag": fmt(tags)}
    # per-cell spread (25 cells; cells share seeds, so this describes the spread, it is not a test)
    archs = sorted(by_arch)
    rates = {a: [per_cell[c][a][0] / per_cell[c][a][1] for c in sorted(per_cell) if per_cell[c][a][1]] for a in archs}
    res["by_archetype_cells"] = {a: {"cells": len(v), "mean": round(float(np.mean(v)), 4), "sd": round(float(np.std(v, ddof=1)), 4)} for a, v in rates.items()}
    slow = [(per_cell[c]["stall"][0] + per_cell[c]["semi_stall"][0]) / (per_cell[c]["stall"][1] + per_cell[c]["semi_stall"][1])
            - (per_cell[c]["hyper_offense"][0] + per_cell[c]["offense"][0]) / (per_cell[c]["hyper_offense"][1] + per_cell[c]["offense"][1])
            for c in sorted(per_cell)]
    # THE CONTROLLED CONTRAST: a team's own strength cancels in (static's win rate holding T) − (legacy's win rate
    # holding T), both from the same mirrored games.
    gap = {a: [per_cell[c][a][0] / per_cell[c][a][1] - per_cell_l[c][a][0] / per_cell_l[c][a][1]
               for c in sorted(per_cell) if per_cell[c][a][1] and per_cell_l[c][a][1]] for a in archs}
    res["skill_gap_by_archetype_cells"] = {
        a: {"cells": len(v), "mean": round(float(np.mean(v)), 4), "sd": round(float(np.std(v, ddof=1)), 4),
            "cells_negative": int(sum(x < 0 for x in v))} for a, v in gap.items()}
    res["skill_gap_note"] = ("per cell: static's win rate HOLDING an archetype's teams minus legacy's win rate HOLDING "
                             "the same archetype's teams (draws are non-wins for both); 0 = equal skill with that archetype")
    res["slow_minus_fast_holding_per_cell"] = {"cells": len(slow), "mean": round(float(np.mean(slow)), 4),
                                               "sd": round(float(np.std(slow, ddof=1)), 4),
                                               "cells_negative": int(sum(x < 0 for x in slow)),
                                               "note": "static's win rate holding a stall/semi-stall team minus holding a hyper-offense/offense team, per cell"}
    (HERE / "h2h_teams.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("rows_used", "teams_seen", "teams_unmapped", "overall")}, indent=1))
    for k, v in res["by_archetype"].items():
        print(k, v, res["by_archetype_cells"][k])
    print("slow - fast", res["slow_minus_fast_holding_per_cell"])
    for k, v in res["skill_gap_by_archetype_cells"].items():
        print("skill gap", k, v)


if __name__ == "__main__":
    sys.exit(main())
