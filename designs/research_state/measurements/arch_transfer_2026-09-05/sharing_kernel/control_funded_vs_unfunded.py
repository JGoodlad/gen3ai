"""WITHIN-ERA CONTROL with no architecture confound — the FUNDED vs UNFUNDED label.

The teacher-content 2x2's funded half (the 8 R5FUND forks) and unfunded half (their 8 R5F parents)
resolve to the SAME 16 teams. So labelling the gen-era parent's taught states "funded" on one side
and "unfunded" on the other names ONE identical state set twice. The kernel ratio must therefore
come back EXACTLY 1.0.

This is an identity check, and it is worth running for exactly that reason: it is the check that
fails if `cross` and `within` are accumulated over misaligned index sets, if the team->row map
drifts, or if a label is silently doing work the states are not. It says nothing about the
hypothesis; it says the meter reads a property of the model and the states rather than of the word.

Step 1 RESOLVES both fleets from run metadata (each arm's own recorded --distill-teacher, whose
':*' expands to each teacher's recorded --trainee-teams) rather than trusting the hardcoded list --
so it also re-derives the taught set independently of gen_states.py.

Run: python control_funded_vs_unfunded.py --kernel kernel_gen.json --out control_gen.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kernel import stats_from_M  # noqa: E402  -- the SAME function, imported, never re-implemented
from gen_states import TAUGHT_SHA  # noqa: E402

MD = "/home/goodlad/dev/gen3ai/models"
FLEETS = {"funded": "ai_v9_160_TCFUNDA_0903", "unfunded": "ai_v9_162_TCUNFA_0903"}
REPLICATES = {"funded_B": "ai_v9_161_TCFUNDB_0903", "unfunded_B": "ai_v9_163_TCUNFB_0903"}


def orig_cmd(run: str) -> str:
    return json.load(open(f"{MD}/{run}/metadata.json")).get("original_command") or ""


def flagval(c: str, flag: str):
    m = re.search(rf"{re.escape(flag)}\s+(\S+)", c)
    return m.group(1) if m else None


def fleet_teams(run: str):
    c = orig_cmd(run)
    dt = flagval(c, "--distill-teacher")
    if not dt:
        raise SystemExit(f"[c] {run}: no --distill-teacher in original_command")
    teachers = [x.split(":")[0] for x in dt.split(";")]
    union, per = [], {}
    for t in teachers:
        tr = os.path.basename(t.rstrip("/"))
        tt = flagval(orig_cmd(tr), "--trainee-teams")
        teams = tt.split(",") if tt else []
        if not teams:
            raise SystemExit(f"[c] teacher {tr}: no --trainee-teams recorded")
        per[tr] = teams
        union.extend(teams)
    return sorted(set(union)), per


def sha10_of(path: str) -> str:
    root = "/home/goodlad/dev/gen3ai/.claude/worktrees/agent-a95ab66ffd73bb722"
    return hashlib.sha1(open(f"{root}/{path}").read().strip().encode()).hexdigest()[:10]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    res = {"_meta": {"what": "FUNDED vs UNFUNDED teacher labels over the gen-era parent's kernel; "
                             "both halves resolve to the SAME 16 teams, so the ratio must be "
                             "EXACTLY 1.0. An identity check on the meter, not a test of the "
                             "hypothesis.",
                     "kernel_artifact": os.path.abspath(a.kernel)}}

    sets = {}
    for tag, run in list(FLEETS.items()) + list(REPLICATES.items()):
        teams, per = fleet_teams(run)
        shas = sorted(sha10_of(p) for p in teams)
        sets[tag] = shas
        res[f"fleet_{tag}"] = {"run": run, "n_teachers": len(per), "n_teams": len(teams),
                               "team_paths": teams, "team_sha10": shas}
        print(f"[c] {tag:11s} {run:26s} {len(per)} teachers -> {len(teams)} teams", flush=True)

    same = all(sets[k] == sets["funded"] for k in sets)
    res["all_four_fleets_same_16_teams"] = bool(same)
    res["matches_gen_states_TAUGHT_SHA"] = bool(sets["funded"] == sorted(TAUGHT_SHA))
    print(f"[c] all four fleets resolve to the same team set: {same}", flush=True)
    print(f"[c] that set == gen_states.py TAUGHT_SHA: {res['matches_gen_states_TAUGHT_SHA']}",
          flush=True)
    if not same or not res["matches_gen_states_TAUGHT_SHA"]:
        raise SystemExit("[c] the funded/unfunded team sets are NOT the same 16 -- the control's "
                         "premise is false and the measurement's taught set must be re-derived")

    K = json.load(open(a.kernel))
    teams_order = K["_meta"]["teams_order"]
    taught_rows = [teams_order.index(s) for s in sorted(TAUGHT_SHA)]

    res["control"] = {}
    for gname, g in K["groups"].items():
        if "team_matrix" not in g:
            res["control"][gname] = {"status": g.get("status", "no matrix")}
            continue
        M = np.array(g["team_matrix"])
        diag = np.array(g["team_diag"])
        st = stats_from_M(M, diag, taught_rows, taught_rows)
        res["control"][gname] = {"cross": st["cross"], "within_pooled": st["within_pooled"],
                                 "ratio": st["ratio"],
                                 "abs_dev_from_1": abs(st["ratio"] - 1.0)}
        print(f"[c] {gname:16s} cross {st['cross']:+.6f}  within {st['within_pooled']:+.6f}  "
              f"ratio {st['ratio']:.9f}", flush=True)

    worst = max(v["abs_dev_from_1"] for v in res["control"].values() if "abs_dev_from_1" in v)
    res["max_abs_dev_from_1"] = float(worst)
    res["PASS"] = bool(worst < 1e-12)
    print(f"[c] max |ratio - 1| = {worst:.3e}   PASS={res['PASS']}", flush=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    return 0 if res["PASS"] else 1


if __name__ == "__main__":
    sys.exit(main())
