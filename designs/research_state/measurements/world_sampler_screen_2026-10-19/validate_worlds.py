"""F2 of ``designs/endstate/design_world_sampler.md`` §7: are the honest builder's WORLDS legal by Showdown's own
``TeamValidator('gen3ou')``? The builder checks the format clauses and engine-runnable entities, never the full
validator (learnset COMBINATIONS, event-only moves, …). POST HOC, descriptive, zero playouts.

Reads the stored F world rows of the same banks as ``screen_samplers.py`` (maturity interim LEAN ids; E2 end2 / prod),
the 8 world teams of each OK state, and the TRUE opponent team as the CONTROL (pool teams: they must pass, else the
check itself is miscalibrated). The engine's gender draw-pad marker (``*`` after a gender, ours, not Showdown's) is
stripped first. Validation is the repo's own bridge (``src/utils/bridge/validate_team.js``) run from the MAIN checkout,
whose ``deps/pokemon-showdown`` is built (read only).

Run: CUDA_VISIBLE_DEVICES="" nice -n 19 taskset -c 5-7,13-15 python -I validate_worlds.py --out result_validate.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))      # `-I` drops the script's own directory
import screen_samplers as SS  # noqa: E402

MAIN_BRIDGE = Path("/home/goodlad/dev/gen3ai/src/utils/bridge/validate_team.js")
BATCH = 200


def unpad(team: str) -> str:
    out = []
    for e in team.split("]"):
        f = e.split("|")
        if len(f) > 7 and f[7].endswith("*"):
            f[7] = f[7][:-1]
        out.append("|".join(f))
    return "]".join(out)


def validate(teams: List[str]) -> List[dict]:
    res: List[dict] = []
    for i in range(0, len(teams), BATCH):
        req = json.dumps({"format": "gen3ou", "team": teams[i:i + BATCH]})
        p = subprocess.run(["node", str(MAIN_BRIDGE)], input=req, capture_output=True, text=True, timeout=600)
        got = json.loads(p.stdout)
        if not isinstance(got, list):        # the bridge's whole-request failure: one object, not a list
            raise RuntimeError(f"validator infrastructure error (never a verdict): {got}")
        res.extend(got)
    return res


def kind(err: str) -> str:
    e = err.lower()
    for k in ("can't learn", "event", "egg", "ability", "evs", "ivs", "banned", "clause", "level", "nature", "item",
              "hidden power", "move"):
        if k in e:
            return k
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(SS.HERE / "result_validate.json"))
    a = ap.parse_args()
    lean = {k for k, v in json.loads(SS.MAT_STATES.read_text())["states"].items() if v["lean"]}
    pops = [("mat_lean", SS.MAT / "bank", SS.MAT / "worlds" / "shared" / "F" / "worlds_0_of_1.jsonl", lean),
            ("e2_end2", SS.E2 / "end2" / "bank", SS.E2 / "end2" / "worlds" / "F" / "worlds_0_of_1.jsonl", None),
            ("e2_prod", SS.E2 / "prod" / "bank", SS.E2 / "prod" / "worlds" / "F" / "worlds_0_of_1.jsonl", None)]
    world_teams: List[str] = []
    world_state: List[str] = []
    true_teams: Dict[str, str] = {}
    for pop, bank, fpath, keep in pops:
        battles = {b["battle_id"]: b for b in SS._jsonl(bank / "battles.jsonl.gz")}
        decs = {d["id"]: d for d in SS._jsonl(bank / "decisions.jsonl.gz")}
        seen = set()
        for r in SS._jsonl(fpath):
            did = r["id"]
            if (keep is not None and did not in keep) or did in seen or not r.get("ok"):
                continue
            seen.add(did)
            d = decs[did]
            opp = "p2" if d["side"] == "p1" else "p1"
            true_teams[did] = battles[d["battle"]][opp]["team"]
            for t in r["teams"]:
                world_teams.append(unpad(t))
                world_state.append(did)
    ctrl_ids = sorted(true_teams)
    ctrl = validate([true_teams[i] for i in ctrl_ids])
    worlds = validate(world_teams)
    bad_states = Counter(s for s, v in zip(world_state, worlds) if not v.get("valid"))
    errs = Counter(kind(e) for v in worlds if not v.get("valid") for e in v.get("errors", []))
    examples = [e for v in worlds if not v.get("valid") for e in v.get("errors", [])][:25]
    ctrl_examples = [e for v in ctrl if not v.get("valid") for e in v.get("errors", [])][:10]
    out = {
        "schema": "world_sampler_validate_v1",
        "control_true_teams": {"n": len(ctrl), "invalid": sum(not v.get("valid") for v in ctrl),
                               "examples": ctrl_examples},
        "worlds": {"n": len(worlds), "invalid": sum(not v.get("valid") for v in worlds),
                   "states": len(set(world_state)), "states_with_an_invalid_world": len(bad_states),
                   "error_kinds": dict(errs.most_common()), "examples": examples},
    }
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "examples"} for k, v in out.items()
                      if isinstance(v, dict)}, indent=1))


if __name__ == "__main__":
    main()
