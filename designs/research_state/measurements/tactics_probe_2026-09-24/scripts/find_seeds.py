"""Seed search for every state's scripted prefix (no model; scripted players only).

For each state: k = 0, 1, 2, ... -> seed = mint_seed(f"{seed_key}:{k}"); play the prefix with both sides
scripted; snapshot at T from BOTH players' own battles; the first k that meets every `expect` predicate is
registered. Writes <out>/seeds.json: {sid: {k, seed, snapshot, tried}}. Deterministic: rerunning finds the
same k. A state with no passing k in --max-k is reported and NOT registered.

    python find_seeds.py --out <dir> [--only sid1,sid2] [--max-k 40]
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tlib  # noqa: E402
import states as ST  # noqa: E402
from snap import check, snapshot  # noqa: E402


def try_state(spec, k):
    seed = tlib.mint_seed(f"{spec.seed_key}:{k}")
    p1, p2 = tlib.pack(spec.p1), tlib.pack(spec.p2)
    s1 = [c for side, c in spec.commands() if side == "p1"]
    s2 = [c for side, c in spec.commands() if side == "p2"]
    r = tlib.run_scripted(p1, p2, s1, s2, seed)
    b1 = list(r["p1"].battles.values())[0]
    b2 = list(r["p2"].battles.values())[0]
    snap = snapshot(b1, b2)
    bad = check(snap, spec.expect)
    if snap["turn"] != spec.T:
        bad.append(f"turn {snap['turn']} != T {spec.T}")
    if r["left"]["p1"] or r["left"]["p2"]:
        bad.append(f"script not consumed: {r['left']}")
    return seed, snap, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    ap.add_argument("--max-k", type=int, default=40)
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "seeds.json")
    reg = json.load(open(path)) if os.path.exists(path) else {}
    only = set(x for x in a.only.split(",") if x)
    for spec in ST.all_states():
        if only and spec.sid not in only:
            continue
        found = None
        fails = []
        for k in range(a.max_k):
            seed, snap, bad = try_state(spec, k)
            if a.verbose or k == 0:
                print(f"  {spec.sid} k={k}: p1 {snap['p1']['active']} {snap['p1']['hp']:.3f} {snap['p1']['boosts']} "
                      f"{snap['p1']['status']} | p2 {snap['p2']['active']} {snap['p2']['hp']:.3f} "
                      f"{snap['p2']['boosts']} {snap['p2']['status']} | p1_first={snap['p1_first']} "
                      f"team1={snap['p1']['team']} bad={bad}", flush=True)
            if not bad:
                found = {"k": k, "seed": seed, "snapshot": snap, "tried": k + 1}
                break
            fails.append(bad)
        if found is None:
            print(f"!! {spec.sid}: NO passing seed in {a.max_k}; first failures: {fails[:3]}", flush=True)
            continue
        reg[spec.sid] = found
        s = found["snapshot"]
        print(f"OK {spec.sid}: k={found['k']} T={spec.T} p1 {s['p1']['active']} hp={s['p1']['hp']:.3f} "
              f"{s['p1']['boosts']} spe={s['p1']['eff_speed']} | p2 {s['p2']['active']} hp={s['p2']['hp']:.3f} "
              f"({s['p2']['hp_abs']}/{s['p2']['hp_max']}) {s['p2']['boosts']} spe={s['p2']['eff_speed']} | "
              f"p2 revealed {s['p1']['foe_revealed_moves']} | p2 sc {s['p2']['side_conditions']}", flush=True)
        json.dump(reg, open(path, "w"), indent=1)


if __name__ == "__main__":
    main()
