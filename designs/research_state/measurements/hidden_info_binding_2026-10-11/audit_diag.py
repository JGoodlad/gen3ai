"""Diagnosis aid for A2's UNEXPECTED cells (no outcome read): for one state of one world dir, every world whose
OPPONENT root row differs outside the opponent's own team block: the differing cells (named), the event rows in full,
the opponent's and the searched side's protocol diff (truth vs world), the leak record, the clock tracker.

    run_at_pin.sh <cp> 1 4 10m audit_diag.py <det|det_b> <decision id> [world j ...]
"""
import difflib
import json
import sys

import numpy as np

import audit_lib as A

C, D = A.C, A.D


def main():
    det_dir, did = sys.argv[1], sys.argv[2]
    js = [int(x) for x in sys.argv[3:]] or list(range(8))
    c = A.cp()
    from agents.battle.core_replay import run_core
    from agents.observation.constants import EVENT_TOKEN_DIM, OFFSET_EVENT_WINDOW, EventCol
    from main.policy_spectrum.truth import cmd_index

    names = A.cell_names()
    bk = A.bank(c)
    d, b = A.decision(bk, did)
    wr = A.world_rows(c, det_dir)[did]
    side = d["side"]
    opp = "p2" if side == "p1" else "p1"
    v = 0 if side == "p1" else 1
    o = 1 - v
    at = cmd_index(b, side, d["n"])
    print("STATE", did, "turn", d["turn"], "leak", wr["leak"], "revealed", wr["revealed"])
    print("TRUE ", getattr(b, opp)["team"])
    res = run_core([A.rec(b, "t", opp, None, b.commands[:at])] + [A.rec(b, f"w{j}", opp, wr["teams"][j], wr["cmds"][j]) for j in js],
                   trackers=True, obs=True)
    rt = res[0]
    row_t = A.row_of(A.obs_entries(rt, o)[-1])
    cols = {int(m): m.name for m in EventCol}
    tc = [x for x in b.commands[:at]]
    for j, rw in zip(js, res[1:]):
        row_w = A.row_of(A.obs_entries(rw, o)[-1])
        diff = [int(i) for i in np.flatnonzero(row_w.view(np.uint32) != row_t.view(np.uint32))
                if names[int(i)]["block"] not in ("own_team",) and not (names[int(i)]["block"] == "obs_facts" and names[int(i)]["field"] == "seen")]
        if not diff:
            continue
        print(f"\n=== world {j}: {wr['teams'][j]}")
        cd = [(k, x, y) for k, (x, y) in enumerate(zip(tc, wr["cmds"][j])) if list(x) != list(y)]
        print("cmd rewrites:", cd)
        for i in diff:
            nm = names[i]
            print("  DIFF", i, nm, "true", float(row_t[i]), "world", float(row_w[i]))
        rows = sorted({(i - OFFSET_EVENT_WINDOW) // EVENT_TOKEN_DIM for i in diff if names[i]["block"] == "event_window"})
        for r in rows:
            base = OFFSET_EVENT_WINDOW + r * EVENT_TOKEN_DIM
            et = {cols[k]: float(row_t[base + k]) for k in range(EVENT_TOKEN_DIM) if row_t[base + k] != 0}
            ew = {cols[k]: float(row_w[base + k]) for k in range(EVENT_TOKEN_DIM) if row_w[base + k] != 0}
            print(f"  EVENT row {r}\n    true  {et}\n    world {ew}")
        for sd, lab in ((o, "OPPONENT"), (v, "SEARCHED")):
            lt = [ln for s, lines in rt["chunks"] if s == sd for ln in lines if not ln.startswith("|request|")]
            lw = [ln for s, lines in rw["chunks"] if s == sd for ln in lines if not ln.startswith("|request|")]
            dl = [x for x in difflib.unified_diff(lt, lw, lineterm="", n=2)]
            print(f"  {lab} protocol diff ({len(dl)} lines):")
            for x in dl[:60]:
                print("    ", x)
        et = A.obs_entries(rt, o)[-1]["trackers"]
        ew = A.obs_entries(rw, o)[-1]["trackers"]
        et = json.loads(et) if isinstance(et, str) else et
        ew = json.loads(ew) if isinstance(ew, str) else ew
        for k in sorted(set(et) | set(ew)):
            if et.get(k) != ew.get(k):
                print("  TRACKER", k, "\n    true ", json.dumps(et.get(k))[:600], "\n    world", json.dumps(ew.get(k))[:600])


if __name__ == "__main__":
    main()
