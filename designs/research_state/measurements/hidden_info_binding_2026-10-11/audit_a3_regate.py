"""A3 RE-GATE DETERMINISM: 100 archived worlds per checkpoint (50 from det/, 50 from det_b/, chosen by a salted hash
over every built world, played or not) are put through the gate AGAIN, two ways, each twice:
  (i)  the BUILDER: ``determinize.build_worlds`` with its sampler replaced (this process only) by one returning the
       ARCHIVED team: it must pass on try 1 and render the ARCHIVED commands exactly;
  (ii) a DIRECT replay of the archived (team, cmds) through ``run_core``: the searched side's rows at every decision
       up to the root (``determinize.reveals``) byte-equal to the true prefix's, and byte-equal between the 2 runs.
PASS = 100 %. No outcome is read.

    run_at_pin.sh <cp> 1 4 1h audit_a3_regate.py [--n 100]
→ here ``audit_a3_regate_<cp>.json``.
"""
import argparse
import hashlib

import audit_lib as A

C, D = A.C, A.D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    c = A.cp()
    from agents.battle.core_replay import run_core
    from main.policy_spectrum.truth import cmd_index

    bk = A.bank(c)
    P = A.priors()
    pick = []
    for det_dir in A.AUDIT_DIRS:
        W = A.world_rows(c, det_dir)
        cand = [(det_dir, did, j) for did, r in W.items() if r.get("ok") for j, t in enumerate(r["teams"]) if t]
        cand.sort(key=lambda x: hashlib.sha256(f"a3:{x[0]}:{x[1]}:{x[2]}".encode()).hexdigest())
        pick += [(x, W[x[1]]) for x in cand[: a.n // 2]]
    res = {"audit": "A3 re-gate determinism", "cp": c, "worlds": len(pick), "builder_try1_and_same_cmds": 0,
           "direct_rows_equal": 0, "direct_runs_identical": 0, "failures": []}
    orig = D.sample_world
    for (det_dir, did, j), wr in pick:
        d, b = A.decision(bk, did)
        side = d["side"]
        opp = "p2" if side == "p1" else "p1"
        v = 0 if side == "p1" else 1
        at = cmd_index(b, side, d["n"])
        team, cmds = wr["teams"][j], wr["cmds"][j]
        D.sample_world = lambda rng, P_, tm, rv, *x, **k: (team, {"new_slots": 0, "redrawn_move_slots": 0})
        try:
            w = D.build_worlds(b, side, at, d["n"], did, P, 1, root_choice_in_log=not wr["other_open"])
        finally:
            D.sample_world = orig
        ok_b = w["teams"][0] == team and w["tries"] == [1] and w["cmds"][0] == cmds
        res["builder_try1_and_same_cmds"] += ok_b
        outs = []
        for rep in range(2):
            rt, rw = run_core([A.rec(b, f"{did}#true", opp, None, b.commands[:at]),
                               A.rec(b, f"{did}#w{j}", opp, team, cmds)], trackers=True, obs=True)
            _, ref = D.reveals(rt, v, d["n"])
            try:
                _, rows = D.reveals(rw, v, d["n"]) if rw["ok"] else (None, None)
            except RuntimeError:
                rows = None
            outs.append((ref, rows, rw.get("chunks")))
        eq = outs[0][1] is not None and outs[0][1] == outs[0][0] and outs[1][1] == outs[1][0]
        same = outs[0] == outs[1]
        res["direct_rows_equal"] += eq
        res["direct_runs_identical"] += same
        if not (ok_b and eq and same) and len(res["failures"]) < 10:
            res["failures"].append({"dir": det_dir, "id": did, "j": j, "builder_ok": ok_b, "builder_tries": w["tries"],
                                    "builder_fails": w["fails"], "rows_equal": eq, "runs_identical": same})
    n = res["worlds"]
    res["pass"] = res["builder_try1_and_same_cmds"] == n and res["direct_rows_equal"] == n and res["direct_runs_identical"] == n
    A.write_json(f"audit_a3_regate_{c}.json", res)
    print({k: v for k, v in res.items() if k != "failures"}, flush=True)


if __name__ == "__main__":
    main()
