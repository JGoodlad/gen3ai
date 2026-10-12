"""Diagnosis aid (no outcome read): BISECT which part of a world produces a hidden divergence the gate missed.
Hybrid teams = the TRUE team with ONE slot (and, for a seen mon, ONE move slot) taken from world j; each hybrid
replayed with the world's commands; the opponent's protocol diffed against the truth.

    run_at_pin.sh <cp> 1 4 10m audit_diag_bisect.py <det|det_b> <decision id> <world j>
"""
import difflib
import sys

import audit_lib as A

C, D = A.C, A.D


def main():
    det_dir, did, j = sys.argv[1], sys.argv[2], int(sys.argv[3])
    c = A.cp()
    from agents.battle.core_replay import run_core
    from main.policy_spectrum.truth import cmd_index

    bk = A.bank(c)
    d, b = A.decision(bk, did)
    wr = A.world_rows(c, det_dir)[did]
    side = d["side"]
    opp = "p2" if side == "p1" else "p1"
    v = 0 if side == "p1" else 1
    o = 1 - v
    at = cmd_index(b, side, d["n"])
    te = getattr(b, opp)["team"].split("]")
    we = wr["teams"][j].split("]")
    hy = []
    for t in range(6):
        if te[t] == we[t]:
            continue
        hy.append((f"slot{t} whole", "]".join(te[:t] + [we[t]] + te[t + 1:])))
        ft, fw = D.parse_set(te[t]), D.parse_set(we[t])
        for g in ("M", "F", "N"):           # the same slot with its GENDER fixed in the packed set (field 7)
            fg = list(fw)
            fg[7] = g
            hy.append((f"slot{t} whole gender={g}", "]".join(te[:t] + ["|".join(fg)] + te[t + 1:])))
        if D.toid(ft[1] or ft[0]) == D.toid(fw[1] or fw[0]):
            mt, mw = ft[4].split(","), fw[4].split(",")
            for k in range(4):
                if mt[k] != mw[k]:
                    m2 = list(mt)
                    m2[k] = mw[k]
                    f2 = list(ft)
                    f2[4] = ",".join(m2)
                    hy.append((f"slot{t} move{k} {mt[k]}->{mw[k]}", "]".join(te[:t] + ["|".join(f2)] + te[t + 1:])))
    cm = wr["cmds"][j]
    res = run_core([A.rec(b, "t", opp, None, b.commands[:at])] + [A.rec(b, lab, opp, tm, cm) for lab, tm in hy],
                   trackers=True, obs=True)
    lt = [ln for s, lines in res[0]["chunks"] if s == o for ln in lines if not ln.startswith("|request|")]
    for (lab, _), r in zip(hy, res[1:]):
        if not r["ok"]:
            print(lab, "REFUSED", str(r.get("error"))[:200])
            continue
        lw = [ln for s, lines in r["chunks"] if s == o for ln in lines if not ln.startswith("|request|")]
        dl = [x for x in difflib.unified_diff(lt, lw, lineterm="", n=0) if x[:1] in "+-" and x[:3] not in ("---", "+++")]
        gate = D.reveals(r, v, d["n"])[1] == D.reveals(res[0], v, d["n"])[1]
        print(f"{lab:40s} gate={gate} opp-protocol-diff={dl[:6]}")


if __name__ == "__main__":
    main()
