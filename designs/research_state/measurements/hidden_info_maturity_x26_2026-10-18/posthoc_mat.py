"""POST HOC, DESCRIPTIVE companions of the registered read (written 2026-10-19 AFTER `read_mat.py`'s output existed; no
rule, no verdict; nothing here changes a registered read). Same frozen table (`score/table.pkl`, sha-checked), the same
common states and the SAME battle-clustered bootstrap (`maturity_rules.Boot`, 2,000 draws, seed 20261011) as
`read_mat.py`, so every contrast is paired with the registered ones.

  1. WHERE the gap's change comes from: G_F = IMP_O - IMP_F, so ΔG_F = ΔIMP_O - ΔIMP_F; each anchor minus the first,
     per scorer (did the oracle's edge over the policy fall, or did honest search rise?);
  2. G_F's change per opponent STRATUM (snapshot / bot): §4 registers G_F per stratum "and its change" as descriptive;
     the reader prints the per-anchor values only;
  3. the network belief's value: IMP_FN - IMP_F per anchor (LEAF-K8, TRUNC-4-K8);
  4. the M8 dose curves per anchor (LEAF-K8 and the continuous TRUNC-4-K8 leaf) on the FN-common states, with a
     P-STAR vs the policy by R-DOSE's definition (§12.13.4: the smallest 0.05-grid p whose lower95 > +0.0005 at p and at
     every larger p) and TODAY on each curve (the TRUNC-4-K8 one is NOT registered: the red team's continuous leaf);
  5. the per-bucket gate statuses per anchor (counts).

    python -I posthoc_mat.py      -> posthoc_mat.json (this folder)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import read_mat as RM

R = RM.R
SC = ("LEAF-K8", "TRUNC-4-K8", "CRITIC")
P_SHOW = (0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)


def s(est, vec):
    return R.summary(est, vec)


def pstar(ce, cv, grid):
    ok = [R.gt(R.ci(cv[:, i])[0], 0.0) for i in range(len(grid))]
    for i in range(len(grid)):
        if all(ok[i:]):
            return float(grid[i])
    return None


def main():
    tab = RM.load_table()
    labels = sorted(tab, key=lambda x: tab[x]["step"])
    ids, _drops, by = RM.common_ids(tab, labels)
    recs = {lab: [by[lab][x] for x in ids] for lab in labels}
    bt = R.Boot([r["battle"] for r in recs[labels[0]]])
    fn_all = np.array([all(RM._ok(by[lab][x], "FN") for lab in labels) for x in ids])
    first = labels[0]
    out: dict = {"anchors": labels, "n_common": len(ids), "n_fn_common": int(fn_all.sum()), "battles": bt.nb,
                 "status": "POST HOC, DESCRIPTIVE (no rule)"}

    # 1. the decomposition of the gap's change
    dec = {}
    for lab in labels:
        dec[lab] = {}
        for z in SC:
            o = RM.vec(recs[lab], "O", z) - RM.vec(recs[first], "O", z)
            f = RM.vec(recs[lab], "F", z) - RM.vec(recs[first], "F", z)
            dec[lab][z] = {"dIMP_O": s(*bt.mean(o)), "dIMP_F": s(*bt.mean(f)), "dG_F": s(*bt.mean(o - f))}
    out["delta_vs_first"] = dec

    # 2. G_F's change per stratum
    st = {}
    for sname in ("snapshot", "bot"):
        m = np.array([r.get("stratum") == sname for r in recs[first]])
        g0 = RM.vec(recs[first], "O", RM.LEAF) - RM.vec(recs[first], "F", RM.LEAF)
        st[sname] = {"n": int(m.sum())}
        for lab in labels[1:]:
            g = RM.vec(recs[lab], "O", RM.LEAF) - RM.vec(recs[lab], "F", RM.LEAF)
            st[sname][f"{lab}-{first}"] = s(*bt.mean(g - g0, m))
    m_s = np.array([r.get("stratum") == "snapshot" for r in recs[first]])
    gl = [RM.vec(recs[lab], "O", RM.LEAF) - RM.vec(recs[lab], "F", RM.LEAF) for lab in (first, labels[-1])]
    d = gl[1] - gl[0]
    es, vs = bt.mean(d, m_s)
    eb, vb = bt.mean(d, ~m_s)
    st["snapshot_minus_bot_change"] = s(es - eb, vs - vb)
    out["G_F_change_by_stratum"] = st

    # 3. the network belief's value over Smogon
    out["FN_minus_F"] = {lab: {z: s(*bt.mean(RM.vec(recs[lab], "FN", z) - RM.vec(recs[lab], "F", z), fn_all))
                               for z in ("LEAF-K8", "TRUNC-4-K8")} for lab in labels}

    # 4. the dose curves, P-STAR vs the policy, TODAY on each
    grid = RM.PF
    curves = {}
    for lab in labels:
        curves[lab] = {}
        for key, z in (("M8|LEAF-K8", "LEAF-K8"), ("M8|TRUNC-4-K8", "TRUNC-4-K8")):
            C = np.stack([r["dk"]["F"][key][::5] if len(r["dk"]["F"][key]) == 101 else r["dk"]["F"][key]
                          for r in recs[lab]])               # LEAF-K8 on 0.01 (sliced to 0.05), TRUNC-4-K8 on 0.05
            cd = [bt.mean(C[:, i], fn_all) for i in range(len(grid))]
            ce = np.array([c[0] for c in cd])
            cv = np.stack([c[1] for c in cd], 1)
            fe, fv = bt.mean(RM.vec(recs[lab], "FN", z), fn_all)
            tp, tv = R.today_clamped(ce, fe, cv, fv, grid)
            curves[lab][key] = {"IMP_p": {f"{p:.2f}": s(ce[i], cv[:, i]) for i, p in enumerate(grid) if p in P_SHOW},
                                "P_STAR_vs_policy": pstar(ce, cv, grid),
                                "TODAY": {"p_eq": tp, "ci": R.ci(tv)}, "IMP_FN": s(fe, fv)}
    out["dose"] = curves

    # 5. gate counts
    res = json.loads((RM.E.HERE / "result_mat.json").read_text())
    out["gate_counts"] = {lab: {k: sum(1 for g in res["per_anchor"][lab]["gates"].values() if g["status"] == k)
                                for k in ("GATE-OPEN", "GATE-WEAK", "GATE-CLOSED")} for lab in labels}
    out["gate_weak_buckets"] = {lab: sorted(b for b, g in res["per_anchor"][lab]["gates"].items()
                                            if g["status"] != "GATE-CLOSED") for lab in labels}
    out["table_sha256"] = (RM.E.ARCH / "score" / "table.sha256").read_text().split()[0]
    (RM.E.HERE / "posthoc_mat.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=float) + "\n")
    print(json.dumps(out, indent=1, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
