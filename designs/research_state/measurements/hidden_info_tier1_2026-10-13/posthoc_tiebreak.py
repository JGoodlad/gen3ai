"""POST HOC (read after unblinding; no verdict, no rule): how much of the dose curves' first step is TIE-BREAKING.

An 8-world LEAF-K8 score is a mean of 8 win / loss / draw outcomes, so several actions often share the maximum; the
registered tie rule then picks the policy's first. Any p > 0 of a second component (the true world in §6.9's M8, the
F+X worlds in Amendment 3's class curves) breaks those ties by THAT component — a lexicographic effect present at an
infinitesimal p, not a dose. This read reports, on the frozen table: the share of states whose honest (F) LEAF-K8 argmax
is tied; IMP at p = 0, 0.001, 0.01, 0.05, 0.1 for M8 (true world) and for the M and U class curves; and the curves'
rise measured from p = 0.001 instead of p = 0 (P-STAR vs the policy and vs the critic at the same p, on the owner's grid).

    python -I posthoc_tiebreak.py   → posthoc_tiebreak.json
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1_common as T  # noqa: E402
import t1_data as D  # noqa: E402
import t1_stats as ST  # noqa: E402
import unblind_t1 as U  # noqa: E402
from score_blind import imp_of  # noqa: E402

GRID = np.array([0.0, 0.001, 0.01, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0])


def main():
    tab, name, _ = U.load()
    P = U.Frame([r for cp in T.CPS for r in tab[cp]["recs"]], True)
    fc = name["F"]
    truth = {}
    for cp in T.CPS:
        meta = D.stage_meta(cp)
        r1 = {**D.phase1_rows(cp, "registered"), **D.phase1_rows(cp, "supp")}
        for did, m in meta.items():
            if r1.get(did, {}).get("ok"):
                S1 = D.truth_arrays(r1[did], m)
                o = np.array(S1["order"])
                truth[did] = {"W": S1["W"][o][:, :8].mean(1), "crit": S1["crit"][o].mean(1)}
    curves = {k: np.full((P.n, len(GRID)), np.nan) for k in ("M8", "M8crit", "M", "Mcrit", "U", "Ucrit")}
    tied = np.zeros(P.n, bool)
    for i, r in enumerate(P.recs):
        wf = r["w"].get(fc)
        if wf is None or r["id"] not in truth:
            continue
        hf, cf = wf["W"].astype(float).mean(1), wf["crit"].astype(float).mean(1)
        tied[i] = np.sum(hf >= hf.max() - D.TIE) > 1
        tr = truth[r["id"]]
        curves["M8"][i] = imp_of(GRID[:, None] * tr["W"][None] + (1 - GRID)[:, None] * hf[None], r["Tp"])
        curves["M8crit"][i] = imp_of(GRID[:, None] * tr["crit"][None] + (1 - GRID)[:, None] * cf[None], r["Tp"])
        for X, lab in (("M", "F+M"), ("U", "F+U")):
            wx = r["w"].get(name[lab])
            if wx is None:
                continue
            hx, cx = wx["W"].astype(float).mean(1), wx["crit"].astype(float).mean(1)
            curves[X][i] = imp_of(GRID[:, None] * hx[None] + (1 - GRID)[:, None] * hf[None], r["Tp"])
            curves[X + "crit"][i] = imp_of(GRID[:, None] * cx[None] + (1 - GRID)[:, None] * cf[None], r["Tp"])
    ok = np.isfinite(curves["M8"][:, 0])
    out = {"status": "POST HOC, descriptive (read after unblinding)", "states": int(ok.sum()),
           "honest_leaf_argmax_tied_share": ST.summary(*P.bt.mean(tied.astype(float), ok)), "curves": {}}
    for k in ("M8", "M", "U"):
        m = ok & np.isfinite(curves[k][:, 0])
        cur = {f"{p:g}": ST.summary(*P.bt.mean(curves[k][:, j], m)) for j, p in enumerate(GRID)}
        jump = P.bt.mean(curves[k][:, 1] - curves[k][:, 0], m)
        chg = P.bt.mean((curves[k][:, 1] != curves[k][:, 0]).astype(float), m)
        # P-STAR measured from the tie-broken start p = 0.001: the smallest owner-grid p >= 0.1 from which the leaf beats
        # the policy / the critic at the same p (rule 8), its rise over p = 0.001
        res = {"n": int(m.sum()), "curve": cur, "jump_0_to_0.001": ST.summary(*jump),
               "share_of_states_whose_pick_moves_at_0.001": ST.summary(*chg)}
        for comp, key in (("vs_policy", None), ("vs_critic_same_p", k + "crit" if k != "M8" else "M8crit")):
            mask = []
            for j, p in enumerate(GRID):
                if p < 0.1:
                    continue
                d = curves[k][:, j] - (curves[key][:, j] if key else 0.0)
                mask.append(ST.gt(ST.ci(P.bt.mean(d, m)[1])[0], 0.0005))
            i0 = U.persistent_first(np.array(mask))
            res[f"P-STAR_from_owner_grid_{comp}"] = None if i0 is None else float(GRID[GRID >= 0.1][i0])
        res["rise_over_0.001_at_0.1"] = ST.summary(*P.bt.mean(curves[k][:, 4] - curves[k][:, 1], m))
        res["rise_over_0.001_at_1"] = ST.summary(*P.bt.mean(curves[k][:, -1] - curves[k][:, 1], m))
        out["curves"][k] = res
    (T.HERE / "posthoc_tiebreak.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: {x: v[x] for x in ("jump_0_to_0.001", "share_of_states_whose_pick_moves_at_0.001",
                                             "rise_over_0.001_at_0.1", "rise_over_0.001_at_1")} for k, v in out["curves"].items()},
                     indent=1))
    print("tied share", out["honest_leaf_argmax_tied_share"])
    for k, v in out["curves"].items():
        print(k, {p: v["curve"][p]["mean"] for p in v["curve"]})


if __name__ == "__main__":
    main()
