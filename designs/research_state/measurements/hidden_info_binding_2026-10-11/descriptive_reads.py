"""THE REGISTERED DESCRIPTIVE READS D1-D8 of the pilot's existing rows (design doc §11; registered at e25a5240 before
this script was written or run). No verdict: every number here is descriptive.

Per state s (both seeds, LEAF-K8 unless stated), with the pilot's own scorer code (`score.state_arrays`,
`score.decisions`, `score_det.det_arrays`, read-only imports):
  IMP_O(s) = T(â_O) − T(top1)   (Phase 1, the oracle world)
  IMP_P(s) = T(â_P) − T(top1)   (Phase 2 Smogon worlds = "det"; Phase 2b belief-species worlds = "det_b", secondary)
  g(s)     = IMP_O(s) − IMP_P(s)
T = the mean win-scale outcome over the 24 truth seeds 16..39. CIs: the pilot's battle-clustered bootstrap
(`score.Boot`, 2,000 draws, seed 20261011), raw (unweighted) and outcome-reweighted (`score.reweights`).

    python -I descriptive_reads.py [--fair det|det_b]   →  descriptive_reads.json (+ a markdown table on stdout)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PILOT = HERE.parent / "value_disc_pilot_2026-10-11"
sys.path.insert(0, str(PILOT))
import score as S1  # noqa: E402
import score_det as SD  # noqa: E402

CPS = ("end2", "prod")
KEYS = ("ARGMAX|LEAF-K8", "ARGMAX|TRUNC-4-K8", "ARGMAX|TRUNC-4-K4", "ARGMAX|CRITIC")


def per_state(cp, fair):
    """One record per state both phases played: IMP_O / IMP_P per key, split-half T, covariates."""
    SD.DET_DIR = fair
    st, r1, _missing, _ = S1.load(cp, True)
    rd = SD.det_rows(cp)
    facts = json.loads((HERE / f"root_facts_{cp}.json").read_text())["states"]
    E = st["states"]
    import gzip
    with gzip.open(S1.ARCH / cp / "bank" / "decisions.jsonl.gz", "rt") as fh:
        bank = {d["id"]: d for d in (json.loads(x) for x in fh)}
    recs, excluded = [], []
    for did in sorted(E):
        if not r1.get(did, {}).get("ok"):
            continue
        So = S1.state_arrays(r1[did], E[did])
        do, _ = S1.decisions(So)
        T, t1 = So["truth"], So["top1"]
        tw = S1.w(So["O"][:, S1.N_LEAF:S1.N_LEAF + S1.N_TRUTH])
        Ta, Tb = tw[:, :12].mean(1), tw[:, 12:].mean(1)            # the split halves: seeds 16-27, 28-39
        f = facts[did]
        rec = {"id": did, "battle": E[did]["battle"], "cycle_opp": E[did]["cycle_opp"], "outcome": E[did]["outcome"],
               "unrevealed": int(E[did]["unrevealed"]), "turn": int(E[did]["turn"]),
               "opp_class": bank[did]["opp_class"],
               "unrevealed_check": int(f["n_unrevealed"]),
               "seen_unseen_moves": int(sum(v for v in f["unseen_moves"].values())),
               "active_unseen_moves": f["opp_active_unseen_moves"],
               "acts": So["acts"]}
        rec["imp_o"] = {k: float(T[do[k]] - T[t1]) for k in KEYS}
        rec["pick_o"] = int(So["acts"][do["ARGMAX|LEAF-K8"]])
        rec["g_half"] = None
        if not rd.get(did, {}).get("ok"):
            excluded.append(rec)
            continue
        _So, Sd = SD.det_arrays(r1[did], rd[did], E[did])
        n16 = S1.N_LEAF
        S1.N_LEAF = Sd["W"].shape[1]
        try:
            dd, _ = S1.decisions(Sd)
        finally:
            S1.N_LEAF = n16
        rec["imp_p"] = {k: float(T[dd[k]] - T[t1]) for k in KEYS}
        rec["pick_p"] = int(So["acts"][dd["ARGMAX|LEAF-K8"]])
        ko, kp = do["ARGMAX|LEAF-K8"], dd["ARGMAX|LEAF-K8"]
        rec["g_half"] = [float((Ta[ko] - Ta[t1]) - (Ta[kp] - Ta[t1])), float((Tb[ko] - Tb[t1]) - (Tb[kp] - Tb[t1]))]
        recs.append(rec)
    return st, recs, excluded


def boot_for(st, recs):
    return S1.Boot([r["battle"] for r in recs], S1.reweights(st, recs))


def stat2(bt, x, sel=None):
    return {"raw": bt.stat(x, sel=sel), "reweighted": bt.stat(x, sel=sel, weighted=True)}


def strata(bt, recs, g, label, fn):
    out = {}
    for lab in sorted({fn(r) for r in recs if fn(r) is not None}, key=str):
        m = np.array([fn(r) == lab for r in recs])
        if m.sum() >= 5:
            out[str(lab)] = {"n": int(m.sum()), "G": bt.stat(g, sel=m),
                             "imp_o": bt.stat(np.array([r["imp_o"]["ARGMAX|LEAF-K8"] for r in recs]), sel=m),
                             "imp_p": bt.stat(np.array([r["imp_p"]["ARGMAX|LEAF-K8"] for r in recs]), sel=m)}
    return {label: out}


def bin_of(x, edges):
    """edges = [(label, lo, hi)] inclusive."""
    if x is None:
        return None
    for lab, lo, hi in edges:
        if lo <= x <= hi:
            return lab
    return None


def read(cp, fair):
    st, recs, excl = per_state(cp, fair)
    bt = boot_for(st, recs)
    g = np.array([r["imp_o"]["ARGMAX|LEAF-K8"] - r["imp_p"]["ARGMAX|LEAF-K8"] for r in recs])
    res = {"cp": cp, "fair": fair, "n_states": len(recs), "n_excluded": len(excl),
           "unrevealed_mismatch": sum(r["unrevealed"] != r["unrevealed_check"] for r in recs + excl),
           "G_all": stat2(bt, g)}
    # D1: unrevealed slots separately
    res.update(strata(bt, recs, g, "D1_unrevealed", lambda r: r["unrevealed"] if r["unrevealed"] < 4 else "4-5"))
    # D2: unseen move slots of SEEN mons, overall and within 0-1 unrevealed
    e2 = [("0-2", 0, 2), ("3-5", 3, 5), ("6-8", 6, 8), ("9+", 9, 99)]
    res.update(strata(bt, recs, g, "D2_seen_unseen_moves", lambda r: bin_of(r["seen_unseen_moves"], e2)))
    sub = [r for r in recs if r["unrevealed"] <= 1]
    if sub:
        bts = boot_for(st, sub)
        gs = np.array([r["imp_o"]["ARGMAX|LEAF-K8"] - r["imp_p"]["ARGMAX|LEAF-K8"] for r in sub])
        res.update(strata(bts, sub, gs, "D2_within_unrev_0-1", lambda r: bin_of(r["seen_unseen_moves"], e2)))
    # D3: the opponent ACTIVE's unseen move slots
    res.update(strata(bt, recs, g, "D3_active_unseen_moves",
                      lambda r: None if r["active_unseen_moves"] is None else
                      (str(r["active_unseen_moves"]) if r["active_unseen_moves"] < 3 else "3-4")))
    # D4: phase by turn, opponent class, traced outcome
    res.update(strata(bt, recs, g, "D4_turn", lambda r: bin_of(r["turn"], [("2-10", 2, 10), ("11-25", 11, 25),
                                                                          ("26-40", 26, 40)])))
    res.update(strata(bt, recs, g, "D4_opp_class", lambda r: r["opp_class"]))
    res.update(strata(bt, recs, g, "D4_outcome", lambda r: r["outcome"]))
    # D5: split-half concentration: rank on seeds 16-27, measure the share of G (seeds 28-39) in the top 10 % / 25 %
    ga = np.array([r["g_half"][0] for r in recs])
    gb = np.array([r["g_half"][1] for r in recs])
    order = np.argsort(-ga, kind="stable")
    d5 = {"sum_g_half_b": float(gb.sum())}
    for q in (0.10, 0.25):
        k = int(round(q * len(recs)))
        top = np.zeros(len(recs), bool)
        top[order[:k]] = True
        nb = len(bt.ub)
        Sa = np.bincount(bt.bi[top], weights=gb[top], minlength=nb)
        Sall = np.bincount(bt.bi, weights=gb, minlength=nb)
        ratio = (bt.C @ Sa) / (bt.C @ Sall)
        d5[f"top_{int(q * 100)}pct_share_of_G"] = {"mean": round(float(Sa.sum() / Sall.sum()), 4),
                                                   "ci": np.percentile(ratio, [2.5, 97.5]).round(4).tolist(),
                                                   "k": k}
    res["D5_concentration_split_half"] = d5
    # D6: the excluded states (no 8 fair worlds)
    allr = recs + excl
    bta = boot_for(st, allr)
    io_all = np.array([r["imp_o"]["ARGMAX|LEAF-K8"] for r in allr])
    is_ex = np.array([False] * len(recs) + [True] * len(excl))
    g_worst = np.concatenate([g, np.zeros(len(excl))])
    res["D6_excluded"] = {"n": len(excl), "imp_o_excluded": bta.stat(io_all, sel=is_ex) if len(excl) >= 2 else None,
                          "imp_o_included": bta.stat(io_all, sel=~is_ex),
                          "R-INFO_worst_case (excluded g = 0)": bta.stat(g_worst),
                          "unrevealed_excluded": sorted(r["unrevealed"] for r in excl)}
    # D7: the oracle pick's kind, and whether the oracle and fair picks differ
    res.update(strata(bt, recs, g, "D7_oracle_pick_kind", lambda r: "move" if r["pick_o"] >= 6 else "switch"))
    res.update(strata(bt, recs, g, "D7_picks_differ", lambda r: r["pick_o"] != r["pick_p"]))
    # D8: H-HORIZON: the gap at TRUNC-4 vs LEAF-K8
    d8 = {}
    for k in ("ARGMAX|TRUNC-4-K8", "ARGMAX|TRUNC-4-K4", "ARGMAX|CRITIC"):
        gk = np.array([r["imp_o"][k] - r["imp_p"][k] for r in recs])
        nb = len(bt.ub)
        Sk = np.bincount(bt.bi, weights=gk, minlength=nb)
        S8 = np.bincount(bt.bi, weights=g, minlength=nb)
        ratio = (bt.C @ Sk) / (bt.C @ S8)
        d8[k] = {"G": bt.stat(gk), "ratio_to_LEAF-K8_G": {"mean": round(float(Sk.sum() / S8.sum()), 4),
                                                          "ci": np.percentile(ratio, [2.5, 97.5]).round(4).tolist()}}
    res["D8_horizon"] = d8
    return res, recs


def pooled(results_recs):
    """G by D1 / D2-within / D3 pooled over both checkpoints (battle ids are distinct across checkpoints)."""
    recs = [r for _, rs in results_recs for r in rs]
    bt = S1.Boot([r["battle"] for r in recs], [1.0] * len(recs))
    g = np.array([r["imp_o"]["ARGMAX|LEAF-K8"] - r["imp_p"]["ARGMAX|LEAF-K8"] for r in recs])
    out = {"n_states": len(recs), "G_all": bt.stat(g)}
    out.update(strata(bt, recs, g, "D1_unrevealed", lambda r: r["unrevealed"] if r["unrevealed"] < 4 else "4-5"))
    e2 = [("0-2", 0, 2), ("3-5", 3, 5), ("6-8", 6, 8), ("9+", 9, 99)]
    out.update(strata(bt, recs, g, "D2_seen_unseen_moves", lambda r: bin_of(r["seen_unseen_moves"], e2)))
    sub = [r for r in recs if r["unrevealed"] <= 1]
    bts = S1.Boot([r["battle"] for r in sub], [1.0] * len(sub))
    gs = np.array([r["imp_o"]["ARGMAX|LEAF-K8"] - r["imp_p"]["ARGMAX|LEAF-K8"] for r in sub])
    out.update(strata(bts, sub, gs, "D2_within_unrev_0-1", lambda r: bin_of(r["seen_unseen_moves"], e2)))
    out.update(strata(bt, recs, g, "D3_active_unseen_moves",
                      lambda r: None if r["active_unseen_moves"] is None else
                      (str(r["active_unseen_moves"]) if r["active_unseen_moves"] < 3 else "3-4")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fair", default="det", choices=("det", "det_b"))
    a = ap.parse_args()
    rr = [read(cp, a.fair) for cp in CPS]
    res = {"fair": a.fair, "per_cp": [r for r, _ in rr], "pooled": pooled(rr)}
    name = "descriptive_reads.json" if a.fair == "det" else f"descriptive_reads_{a.fair}.json"
    (HERE / name).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    for r in res["per_cp"]:
        print(r["cp"], "G_all", r["G_all"]["raw"], "excluded", r["n_excluded"], "unrevealed mismatches",
              r["unrevealed_mismatch"])


if __name__ == "__main__":
    main()
