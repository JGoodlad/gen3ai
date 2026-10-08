"""POST-HOC diagnostics of T1 (README §3). These are NOT part of the registered verdict. They were written
AFTER result.md showed T1 = BITING for blob, to separate two accounts that produce the same signature:

  (M) MOTIVATED COGNITION: route C bends the belief toward what justifies the preferred action;
  (N) ACTING ON NOISE: the head is honest but noisy, the policy reads it, so rows where the head
      happened to under-estimate the threat are the rows where the policy stays (selection on the
      belief's own error; no gradient from PPO into the head is needed).

D0  determinism: rows_v2 reproduces rows/ (top1, e1, d2, e3) exactly.
D1  cross-policy split: head of seed s, STAY/SWITCH from the policy of ANOTHER seed s' of the SAME arm
    (mean over s' != s), and from the matched seed of the OTHER arm. (N) predicts the gap collapses to
    the state-selection level (other policies do not act on THIS head's noise); (M) predicts it survives
    within the arm (the arm's heads are bent in the states its policies like to stay in).
D2  within-belief-bin gap (unit level, bins of p_arm on REL_EDGES): mean (p − y) | STAY − | SWITCH inside
    each p bin, unit-weighted. A head calibrated given its own p, read by a policy, gives ~0 here under
    (N) when the stay choice runs through p; (M) leaves a gap at fixed p.
D3  decomposition: Δ(Σp_arm), Δ(Σp_prior), Δ(Σy) between STAY and SWITCH rows (Δ1 = Δ(Σp) − Δ(Σy)).

    python3 explore.py [--rows rows_v2] [--ref rows]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analyze import ARMS, BOOT_SEED, SEEDS, TIE_EPS, tci

REL = np.array([0, .01, .02, .05, .1, .2, .3, .5, .7, .9, 1.0])


def main():
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    ap.add_argument("--rows", default=str(here / "rows_v2"))
    ap.add_argument("--ref", default=str(here / "rows"))
    ap.add_argument("--boot", type=int, default=1000)
    a = ap.parse_args()
    rows, ref = Path(a.rows), Path(a.ref)
    meta = np.load(rows / "_bank_meta.npz")
    bi, free, masks = meta["battle_index"], meta["free"], meta["masks"]
    nb = int(bi.max()) + 1
    base = free & masks[:, 6:11].any(1) & masks[:, 0:6].any(1)
    D = {}
    d0 = {}
    for arm in ARMS:
        for s in SEEDS:
            lab = f"{arm}_s{s}"
            R = dict(np.load(rows / f"{lab}.npz"))
            Q = dict(np.load(ref / f"{lab}.npz"))
            d0[lab] = all(np.array_equal(R[k], Q[k], equal_nan=True)
                          for k in ("top1", "e1_arm", "e1_prior", "d2_arm", "d2_prior", "e3_arm", "e3_prior"))
            R["elig"] = base & (R["margin"] >= TIE_EPS)
            R["stay"] = R["top1"] >= 6
            D[lab] = R
    rng = np.random.default_rng(BOOT_SEED)
    W = np.stack([np.bincount(rng.integers(0, nb, nb), minlength=nb).astype(float) for _ in range(a.boot)])

    def gap(v, ok, stay, w=None):
        """STAY − SWITCH mean of v over ok rows; w: [B, nb] battle weights → [B]."""
        m1, m0 = ok & stay, ok & ~stay
        if w is None:
            return v[m1].mean() - v[m0].mean()
        s1 = np.bincount(bi, np.where(m1, v, 0), nb); n1 = np.bincount(bi, m1.astype(float), nb)
        s0 = np.bincount(bi, np.where(m0, v, 0), nb); n0 = np.bincount(bi, m0.astype(float), nb)
        return (w @ s1) / (w @ n1) - (w @ s0) / (w @ n0)

    out = {"D0_identical_to_registered_rows": d0, "D1": {}, "D2": {}, "D3": {}}
    for arm in ARMS:
        other = "fm" if arm == "blob" else "blob"
        own, same, cross = [], [], []
        bo_own, bo_same, bo_cross = np.zeros(a.boot), np.zeros(a.boot), np.zeros(a.boot)
        for s in SEEDS:
            H = D[f"{arm}_s{s}"]
            v = H["e1_arm"] - H["e1_prior"]
            fin = np.isfinite(v)

            def g(P, w=None):
                ok = fin & H["elig"] & P["elig"]
                return gap(v, ok, P["stay"], w)
            own.append(g(H)); bo_own += g(H, W) / len(SEEDS)
            sv = [g(D[f"{arm}_s{t}"]) for t in SEEDS if t != s]
            same.append(np.mean(sv))
            bo_same += np.mean([g(D[f"{arm}_s{t}"], W) for t in SEEDS if t != s], axis=0) / len(SEEDS)
            cross.append(g(D[f"{other}_s{s}"])); bo_cross += g(D[f"{other}_s{s}"], W) / len(SEEDS)
        res = {}
        for name, x, bo in (("own_policy", own, bo_own), ("same_arm_other_seed_policy", same, bo_same),
                            ("other_arm_matched_seed_policy", cross, bo_cross)):
            r = tci(x)
            r["boot_lo"], r["boot_hi"] = (float(q) for q in np.percentile(bo, [2.5, 97.5]))
            res[name] = r
        out["D1"][arm] = res
        # D2: within p_arm bins, unit level (Δ_delta-free: the arm's own p and its own error)
        d2 = {"arm": [], "prior": []}
        for s in SEEDS:
            H = D[f"{arm}_s{s}"]
            r_ = H["u_row"]
            ok = H["elig"][r_]
            st = H["stay"][r_]
            for c in ("arm", "prior"):
                p = H[f"u_p_{c}"]
                e = p - H["u_y"]
                b = np.clip(np.searchsorted(REL, p, side="right") - 1, 0, 9)
                tot, acc = 0.0, 0.0
                for k in range(10):
                    m1 = ok & st & (b == k)
                    m0 = ok & ~st & (b == k)
                    if m1.sum() and m0.sum():
                        n = m1.sum() + m0.sum()
                        acc += n * (e[m1].mean() - e[m0].mean())
                        tot += n
                d2[c].append(acc / tot)
        out["D2"][arm] = {c: tci(d2[c]) for c in d2}
        out["D2"][arm]["delta"] = tci(np.subtract(d2["arm"], d2["prior"]))
        # D3: decomposition of Δ1 into the belief's move and the truth's move
        d3 = {"sum_p_arm": [], "sum_p_prior": [], "sum_y": [], "n_theta": [], "p_stay_mean": []}
        for s in SEEDS:
            H = D[f"{arm}_s{s}"]
            ok = H["elig"] & np.isfinite(H["e1_arm"])
            d3["sum_p_arm"].append(gap(H["s1_arm"], ok, H["stay"]))
            d3["sum_p_prior"].append(gap(H["s1_prior"], ok, H["stay"]))
            d3["sum_y"].append(gap(H["y1"], ok, H["stay"]))
            d3["n_theta"].append(gap(H["n_theta"], ok, H["stay"]))
            d3["p_stay_mean"].append(float(H["p_stay"][ok].mean()))
        out["D3"][arm] = {k: tci(v) for k, v in d3.items()}
    (rows.parent / "explore.json").write_text(json.dumps(out, indent=1))
    L = ["# Post-hoc T1 diagnostics (generated by `explore.py`; NOT part of the registered verdict)", "",
         f"D0, rows_v2 reproduces the registered rows exactly: {all(d0.values())} "
         f"({sum(d0.values())}/{len(d0)} checkpoints)", "",
         "## D1: Δ1_delta with the STAY/SWITCH split taken from a DIFFERENT policy", "",
         "| head | split by | Δ1_delta [t95] | bootstrap 95 % |", "|---|---|---|---|"]
    for arm, res in out["D1"].items():
        for k, r in res.items():
            L.append(f"| {arm} | {k} | {r['mean']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] | "
                     f"[{r['boot_lo']:+.4f}, {r['boot_hi']:+.4f}] |")
    L += ["", "## D2: STAY − SWITCH error at FIXED belief (unit level, p bins)", "",
          "| arm | ARM column | PRIOR column (binned on its own p) | ARM − PRIOR |", "|---|---|---|---|"]
    for arm, r in out["D2"].items():
        L.append(f"| {arm} | " + " | ".join(f"{r[c]['mean']:+.4f} [{r[c]['lo']:+.4f}, {r[c]['hi']:+.4f}]"
                                            for c in ("arm", "prior", "delta")) + " |")
    L += ["", "## D3: what moves between STAY and SWITCH rows (STAY − SWITCH)", "",
          "| arm | Σp arm | Σp prior | Σy (truth) | |Θ| | mean P(stay) |", "|---|---|---|---|---|---|"]
    for arm, r in out["D3"].items():
        L.append(f"| {arm} | " + " | ".join(f"{r[k]['mean']:+.4f} [{r[k]['lo']:+.4f}, {r[k]['hi']:+.4f}]"
                                            for k in ("sum_p_arm", "sum_p_prior", "sum_y", "n_theta",
                                                      "p_stay_mean")) + " |")
    (rows.parent / "explore.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
