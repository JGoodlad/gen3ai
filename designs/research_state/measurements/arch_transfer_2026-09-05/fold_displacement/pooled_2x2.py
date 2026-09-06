"""The POOLED 2x2 contrast against its own REPLICATE floor, on the actual off-slice KL.

Two funded legs and two unfunded legs, so the funded-vs-unfunded contrast has a floor made of the
SAME experiment: |funded A - funded B| and |unfunded A - unfunded B|, two arms that differ only in
seed. A pooled gap inside that band is WITHIN FLOOR, never a direction.

Also: does |Delta_theta| predict the actual off-slice KL across ALL 17 arm-depths?

Run: python pooled_2x2.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
S = os.environ.get("FD_SCRATCH", "/tmp/claude-1000/-home-goodlad-dev-gen3ai/"
                                 "1bdd246b-96e2-4e73-8e09-1d9427eea286/scratchpad/fd")
d = np.load(os.path.join(S, "per_state.npz"), allow_pickle=False)
KLact, KL1 = d["KLact"], d["KL1"]
team, grp = d["team"], d["group"]
DISP = json.load(open(os.path.join(HERE, "displacement.json")))
KEYS = DISP["_meta"]["arm_keys"]
IDX = {k: i for i, k in enumerate(KEYS)}
rng = np.random.default_rng(20260905)

out = {}
for sl in ("untaught", "taught"):
    teams = sorted(set(team[grp == sl].tolist()))
    rows = [np.flatnonzero(team == t) for t in teams]
    idx = rng.integers(0, len(teams), size=(20000, len(teams)))

    def pt(keys):                                    # per-team mean, averaged over the legs
        M = np.stack([np.array([KLact[r, IDX[k]].mean() for r in rows]) for k in keys])
        return M.mean(axis=0)

    F, U = pt(["TCFUNDA@end", "TCFUNDB@end"]), pt(["TCUNFA@end", "TCUNFB@end"])
    rF = pt(["TCFUNDA@end"]) - pt(["TCFUNDB@end"])
    rU = pt(["TCUNFA@end"]) - pt(["TCUNFB@end"])

    def ci(v):
        dr = v[idx].mean(axis=1)
        return [float(np.percentile(dr, 2.5)), float(np.percentile(dr, 97.5))]

    gap, cig = float((F - U).mean()), ci(F - U)
    r1, r2 = float(rF.mean()), float(rU.mean())
    floor = max(abs(r1), abs(r2))
    verdict = ("NOT DETECTED" if cig[0] <= 0 <= cig[1]
               else ("WITHIN FLOOR" if abs(gap) <= floor else "SIGNIFICANT"))
    out[sl] = dict(n_teams=len(teams), funded_mean=float(F.mean()), unfunded_mean=float(U.mean()),
                   gap=gap, gap_ci95=cig,
                   replicate_funded=dict(delta=r1, ci95=ci(rF)),
                   replicate_unfunded=dict(delta=r2, ci95=ci(rU)),
                   replicate_floor=floor, verdict=verdict)
    print(f"=== {sl} ({len(teams)} teams)  actual KL(parent||arm)")
    print(f"  funded (A,B pooled)   {F.mean():.5f}")
    print(f"  unfunded (A,B pooled) {U.mean():.5f}")
    print(f"  GAP funded-unfunded   {gap:+.5f}  CI95 [{cig[0]:+.5f},{cig[1]:+.5f}]")
    print(f"  replicate FUNDED  A-B {r1:+.5f}  CI95 {['%+.5f' % x for x in ci(rF)]}")
    print(f"  replicate UNFUND  A-B {r2:+.5f}  CI95 {['%+.5f' % x for x in ci(rU)]}")
    print(f"  floor = {floor:.5f}   -> {verdict}")

# does raw weight-space distance predict off-slice damage across all 17 arm-depths?
L2 = np.array([DISP["arms"][k]["l2"] for k in KEYS])
for sl in ("untaught", "taught"):
    sel = np.flatnonzero(grp == sl)
    A = KLact[sel].mean(axis=0)
    K1 = KL1[sel].mean(axis=0)

    def rho(a, b):
        def r(x):
            o = np.argsort(np.asarray(x, float), kind="mergesort")
            q = np.empty(len(x))
            q[o] = np.arange(len(x))
            return q
        return float(np.corrcoef(r(a), r(b))[0, 1])
    out[sl]["across_17_arms"] = dict(
        pearson_l2_vs_actual=float(np.corrcoef(L2, A)[0, 1]), spearman_l2_vs_actual=rho(L2, A),
        pearson_kl1_vs_actual=float(np.corrcoef(K1, A)[0, 1]), spearman_kl1_vs_actual=rho(K1, A),
        pearson_l2_vs_kl1=float(np.corrcoef(L2, K1)[0, 1]))
    v = out[sl]["across_17_arms"]
    print(f"[{sl}] across all 17 arm-depths: |Delta| vs actual r={v['pearson_l2_vs_actual']:+.3f} "
          f"rho={v['spearman_l2_vs_actual']:+.3f} | KL1 vs actual "
          f"r={v['pearson_kl1_vs_actual']:+.3f} rho={v['spearman_kl1_vs_actual']:+.3f} | "
          f"|Delta| vs KL1 r={v['pearson_l2_vs_kl1']:+.3f}")

with open(os.path.join(HERE, "pooled_2x2.json"), "w") as f:
    json.dump(out, f, indent=1)
print("wrote pooled_2x2.json")
