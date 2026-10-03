import numpy as np, json
from scipy import stats
from gs_sim import ALPHA, LOOKS, obf_z, runs_cost, sim, t_bounds
C, zb = obf_z(ALPHA); SC = 1.02
dfs = [2 * (m - 1) for m in LOOKS]; tb = t_bounds(zb * SC, dfs)
print("FINAL t-bounds", np.round(tb, 3), "nominal one-sided p", [f"{1-stats.t.cdf(b,df):.4g}" for b, df in zip(tb, dfs)])
N = 200_000
print("type I at small sigma (cell noise dominant):")
for s in (0.5, 1.0, 1.5):
    for si in (0.0, 1.5):
        print(f"  sig {s} sig_int {si}: {np.mean(sim('cross', s, 3.5, -3.5, tb, sig_int=si, n=N)[0]==1):.4f}")
rows = []
for delta in (3.5, 4.5, 5.5):
    for s in (2.5, 3.43, 4.8):
        for fut in (None, 1.0, 0.5):
            r0, l0 = sim("cross", s, delta, -delta, tb, fut=fut, n=N)
            r1, l1 = sim("cross", s, delta, 0.0, tb, fut=fut, n=N)
            rec = dict(delta=delta, sig=s, fut=fut, typeI=float(np.mean(r0 == 1)), power=float(np.mean(r1 == 1)),
                       stop_by_look_H1=[float(np.mean((r1 == 1) & (l1 == k))) for k in range(3)],
                       futility_stop_H1=float(np.mean(r1 == 2)), futility_stop_H0=float(np.mean(r0 == 2)),
                       EGPUh_H1=float(runs_cost(l1)), EGPUh_H0=float(runs_cost(l0)))
            rows.append(rec)
            print(f"delta {delta} sig {s} fut {fut}: typeI {rec['typeI']:.4f} power {rec['power']:.3f} "
                  f"(NI at looks {np.round(rec['stop_by_look_H1'],3)}) fut-stop H1 {rec['futility_stop_H1']:.3f} H0 {rec['futility_stop_H0']:.3f} "
                  f"E[GPU-h] H1 {rec['EGPUh_H1']:.1f} H0 {rec['EGPUh_H0']:.1f}")
# paired design power for comparison (its own boundaries, sim-verified earlier ~0.053-0.055 unscaled)
tbp = t_bounds(zb * SC, [m - 1 for m in LOOKS])
for s in (2.5, 3.43, 4.8):
    r1, _ = sim("paired", s, 3.5, 0.0, tbp, n=N); r0, _ = sim("paired", s, 3.5, -3.5, tbp, n=N)
    print(f"PAIRED delta 3.5 sig {s}: typeI {np.mean(r0==1):.4f} power {np.mean(r1==1):.3f}")
# fixed-n (single look at n=8) reference
for s in (2.5, 3.43, 4.8):
    tb8 = np.array([np.inf, np.inf, stats.t.ppf(0.95, 14)])
    r1, _ = sim("cross", s, 3.5, 0.0, tb8, n=N); r0, _ = sim("cross", s, 3.5, -3.5, tb8, n=N)
    print(f"FIXED n=8 cross delta 3.5 sig {s}: typeI {np.mean(r0==1):.4f} power {np.mean(r1==1):.3f}")
json.dump({"t_bounds": tb.tolist(), "scale": SC, "z_obf": (zb*SC).tolist(), "rows": rows}, open("../out/gs_results.json", "w"), indent=1)
