import numpy as np
from gs_sim import ALPHA, LOOKS, obf_z, runs_cost, sim, t_bounds
C, zb = obf_z(ALPHA); dfs = [2 * (m - 1) for m in LOOKS]; tb = t_bounds(zb * 1.02, dfs)
for s in (5.21, 8.0):
    for delta in (3.5, 4.5, 5.5):
        r0, l0 = sim("cross", s, delta, -delta, tb, fut=1.0, n=200_000); r1, l1 = sim("cross", s, delta, 0.0, tb, fut=1.0, n=200_000)
        print(f"sig {s} delta {delta}: typeI {np.mean(r0==1):.4f} power {np.mean(r1==1):.3f} E[GPU-h] H1 {runs_cost(l1):.1f} H0 {runs_cost(l0):.1f}")
