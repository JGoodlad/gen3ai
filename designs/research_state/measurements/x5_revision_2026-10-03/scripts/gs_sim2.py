import numpy as np
from gs_sim import ALPHA, LOOKS, obf_z, sim, t_bounds
C, zb = obf_z(ALPHA)
dfs = [2 * (m - 1) for m in LOOKS]
# calibrate a common scale on the OBF z-constant so the WORST type I over the sigma x sig_int grid is <= 0.05
grid = [(s, si) for s in (2.5, 3.43, 4.8) for si in (0.0, 1.5)]
def worst(scale, n=300_000):
    tb = t_bounds(zb * scale, dfs)
    return max(np.mean(sim("cross", s, 3.5, -3.5, tb, sig_int=si, n=n)[0] == 1) for s, si in grid), tb
for sc in (1.0, 1.01, 1.02, 1.03):
    w, tb = worst(sc); print(f"scale {sc}: worst typeI {w:.4f} t-bounds {np.round(tb,3)}")
