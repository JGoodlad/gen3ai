"""OC of D1's registered rule (main.x35.d1): H (Kakuna, unfamiliar panels, pooled binomial + sign rule) and the
co-primary P (Welch 97.5 % non-inferiority at -2.0 pp). Uses the module's own functions (look 1 = 3 seeds; look 2 = 5)."""
import sys, numpy as np
sys.path.insert(0, sys.argv[1])
from main.x35 import d1 as D, readout as R
rng = np.random.default_rng(7)
BASE = {"heldout72_hpfix": 0.395, "forum72": 0.40, "p72": 0.42}   # ASSUMED levels (heldout from the X26 read)
NG = 140
def sample(delta, sig_s, nseeds):
    out = {}
    for s in range(nseeds):
        u = rng.normal(0, sig_s)
        out[str(s)] = {k: (int(rng.binomial(NG, min(max(p + delta + u, 0.01), 0.99))), NG) for k, p in BASE.items()}
    return out
def run(delta, sig_s, n=1500):
    c = {}
    for _ in range(n):
        on, off = sample(delta, sig_s, 5), sample(0.0, sig_s, 5)
        h1 = D.d1_primary(1, {k: on[k] for k in "012"}, {k: off[k] for k in "012"})
        h = D.d1_primary(2, on, off) if h1.outcome == R.CONTINUE else h1
        c[h.outcome] = c.get(h.outcome, 0) + 1
    return {k: round(v / n, 3) for k, v in sorted(c.items())}
print("H: Δ(pp)  σ_seed(pp) -> final outcome shares (look 1, then look 2 on CONTINUE)")
for sig in (0.0, 0.02, 0.04):
    for d in (0.0, 0.03, 0.05, 0.08):
        print(f"  {100*d:4.0f} {100*sig:4.0f}  {run(d, sig)}")
# P: non-inferiority, Welch 97.5 %, lower > -2.0, Δ_P true 0
print("P (non-inferiority, true Δ_P = 0): P(NON-INFERIOR) at 3 and 5 seeds")
for sig in (0.8, 1.2, 1.6, 2.0):
    hit3 = hit5 = 0; N = 4000
    for _ in range(N):
        on, off = rng.normal(0, sig, 5), rng.normal(0, sig, 5)
        hit3 += R.welch_interval(list(on[:3]), list(off[:3])).ci[0] > -2.0
        hit5 += R.welch_interval(list(on), list(off)).ci[0] > -2.0
    print(f"  σ {sig}: 3v3 {hit3/N:.3f}  5v5 {hit5/N:.3f}")
