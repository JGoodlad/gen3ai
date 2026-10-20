"""D1 H with an overdispersion widening (quasi-binomial): φ = mean over arms of (between-seed variance of per-seed
panel means) / (binomial variance expected), df = 2(n-1); the Newcombe half-widths are scaled by sqrt(max(1, φ)).
Compared with the registered H (no widening). Look 1 (3 seeds) then look 2 (5) on CONTINUE (Δ̂ > 0)."""
import sys, math, numpy as np
sys.path.insert(0, sys.argv[1])
from main.x35 import d1 as D, readout as R
rng = np.random.default_rng(11)
BASE = {"heldout72_hpfix": 0.395, "forum72": 0.40, "p72": 0.42}; NG = 140
def sample(delta, sig, ns):
    o = {}
    for s in range(ns):
        u = rng.normal(0, sig)
        o[str(s)] = {k: (int(rng.binomial(NG, min(max(p + delta + u, .01), .99))), NG) for k, p in BASE.items()}
    return o
def phi(on, off):
    r = []
    for d in (on, off):
        m = np.array([D.panel_mean(c) for c in d.values()])
        exp = np.mean([sum(x / n * (1 - x / n) / n for x, n in c.values()) / 9 for c in d.values()])
        r.append(m.var(ddof=1) / exp)
    return max(1.0, float(np.mean(r)))
def h(look, on, off, widen):
    hd = D.d1_primary(look, on, off)
    if not widen or hd.ci_pp is None:
        return hd.outcome
    f = math.sqrt(phi(on, off)); d = hd.delta_pp
    lo, hi = d - (d - hd.ci_pp[0]) * f, d + (hd.ci_pp[1] - d) * f
    cls, _ = D.classify_h((lo, hi), hd.sign_ok)
    if cls == R.UNDECIDED and look == 1 and d > 0:
        return R.CONTINUE
    return cls
def run(delta, sig, widen, n=1200):
    c = {}
    for _ in range(n):
        on, off = sample(delta, sig, 5), sample(0, sig, 5)
        o = h(1, {k: on[k] for k in "012"}, {k: off[k] for k in "012"}, widen)
        if o == R.CONTINUE:
            o = h(2, on, off, widen)
        c[o] = c.get(o, 0) + 1
    return round(c.get("GAIN", 0) / n, 3), round(c.get("LOSS", 0) / n, 3)
print("Δ σ_seed | registered (GAIN, LOSS) | widened (GAIN, LOSS)")
for sig in (0.0, 0.02, 0.04):
    for d in (0.0, 0.05):
        print(f"{100*d:3.0f} {100*sig:3.0f} | {run(d, sig, False)} | {run(d, sig, True)}")
