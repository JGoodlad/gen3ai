"""P(PER-STEP ONLY) when X47 gains NOTHING between s_eq and its final (g = 0): two S reads of the same seed differ only
by measurement noise. Vectorised replica of main.x35.x47.decide's primary/secondary logic, cross-checked against the
kit's own functions on 2,000 draws."""
import sys, numpy as np
sys.path.insert(0, sys.argv[1])
from scipy.stats import t as T
from main.x35 import readout as R, x47 as X
rng = np.random.default_rng(3601)
SIG_T, SIG_M = 0.89, 0.59
def welch(a, b):
    va, vb = a.var(1, ddof=1), b.var(1, ddof=1); n = a.shape[1]
    A, B = va/n, vb/n; v = A+B; df = v*v/((A*A+B*B)/(n-1)); q = T.ppf(0.9875, df); d = a.mean(1)-b.mean(1)
    return d, d-q*np.sqrt(v), d+q*np.sqrt(v)
def cls(lo, hi):
    return np.where(lo > 2, "GAIN", np.where(hi < -2, "LOSS", np.where((lo > -2) & (hi < 2), "NULL", "UNDECIDED")))
def verdict(seq, fin, off):
    d1, l1, h1 = welch(seq[:, :3], off[:, :3]); c1 = cls(l1, h1); cont = (c1 == "UNDECIDED") & (d1 > 0)
    _, sl1, sh1 = welch(fin[:, :3], off[:, :3]); s1 = cls(sl1, sh1)
    d2, l2, h2 = welch(seq, off); c2 = cls(l2, h2); _, sl2, sh2 = welch(fin, off); s2 = cls(sl2, sh2)
    p = np.where(cont, c2, c1); s = np.where(cont, s2, s1)
    out = np.where(p == "GAIN", "ADOPT/TRADE", np.where(np.isin(p, ["NULL", "UNDECIDED"]) & (s == "GAIN"), "PER-STEP ONLY", p))
    return out, cont
N = 200000
print("delta  g | P(PER-STEP ONLY) P(primary GAIN) P(secondary GAIN at the deciding look)")
for g in (0.0, 0.5):
    for delta in (3, 4, 5, 6):
        L = rng.normal(delta, SIG_T, (N, 5)); seq = L + rng.normal(0, SIG_M, (N, 5)); fin = L + g + rng.normal(0, SIG_M, (N, 5))
        off = rng.normal(0, SIG_T, (N, 5)) + rng.normal(0, SIG_M, (N, 5))
        v, cont = verdict(seq, fin, off)
        print(f"{delta:5.1f} {g:3.1f} | {np.mean(v=='PER-STEP ONLY'):.3f}  {np.mean(v=='ADOPT/TRADE'):.3f}")
# cross-check against the kit on 2000 draws (delta 4, g 0)
L = rng.normal(4, SIG_T, (2000, 5)); seq = L + rng.normal(0, SIG_M, (2000, 5)); fin = L + rng.normal(0, SIG_M, (2000, 5)); off = rng.normal(0, SIG_T, (2000, 5)) + rng.normal(0, SIG_M, (2000, 5))
v, cont = verdict(seq, fin, off); mism = 0
seeds = [str(3601+i) for i in range(5)]
for k in range(2000):
    look = 2 if cont[k] else 1; n = 5 if look == 2 else 3
    on = {s: float(seq[k, i]) for i, s in enumerate(seeds[:n])}; of = {s: float(off[k, i]) for i, s in enumerate(seeds[:n])}
    fi = {s: float(fin[k, i]) for i, s in enumerate(seeds[:n])}
    pr = R.decide_r1(look, on, of); se = R.decide_r1(look, fi, of)
    gp = [R.GuardResult("G-F", R.PASS), R.GuardResult("G-C", R.PASS)]
    kv = X.decide(look, pr, se, gp).verdict
    kv = "ADOPT/TRADE" if kv in (R.ADOPT, R.TRADE) else kv
    if look == 1 and kv == R.CONTINUE: kv = None
    if kv is not None and kv != v[k]: mism += 1
print("cross-check mismatches vs main.x35.x47.decide:", mism, "of 2000")
