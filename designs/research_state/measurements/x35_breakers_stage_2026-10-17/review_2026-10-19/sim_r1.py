"""OC of X35's R1 (Welch 97.5 %, ±2.0 bar, look 1 = 3v3, look 2 = 5v5 if UNDECIDED and Δ̂>0) under a normal
per-final model S ~ N(mu, sigma^2). Vectorised; cross-checked against main.x35.readout.decide_r1."""
import sys, numpy as np
from scipy.stats import t as T
sys.path.insert(0, sys.argv[1])
from main.x35 import readout as R

rng = np.random.default_rng(20261019)
N = 40000
BAR = 2.0

def welch(on, off, conf=0.975):
    n1, n2 = on.shape[1], off.shape[1]
    a = on.var(axis=1, ddof=1) / n1; b = off.var(axis=1, ddof=1) / n2
    v = a + b; df = v * v / (a * a / (n1 - 1) + b * b / (n2 - 1))
    q = T.ppf(1 - (1 - conf) / 2, df); d = on.mean(1) - off.mean(1); se = np.sqrt(v)
    return d, d - q * se, d + q * se

def cls(lo, hi):
    c = np.full(lo.shape, "UND", dtype=object)
    c[lo > BAR] = "GAIN"; c[hi < -BAR] = "LOSS"; c[(lo > -BAR) & (hi < BAR)] = "NULL"
    return c

def oc(delta, sigma, n1=3, n2=5):
    on = rng.normal(delta, sigma, (N, n2)); off = rng.normal(0.0, sigma, (N, n2))
    d1, lo1, hi1 = welch(on[:, :n1], off[:, :n1]); c1 = cls(lo1, hi1)
    cont = (c1 == "UND") & (d1 > 0)
    d2, lo2, hi2 = welch(on, off); c2 = cls(lo2, hi2)
    final = np.where(cont, c2, c1)
    out = {k: float(np.mean(final == k)) for k in ("GAIN", "NULL", "LOSS", "UND")}
    out["look1_decided"] = float(np.mean(c1 != "UND")); out["went_look2"] = float(np.mean(cont))
    out["hw1_median"] = float(np.median((hi1 - lo1) / 2)); out["hw2_median"] = float(np.median((hi2 - lo2) / 2))
    return out, (on, off, c1)

# cross-check the vectorised class against the registered decide_r1 on 2,000 draws
_, (on, off, c1) = oc(2.5, 1.2)
bad = 0
for i in range(2000):
    r = R.decide_r1(1, {str(s): float(on[i, s]) for s in range(3)}, {str(s): float(off[i, s]) for s in range(3)})
    want = {"GAIN": "GAIN", "NULL": "NULL", "LOSS": "LOSS", "UNDECIDED": "UND"}[r.cls]
    bad += want != c1[i]
print(f"cross-check vs decide_r1: {bad} mismatches / 2000")

print("sigma  Delta | P(GAIN) P(NULL) P(LOSS) P(UND) | look1 decided  ->look2 | median half-width L1 / L2")
for sigma in (0.8, 1.2, 1.6, 2.0, 2.5):
    for delta in (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0):
        o, _ = oc(delta, sigma)
        print(f"{sigma:4.1f} {delta:5.1f} | {o['GAIN']:.3f} {o['NULL']:.3f} {o['LOSS']:.3f} {o['UND']:.3f} | "
              f"{o['look1_decided']:.3f} {o['went_look2']:.3f} | {o['hw1_median']:.2f} / {o['hw2_median']:.2f}")
