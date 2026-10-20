import numpy as np
from scipy.stats import t as T
rng = np.random.default_rng(20261020)
N = 200000
def welch(a, b, conf=0.975):
    va, vb = a.var(axis=1, ddof=1), b.var(axis=1, ddof=1)
    na, nb = a.shape[1], b.shape[1]
    A, B = va/na, vb/nb
    v = A + B
    df = v*v/(A*A/(na-1) + B*B/(nb-1))
    q = T.ppf(1-(1-conf)/2, df)
    d = a.mean(1) - b.mean(1)
    se = np.sqrt(v)
    return d, d - q*se, d + q*se
def cls(lo, hi, bar=2.0):
    return np.where(lo > bar, "GAIN", np.where(hi < -bar, "LOSS", np.where((lo > -bar) & (hi < bar), "NULL", "UND")))
print("sigma  delta | look1: GAIN NULL LOSS UND(<=0,final) CONT | final (after look2): GAIN NULL LOSS UND | P(any NULL)")
for sig in (0.8, 1.07, 1.2, 1.6):
    for delta in (0, 2, 3, 4, 5, 6, 8):
        on = rng.normal(delta, sig, (N, 5)); off = rng.normal(0, sig, (N, 5))
        d1, lo1, hi1 = welch(on[:, :3], off[:, :3]); c1 = cls(lo1, hi1)
        cont = (c1 == "UND") & (d1 > 0)
        d2, lo2, hi2 = welch(on, off); c2 = cls(lo2, hi2)
        fin = np.where(cont, c2, c1)
        r1 = [np.mean(c1 == k) for k in ("GAIN", "NULL", "LOSS")] + [np.mean((c1 == "UND") & ~cont), np.mean(cont)]
        rf = [np.mean(fin == k) for k in ("GAIN", "NULL", "LOSS", "UND")]
        print(f"{sig:4.2f} {delta:5.1f} | " + " ".join(f"{x:5.3f}" for x in r1) + " | " + " ".join(f"{x:5.3f}" for x in rf))
# half-widths at sigma, n
for sig in (0.8, 1.07, 1.2):
    for n in (3, 5):
        se = sig*np.sqrt(2/n); df = 2*(n-1); q = T.ppf(0.9875, df)
        print(f"sigma {sig} n {n}: expected half-width (equal var, nominal df {df}) {q*se:.2f} pp; GAIN needs dhat > {2+q*se:.2f}")
