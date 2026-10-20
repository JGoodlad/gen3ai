import sys, numpy as np, torch
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
torch.set_num_threads(1)
from checks_lib import ref_tau, tau, ulp32, N
rng = np.random.default_rng(0)
rows = []
for trial in range(300):
    k = int(rng.integers(1, 7)); scale = float(rng.choice([0.5, 2, 5, 10]))
    s = (rng.normal(0, scale, N) - 8).astype(np.float32)
    lead = float(rng.choice([0, 0, 10, 20, 30, 50, 80])); s[:k] += lead
    r = ref_tau(s, k); a = tau(s, k); f = tau(s, k, "fp64")
    x = s.astype(np.float64) + r; slope = float((1/(1+np.exp(-x)) * 1/(1+np.exp(x))).sum())
    rows.append((abs(a - r), abs(a - r) / ulp32(r), abs(f - r), r, k, scale, lead, slope, float(np.abs(x).max()),
                 ulp32(float(np.abs(s).max()))))
rows.sort(key=lambda t: -t[0])
print("abs_err  ulp(tau)s  fp64_err  tau  k scale lead slope max|x|  ulp(max|a|)")
for t in rows[:8]: print(" ".join(f"{v:.3g}" for v in t))
print("max abs err alt", max(t[0] for t in rows), "fp64", max(t[2] for t in rows))
print("max err / ulp(max|a|):", max(t[0] / t[9] for t in rows))
