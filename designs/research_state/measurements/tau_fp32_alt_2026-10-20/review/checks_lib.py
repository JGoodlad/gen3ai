"""Shared helpers of the review checks (the head of checks.py): the float64 split-residual reference root, ulp, tau."""
import math, sys, numpy as np, torch
torch.set_num_threads(1)
from agents.model.hypothesis_set import fixed_size_tau, fixed_mass_presence
N = 400
u = 2.0 ** -24

def ref_tau(s, k):
    s = np.asarray(s, dtype=np.float64).reshape(-1)
    lo, hi = -1000.0, 1000.0
    for _ in range(200):
        mid = 0.5 * (lo + hi); x = s + mid
        t = 1.0 / (1.0 + np.exp(np.abs(x)))
        if np.where(x > 0, -t, t).sum() > k - int((x > 0).sum()): hi = mid
        else: lo = mid
    return 0.5 * (lo + hi)

def ulp32(x): return float(np.spacing(np.float32(abs(x))))

def tau(s, k, solve="fp32_alt", cand=None):
    s = torch.as_tensor(s, dtype=torch.float32).reshape(1, -1)
    cand = torch.ones_like(s, dtype=torch.bool) if cand is None else cand.reshape(1, -1)
    return float(fixed_size_tau(s, cand, torch.tensor([k]), solve=solve))

