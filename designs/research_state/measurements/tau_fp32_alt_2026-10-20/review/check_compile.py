import os, sys, numpy as np, torch
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
torch.set_num_threads(1)
import torch._dynamo as dynamo
from agents.model.hypothesis_set import fixed_size_tau, fixed_mass_presence, set_bce
from agents.model.inductor_compile import compile_on
N = 400
rng = np.random.default_rng(7)
B = 256
k = torch.tensor(rng.integers(1, 7, B))
scale = rng.choice([0.5, 2, 5, 10], B)[:, None]; lead = rng.choice([0, 0, 10, 20, 30, 50, 80], B)[:, None]
s = rng.normal(0, 1, (B, N)) * scale - 8
for i in range(B): s[i, :k[i]] += lead[i, 0]
s = torch.tensor(s, dtype=torch.float32)
cand = torch.tensor(rng.random((B, N)) > 0.05)
cand[:, :6] = True
fn = compile_on(lambda a, b, c: fixed_size_tau(a, b, c, solve="fp32_alt"), device="cpu", backend="inductor", dynamic=False)
try:
    comp = fn(s, cand, k)
finally:
    dynamo.reset()
eager = fixed_size_tau(s, cand, k, solve="fp32_alt")
ref = fixed_size_tau(s.double(), cand, k, solve="fp32_alt")   # split residual in fp64: the exact root of the fp32 scores
d = (comp.double() - eager.double())
ulp = torch.tensor([float(np.spacing(np.float32(abs(float(t))))) for t in eager])
print(f"compiled vs eager: rows differing {int((d != 0).sum())}/{B}; max |d| {float(d.abs().max()):.3g}; max |d|/ulp {float((d.abs()/ulp).max()):.2f}")
print(f"   signed mean d/ulp {float((d/ulp).mean()):+.3f} (n_pos {int((d>0).sum())}, n_neg {int((d<0).sum())})")
sat = (lead[:, 0] >= 20)
for name, m in (("lead>=20", torch.tensor(sat)), ("lead<20", torch.tensor(~sat))):
    dd = (d / ulp)[m]
    print(f"   {name}: mean {float(dd.mean()):+.3f} ulp, pos {int((dd>0).sum())} neg {int((dd<0).sum())} of {int(m.sum())}")
print(f"eager vs fp64-split ref: max |err| {float((eager.double()-ref).abs().max()):.3g}; compiled {float((comp.double()-ref).abs().max()):.3g}")
# gradient: set BCE grad wrt scores, fp32_alt vs fp64 solve
y = torch.zeros(B, N)
for i in range(B):
    idx = torch.nonzero(cand[i]).flatten()[torch.randperm(int(cand[i].sum()), generator=torch.Generator().manual_seed(i))[:int(k[i])]]
    y[i, idx] = 1.0
grads = {}
for solve in ("fp32_alt", "fp64", "fp32_naive"):
    a = s.clone().requires_grad_(True)
    p = fixed_mass_presence(a, cand, k, solve=solve)
    loss, n, mm = set_bce(p.logits, p, y)
    loss.backward(); grads[solve] = a.grad.clone()
    print(f"   {solve}: loss {float(loss):.9g}, |sum(pi)-k| max {float((p.pi.sum(-1)-k).abs().max()):.3g}")
g64 = grads["fp64"]
for sv in ("fp32_alt", "fp32_naive"):
    rel = (grads[sv] - g64).abs().max() / g64.abs().max()
    print(f"   grad {sv} vs fp64: max abs diff {float((grads[sv]-g64).abs().max()):.3g} (rel to max grad {float(rel):.3g})")
