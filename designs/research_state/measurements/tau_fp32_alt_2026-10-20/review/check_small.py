import sys, numpy as np, torch
sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
torch.set_num_threads(1)
from checks_lib import ref_tau, ulp32
from agents.model.hypothesis_set import fixed_size_tau
def T(s, k, solve="fp32_alt", dt=torch.float32):
    s = torch.tensor([s], dtype=dt); return float(fixed_size_tau(s, torch.ones_like(s, dtype=torch.bool), torch.tensor([k]), solve=solve))
for n, k, gap in ((2, 1, 30.0), (4, 3, 30.0), (5, 4, 25.0), (8, 4, 40.0)):
    s = np.zeros(n, np.float32); s[k:] = -gap; s[:k] += np.arange(k, dtype=np.float32) * 0.25
    r = ref_tau(s, k)
    print(n, k, gap, "alt err", abs(T(s, k) - r), "naive err", abs(T(s, k, "fp32_naive") - r), "ulp", ulp32(r), "tau", r)
# bf16 teeth: solve in bf16 vs fp32 on a saturated bf16 row
g = torch.Generator().manual_seed(62)
a = (-23.0 + 2.0 * torch.randn(400, generator=g)); a = a - (a.max() + 23.0); a[17] = 0.0
sb = a.bfloat16().unsqueeze(0); c = torch.ones(1, 400, dtype=torch.bool); k = torch.tensor([1])
good = fixed_size_tau(sb, c, k)
ref32 = fixed_size_tau(sb.float(), c, k).bfloat16()
print("bf16 return equals fp32 solve rounded:", torch.equal(good, ref32), float(good), float(ref32))
