"""Adversarial CPU checks of FP32_ALT (review, 2026-10-20)."""
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

print("== 1. random rows, many shapes: |tau - ref| in ulp(tau)")
rng = np.random.default_rng(0)
worst = 0.0; worst_naive = 0.0
for trial in range(300):
    k = int(rng.integers(1, 7)); scale = float(rng.choice([0.5, 2, 5, 10]))
    s = (rng.normal(0, scale, N) - 8).astype(np.float32)
    lead = float(rng.choice([0, 0, 10, 20, 30, 50, 80]))
    s[:k] += lead
    r = ref_tau(s, k)
    e = abs(tau(s, k) - r) / ulp32(r); worst = max(worst, e)
    worst_naive = max(worst_naive, abs(tau(s, k, "fp32_naive") - r) / ulp32(r))
print(f"   fp32_alt worst {worst:.2f} ulp; naive worst {worst_naive:.3g} ulp")

print("== 2. worst-case SEQUENTIAL fp32 accumulation (depth n): the proof's bound with depth = n-1")
def seq_tau(s, k, order):
    s = np.asarray(s, np.float32)[order]
    lo = np.float32(math.log(k / (len(s) - k)) - s.max()); hi = np.float32(math.log(k / (len(s) - k)) - s.min())
    for _ in range(64):
        mid = np.float32(np.float32(0.5) * (lo + hi)); x = (s + mid).astype(np.float32)
        t = (np.float32(1) / (np.float32(1) + np.exp(np.abs(x)).astype(np.float32))).astype(np.float32)
        h = np.where(x > 0, -t, t).astype(np.float32)
        acc = np.float32(0)
        for v in h: acc = np.float32(acc + v)
        if acc > np.float32(k - int((x > 0).sum())): hi = mid
        else: lo = mid
    return float(np.float32(0.5) * (lo + hi))
for lead in (23.0, 60.0):
    g = np.random.default_rng(1); s = (-lead + 2 * g.normal(size=N)); s = s - (s.max() + lead); s[17] = 0.0
    s = s.astype(np.float32); r = ref_tau(s, 1)
    x = s + r; t = 1 / (1 + np.exp(np.abs(x)))
    errs = {}
    for name, order in (("asc|t|", np.argsort(t)), ("desc|t|", np.argsort(-t)), ("index", np.arange(N)),
                        ("rand", g.permutation(N))):
        errs[name] = abs(seq_tau(s, 1, order) - r)
    print(f"   lead {lead}: tau {r:.4f} ulp {ulp32(r):.2e}; seq-sum |err| " + ", ".join(f"{k}={v:.2e}" for k, v in errs.items())
          + f"; proven worst 2(n+1)u = {2*(N+1)*u:.2e}")

print("== 3. edge cases (finite, structural)")
def pres(s, cand, k):
    p = fixed_mass_presence(torch.tensor([s], dtype=torch.float32), torch.tensor([cand]), torch.tensor([k]))
    return p
for name, s, cand, k in (
    ("k=0", [0., 1, 2], [1, 1, 1], 0), ("k=n", [0., 1, 2], [1, 1, 1], 3), ("n=0", [0., 1, 2], [0, 0, 0], 1),
    ("n=1,k=1", [0., 1, 2], [1, 0, 0], 1), ("n=2,k=1 tie", [3., 3., 9], [1, 1, 0], 1),
    ("all equal k=2 n=4 (x=0 at root)", [1.5] * 4, [1] * 4, 2),
    ("NaN in non-cand", [0., float('nan'), 2], [1, 0, 1], 1), ("+inf in non-cand", [0., float('inf'), 2], [1, 0, 1], 1),
    ("-inf in non-cand", [0., float('-inf'), 2], [1, 0, 1], 1),
    ("-inf CANDIDATE", [0., float('-inf'), 2], [1, 1, 1], 1), ("+inf CANDIDATE", [0., float('inf'), 2], [1, 1, 1], 1),
    ("NaN CANDIDATE", [0., float('nan'), 2], [1, 1, 1], 1), ("-1e30 CANDIDATE", [0., -1e30, 2], [1, 1, 1], 1)):
    cb = [bool(c) for c in cand]
    t = float(fixed_size_tau(torch.tensor([s], dtype=torch.float32), torch.tensor([cb]), torch.tensor([k])))
    tn = float(fixed_size_tau(torch.tensor([s], dtype=torch.float32), torch.tensor([cb]), torch.tensor([k]), solve="fp64"))
    p = pres(s, cb, k)
    print(f"   {name:34s} tau alt {t:+.6g} fp64 {tn:+.6g}  pi {p.pi.tolist()}  sum {float(p.pi.sum()):.7g}")

print("== 4. bf16 scores: computed in fp32, returned in bf16")
g = np.random.default_rng(3); s = torch.tensor(g.normal(0, 3, (4, N)), dtype=torch.bfloat16); c = torch.ones(4, N, dtype=torch.bool); kk = torch.tensor([1, 2, 3, 6])
tb = fixed_size_tau(s, c, kk); tf = fixed_size_tau(s.float(), c, kk)
print("   dtype", tb.dtype, "equal to fp32-solve rounded to bf16:", torch.equal(tb, tf.bfloat16()), "| max |tau_bf16 - tau_fp32|", float((tb.float() - tf).abs().max()))

print("== 5. underflow: k=1 lead L (one top candidate at 0, the next L below; rest 2 nats further)")
for L in (100, 140, 150, 160, 165, 170, 175, 180, 200):
    s = np.full(N, -L - 2.0, np.float32); s[0] = 0.0; s[1] = -L
    r = ref_tau(s, 1); print(f"   L {L}: |alt-ref| {abs(tau(s, 1) - r):.3g}  |fp64-ref| {abs(tau(s, 1, 'fp64') - r):.3g}  ref {r:.4f}")
