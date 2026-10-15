"""Which of the move_facts-shaped candidates make Inductor's CPU backend choose the tile2d (transpose_mxn) kernel,
and does any of them trip the undeclared-`tmp` codegen bug? Prints the C++ facts per candidate."""
import os, sys, tempfile, re
os.environ["CUDA_VISIBLE_DEVICES"] = ""             # standing rule 5: no lease, so nothing may touch the card
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", tempfile.mkdtemp(prefix="mini_ind_"))
import torch
assert not torch.cuda.is_available(), "CUDA must be invisible in this script (no GPU lease)"
from torch._inductor.utils import run_and_get_cpp_code

torch.set_num_threads(1)
B, S, K = 16, 6, 18
g = torch.Generator().manual_seed(0)
X = torch.rand(B, S, K, generator=g)
A = torch.rand(B, S, K, 14, generator=g)
P = torch.rand(B, K, 13, generator=g)
Q = torch.rand(B, K, 6, generator=g)
act = torch.randint(0, S, (B,), generator=g)
s1, s2, s3, s4 = (torch.rand(B, generator=g) for _ in range(4))
ar_global = torch.arange(B)


def run(name, f, *args):
    torch._dynamo.reset()
    c = torch.compile(f, backend="inductor", fullgraph=True)
    try:
        out, code = run_and_get_cpp_code(c, *args)
        ref = f(*args)
        out = out if isinstance(out, tuple) else (out,)
        ref = ref if isinstance(ref, tuple) else (ref,)
        d = max((a.float() - b.float()).abs().max().item() for a, b in zip(out, ref))
        n_tr, n_k = code.count("transpose_mxn<"), code.count("cpp_fused")
        print(f"{name}: OK d={d:.1e}  transpose_mxn x{n_tr}  cpp_fused mentions={n_k}", flush=True)
    except Exception as e:
        msg = str(e)
        print(f"{name}: FAIL {type(e).__name__}: {msg.splitlines()[0][:80]}"
              + (" | UNDECLARED " + re.findall(r"error: .(\w+). was not declared", msg)[0] if "was not declared" in msg else ""),
              flush=True)


def common(X, A, P, Q, act, s1, s2, s3, s4):
    ar = torch.arange(B)
    tm = X[ar, act]
    pin = A[ar, act]
    ok = tm > 0
    hits = ok.float() * pin[..., 1] * P[..., 9]
    blocked = 1 - (1 - pin[..., 2]) * (1 - pin[..., 3] * s1[:, None]) * (1 - hits)
    return tm, pin, ok, hits, blocked


def v1(X, A, P, Q, act, s1, s2, s3, s4):
    tm, pin, ok, hits, blocked = common(X, A, P, Q, act, s1, s2, s3, s4)
    r0 = (hits * Q[..., 5]).sum(-1)
    r1 = (blocked * Q[..., 4] * s2[:, None]).sum(-1)
    r2 = (hits * blocked * s4[:, None]).sum(-1)
    return r0, r1, r2, ok, hits, blocked


def v2(X, A, P, Q, act, s1, s2, s3, s4):               # reductions read the [B,K] stored intermediates AND strided cols
    tm, pin, ok, hits, blocked = common(X, A, P, Q, act, s1, s2, s3, s4)
    act_k = 1.0 - s2[:, None] * blocked
    r0 = (hits * Q[..., 5]).sum(-1)
    r1 = (act_k * Q[..., 4]).sum(-1)
    r2 = (act_k * hits).sum(-1) * s3
    return r0, r1, r2, ok, hits, blocked, act_k


def v3(X, A, P, Q, act, s1, s2, s3, s4):               # only the bool gather + strided-col reductions
    ar = torch.arange(B)
    tm = X[ar, act]
    ok = tm > 0
    h = ok.float() * P[..., 9]
    return (h * Q[..., 5]).sum(-1), (h * Q[..., 4] * s1[:, None]).sum(-1), (Q[..., 3] * h).sum(-1) * s2, ok, h


for nm, f in [("v1", v1), ("v2", v2), ("v3", v3)]:
    run(nm, f, X, A, P, Q, act, s1, s2, s3, s4)
