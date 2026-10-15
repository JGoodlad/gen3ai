"""Minimise the CPU-Inductor tile2d + indirect-gather codegen bug (torch 2.8.0): which ingredients are needed."""
import os, sys, tempfile, re
os.environ["CUDA_VISIBLE_DEVICES"] = ""             # standing rule 5: no lease, so nothing may touch the card
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", tempfile.mkdtemp(prefix="mini_ind_"))
import torch
assert not torch.cuda.is_available(), "CUDA must be invisible in this script (no GPU lease)"

torch.set_num_threads(1)
B, S, K = int(os.environ.get("B", 16)), 6, int(os.environ.get("K", 18))
g = torch.Generator().manual_seed(0)
X = torch.rand(B, S, K, generator=g)
Q = torch.rand(B, K, 6, generator=g)
V = torch.rand(B, K, generator=g)
act = torch.randint(0, S, (B,), generator=g)
s1 = torch.rand(B, generator=g)


def run(name, f, *args):
    torch._dynamo.reset()
    c = torch.compile(f, backend="inductor", fullgraph=True)
    try:
        out = c(*args)
        ref = f(*args)
        out = out if isinstance(out, tuple) else (out,)
        ref = ref if isinstance(ref, tuple) else (ref,)
        d = max((a.float() - b.float()).abs().max().item() for a, b in zip(out, ref))
        print(f"{name}: OK d={d:.1e}", flush=True)
    except Exception as e:
        msg = str(e)
        print(f"{name}: FAIL {type(e).__name__}: {msg.splitlines()[0][:70]}"
              + (" | UNDECLARED " + re.findall(r"error: .(\w+). was not declared", msg)[0] if "was not declared" in msg else ""),
              flush=True)


def m1(X, act, Q, V, s1):                # the smallest guess: gather + strided-col reduction + the gather stored
    gt = X[torch.arange(B), act]
    return (gt * Q[..., 5]).sum(-1), gt


def m2(X, act, Q, V, s1):                # same, two reductions
    gt = X[torch.arange(B), act]
    return (gt * Q[..., 5]).sum(-1), (gt * Q[..., 4]).sum(-1), gt


def m3(X, act, Q, V, s1):                # gather only consumed by reductions (not stored)
    gt = X[torch.arange(B), act]
    return (gt * Q[..., 5]).sum(-1), (gt * Q[..., 4] * s1[:, None]).sum(-1)


def m4(X, act, Q, V, s1):                # contiguous [B,K] operand instead of a strided column
    gt = X[torch.arange(B), act]
    return (gt * V).sum(-1), gt


def m5(X, act, Q, V, s1):                # bool-gather, as v3
    ok = X[torch.arange(B), act] > 0
    h = ok.float() * Q[..., 5]
    return (h * Q[..., 4]).sum(-1), (h * V * s1[:, None]).sum(-1), ok, h


def m6(X, act, Q, V, s1):                # no reduction at all: pointwise only
    gt = X[torch.arange(B), act]
    return gt * Q[..., 5], gt


for nm, f in [("m1", m1), ("m2", m2), ("m3", m3), ("m4", m4), ("m5", m5), ("m6", m6)]:
    run(nm, f, X, act, Q, V, s1)
