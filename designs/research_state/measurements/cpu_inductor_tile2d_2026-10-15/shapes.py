"""Which (B, K) shapes trip the bug on the smallest repro (CPU, AVX512 box: 16 fp32 lanes)?"""
import os, sys, tempfile, re
os.environ["CUDA_VISIBLE_DEVICES"] = ""             # standing rule 5: no lease, so nothing may touch the card
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", tempfile.mkdtemp(prefix="mini_ind_"))
import torch
assert not torch.cuda.is_available(), "CUDA must be invisible in this script (no GPU lease)"
import torch._inductor.config as icfg

torch.set_num_threads(1)
icfg.fx_graph_cache = False


def make(B):
    def m1(X, act, Q):
        gt = X[torch.arange(B), act]
        return (gt * Q[..., 5]).sum(-1), gt
    return m1


for B, K in [(16, 18), (8, 18), (15, 18), (17, 18), (32, 18), (64, 18), (16, 8), (16, 16), (16, 17), (16, 32), (256, 18), (256, 64)]:
    g = torch.Generator().manual_seed(0)
    X, Q = torch.rand(B, 6, K, generator=g), torch.rand(B, K, 6, generator=g)
    act = torch.randint(0, 6, (B,), generator=g)
    f = make(B)
    torch._dynamo.reset()
    try:
        torch.compile(f, backend="inductor", fullgraph=True)(X, act, Q)
        print(f"B={B} K={K}: OK", flush=True)
    except Exception as e:
        print(f"B={B} K={K}: FAIL" + (" UNDECLARED" if "was not declared" in str(e) else " " + str(e)[:60]), flush=True)
