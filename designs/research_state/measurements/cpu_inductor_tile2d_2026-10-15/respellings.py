"""Would a RESPELLING of the per-row gather (option b) step around the bug? Each spelling on the smallest repro,
B=16, K=18, CPU Inductor, torch 2.8.0, no options. Also: is the one-hot spelling bit-identical to the index in eager?"""
import os, sys, tempfile, re
os.environ["CUDA_VISIBLE_DEVICES"] = ""             # standing rule 5: no lease, so nothing may touch the card
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", tempfile.mkdtemp(prefix="mini_ind_"))
import torch
assert not torch.cuda.is_available(), "CUDA must be invisible in this script (no GPU lease)"
import torch._inductor.config as icfg

torch.set_num_threads(1)
icfg.fx_graph_cache = False
B, K = 16, 18
g = torch.Generator().manual_seed(0)
X = torch.rand(B, 6, K, generator=g)
Q = torch.rand(B, K, 6, generator=g)
act = torch.randint(0, 6, (B,), generator=g)


def idx(X, act, Q):
    gt = X[torch.arange(B), act]
    return (gt * Q[..., 5]).sum(-1), gt


def gat(X, act, Q):
    gt = X.gather(1, act[:, None, None].expand(-1, 1, K)).squeeze(1)
    return (gt * Q[..., 5]).sum(-1), gt


def isel(X, act, Q):
    gt = torch.stack([X[b].index_select(0, act[b:b + 1])[0] for b in range(B)])
    return (gt * Q[..., 5]).sum(-1), gt


def onehot(X, act, Q):
    oh = (torch.arange(6)[None, :] == act[:, None]).to(X.dtype)           # [B,6]
    gt = (X * oh[:, :, None]).sum(1)
    return (gt * Q[..., 5]).sum(-1), gt


print("eager: one-hot == index bitwise:", torch.equal(onehot(X, act, Q)[1], idx(X, act, Q)[1]),
      "| gather == index bitwise:", torch.equal(gat(X, act, Q)[1], idx(X, act, Q)[1]))
for nm, f in [("index", idx), ("gather", gat), ("one-hot", onehot)]:
    torch._dynamo.reset()
    try:
        torch.compile(f, backend="inductor", fullgraph=True)(X, act, Q)
        print(f"{nm}: OK", flush=True)
    except Exception as e:
        print(f"{nm}: FAIL" + (" UNDECLARED" if "was not declared" in str(e) else " " + str(e)[:60]), flush=True)
