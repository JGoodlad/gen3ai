"""Candidate CPU-Inductor configs for the tile2d + indirect-gather codegen bug, on the SMALLEST repro (m1).
Each variant is compiled fresh; reports OK / FAIL, whether tile2d (transpose_mxn) is still in the C++, and the
max |compiled - eager| (fp32)."""
import contextlib, os, sys, tempfile, re
os.environ["CUDA_VISIBLE_DEVICES"] = ""             # standing rule 5: no lease, so nothing may touch the card
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", tempfile.mkdtemp(prefix="mini_ind_"))
import torch
assert not torch.cuda.is_available(), "CUDA must be invisible in this script (no GPU lease)"
import torch._inductor.config as icfg
from torch._inductor.utils import run_and_get_cpp_code

torch.set_num_threads(1)
icfg.fx_graph_cache = False
B, S, K = int(os.environ.get("B", 16)), 6, int(os.environ.get("K", 18))
g = torch.Generator().manual_seed(0)
X = torch.rand(B, S, K, generator=g)
Q = torch.rand(B, K, 6, generator=g)
act = torch.randint(0, S, (B,), generator=g)


def m1(X, act, Q):
    gt = X[torch.arange(B), act]                   # [B,K]: a per-row gather at a data-dependent slot
    return (gt * Q[..., 5]).sum(-1), gt            # a [B] reduction over K of a strided column, and the gather stored


@contextlib.contextmanager
def cfg(**kw):
    prev = {}
    for k, v in kw.items():
        obj, name = (icfg.cpp, k[4:]) if k.startswith("cpp_") else (icfg, k)
        prev[k] = (obj, name, getattr(obj, name))
        setattr(obj, name, v)
    try:
        yield
    finally:
        for obj, name, v in prev.values():
            setattr(obj, name, v)


def run(name, **kw):
    torch._dynamo.reset()
    with cfg(**kw):
        c = torch.compile(m1, backend="inductor", fullgraph=True)
        try:
            out, code = run_and_get_cpp_code(c, X, act, Q)
            ref = m1(X, act, Q)
            d = max((a - b).abs().max().item() for a, b in zip(out, ref))
            print(f"{name}: OK d={d:.1e} transpose_mxn={code.count('transpose_mxn<')} "
                  f"vec_loads={code.count('Vectorized<float>')}", flush=True)
        except Exception as e:
            msg = str(e)
            print(f"{name}: FAIL {type(e).__name__}: {msg.splitlines()[0][:60]}"
                  + (" | UNDECLARED " + re.findall(r"error: .(\w+). was not declared", msg)[0]
                     if "was not declared" in msg else ""), flush=True)


which = sys.argv[1:] or ["base", "vec_off", "simdlen1", "heur_off", "maxfusion1", "no_fuse_pw_red"]
V = {
    "base": {},
    "vec_off": dict(cpp_vec_isa_ok=False),
    "simdlen1": dict(cpp_simdlen=1),
    "heur_off": dict(cpp_enable_tiling_heuristics=False),
    "maxfusion1": dict(max_fusion_size=1),
    "no_fuse_pw_red": dict(cpp_tiling_prevents_pointwise_fusion=True) if hasattr(icfg.cpp, "tiling_prevents_pointwise_fusion") else {},
}
for w in which:
    run(w, **V[w])
