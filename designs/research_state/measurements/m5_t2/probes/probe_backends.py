"""Probe: eager / eager+CUDA graph / inductor / inductor+CUDA graph / AOT per bucket, real ckpt."""
import os, sys, time, json, functools, torch, numpy as np
from agents.inference.service.decision import DecisionModule, policy_reference
from agents.model.compile_parity_fixture import load_parity_rows
from agents.model.snapshot import current_model_version, load_foreign_opponent
from agents.observation.state_encoder import load_mappings
CK = os.environ.get("CKPT", "/home/goodlad/dev/gen3ai/models/ai_v14_06_lbat_ctrl_fix/final_model.zip")
BK = [int(x) for x in os.environ.get("BUCKETS", "8,48").split(",")]
ARMS = os.environ.get("ARMS", "eager,eager_graph,inductor,inductor_graph,aot").split(",")
model, _ = load_foreign_opponent(CK, current_version=current_model_version(load_mappings()), device="cpu")
pol = model.policy.eval()
for m in pol.modules():
    if hasattr(m, "_debugger"): m._debugger = None
pol = pol.to("cuda")
obs, mask = load_parity_rows(pol.observation_space["observation"].shape[0])
print("torch", torch.__version__, "ckpt", CK, flush=True)
dm = DecisionModule(pol).eval()

def fix_fill(gm):
    for n in list(gm.graph.nodes):
        if n.op == "call_function" and n.target is torch.ops.aten.fill.Tensor:
            src = n.args[1]
            if src.op == "call_function" and src.target is torch.ops.aten.lift_fresh_copy.default:
                c = functools.reduce(getattr, src.args[0].target.split("."), gm)
                n.target = torch.ops.aten.fill.Scalar; n.args = (n.args[0], float(c.item()))
    gm.graph.eliminate_dead_code(); gm.recompile(); return gm

def graphed(fn, o, m):
    so, sm = o.clone(), m.clone()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad():
        for _ in range(3): fn(so, sm)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(g):
        out = fn(so, sm)
    def run(o2, m2):
        so.copy_(o2); sm.copy_(m2); g.replay(); return out
    return run

def timeit(f, o, m, n=30):
    with torch.no_grad():
        for _ in range(3): f(o, m)
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(n): f(o, m)
        torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3

rows = {}
for B in BK:
    idx = np.arange(B) % len(obs)
    o = torch.as_tensor(obs[idx]).cuda(); m = torch.as_tensor(mask[idx]).cuda()
    ref = policy_reference(pol, o, m); fin = torch.isfinite(ref[0])
    for arm in ARMS:
        t0 = time.time()
        try:
            if arm == "eager": f = dm
            elif arm == "eager_graph": f = graphed(dm, o, m)
            elif arm in ("inductor", "inductor_graph"):
                torch._dynamo.reset(); c = torch.compile(dm, dynamic=False)
                with torch.no_grad(): c(o, m)
                f = c if arm == "inductor" else graphed(c, o, m)
            elif arm == "inductor_ro":
                torch._dynamo.reset(); f = torch.compile(dm, dynamic=False, mode="reduce-overhead")
                with torch.no_grad():
                    for _ in range(3): f(o, m)
            elif arm == "aot":
                with torch.no_grad():
                    ep = torch.export.export(dm, (o, m), strict=False)
                gm = fix_fill(ep.module())
                if hasattr(torch._inductor, "aoti_compile_and_package"):
                    ep2 = torch.export.export(gm, (o, m), strict=False)
                    pkg = torch._inductor.aoti_compile_and_package(ep2, package_path=f"/tmp/m5t2/aoti_{torch.__version__}_{B}.pt2")
                    f = torch._inductor.aoti_load_package(pkg)
                else:
                    so = torch._export.aot_compile(gm, (o, m), options={"aot_inductor.output_path": f"/tmp/m5t2/aoti_{torch.__version__}_b{B}.so"})
                    f = torch._export.aot_load(so, "cuda")
            build = time.time() - t0
            with torch.no_grad(): out = [t.clone() for t in f(o, m)]
            dl = float((out[0] - ref[0])[fin].abs().max()); dv = float((out[1] - ref[1]).abs().max())
            am = float((out[0].argmax(-1) == ref[0].argmax(-1)).float().mean())
            ms = timeit(f, o, m)
            r = dict(B=B, arm=arm, build_s=round(build, 1), ms=round(ms, 3), us_row=round(ms / B * 1e3, 1), dlogp=dl, dv=dv, argmax=am)
        except Exception as e:
            r = dict(B=B, arm=arm, error=f"{type(e).__name__}: {str(e)[-2500:]}")
        print(json.dumps(r), flush=True)
