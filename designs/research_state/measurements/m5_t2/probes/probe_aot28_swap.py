import os, sys, time, json, torch, traceback
from agents.inference.service.decision import DecisionModule, policy_reference
from agents.inference.service.parity import fixture_rows
from agents.inference.service.fixtures import perturbed_fresh_policy
from agents.model.snapshot import current_model_version, load_foreign_opponent
from agents.observation.state_encoder import load_mappings
os.environ.setdefault("CUDA_HOME", os.path.join(sys.prefix, "targets", "x86_64-linux"))
MODE = os.environ.get("MODE", "swap"); B = int(os.environ.get("B", "8"))
def prep(p):
    for m in p.modules():
        if hasattr(m, "_debugger"): m._debugger = None
    return p.eval().cuda()
real = prep(load_foreign_opponent("/home/goodlad/dev/gen3ai/models/ai_v14_06_lbat_ctrl_fix/final_model.zip", current_version=current_model_version(load_mappings()), device="cpu")[0].policy)
fresh = prep(perturbed_fresh_policy(0))
D = real.observation_space["observation"].shape[0]
obs, mask = fixture_rows(D, B); o = torch.as_tensor(obs).cuda(); m = torch.as_tensor(mask).cuda()
dm = DecisionModule(real).eval()
def err(out, ref):
    fin = torch.isfinite(ref[0]); return float((out[0]-ref[0])[fin].abs().max()), float((out[1]-ref[1]).abs().max())
cfg = {"aot_inductor.use_runtime_constant_folding": True} if os.environ.get("RCF") == "1" else {}
if os.environ.get("NOPKG") == "1": cfg["aot_inductor.package_constants_in_so"] = False
with torch.no_grad(): ep = torch.export.export(dm, (o, m), strict=False)
t = time.time()
pkg = torch._inductor.aoti_compile_and_package(ep, package_path=f"/tmp/m5t2/aot28/swap_{MODE}_{B}_{os.environ.get('RCF','0')}{os.environ.get('NOPKG','0')}.pt2", inductor_configs=cfg)
print("build", round(time.time()-t, 1), "size_mb", round(os.path.getsize(pkg)/1e6, 1), cfg, flush=True)
f = torch._inductor.aoti_load_package(pkg)
fdm = DecisionModule(fresh).eval()
fsd = dict(fdm.named_parameters()); fsd.update(dict(fdm.named_buffers()))
rsd = dict(dm.named_parameters()); rsd.update(dict(dm.named_buffers()))
names = f.get_constant_fqns()
if os.environ.get("NOPKG") == "1":
    f.load_constants({n: rsd[n] for n in names}, check_full_update=True)
with torch.no_grad(): print("real", err(f(o, m), policy_reference(real, o, m)), flush=True)
if MODE == "swap":
    f.load_constants({n: fsd[n] for n in names}, check_full_update=True)
    with torch.no_grad(): print("swapped->fresh", err(f(o, m), policy_reference(fresh, o, m)), flush=True)
    f.load_constants({n: rsd[n] for n in names}, check_full_update=True)
    with torch.no_grad(): print("swapped->real", err(f(o, m), policy_reference(real, o, m)), flush=True)
else:
    torch.cuda.set_sync_debug_mode("error")
    try:
        with torch.no_grad(): f(o, m)
        print("no host sync in AOT run")
    except Exception as e:
        print("SYNC in AOT run:", str(e)[:200])
    torch.cuda.set_sync_debug_mode(0)
    so, sm = o.clone(), m.clone()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad():
        for _ in range(3): f(so, sm)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    try:
        with torch.no_grad(), torch.cuda.graph(g, capture_error_mode="thread_local"):
            gout = f(so, sm)
        g.replay(); torch.cuda.synchronize(); print("graph", err(gout, policy_reference(real, o, m)))
    except Exception as e:
        print("CAPTURE FAIL", type(e).__name__, str(e)[:300])
