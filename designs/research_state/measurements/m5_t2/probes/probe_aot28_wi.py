"""AOT with WEIGHTS AS INPUTS (one artifact per bucket serves every slot): parity real + fresh (the
same artifact), latency, per-call overhead. Also dumps raw inputs/outputs for the C++ loader test."""
import os, sys, time, json, torch, numpy as np
from torch.func import functional_call
from agents.inference.service.decision import DecisionModule, policy_reference
from agents.inference.service.parity import fixture_rows
from agents.inference.service.fixtures import perturbed_fresh_policy
from agents.model.snapshot import current_model_version, load_foreign_opponent
from agents.observation.state_encoder import load_mappings
os.environ.setdefault("CUDA_HOME", os.path.join(sys.prefix, "targets", "x86_64-linux"))
BK = [int(x) for x in os.environ.get("BUCKETS", "8,48,128").split(",")]
OUT = "/tmp/m5t2/aot28"; os.makedirs(OUT, exist_ok=True)
def prep(p):
    for m in p.modules():
        if hasattr(m, "_debugger"): m._debugger = None
    return p.eval().cuda()
real = prep(load_foreign_opponent("/home/goodlad/dev/gen3ai/models/ai_v14_06_lbat_ctrl_fix/final_model.zip", current_version=current_model_version(load_mappings()), device="cpu")[0].policy)
fresh = prep(perturbed_fresh_policy(0))
D = real.observation_space["observation"].shape[0]
dm, fdm = DecisionModule(real).eval(), DecisionModule(fresh).eval()
persist = set(dm.state_dict().keys())
names = [n for n, _ in dm.named_parameters()] + [n for n, _ in dm.named_buffers() if n in persist]
def weights(mod):
    d = dict(mod.named_parameters()); d.update(dict(mod.named_buffers())); return [d[n] for n in names]
class WI(torch.nn.Module):
    def __init__(self, dm):
        super().__init__(); self.dm = dm
    def forward(self, obs, mask, *w):
        return functional_call(self.dm, dict(zip(names, w)), (obs, mask))
wi = WI(dm).eval()
def err(out, ref):
    fin = torch.isfinite(ref[0]); return float((out[0]-ref[0])[fin].abs().max()), float((out[1]-ref[1]).abs().max())
wr, wf = weights(dm), weights(fdm)
print(json.dumps({"n_weight_inputs": len(names)}), flush=True)
for B in BK:
    obs, mask = fixture_rows(D, B); o = torch.as_tensor(obs).cuda(); m = torch.as_tensor(mask).cuda()
    row = {"B": B}
    try:
        t = time.time()
        with torch.no_grad(): ep = torch.export.export(wi, (o, m, *wr), strict=False)
        row["export_s"] = round(time.time()-t, 1); t = time.time()
        pkg = torch._inductor.aoti_compile_and_package(ep, package_path=f"{OUT}/wi_b{B}.pt2")
        row["build_s"] = round(time.time()-t, 1); row["size_mb"] = round(os.path.getsize(pkg)/1e6, 2)
        t = time.time(); f = torch._inductor.aoti_load_package(pkg); row["load_s"] = round(time.time()-t, 3)
        row["constants"] = len(f.get_constant_fqns())
        with torch.no_grad():
            row["real"] = err(f(o, m, *wr), policy_reference(real, o, m))
            row["fresh_same_artifact"] = err(f(o, m, *wf), policy_reference(fresh, o, m))
            for _ in range(5): f(o, m, *wr)
            torch.cuda.synchronize(); t = time.perf_counter()
            for _ in range(50): f(o, m, *wr)
            torch.cuda.synchronize(); row["ms"] = round((time.perf_counter()-t)/50*1e3, 3)
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {str(e)[-500:]}"
    print(json.dumps(row), flush=True)
