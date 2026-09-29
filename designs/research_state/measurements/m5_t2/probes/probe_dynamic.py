import os, time, torch, numpy as np, json
from agents.inference.service.decision import DecisionModule, policy_reference
from agents.model.compile_parity_fixture import load_parity_rows
from agents.model.snapshot import current_model_version, load_foreign_opponent
from agents.observation.state_encoder import load_mappings
from agents.model.compile_control import dynamo_graphs_total
cv = current_model_version(load_mappings())
model, _ = load_foreign_opponent("/home/goodlad/dev/gen3ai/models/ai_v14_06_lbat_ctrl_fix/final_model.zip", current_version=cv, device="cpu")
A = model.policy.eval()
for mm in A.modules():
    if hasattr(mm, "_debugger"): mm._debugger = None
A = A.to("cuda"); dA = DecisionModule(A).eval()
obs, mask = load_parity_rows(A.observation_space["observation"].shape[0])
DYN = os.environ.get("DYN", "1") == "1"
c = torch.compile(lambda mod, o, m: mod(o, m), dynamic=DYN if DYN else False)
pool = torch.cuda.graph_pool_handle()
BK = [int(x) for x in os.environ.get("BUCKETS", "2,8,48,128").split(",")]
t0 = time.time(); keep = []
for B in BK:
    idx = np.arange(B) % len(obs)
    o = torch.as_tensor(obs[idx]).cuda(); m = torch.as_tensor(mask[idx]).cuda()
    if DYN:
        torch._dynamo.mark_dynamic(o, 0, min=2); torch._dynamo.mark_dynamic(m, 0, min=2)
    t = time.time(); g0 = dynamo_graphs_total()
    with torch.no_grad(): c(dA, o, m)
    tc = time.time() - t; g1 = dynamo_graphs_total()
    so, sm = o.clone(), m.clone()
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad():
        for _ in range(2): c(dA, so, sm)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph(); t = time.time()
    with torch.no_grad(), torch.cuda.graph(g, pool=pool):
        out = c(dA, so, sm)
    tcap = time.time() - t; keep.append((g, out, so, sm))
    ref = policy_reference(A, o, m); fin = torch.isfinite(ref[0])
    so.copy_(o); sm.copy_(m); g.replay(); torch.cuda.synchronize()
    dl = float((out[0]-ref[0])[fin].abs().max()); dv = float((out[1]-ref[1]).abs().max())
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(50): so.copy_(o); sm.copy_(m); g.replay()
    torch.cuda.synchronize(); ms = (time.perf_counter()-t)/50*1e3
    t = time.perf_counter()
    with torch.no_grad():
        for _ in range(10): dA(o, m)
    torch.cuda.synchronize(); ems = (time.perf_counter()-t)/10*1e3
    print(json.dumps(dict(dyn=DYN, B=B, compile_s=round(tc,1), new_graphs=g1-g0, capture_s=round(tcap,2), graph_ms=round(ms,3), us_row=round(ms/B*1e3,1), eager_ms=round(ems,2), dlogp=dl, dv=dv, mem_mb=round(torch.cuda.max_memory_allocated()/2**20))), flush=True)
print("total s", round(time.time()-t0,1))
