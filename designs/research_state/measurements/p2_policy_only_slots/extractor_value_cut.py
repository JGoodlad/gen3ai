"""Run on branch `p2` (e9d17f06; PolicyOnlyDecisionModule lives there, not on main): an UPPER BOUND on what an extractor-level policy-only mode could save.
Variant B = PolicyOnlyDecisionModule on a replica whose extractor has its value-only pieces cut by
in-process surgery (value routes, win head, value pre-norm + projection). The 1-query value CLS attention
(inside CLSPool) is NOT cut, so B is still a little heavier than a true policy-only extractor."""
import json   # run with PYTHONPATH=<a checkout at branch p2>/src under scripts/ops/gpu_lock.sh
import torch
from agents.model.compile_cache import ensure_hermetic_cache
ensure_hermetic_cache("p2_extractor_bound")
from agents.inference.service.fixtures import perturbed_fresh_policy
from agents.inference.service.decision import DecisionModule, PolicyOnlyDecisionModule
from agents.inference.service.engine import decide
from agents.inference.service.parity import fixture_rows
from agents.inference.service.slots import served_replica

dev = torch.device("cuda")
tmpl = perturbed_fresh_policy(0)
def rep():
    r = served_replica(tmpl); r.optimizer = None
    return r.to(dev).eval()
ra, rb, rc = rep(), rep(), rep()
fe = rb.features_extractor
print("value routes on:", [n for n in ("value_entity_pool","value_threat_proj","pair_value_proj") if getattr(fe, n, None) is not None or getattr(getattr(fe,'cls_pool',None), n, None) is not None])
fe._value_pooled_routes = lambda *a, **k: iter(())
fe.win_head = None
fe.value_pre_norm = torch.nn.Identity()
fe.value_projection = torch.nn.Identity()
mods = {"full": DecisionModule(ra).eval(), "policy_only(DecisionModule-level)": PolicyOnlyDecisionModule(rc).eval(),
        "policy_only+extractor_value_cut": PolicyOnlyDecisionModule(rb).eval()}
limit = 64
torch._dynamo.config.cache_size_limit = limit
res = {}
main = torch.cuda.current_stream(dev)
ls = torch.cuda.Stream(dev)
pool = torch.cuda.graph_pool_handle()
outs_by = {}
for b in (8, 64):
    o, m = fixture_rows(int(ra.observation_space["observation"].shape[0]), b)
    so, sm = torch.as_tensor(o).to(dev).clone(), torch.as_tensor(m).to(dev).clone()
    graphs = {}
    for name, mod in mods.items():
        compiled = torch.compile(decide, dynamic=False)
        with torch.no_grad():
            ls.wait_stream(main)
            with torch.cuda.stream(ls):
                for _ in range(2): compiled(mod, so, sm)
            main.wait_stream(ls)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=ls):
                out = compiled(mod, so, sm)
        graphs[name] = (g, out)
    torch.cuda.synchronize()
    for name, (g, out) in graphs.items():
        outs_by[(name, b)] = out[0].clone()
    # replay timing, interleaved rounds
    us = {k: [] for k in graphs}
    for rnd in range(15):
        for name, (g, _) in graphs.items():
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(ls):
                for _ in range(5): g.replay()
                e0.record(ls)
                for _ in range(100): g.replay()
                e1.record(ls)
            e1.synchronize()
            us[name].append(e0.elapsed_time(e1) / 100 * 1e3)
    res[b] = {k: round(sorted(v)[len(v)//2], 1) for k, v in us.items()}
    base = res[b]["full"]
    res[b]["delta_pct_vs_full"] = {k: round((v - base) / base * 100, 2) for k, v in res[b].items() if k != "delta_pct_vs_full"}
    res[b]["logp_max_abs_diff_vs_full"] = {n: float((outs_by[(n, b)] - outs_by[("full", b)])[torch.isfinite(outs_by[("full", b)])].abs().max()) for n in mods}
print(json.dumps(res, indent=2))
