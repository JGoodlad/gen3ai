"""Unit 3 probe: S slots x small bucket — sequential replays vs replays on L concurrent streams."""
import os, time, json, torch, numpy as np
from agents.inference.service.decision import DecisionModule, policy_reference
from agents.inference.service.parity import fixture_rows
from agents.inference.service.fixtures import perturbed_fresh_policy
from agents.inference.service.service import _decide
S = int(os.environ.get("S", "8")); BK = [int(x) for x in os.environ.get("BK", "8,16").split(",")]
pols = [perturbed_fresh_policy(s).cuda() for s in range(S)]
mods = [DecisionModule(p).eval() for p in pols]
D = pols[0].observation_space["observation"].shape[0]
c = torch.compile(_decide, dynamic=False)
res = []
for B in BK:
    obs, mask = fixture_rows(D, B)
    o = torch.as_tensor(obs).cuda(); m = torch.as_tensor(mask).cuda()
    for L in (1, 2, 4, 8):
        if L > S: continue
        pools = [torch.cuda.graph_pool_handle() for _ in range(L)]
        streams = [torch.cuda.Stream() for _ in range(L)]
        graphs = []
        with torch.no_grad():
            for s in range(S):
                lane = s % L
                so, sm = o.clone(), m.clone()
                side = torch.cuda.Stream(); side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    for _ in range(2): c(mods[s], so, sm)
                torch.cuda.current_stream().wait_stream(side)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=pools[lane], stream=streams[lane]):
                    out = c(mods[s], so, sm)
                graphs.append((g, out, lane, so, sm))
        torch.cuda.synchronize()
        # correctness under concurrency
        main = torch.cuda.current_stream()
        def run():
            for st in streams: st.wait_stream(main)
            for g, out, lane, *_ in graphs:
                with torch.cuda.stream(streams[lane]): g.replay()
            for st in streams: main.wait_stream(st)
        run(); torch.cuda.synchronize()
        worst = 0.0
        for s, (g, out, lane, *_) in enumerate(graphs):
            r = policy_reference(pols[s], o, m); fin = torch.isfinite(r[0])
            worst = max(worst, float((out[0]-r[0])[fin].abs().max()), float((out[1]-r[1]).abs().max()))
        for _ in range(3): run()
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(50): run()
        torch.cuda.synchronize(); ms = (time.perf_counter()-t)/50*1e3
        r = dict(S=S, B=B, lanes=L, ms_all_slots=round(ms, 3), ms_per_slot=round(ms/S, 3), worst_parity=worst,
                 mem_mib=round(torch.cuda.memory_allocated()/2**20))
        print(json.dumps(r), flush=True)
        del graphs; torch.cuda.synchronize()
