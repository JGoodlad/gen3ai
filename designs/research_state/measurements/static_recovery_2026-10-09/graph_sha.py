"""The extractor's dynamo FX graph sha + state_dict sha + output sha, for the flags-OFF identity proof of the static
RECOVERY levers (`gen3_static_recovery_v1`). The F6b unit's `op_reduction_f6b_2026-10-08/graph_sha.py`, generalised to
any toggle overlay.

    cd <checkout>/src && python -c "import runpy,sys; sys.argv=['graph_sha.py', '<json overlay>']; \
        runpy.run_path('../designs/research_state/measurements/static_recovery_2026-10-09/graph_sha.py', \
        run_name='__main__')"

(`-c` puts the cwd, i.e. THIS checkout's `src`, first on sys.path.) The overlay is a JSON object of extractor kwargs
over the production toggles (`{}` = production). Prints one JSON line."""
import hashlib
import json
import sys

import torch

import agents.model.extractor_build as EB
from agents.model.extractor_compiles_test import _build_production_extractor

over = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
torch.set_num_threads(1)
fe, layout = _build_production_extractor(**over)
obs = {"observation": torch.rand(8, layout["total_dim"], generator=torch.Generator().manual_seed(7))}
graphs = []


def backend(gm, example_inputs):
    graphs.append(gm.code)
    return gm.forward


torch._dynamo.reset()
with torch.no_grad():
    torch.compile(fe.forward, backend=backend)(obs)
    out_e = fe(obs)
g = "\n".join(graphs)
sd = fe.state_dict()
h = hashlib.sha256()
for k in sorted(sd):
    h.update(k.encode())
    h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
ho = hashlib.sha256()
for t in out_e:
    ho.update(t.detach().cpu().contiguous().numpy().tobytes())
print(json.dumps({"src": EB.__file__, "overlay": over, "n_graphs": len(graphs), "graph_lines": g.count("\n") + 1,
                  "graph_sha": hashlib.sha256(g.encode()).hexdigest()[:16],
                  "state_sha": h.hexdigest()[:16], "out_sha": ho.hexdigest()[:16],
                  "n_params": sum(p.numel() for p in fe.parameters())}))
