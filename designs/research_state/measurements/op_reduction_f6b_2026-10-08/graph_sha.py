"""The production extractor's dynamo FX graph sha + state_dict sha + output sha (the `max` identity proof).

Run with PYTHONPATH pointing at the tree under test; prints one JSON line."""
import hashlib
import io
import json
import sys

import torch

from agents.model.extractor_compiles_test import _build_production_extractor

over = {}
if len(sys.argv) > 1:
    over["op_reduction"] = sys.argv[1]
if len(sys.argv) > 2 and sys.argv[2] == "bundle":
    over.update(token_encoding="static", move_resolution="on", speed_physics="on")
torch.set_num_threads(1)
fe, layout = _build_production_extractor(**over)
obs = {"observation": torch.rand(8, layout["total_dim"], generator=torch.Generator().manual_seed(7))}
graphs = []


def backend(gm, example_inputs):
    graphs.append(gm.code)
    return gm.forward


torch._dynamo.reset()
with torch.no_grad():
    out_c = torch.compile(fe.forward, backend=backend)(obs)
    out_e = fe(obs)
g = "\n".join(graphs)
import os
if os.environ.get("GDUMP"): open(os.environ["GDUMP"], "w").write(g)
buf = io.BytesIO()
sd = fe.state_dict()
h = hashlib.sha256()
for k in sorted(sd):
    h.update(k.encode())
    h.update(sd[k].detach().cpu().contiguous().numpy().tobytes())
ho = hashlib.sha256()
for t in out_e:
    ho.update(t.detach().cpu().contiguous().numpy().tobytes())
print(json.dumps({"n_graphs": len(graphs), "graph_lines": g.count("\n") + 1,
                  "graph_sha": hashlib.sha256(g.encode()).hexdigest()[:16],
                  "state_sha": h.hexdigest()[:16], "out_sha": ho.hexdigest()[:16],
                  "n_params": sum(p.numel() for p in fe.parameters())}))
