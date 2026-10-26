import sys
from agents.model.snapshot import historical_load_kwargs, load_checkpoint_strict
from main.prober import arch_status as A
for p in sys.argv[1:]:
    m = load_checkpoint_strict(p, device="cpu", **historical_load_kwargs(p))
    v = A.checkpoint_verdict(p)
    print("LOADED", p.split("/models/")[1], "| prober verdict:", v.status, getattr(v, "kind", None))
