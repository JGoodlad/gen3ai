"""One learner-golden update under WHICHEVER tree is on PYTHONPATH; dumps the post-update parameters
and every logged scalar. Run once per tree (legacy fold vs the K8 micro-step) and diff with
`compare.py`. Usage: one_update.py <out.npz> [--float64] [--no-intent] [--seed N] [--threads N]

The matched-NOISE control: the SAME tree at a different CPU thread count changes only the
reduction order of the matmuls — the fp32 rounding floor a pure refactor is judged against."""
import json
import sys

import numpy as np
import torch as th

from agents.training import learner_golden as LG

out = sys.argv[1]
f64 = "--float64" in sys.argv
no_intent = "--no-intent" in sys.argv
seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else LG.UPDATE_SEED

th.set_num_threads(int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 1)
if f64:
    th.set_default_dtype(th.float64)
model = LG.build_learner()
if no_intent:
    model.opp_intent_coef = 0.0
LG.load_buffer_into(model)
if f64:
    model.policy.double()
    rb = model.rollout_buffer
    for k, v in list(rb.observations.items()):
        if v.dtype == np.float32:
            rb.observations[k] = v.astype(np.float64)
    for name in ("rewards", "values", "log_probs", "advantages", "returns", "action_masks"):
        a = getattr(rb, name)
        if a.dtype == np.float32:
            setattr(rb, name, a.astype(np.float64))
np.random.seed(seed)
th.manual_seed(seed)
model.train()
logged = {k: float(v) for k, v in model.logger.name_to_value.items()
          if isinstance(v, (int, float, np.floating, np.integer))}
params = {n: p.detach().cpu().double().numpy() for n, p in model.policy.named_parameters()}
np.savez(out, **params)
with open(out + ".json", "w") as fh:
    json.dump(logged, fh, indent=1, sort_keys=True)
print("ok", out, len(params), "params", len(logged), "scalars")
