"""F-XC-4 harness: R1 compiled vs eager, per loss TERM, for one arm (blob | fixed_mass).
Usage: terms.py <arm> <device> <B>"""
import json
import sys
import time

import torch as th

from agents.model import compile_regions as cr, compile_trainer as ct
from agents.training import learner_golden as LG
from agents.training.instrumented_ppo.micro_step import micro_step

arm, dev, B = sys.argv[1], sys.argv[2], int(sys.argv[3])
th.set_float32_matmul_precision("highest")
from agents.model.compile_cache import ensure_hermetic_cache  # noqa: E402
ensure_hermetic_cache("fxc4 harness")
m = LG.build_learner() if arm == "blob" else LG.build_arm_learner(arm)
if arm == "blob":
    LG.load_buffer_into(m)
else:
    LG.load_buffer_into(m, LG.arm_buffer(arm))
if dev != "cpu":
    m.policy.to(dev)
    m.device = th.device(dev)
m.policy.set_training_mode(True)
args = cr._r1_args(m, cr.r1_batch(m, B))


def run(fn):
    params = [p for _, p in ct.grad_parameters(m, m.policy.features_extractor)]
    for p in params:
        p.grad = None
    out = fn(*args)
    out.loss.backward()
    g = th.cat([(p.grad if p.grad is not None else th.zeros_like(p)).detach().double().flatten()
                for p in params])
    for p in params:
        p.grad = None
    terms = {k: v.detach().double().cpu() for k, v in out.terms.items() if th.is_tensor(v)}
    return out, terms, g


t0 = time.time()
eo, et, eg = run(micro_step)
th._dynamo.reset()
from torch._dynamo.utils import counters  # noqa: E402
counters.clear()
comp = th.compile(micro_step, fullgraph=True, dynamic=False)
co, ctm, cg = run(comp)
row = {"arm": arm, "dev": dev, "B": B, "secs": round(time.time() - t0, 1),
       "loss_e": float(eo.loss.detach().double()), "loss_c": float(co.loss.detach().double()),
       "loss_bits_equal": bool(th.equal(eo.loss.detach().cpu(), co.loss.detach().cpu())),
       "terms": {k: {"e": float(et[k]), "c": float(ctm[k]), "d": float(ctm[k] - et[k])} for k in et},
       "log_prob_maxdiff": float((co.log_prob - eo.log_prob).abs().max()),
       "values_maxdiff": float((co.values - eo.values).abs().max()),
       "grad_cos": ct._cos(cg, eg), "grad_norm_e": float(eg.norm()), "grad_norm_c": float(cg.norm()),
       "counters": {k: dict(v) for k, v in counters.items()
                    if k in ("stats", "graph_break", "inductor", "aot_autograd", "recompiles")}}
print(json.dumps(row, indent=1, default=str))
