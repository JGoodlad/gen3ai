"""F-XC-4: localise the non-finite compiled backward. Every module of the policy (to ``depth``) gets a
scalar ones-probe on each floating tensor INPUT and OUTPUT (per call); after one compiled R1 backward,
a probe's grad is sum(dL/dx * x) — non-finite iff the gradient reaching that tensor is.
Usage: probe_modules.py <arm> <fresh|golden> <depth> <out.json>"""
import json
import sys

import torch as th

from agents.model import compile_regions as cr, parity_probe as PP
from agents.model.compile_control import apply_compile_config
from agents.training import learner_golden as LG
from agents.training.instrumented_ppo.micro_step import micro_step

arm, weights, depth, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
METHODS_OF = sys.argv[5] if len(sys.argv) > 5 else ""
B = 2048
th.set_float32_matmul_precision("highest")
if weights == "fresh":
    PP.PERTURB_SCALE = 0.0
m = LG.build_learner() if arm == "blob" else LG.build_arm_learner(arm)
m.policy.to("cuda")
m.device = th.device("cuda")
from agents.model.compile_cache import ensure_hermetic_cache  # noqa: E402
ensure_hermetic_cache("fxc4 probe modules")
apply_compile_config()

MAXCALLS = 8
probes = {}
order = []
counts = {}


def probe(key):
    if key not in probes:
        probes[key] = th.zeros((), device="cuda", requires_grad=True)
        order.append(key)
    return probes[key]


mods = [(n, mod) for n, mod in m.policy.named_modules() if n and n.count(".") < depth]
meths = []
if METHODS_OF:
    tgt = m.policy.get_submodule(METHODS_OF)
    for attr in dir(type(tgt)):
        fn = getattr(type(tgt), attr, None)
        if attr == "forward" or not callable(fn) or not (getattr(fn, "__module__", None) or "").startswith("agents.model"):
            continue
        if isinstance(getattr(type(tgt), attr), (staticmethod, classmethod, property)):
            continue
        import inspect
        if not inspect.isfunction(fn):
            continue
        meths.append((METHODS_OF + "::" + attr, tgt, attr))
    mods = []
for i, n in enumerate([x[0] for x in mods] + [x[0] for x in meths]):
    for c in range(MAXCALLS):
        for side in ("in", "out"):
            for j in range(6):
                probe((n, c, side, j))


def wrap(name, mod, attr="forward"):
    orig = getattr(mod, attr)

    def fwd(*a, **k):
        c = counts.get(name, 0)
        counts[name] = c + 1
        c = min(c, MAXCALLS - 1)
        a2 = []
        j = 0
        for x in a:
            if isinstance(x, th.Tensor) and x.is_floating_point() and j < 6:
                x = x + probes[(name, c, "in", j)]
                j += 1
            a2.append(x)
        for kk in sorted(k):
            x = k[kk]
            if isinstance(x, th.Tensor) and x.is_floating_point() and j < 6:
                k[kk] = x + probes[(name, c, "in", j)]
                j += 1
        o = orig(*a2, **k)
        if isinstance(o, th.Tensor) and o.is_floating_point():
            return o + probes[(name, c, "out", 0)]
        if isinstance(o, tuple) and not hasattr(o, "_fields"):
            res = []
            j = 0
            for x in o:
                if isinstance(x, th.Tensor) and x.is_floating_point() and j < 6:
                    x = x + probes[(name, c, "out", j)]
                    j += 1
                res.append(x)
            return tuple(res)
        return o
    setattr(mod, attr, fwd)


for n, mod in mods:
    wrap(n, mod)
for n, mod, attr in meths:
    wrap(n, mod, attr)
print("wrapped methods", [x[0] for x in meths], flush=True)
b = cr.r1_batch(m, B)
m.policy.set_training_mode(True)
args = cr._r1_args(m, b)


def run(fn):
    counts.clear()
    for p in probes.values():
        p.grad = None
    o = fn(*args)
    o.loss.backward()
    nf = [n for n, q in m.policy.named_parameters() if q.grad is not None and not bool(th.isfinite(q.grad).all())]
    print("param nonfinite:", len(nf), nf[:3], flush=True)
    m.policy.zero_grad(set_to_none=True)
    return {f"{k[0]}#{k[1]}.{k[2]}{k[3]}": float(p.grad) for k, p in probes.items() if p.grad is not None}


e = run(micro_step)
comp = th.compile(micro_step, fullgraph=True, dynamic=False)
c = run(comp)
bad = {k: (e.get(k), v) for k, v in c.items() if not (v == v and abs(v) != float("inf"))}
json.dump({"eager": e, "compiled": c, "nonfinite_compiled": bad}, open(out, "w"), indent=1)
print("NONFINITE", len(bad), "of", len(c), flush=True)
for k in sorted(bad):
    print("  ", k, bad[k], flush=True)
