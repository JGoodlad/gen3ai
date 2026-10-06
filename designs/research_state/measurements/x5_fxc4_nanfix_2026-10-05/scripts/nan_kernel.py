"""F-XC-4 follow-up: WHICH generated Triton kernel first writes a NaN in the compiled fixed_mass R1?
Every Triton kernel call in the generated wrapper is bracketed by a host hook that counts NaN / inf
in each tensor argument before and after the call (isnan only decides; the forward's -inf masks are
legitimate). Codegen-only instrumentation: it runs after scheduling and fusion, so the kernels are the
ones the real compile generates. Usage: nan_kernel.py <arm> <out_prefix> [inductor_cfg k=v,...]"""
import builtins
import json
import sys

import torch as th
import torch._functorch.config as FC
import torch._inductor.config as IC
from torch._inductor.codegen import triton as TR
from torch._inductor.virtualized import V

from agents.model import compile_regions as cr, compile_trainer as ct, parity_probe as PP
from agents.model.compile_control import apply_compile_config
from agents.training import learner_golden as LG
from agents.training.instrumented_ppo.micro_step import micro_step

arm, outp = sys.argv[1:3]
cfg = dict(kv.split("=") for kv in sys.argv[3].split(",")) if len(sys.argv) > 3 and sys.argv[3] else {}
B = 2048
th.set_float32_matmul_precision("highest")
PP.PERTURB_SCALE = 0.0
m = LG.build_learner() if arm == "blob" else LG.build_arm_learner(arm)
m.policy.to("cuda")
m.device = th.device("cuda")
m.batch_size = B
from agents.model.compile_cache import ensure_hermetic_cache  # noqa: E402
ensure_hermetic_cache("fxc4 nan kernel")
apply_compile_config()
IC.fx_graph_cache = False
FC.enable_autograd_cache = False
for k, v in cfg.items():
    setattr(IC, k, {"True": True, "False": False}.get(v, v))

EVENTS: list = []
FIRST: dict = {}
SEEN_NAN = [False]


def _stat(t):
    if not isinstance(t, th.Tensor) or not t.is_floating_point():
        return None
    return [int(th.isnan(t).sum()), int(th.isinf(t).sum()), list(t.shape)]


def _hook(phase, f, kname, names, *tensors):
    if SEEN_NAN[0] and phase == "pre":
        return
    st = {n: _stat(t) for n, t in zip(names, tensors)}
    if phase == "pre":
        FIRST["_pre"] = (f, kname, st)
        return
    if any(s and s[0] for s in st.values()) and len(EVENTS) < 6:
        SEEN_NAN[0] = True
        pf, pk, pst = FIRST.get("_pre", (None, None, {}))
        EVENTS.append({"file": f, "kernel": kname, "after": st,
                       "before": pst if pk == kname else None})


builtins._g3nan = _hook
_orig_call = TR.TritonKernel.call_kernel


def call_kernel(self, name, node=None):
    w = V.graph.wrapper_code
    _, call_args, arg_types, _ = self.args.python_argdefs()
    tens = [a for a, s in zip(call_args, arg_types) if isinstance(s, TR.TensorArg)]
    names = ", ".join(repr(a) for a in tens)
    vals = ", ".join(tens)
    w.writeline(f"__import__('builtins')._g3nan('pre', __file__, {name!r}, ({names}{',' if tens else ''}), {vals})")
    _orig_call(self, name, node)
    w.writeline(f"__import__('builtins')._g3nan('post', __file__, {name!r}, ({names}{',' if tens else ''}), {vals})")


TR.TritonKernel.call_kernel = call_kernel
names = [n for n, _ in ct.grad_parameters(m, m.policy.features_extractor)]
b = cr.r1_batch(m, B)
m.policy.set_training_mode(True)
args = cr._r1_args(m, b)
comp = th.compile(micro_step, fullgraph=True, dynamic=False)
c = cr._r1_arm(m, comp, args)
sizes = [p.numel() for _, p in ct.grad_parameters(m, m.policy.features_extractor)]
cg = th.split(c["grad"], sizes) if sum(sizes) == c["grad"].numel() else []
nonfinite = [names[i] for i, g in enumerate(cg) if not bool(th.isfinite(g).all())]
from torch._inductor.codecache import PyCodeCache  # noqa: E402
files = sorted({getattr(mod, "__file__", "") for mod in PyCodeCache.modules})
row = {"arm": arm, "cfg": cfg, "n_nonfinite": len(nonfinite), "nonfinite": nonfinite[:50],
       "first_nan_events": EVENTS[:3], "wrapper_files": files}
json.dump(row, open(outp + ".json", "w"), indent=1)
print(json.dumps({k: row[k] for k in ("arm", "cfg", "n_nonfinite")}), flush=True)
print("FIRST", json.dumps(EVENTS[:1]), flush=True)
import shutil  # noqa: E402
if EVENTS:
    shutil.copy(EVENTS[0]["file"], outp + "_wrapper.py")
for i, f in enumerate(files):
    if f:
        shutil.copy(f, f"{outp}_mod{i}.py")
