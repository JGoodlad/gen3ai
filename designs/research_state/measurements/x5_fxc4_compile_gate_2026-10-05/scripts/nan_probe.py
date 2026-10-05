"""F-XC-4: where does the compiled R1 backward go non-finite? (fixed_mass, CUDA)
Usage: nan_probe.py <arm> <weights fresh|golden> <backend> <out.json> [inductor_cfg k=v,...]"""
import json
import sys

import torch as th

from agents.model import compile_regions as cr, compile_trainer as ct, parity_probe as PP
from agents.training import learner_golden as LG
from agents.training.instrumented_ppo.micro_step import micro_step
from agents.model.compile_control import apply_compile_config

arm, weights, backend, out = sys.argv[1:5]
cfg = dict(kv.split("=") for kv in sys.argv[5].split(",")) if len(sys.argv) > 5 and sys.argv[5] else {}
B = 2048
th.set_float32_matmul_precision("highest")
if weights == "fresh":
    PP.PERTURB_SCALE = 0.0
m = LG.build_learner() if arm == "blob" else LG.build_arm_learner(arm)
m.policy.to("cuda")
m.device = th.device("cuda")
m.batch_size = B
from agents.model.compile_cache import ensure_hermetic_cache  # noqa: E402
ensure_hermetic_cache("fxc4 nan probe")
apply_compile_config()
import torch._inductor.config as IC  # noqa: E402
for k, v in cfg.items():
    setattr(IC, k, {"True": True, "False": False}.get(v, v))
PROBES = [th.ones((), device="cuda", requires_grad=True) for _ in range(3)]
_pe = m.policy.features_extractor.pokemon_encoder
_orig = type(_pe).forward
_calls = [0]


def _fwd(ctx, emb):
    out = _orig(_pe, ctx, emb)
    k = _calls[0] % 2
    _calls[0] += 1
    return out * PROBES[k]


if "probe" in cfg.pop("__probe", "") or True:
    _pe.forward = _fwd
_fe = m.policy.features_extractor
_obhs = type(_fe)._build_hypothesis_species


def _bhs(ctx, role_tokens):
    return _obhs(_fe, ctx, role_tokens * PROBES[2])


_fe._build_hypothesis_species = _bhs
names = [n for n, _ in ct.grad_parameters(m, m.policy.features_extractor)]
b = cr.r1_batch(m, B)
m.policy.set_training_mode(True)
args = cr._r1_args(m, b)
fn = micro_step if backend == "eager" else th.compile(micro_step, backend=backend, fullgraph=True, dynamic=False)
def probe_read(tag):
    r = {}
    for i, p in enumerate(PROBES):
        r[f"{tag}_probe{i}"] = None if p.grad is None else float(p.grad)
        p.grad = None
    return r


e = cr._r1_arm(m, micro_step, args)
pr = probe_read("eager")
c = cr._r1_arm(m, fn, args)
pr.update(probe_read("compiled"))
print("PROBES", json.dumps(pr), flush=True)
sizes = [int(x) for x in e["grad_sizes"].tolist()]
cg = th.split(c["grad"], sizes)
nonfinite = [names[i] for i, g in enumerate(cg) if not bool(th.isfinite(g).all())]
finite = [names[i] for i, g in enumerate(cg) if bool(th.isfinite(g).all())]
frac = {names[i]: float((~th.isfinite(g)).float().mean()) for i, g in enumerate(cg) if not bool(th.isfinite(g).all())}
row = {"arm": arm, "weights": weights, "backend": backend, "cfg": cfg, "loss_c": c["loss"].item(),
       "loss_e": e["loss"].item(), "n_nonfinite": len(nonfinite), "nonfinite": nonfinite,
       "nonfinite_frac": frac, "finite": finite,
       "cos": ct._cos(c["grad"], e["grad"])}
json.dump(row, open(out, "w"), indent=1)
print(json.dumps({k: row[k] for k in ("backend", "cfg", "n_nonfinite", "cos", "loss_c", "loss_e")}), flush=True)
