"""F-XC-4: the PRODUCTION gate path (compile_control + compile_regions.install), instrumented.
Usage: gate.py <arm: blob|fixed_mass> <weights: fresh|golden> <reps> [B]"""
import json
import sys

import torch as th

from agents.model import compile_regions as cr, compile_trainer as ct, parity_probe as PP
from agents.training import learner_golden as LG
from agents.training.instrumented_ppo.micro_step import micro_step

arm, weights, reps = sys.argv[1], sys.argv[2], int(sys.argv[3])
B = int(sys.argv[4]) if len(sys.argv) > 4 else 2048
th.set_float32_matmul_precision("highest")
if weights == "fresh":
    PP.PERTURB_SCALE = 0.0          # the golden builder's perturbation becomes a no-op: fresh init
m = LG.build_learner() if arm == "blob" else LG.build_arm_learner(arm)
m.policy.to("cuda")
m.device = th.device("cuda")
m.batch_size = B
from agents.model.compile_control import control, set_strict_errors  # noqa: E402
ctl = control(None).install()
set_strict_errors()
print(ctl.reset(), flush=True)
print("installed", cr.install(m), flush=True)
names = [n for n, _ in ct.grad_parameters(m, m.policy.features_extractor)]
regime = cr.weights_regime(m)
b = cr.r1_batch(m, B)
m.policy.set_training_mode(True)
args = cr._r1_args(m, b)
r1 = m._compiled_micro_step
inner = r1._gen3_compiled
print("identity:", {"dispatcher": type(r1).__name__, "inner": type(inner).__name__,
                    "inner_is_micro_step": inner is micro_step,
                    "inner_orig": getattr(getattr(inner, "_compile_control_inner", None),
                                          "_torchdynamo_orig_callable", None) is micro_step,
                    "regime": regime}, flush=True)


def summary(tag, c, e):
    sizes = [int(x) for x in e["grad_sizes"].tolist()]
    cg = th.split(c["grad"], sizes)
    eg = th.split(e["grad"], sizes)
    nan_c = [names[i] for i, g in enumerate(cg) if not bool(th.isfinite(g).all())]
    nan_e = [names[i] for i, g in enumerate(eg) if not bool(th.isfinite(g).all())]
    zero_c = sum(1 for g in cg if not bool(g.any()))
    zero_e = sum(1 for g in eg if not bool(g.any()))
    row = {"tag": tag, "loss_c": c["loss"].item(), "loss_e": e["loss"].item(),
           "loss_bits_equal": bool(th.equal(c["loss"], e["loss"])),
           "loss_same_storage": c["loss"].data_ptr() == e["loss"].data_ptr(),
           "grad_bits_equal": bool(th.equal(c["grad"], e["grad"])),
           "cos": ct._cos(c["grad"], e["grad"]), "norm_c": float(c["grad"].double().norm()),
           "norm_e": float(e["grad"].double().norm()),
           "nonfinite_params_c": nan_c[:12], "n_nonfinite_c": len(nan_c),
           "nonfinite_params_e": nan_e[:12], "zero_params_c": zero_c, "zero_params_e": zero_e}
    print(json.dumps(row), flush=True)


from torch._dynamo.utils import counters  # noqa: E402
for rep in range(reps):
    c = cr._r1_arm(m, r1, args)
    e = cr._r1_arm(m, micro_step, args)
    summary(f"live rep{rep}", c, e)
    scale, k = PP.PERTURB_LADDER[0]
    with PP.perturbed_parameters(m.policy, seed=PP.rung_seed(k), scale=scale):
        c = cr._r1_arm(m, r1, args)
        e = cr._r1_arm(m, micro_step, args)
    summary(f"perturbed rep{rep}", c, e)
print("counters", json.dumps({k: dict(v) for k, v in counters.items()
                              if k in ("stats", "graph_break", "aot_autograd", "recompiles")}), flush=True)
print("cache entries", cr.assert_inventory(m) if True else "", flush=True)
try:
    cr.gate_regions(m, batch_size=B, say=lambda s: print(s, flush=True))
except Exception as exc:  # noqa: BLE001
    print("GATE RAISED", type(exc).__name__, str(exc)[:600], flush=True)
