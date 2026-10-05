"""F-XC-4: CompilerBisector over the compiled R1 backward's non-finite gradient (fixed_mass, fresh, CUDA)."""
import json
import sys

import torch as th

from agents.model import compile_regions as cr, parity_probe as PP
from agents.model.compile_control import apply_compile_config
from agents.training import learner_golden as LG
from agents.training.instrumented_ppo.micro_step import micro_step
from torch._inductor.compiler_bisector import CompilerBisector

B = 2048
th.set_float32_matmul_precision("highest")
PP.PERTURB_SCALE = 0.0
m = LG.build_arm_learner("fixed_mass")
m.policy.to("cuda")
m.device = th.device("cuda")
from agents.model.compile_cache import ensure_hermetic_cache  # noqa: E402
ensure_hermetic_cache("fxc4 bisect")
apply_compile_config()
b = cr.r1_batch(m, B)
m.policy.set_training_mode(True)
args = cr._r1_args(m, b)
n = [0]


def test() -> bool:
    th._dynamo.reset()
    comp = th.compile(micro_step, fullgraph=True, dynamic=False)
    c = cr._r1_arm(m, comp, args)
    ok = bool(th.isfinite(c["grad"]).all())
    n[0] += 1
    print(f"BISECT run {n[0]}: backend={CompilerBisector.get_backend()} "
          f"subsystem={CompilerBisector.get_subsystem()} finite={ok}", flush=True)
    return ok


res = CompilerBisector.do_bisect(test)
print("RESULT", res, flush=True)
if res is not None:
    print("RESULT_JSON", json.dumps({"backend": res.backend, "subsystem": res.subsystem,
                                     "bisect_number": res.bisect_number,
                                     "debug_info": str(res.debug_info)[:4000]}), flush=True)
sys.exit(0)
