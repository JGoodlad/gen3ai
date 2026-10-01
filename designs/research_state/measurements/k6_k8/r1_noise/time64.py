"""Time one fp64 eager R1 forward+backward at the gate's B (CPU; the fp64 arm needs the two patches)."""
import sys, time
sys.path.insert(0, "/home/goodlad/gen3ai_archive/k6_k8/r1bar")
import torch as th
import noise
from agents.model import compile_regions as cr
from agents.training.instrumented_ppo.micro_step import micro_step
B = int(sys.argv[1]); dev = th.device(sys.argv[2] if len(sys.argv) > 2 else "cpu")
m, _ = noise.build("C_final")
b = cr.r1_batch(m, B)
pol = m.policy.double().to(dev); pol.set_training_mode(True)
import stable_baselines3.common.policies as _sbp
real_pre, real_float = _sbp.preprocess_obs, th.Tensor.float
_sbp.preprocess_obs = lambda o, s, normalize_images=True: {k: (v.double() if v.is_floating_point() else v) for k, v in real_pre(o, s, normalize_images).items()}
th.Tensor.float = lambda self, *a, **k: self.to(th.get_default_dtype())
th.set_default_dtype(th.float64)
args = noise.args_for(m, pol, b, dev, th.float64)
for i in range(2):
    t0 = time.time()
    _, g, _ = noise.grad_of(pol, micro_step, args)
    if dev.type == "cuda": th.cuda.synchronize()
    print(f"fp64 R1 fwd+bwd B={B} on {dev}: {time.time() - t0:.1f} s (call {i})", flush=True)
