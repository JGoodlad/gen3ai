"""N0@9.5M: 17 parameters (all hidden_opp_belief.decoder + 2 edge_bias biases) where the compiled R1's
gradient is ~3e-4 off fp64 while CUDA eager is ~2e-6 off. Is it the ATTENTION KERNEL choice?
Arms (CUDA fp32 highest) vs the fp64 reference: eager default; eager under SDPA MATH; eager under
SDPA EFFICIENT; compiled default; compiled traced under SDPA MATH."""
import sys, json
sys.path.insert(0, "/home/goodlad/gen3ai_archive/k6_k8/r1bar")
import torch as th
from torch.nn.attention import sdpa_kernel, SDPBackend
import noise
from agents.model import compile_regions as cr
from agents.training.instrumented_ppo.micro_step import micro_step
state = sys.argv[1]
th.set_float32_matmul_precision("highest")
from agents.model.compile_cache import ensure_hermetic_cache
ensure_hermetic_cache("k8 r1 sdpa probe")
model, _ = noise.build(state)
names = [n for n, _ in model.policy.named_parameters() if not n.startswith("ridealong.")]
b = cr.r1_batch(model, 2048)
model.policy.set_training_mode(True)
# fp64 reference (CPU)
m64, _ = noise.build(state); pol64 = m64.policy.double(); pol64.set_training_mode(True)
import stable_baselines3.common.policies as _sbp
rp, rf = _sbp.preprocess_obs, th.Tensor.float
_sbp.preprocess_obs = lambda o, s, normalize_images=True: {k: (v.double() if v.is_floating_point() else v) for k, v in rp(o, s, normalize_images).items()}
th.Tensor.float = lambda self, *a, **k: self.to(th.get_default_dtype())
th.set_default_dtype(th.float64)
try:
    _, g64, sizes = noise.grad_of(pol64, micro_step, noise.args_for(m64, pol64, b, th.device("cpu"), th.float64))
finally:
    th.Tensor.float = rf; _sbp.preprocess_obs = rp; th.set_default_dtype(th.float32)
dev = th.device("cuda"); model.policy.to(dev); model.device = dev; model.rollout_buffer.device = dev
a = noise.args_for(model, model.policy, b, dev)
arms = {}
_, arms["eager"], _ = noise.grad_of(model.policy, micro_step, a)
with sdpa_kernel(SDPBackend.MATH):
    _, arms["eager_MATH"], _ = noise.grad_of(model.policy, micro_step, a)
try:
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        _, arms["eager_EFF"], _ = noise.grad_of(model.policy, micro_step, a)
except Exception as e:
    print("eager EFFICIENT failed:", type(e).__name__, str(e)[:200])
th._dynamo.reset()
comp = th.compile(micro_step, fullgraph=True, dynamic=False)
_, arms["comp"], _ = noise.grad_of(model.policy, comp, a)
th._dynamo.reset()
def comp_math(*x):
    with sdpa_kernel(SDPBackend.MATH):
        return micro_step(*x)
comp2 = th.compile(comp_math, fullgraph=True, dynamic=False)
try:
    _, arms["comp_MATH"], _ = noise.grad_of(model.policy, comp2, a)
except Exception as e:
    print("compiled MATH failed:", type(e).__name__, str(e)[:300])
th._dynamo.reset()
focus = [i for i, n in enumerate(names) if "hidden_opp_belief.decoder.linear1.weight" in n or "hidden_opp_belief.queries" in n
         or "decoder.self_attn.in_proj_weight" in n or "edge_bias.x_map.bias" in n]
out = {}
for k, g in arms.items():
    r = noise.rel(g, g64, sizes)
    out[k] = {"max": max(r.values()), "argmax": names[max(r, key=r.get)],
              **{names[i].replace("features_extractor.", ""): r.get(i) for i in focus}}
    print(k, json.dumps({kk: (f"{v:.2e}" if isinstance(v, float) else v) for kk, v in out[k].items()}))
json.dump(out, open(f"/home/goodlad/gen3ai_archive/k6_k8/r1bar/sdpa_{state.replace('/', '_')}.json", "w"), indent=1)
