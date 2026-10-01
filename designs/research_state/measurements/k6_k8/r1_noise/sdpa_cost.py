"""Speed cost of forcing the SDPA MATH backend inside region R1 (CUDA, fp32 'highest', B = 2048,
arm C's weights): compiled R1 forward+backward, default kernel choice vs traced under MATH."""
import io, time, zipfile
import torch as th
from torch.nn.attention import sdpa_kernel, SDPBackend
from agents.model import compile_regions as cr
from agents.training import learner_golden as LG
from agents.training.instrumented_ppo.micro_step import micro_step
th.set_float32_matmul_precision("highest")
from agents.model.compile_cache import ensure_hermetic_cache
ensure_hermetic_cache("k8 sdpa cost")
m = LG.build_learner(); LG.load_buffer_into(m)
sd = th.load(io.BytesIO(zipfile.ZipFile("/home/goodlad/dev/gen3ai/models/ai_v14_02_lbat_ctrl/final_model.zip").read("policy.pth")), map_location="cpu")
m.policy.load_state_dict(sd, strict=True)
dev = th.device("cuda"); m.policy.to(dev); m.device = dev; m.rollout_buffer.device = dev
m.policy.set_training_mode(True)
args = cr._r1_args(m, cr.r1_batch(m, 2048))
def math_step(*a):
    with sdpa_kernel(SDPBackend.MATH):
        return micro_step(*a)
for name, fn in (("default", micro_step), ("MATH", math_step), ("default2", micro_step), ("MATH2", math_step)):
    th._dynamo.reset()
    c = th.compile(fn, fullgraph=True, dynamic=False)
    for _ in range(3):
        out = c(*args); out.loss.backward(); m.policy.zero_grad(set_to_none=True)
    th.cuda.synchronize(); t0 = time.perf_counter(); n = 30
    for _ in range(n):
        out = c(*args); out.loss.backward(); m.policy.zero_grad(set_to_none=True)
    th.cuda.synchronize()
    print(f"{name}: {(time.perf_counter() - t0) / n * 1000:.1f} ms per R1 fwd+bwd at B=2048", flush=True)
th._dynamo.reset()
