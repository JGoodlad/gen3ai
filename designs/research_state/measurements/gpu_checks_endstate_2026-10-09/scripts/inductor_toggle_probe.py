"""WHICH Inductor transformation carries the E-config compiled-gradient error? (gpu_checks_endstate, 2026-10-09).
GPU, under the lease.

    python inductor_toggle_probe.py --toggles base,no_pattern_matcher,no_split_reductions,... [--cfg E]

One launch-equivalent weight state (seed 42 + the gate's perturbed rung 0), ONE eager fp32 CUDA reference R1
gradient, then per toggle a fresh `torch.compile(micro_step, fullgraph=True, dynamic=False)` under that Inductor
config patch (or SDPA backend restriction), and the gate's worst parameters against the eager reference. `aot_eager`
is bit-equal to eager on this state (`noise_cfg.py`), so the error is Inductor's; a toggle that removes it names
the transformation. Measurement only — no production setting is changed.
"""
import argparse
import contextlib
import os
import sys

import torch as th

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import noise_cfg as NC                                     # noqa: E402
from agents.model import compile_regions as cr            # noqa: E402
from agents.training.instrumented_ppo.micro_step import micro_step   # noqa: E402


def toggle_ctx(name):
    import torch._inductor.config as ic
    from torch.nn.attention import SDPBackend, sdpa_kernel
    if name == "base":
        return contextlib.nullcontext()
    if name == "no_pattern_matcher":
        return ic.patch(pattern_matcher=False)
    if name == "no_split_reductions":
        return ic.patch(split_reductions=False)
    if name == "no_persistent_reductions":
        return ic.patch({"triton.persistent_reductions": False})
    if name == "no_fusion":
        return ic.patch(max_fusion_size=1)
    if name == "sdpa_math":
        return sdpa_kernel([SDPBackend.MATH])
    if name == "sdpa_efficient":
        return sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION])
    if name == "emulate_precision_casts":
        return ic.patch(emulate_precision_casts=True)
    if name == "deterministic":
        @contextlib.contextmanager
        def det():
            th.use_deterministic_algorithms(True, warn_only=True)
            try:
                yield
            finally:
                th.use_deterministic_algorithms(False)
        return det()
    if name == "no_autotune_pointwise":
        return ic.patch({"triton.autotune_pointwise": False})
    if name == "no_mm_decompose":
        return ic.patch({"post_grad_fusion_options": {}, "pre_grad_fusion_options": {}})
    raise KeyError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="E")
    ap.add_argument("--toggles", required=True)
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--model-seed", type=int, default=42)
    a = ap.parse_args()
    NC.MODEL_SEED = a.model_seed
    th.set_float32_matmul_precision("highest")
    from agents.model.compile_cache import ensure_hermetic_cache
    ensure_hermetic_cache("gpu checks endstate inductor toggles")
    print("tf32 matmul", th.backends.cuda.matmul.allow_tf32, "cudnn", th.backends.cudnn.allow_tf32,
          "precision", th.get_float32_matmul_precision(), flush=True)
    dev = th.device("cuda")
    model, cm, _ = NC.build(a.cfg, a.k)
    cm.__enter__()
    names = [n for n, _ in model.policy.named_parameters() if not n.startswith("ridealong.")]
    b = cr.r1_batch(model, 2048)
    model.policy.set_training_mode(True)
    model.policy.to(dev)
    model.device = dev
    model.rollout_buffer.device = dev
    args = NC.args_for(model, model.policy, b, dev)
    refs = {}
    for t in a.toggles.split(","):
        with toggle_ctx(t):
            key = "sdpa_math" if t == "sdpa_math" else ("sdpa_efficient" if t == "sdpa_efficient" else "default")
            if key not in refs:                         # the eager reference under the SAME SDPA restriction
                _, refs[key], sizes = NC.N.grad_of(model.policy, micro_step, args)
            th._dynamo.reset()
            try:
                comp = th.compile(micro_step, fullgraph=True, dynamic=False)
                _, g, sizes = NC.N.grad_of(model.policy, comp, args)
                gate = NC.N.rel(g, refs[key], sizes)
                top = sorted(gate.items(), key=lambda x: -x[1])[:4]
                print(t, "worst:", [(names[i].replace("features_extractor.", ""), f"{e:.2e}") for i, e in top],
                      flush=True)
            except Exception as exc:  # noqa: BLE001 — a toggle that cannot compile is reported, not fatal
                print(t, "FAILED:", repr(exc)[:300], flush=True)
            th._dynamo.reset()


if __name__ == "__main__":
    main()
