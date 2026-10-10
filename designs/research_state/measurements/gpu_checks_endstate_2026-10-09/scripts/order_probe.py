"""WHY does the E-config gate failure reproduce in `noise_cfg.py` but not in `ablate.py`? (gpu_checks_endstate,
2026-10-09). GPU, under the lease.

    python order_probe.py --cfg E --steps cpu_eager,fp64,cuda_eager,compile,compile2 [--k 0] [--model-seed 42]

Runs the steps in the given order on ONE launch-equivalent weight state (seed 42 + the gate's perturbed rung k):
  cpu_eager  — R1 eager fp32 on the CPU (noise_cfg's X32 arm, BEFORE the model moves to CUDA)
  fp64       — noise_cfg's float64 arm on a SECOND build (default dtype float64, `Tensor.float` patched)
  cuda_eager — R1 eager fp32 on CUDA (moves the model)
  compile    — a fresh `torch.compile(micro_step, fullgraph=True, dynamic=False)` (after `dynamo.reset`) on CUDA
  compile2   — another fresh compile (a second, independent compilation of the same graph)
  compile_reuse — a fresh compile run on the SAME argument tuple the previous step used (no `args_for` rebuild)
and prints, per compiled arm, the gate's worst parameters against the FIRST cuda_eager arm.
"""
import argparse
import os
import sys

import torch as th

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import noise_cfg as NC                                     # noqa: E402
from agents.model import compile_regions as cr            # noqa: E402
from agents.training.instrumented_ppo.micro_step import micro_step   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="E")
    ap.add_argument("--steps", required=True)
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--model-seed", type=int, default=42)
    a = ap.parse_args()
    if a.cfg not in NC.CFG:                              # an ablation variant (`ablate.VARIANTS`)
        import ablate
        NC.CFG[a.cfg] = ablate.VARIANTS[a.cfg]
    NC.MODEL_SEED = a.model_seed
    th.set_float32_matmul_precision("highest")
    from agents.model.compile_cache import ensure_hermetic_cache
    ensure_hermetic_cache("gpu checks endstate order probe")
    dev = th.device("cuda")
    model, cm, _ = NC.build(a.cfg, a.k)
    cm.__enter__()
    names = [n for n, _ in model.policy.named_parameters() if not n.startswith("ridealong.")]
    b = cr.r1_batch(model, 2048)
    model.policy.set_training_mode(True)
    ref = None
    on_cuda = False
    for s in a.steps.split(","):
        if s == "cpu_eager":
            _, g, sizes = NC.N.grad_of(model.policy, micro_step, NC.args_for(model, model.policy, b, th.device("cpu")))
            print("cpu_eager done", flush=True)
        elif s == "fp64":
            m64, cm64, _ = NC.build(a.cfg, a.k)
            cm64.__enter__()
            pol64 = m64.policy.double()
            pol64.set_training_mode(True)
            prev = th.get_default_dtype()
            th.set_default_dtype(th.float64)
            import stable_baselines3.common.policies as _sbp
            real_pre = _sbp.preprocess_obs

            def pre64(obs, space, normalize_images=True):
                o = real_pre(obs, space, normalize_images)
                return {kk: (v.double() if th.is_tensor(v) and v.is_floating_point() else v) for kk, v in o.items()}
            _sbp.preprocess_obs = pre64
            real_float = th.Tensor.float
            th.Tensor.float = lambda self, *x, **kw: self.to(th.get_default_dtype())
            try:
                NC.N.grad_of(pol64, micro_step, NC.args_for(m64, pol64, b, th.device("cpu"), th.float64))
            finally:
                th.Tensor.float = real_float
                _sbp.preprocess_obs = real_pre
                th.set_default_dtype(prev)
            print("fp64 done", flush=True)
        else:
            if not on_cuda:
                model.policy.to(dev)
                model.device = dev
                model.rollout_buffer.device = dev
                on_cuda = True
            if s != "compile_reuse":
                args = NC.args_for(model, model.policy, b, dev)
            if s == "cuda_eager":
                _, g, sizes = NC.N.grad_of(model.policy, micro_step, args)
                if ref is None:
                    ref = g
                else:
                    print("cuda_eager vs first cuda_eager: max", max(NC.N.rel(g, ref, sizes).values()), flush=True)
                print("cuda_eager done", flush=True)
            elif s in ("compile", "compile2", "compile_reuse"):
                th._dynamo.reset()
                comp = th.compile(micro_step, fullgraph=True, dynamic=False)
                _, g, sizes = NC.N.grad_of(model.policy, comp, args)
                th._dynamo.reset()
                gate = NC.N.rel(g, ref, sizes)
                top = sorted(gate.items(), key=lambda t: -t[1])[:5]
                print(s, "worst:", [(names[i].replace("features_extractor.", ""), f"{e:.2e}") for i, e in top],
                      flush=True)


if __name__ == "__main__":
    main()
