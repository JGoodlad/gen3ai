"""R1's per-parameter gradient: the MATCHED-NOISE CONTROL for one CONFIGURATION (gpu_checks_endstate, 2026-10-09).

The method is `designs/research_state/measurements/k6_k8/r1_noise/noise.py`'s (the evidence the R1 bar rests on),
unchanged except that the learner is built for a configuration's RESOLVED namespace (P / R / E, as
`cuda_parity_targets.py`) and the weight state is the STARTUP GATE'S OWN perturbed rung: the fresh build,
unperturbed, then `parity_probe.perturbed_parameters(policy, seed=rung_seed(k), scale)` at the ladder's first rung
(`compile_regions._r1_perturbed`), or another seed offset k for a replicate.

For one weight state and the gate's batch (`compile_regions.r1_batch`, the K9 golden rows tiled to B), R1's flat
gradient over every policy parameter in five arms: R64 (eager float64, CPU — the reference), E32 (eager fp32 on
CUDA), C32 (the compiled R1, Inductor, fullgraph, on CUDA), X32 (eager fp32 on the CPU), AOT (aot_eager: the same
traced graph run eagerly), with the per-parameter relative error of the gate's own function. A compiled gradient
whose error against float64 is within 2x the WORSE of the two correct fp32 implementations' own (CUDA eager, CPU
eager) is fp32 noise of an ill-conditioned gradient, not a miscompile (the k6_k8 rule).

    GEN3AI_TEST_ALLOW_GPU=1 python noise_cfg.py <out.jsonl> <cfg> <B> <k>[,<k>...] [--focus <param substring>]
"""
import contextlib
import io
import json
import os
import sys
import time

import torch as th

HERE = os.path.dirname(os.path.abspath(__file__))
# THIS checkout's src first: a script run by path gets its own dir as sys.path[0], and the editable install
# resolves `agents` to the MAIN checkout (whose code moves under a running diagnosis — it did, 2026-10-09).
sys.path.insert(0, os.environ.get("DIAG_SRC") or os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "..", "src")))
sys.path.insert(0, os.path.join(HERE, "..", "..", "k6_k8", "r1_noise"))
sys.path.insert(0, HERE)
import noise as N                                    # noqa: E402  the k6_k8 method's own helpers
from cuda_parity_targets import CFG, resolved_args   # noqa: E402

from agents.model import compile_regions as cr      # noqa: E402
from agents.training import learner_golden as LG    # noqa: E402
from agents.training.instrumented_ppo.micro_step import micro_step   # noqa: E402


def args_for(model, policy, b, dev, dtype=None):
    """R1's arguments as `compile_regions._r1_args` builds them at HEAD (noise.py's predates the
    `_micro_static(flags)` signature), moved to ``dev`` (floating tensors cast to ``dtype`` when given)."""
    st = model._micro_static(model._resolve_fold_flags())
    T = N.to_dev
    return (policy, T(b.obs, dev, dtype), b.actions.to(dev), T(b.action_masks, dev), T(b.old_log_prob, dev, dtype),
            T(b.old_values, dev, dtype), T(b.advantages, dev, dtype), T(b.returns, dev, dtype), st)


MODEL_SEED = None     # --model-seed: the trainer's `--seed` (42) reproduces a launch's fresh weights (init_match.py)


def build(cfg, k):
    import agents.model.parity_probe as pp
    real, seed0 = pp.perturb_, LG.MODEL_SEED
    pp.perturb_ = lambda *a, **kw: None                 # a fresh launch is unperturbed
    if MODEL_SEED is not None:
        LG.MODEL_SEED = MODEL_SEED
    try:
        model = LG.build_learner(args=resolved_args(CFG[cfg]))
    finally:
        pp.perturb_, LG.MODEL_SEED = real, seed0
    LG.load_buffer_into(model)
    scale, _k0 = pp.PERTURB_LADDER[0]
    cm = pp.perturbed_parameters(model.policy, seed=pp.rung_seed(k), scale=scale)
    return model, cm, f"fresh build + perturbed_parameters(seed=rung_seed({k}), scale={scale})"


def main():
    out, cfg, B = sys.argv[1], sys.argv[2], int(sys.argv[3])
    ks = [int(x) for x in sys.argv[4].split(",")]
    focus = sys.argv[sys.argv.index("--focus") + 1] if "--focus" in sys.argv else None
    global MODEL_SEED
    if "--model-seed" in sys.argv:
        MODEL_SEED = int(sys.argv[sys.argv.index("--model-seed") + 1])
    dev = th.device("cuda")
    th.set_float32_matmul_precision("highest")
    from agents.model.compile_cache import ensure_hermetic_cache
    ensure_hermetic_cache("gpu checks endstate noise control")
    for k in ks:
        t0 = time.time()
        model, cm, note = build(cfg, k)
        m64, cm64, _ = build(cfg, k)
        # entered and NEVER exited: the models are discarded, and the fp64 copy's restore would meet a dtype change
        cm.__enter__()
        cm64.__enter__()
        if True:
            names = [n for n, _ in model.policy.named_parameters() if not n.startswith("ridealong.")]
            b_cpu = cr.r1_batch(model, B)
            model.policy.set_training_mode(True)
            lx, gx, sizes = N.grad_of(model.policy, micro_step, args_for(model, model.policy, b_cpu, th.device("cpu")))
            pol64 = m64.policy.double()
            pol64.set_training_mode(True)
            prev = th.get_default_dtype()
            th.set_default_dtype(th.float64)
            import stable_baselines3.common.policies as _sbp
            real_pre = _sbp.preprocess_obs

            def pre64(obs, space, normalize_images=True):
                o = real_pre(obs, space, normalize_images)
                if isinstance(o, dict):
                    return {kk: (v.double() if th.is_tensor(v) and v.is_floating_point() else v) for kk, v in o.items()}
                return o.double() if o.is_floating_point() else o
            _sbp.preprocess_obs = pre64
            real_float = th.Tensor.float
            th.Tensor.float = lambda self, *a, **kw: self.to(th.get_default_dtype())
            try:
                l64, g64, _ = N.grad_of(pol64, micro_step, args_for(m64, pol64, b_cpu, th.device("cpu"), th.float64))
            finally:
                th.Tensor.float = real_float
                _sbp.preprocess_obs = real_pre
                th.set_default_dtype(prev)
            model.policy.to(dev)
            model.device = dev
            model.rollout_buffer.device = dev
            a = args_for(model, model.policy, b_cpu, dev)
            le, ge, _ = N.grad_of(model.policy, micro_step, a)
            le2, ge2, _ = N.grad_of(model.policy, micro_step, a)
            th._dynamo.reset()
            comp = th.compile(micro_step, fullgraph=True, dynamic=False)
            lc, gc, _ = N.grad_of(model.policy, comp, a)
            lc2, gc2, _ = N.grad_of(model.policy, comp, a)
            th._dynamo.reset()
            aot = th.compile(micro_step, fullgraph=True, dynamic=False, backend="aot_eager")
            la, ga, _ = N.grad_of(model.policy, aot, a)
            th._dynamo.reset()
        R = {"gate": N.rel(gc, ge, sizes), "c64": N.rel(gc, g64, sizes), "e64": N.rel(ge, g64, sizes),
             "x64": N.rel(gx, g64, sizes), "ee_rep": N.rel(ge2, ge, sizes), "cc_rep": N.rel(gc2, gc, sizes),
             "ee_cpu": N.rel(gx, ge, sizes), "aot_e": N.rel(ga, ge, sizes), "aot64": N.rel(ga, g64, sizes)}
        keys = sorted(set(R["gate"]) | set(R["c64"]))
        norms64 = [float(x.norm()) for x in th.split(g64, sizes)]
        top64 = max(norms64)
        per = {names[i]: {**{kk: R[kk].get(i) for kk in R}, "norm_frac": norms64[i] / top64} for i in keys}
        summ = {kk: {"max": max(v.values()) if v else None, "n": len(v),
                     "argmax": names[max(v, key=v.get)] if v else None} for kk, v in R.items()}
        env_viol = []
        for i in R["c64"]:
            envelope = max(R["e64"].get(i, 0.0), R["x64"].get(i, 0.0))
            if R["c64"][i] > 2.0 * envelope and R["c64"][i] > 1e-5:
                env_viol.append((names[i], R["c64"][i], R["e64"].get(i), R["x64"].get(i)))
        row = {"cfg": cfg, "k": k, "model_seed": MODEL_SEED, "B": B, "torch": th.__version__, "note": note,
               "loss": {"R64": l64, "E32": le, "C32": lc, "X32": lx, "AOT": la}, "summary": summ,
               "outside_2x_eager_envelope": env_viol, "per_param": per, "secs": round(time.time() - t0, 1)}
        if focus:
            row["focus"] = {n: v for n, v in per.items() if focus in n}
        with open(out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(cfg, k, json.dumps({kk: (v["max"] and round(v["max"], 6), v["argmax"]) for kk, v in summ.items()}),
              "outside envelope:", env_viol[:6], f"{row['secs']}s", flush=True)
        if focus:
            for n, v in row["focus"].items():
                print("  ", n, {kk: (round(x, 7) if isinstance(x, float) else x) for kk, x in v.items()}, flush=True)
        del model, m64
        th.cuda.empty_cache()


if __name__ == "__main__":
    with contextlib.redirect_stderr(io.StringIO()) if os.environ.get("QUIET") else contextlib.nullcontext():
        main()
