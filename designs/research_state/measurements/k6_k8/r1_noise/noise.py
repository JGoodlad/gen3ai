"""K8 R1 per-parameter gradient: MATCHED-NOISE CONTROL (orchestrator 2026-10-01).

For one weight state W and one micro-batch X (the R1 gate's own: the K9 golden rows tiled to B; or
B real rows of arm C's pinned buffer), compute R1's flat gradient over every policy parameter in:
  R64  eager micro_step, float64 (the reference)
  E32  eager, fp32 'highest', on DEVICE
  C32  the compiled R1 (Inductor, fullgraph) on DEVICE
  E32b eager fp32 on DEVICE, deterministic algorithms ON (a reduction-order change)
  X32  eager fp32 on CPU (a reduction-order change: another BLAS / reduction tree)
and per parameter (above the gate's floor, the gate's own relative-error function):
  gate  = rel(C32 | E32)     -- what the R1 gate reads
  c64   = rel(C32 | R64)     -- compiled vs the fp64 truth
  e64   = rel(E32 | R64)     -- eager's own fp32 error
  ee_d  = rel(E32b | E32)    -- eager vs eager, deterministic on/off
  ee_x  = rel(X32 | E32)     -- eager vs eager, CPU vs device
One JSON row per (state, batch) is APPENDED to the out file (resumable: a done key is skipped).
Usage: noise.py <out.jsonl> <device> <B> <state>[,<state>...] [gate|real]"""
import copy, io, json, os, sys, time, zipfile
from pathlib import Path
import numpy as np
import torch as th

from agents.model import compile_regions as cr, compile_trainer as ct
from agents.model.compile_gate_probe import per_param_grad_errors
from agents.training import learner_golden as LG
from agents.training.instrumented_ppo.micro_step import micro_step

M = Path("/home/goodlad/dev/gen3ai/models")
STATES = {
    "fresh": None, "golden": "golden",
    "C_final": M / "ai_v14_02_lbat_ctrl/final_model.zip",
    "C_mid": M / "ai_v14_02_lbat_ctrl/checkpoints/checkpoint_75505968_steps.zip",
    "N0_final": M / "ai_v14_01_base/final_model.zip",
    "E5_final": M / "ai_v14_03_lbat_e5/final_model.zip",
    "L95_final": M / "ai_v14_05_lbat_l95/final_model.zip",
}
FLOOR = ct._PARAM_GRAD_FLOOR


def build(state):
    if state.startswith("fresh_s"):                         # another fresh SEED (unperturbed)
        import agents.model.parity_probe as pp
        real, seed0 = pp.perturb_, LG.MODEL_SEED
        pp.perturb_ = lambda *a, **k: None
        LG.MODEL_SEED = int(state[len("fresh_s"):])
        try:
            model = LG.build_learner()
        finally:
            pp.perturb_, LG.MODEL_SEED = real, seed0
        LG.load_buffer_into(model)
        return model, f"seed {state[len('fresh_s'):]}"
    if state.startswith("golden_s"):                        # another PERTURBATION seed of the fresh build
        import agents.model.parity_probe as pp
        real = pp.perturb_
        k = int(state[len("golden_s"):])
        pp.perturb_ = lambda m, seed, scale: real(m, seed=seed + 1000 * k, scale=scale)
        try:
            model = LG.build_learner()
        finally:
            pp.perturb_ = real
        LG.load_buffer_into(model)
        return model, f"perturb seed +{1000 * k}"
    if state == "fresh":
        import agents.model.parity_probe as pp
        real = pp.perturb_
        pp.perturb_ = lambda *a, **k: None                     # a fresh launch is unperturbed
        try:
            model = LG.build_learner()
        finally:
            pp.perturb_ = real
    else:
        model = LG.build_learner()
    LG.load_buffer_into(model)
    note = ""
    if state not in ("fresh", "golden"):
        path = M / state if "/" in state else STATES[state]      # "<run>/<relative .zip>"
        z = zipfile.ZipFile(path)
        sd = th.load(io.BytesIO(z.read("policy.pth")), map_location="cpu")
        res = model.policy.load_state_dict(sd, strict=False)
        note = f"missing {len(res.missing_keys)} {res.missing_keys[:4]} unexpected {len(res.unexpected_keys)} {res.unexpected_keys[:4]}"
    return model, note


def real_batch(model, B, seed=7):
    import pickle
    buf = pickle.load(open("/home/goodlad/gen3ai_archive/learner_bench/20260928_135948_cuda/rollout_buffer.pkl", "rb"))
    raise NotImplementedError("real buffer: see inspect step")


def to_dev(x, dev, dtype=None):
    if th.is_tensor(x):
        y = x.to(dev)
        return y.to(dtype) if (dtype is not None and y.is_floating_point()) else y
    if isinstance(x, dict):
        return {k: to_dev(v, dev, dtype) for k, v in x.items()}
    return x


def args_for(model, policy, b, dev, dtype=None):
    f = model._resolve_fold_flags()
    st = model._micro_static(f, getattr(policy, "popart", None), None, False)
    var = to_dev(model._micro_var(st, None), dev, dtype)
    return (policy, getattr(policy, "popart", None), to_dev(b.obs, dev, dtype), b.actions.to(dev),
            to_dev(b.action_masks, dev), to_dev(b.old_log_prob, dev, dtype), to_dev(b.old_values, dev, dtype),
            to_dev(b.advantages, dev, dtype), to_dev(b.returns, dev, dtype), var, st)


def grad_of(policy, fn, args, keep64=False):
    params = [p for n, p in policy.named_parameters() if not n.startswith("ridealong.")]
    for p in params:
        p.grad = None
    out = fn(*args)
    out.loss.backward()
    gs = [(p.grad if p.grad is not None else th.zeros_like(p)).detach().double().flatten().cpu() for p in params]
    for p in params:
        p.grad = None
    return float(out.loss.detach().double()), th.cat(gs), [g.numel() for g in gs]


def rel(a, b, sizes):
    """per-param ||a-b||/||b|| over params above the floor (of b's own top norm) -> {i: err}"""
    errs = per_param_grad_errors(a.double(), b.double(), sizes, floor_frac=FLOOR)
    return {i: e for i, e in errs}


def main():
    out, dev, B, states = sys.argv[1], th.device(sys.argv[2]), int(sys.argv[3]), sys.argv[4].split(",")
    src = sys.argv[5] if len(sys.argv) > 5 else "gate"
    done = set()
    if os.path.exists(out):
        done = {(json.loads(l)["state"], json.loads(l)["batch"], json.loads(l)["B"]) for l in open(out)}
    th.set_float32_matmul_precision("highest")
    from agents.model.compile_cache import ensure_hermetic_cache
    ensure_hermetic_cache("k8 r1 noise control")
    for state in states:
        if (state, src, B) in done:
            print("skip", state, src, B); continue
        t0 = time.time()
        model, note = build(state)
        names = [n for n, _ in model.policy.named_parameters() if not n.startswith("ridealong.")]
        b_cpu = cr.r1_batch(model, B)                     # built on the CPU model
        model.policy.set_training_mode(True)
        # X32: CPU eager fp32
        lx, gx, sizes = grad_of(model.policy, micro_step, args_for(model, model.policy, b_cpu, th.device("cpu")))
        # R64: eager float64 (CPU), default dtype float64 for tensors created inside
        m64, _ = build(state)                             # a separate copy (the policy holds stashes)
        pol64 = m64.policy.double()
        pol64.set_training_mode(True)
        prev = th.get_default_dtype(); th.set_default_dtype(th.float64)
        # sb3's preprocess_obs casts a Box obs to float32 (`.float()`); keep the fp64 arm fp64
        import stable_baselines3.common.policies as _sbp
        real_pre = _sbp.preprocess_obs

        def pre64(obs, space, normalize_images=True):
            out = real_pre(obs, space, normalize_images)
            if isinstance(out, dict):
                return {k: (v.double() if th.is_tensor(v) and v.is_floating_point() else v) for k, v in out.items()}
            return out.double() if out.is_floating_point() else out
        _sbp.preprocess_obs = pre64
        # the model casts with `.float()` in ~300 places (an fp32 program): inside the fp64 arm ONLY,
        # `.float()` means "the default float dtype" (float64 here); restored in `finally`
        real_float = th.Tensor.float
        th.Tensor.float = lambda self, *a, **k: self.to(th.get_default_dtype())
        try:
            l64, g64, _ = grad_of(pol64, micro_step, args_for(m64, pol64, b_cpu, th.device("cpu"), th.float64))
        finally:
            th.Tensor.float = real_float
            _sbp.preprocess_obs = real_pre
            th.set_default_dtype(prev)
        assert g64.dtype == th.float64
        del pol64, m64
        # device arms
        model.policy.to(dev); model.device = dev; model.rollout_buffer.device = dev
        a = args_for(model, model.policy, b_cpu, dev)
        le, ge, _ = grad_of(model.policy, micro_step, a)
        le2, ge2, _ = grad_of(model.policy, micro_step, a)
        th.use_deterministic_algorithms(True, warn_only=True)
        try:
            ld, gd, _ = grad_of(model.policy, micro_step, a)
        finally:
            th.use_deterministic_algorithms(False)
        th._dynamo.reset()
        comp = th.compile(micro_step, fullgraph=True, dynamic=False)
        lc, gc, _ = grad_of(model.policy, comp, a)
        lc2, gc2, _ = grad_of(model.policy, comp, a)
        th._dynamo.reset()
        aot = th.compile(micro_step, fullgraph=True, dynamic=False, backend="aot_eager")
        la, ga, _ = grad_of(model.policy, aot, a)
        th._dynamo.reset()
        R = {"gate": rel(gc, ge, sizes), "c64": rel(gc, g64, sizes), "e64": rel(ge, g64, sizes),
             "x64": rel(gx, g64, sizes), "ee_rep": rel(ge2, ge, sizes), "cc_rep": rel(gc2, gc, sizes),
             "ee_det": rel(gd, ge, sizes), "ee_cpu": rel(gx, ge, sizes),
             "aot_e": rel(ga, ge, sizes), "aot64": rel(ga, g64, sizes)}
        keys = sorted(set(R["gate"]) | set(R["c64"]))
        norms64 = [float(x.norm()) for x in th.split(g64, sizes)]
        top64 = max(norms64)
        per = {names[i]: {**{k: R[k].get(i) for k in R}, "norm_frac": norms64[i] / top64} for i in keys}
        summ = {k: {"max": max(v.values()) if v else None, "p99": float(np.quantile(list(v.values()), 0.99)) if v else None,
                    "n": len(v), "argmax": names[max(v, key=v.get)] if v else None} for k, v in R.items()}
        # envelope: compiled's error vs fp64 against eager's own (per param)
        ratio = [R["c64"][i] / max(R["e64"][i], 1e-30) for i in R["c64"] if i in R["e64"]]
        row = {"state": state, "batch": src, "B": B, "device": str(dev), "torch": th.__version__, "note": note,
               "loss": {"R64": l64, "E32": le, "C32": lc, "X32": lx, "Edet": ld, "AOT": la}, "summary": summ,
               "c64_over_e64": {"max": max(ratio), "p99": float(np.quantile(ratio, 0.99)), "median": float(np.median(ratio))},
               "per_param": per, "secs": round(time.time() - t0, 1)}
        with open(out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(state, src, B, json.dumps({k: (round(v["max"], 6) if v["max"] is not None else None, v["argmax"]) for k, v in summ.items()}),
              "c64/e64", {k: round(v, 3) for k, v in row["c64_over_e64"].items()}, note, f"{row['secs']}s", flush=True)
        del model
        if dev.type == "cuda":
            th.cuda.empty_cache()


if __name__ == "__main__":
    main()
