"""The oracle canary FATAL (2026-10-04): a CPU matched-noise control on the fatal weights.

For each weight state (the oracle run's `final_model_canary_fatal.zip`, the blob seed's checkpoint at
~1M, and each again under the startup gate's seeded perturbation), R1's gradient over every policy
parameter on the canary's own rows (`compile_regions.r1_batch`, slice 0, B rows) by three arms:
  R64  eager micro_step in float64 (CPU; k6_k8/r1_noise/noise.py's two measurement-only patches)
  X32  eager fp32 on the CPU
  C32  the compiled R1 on the CPU (Inductor, fullgraph, static) -- CPU Inductor carries a KNOWN
       unresolved belief-path deviation (k6_k8/r1_noise README), so C32 here is NOT the CUDA graph
and per parameter the gate's own relative error. For `belief_head.species_head.weight` also: the
absolute gradient norm, its fraction of the top parameter's, and the CONDITION NUMBER of its
gradient as a sum over (row, slot) terms: ||(|dlogits|^T |x|)|| / ||dlogits^T x||, from the fp64 arm
(x = the head's input, dlogits = dL/d its output).
Usage: repro.py <out.jsonl> <B> [compile|eager] [live,all,unmoved] [oracle_fatal,blob_1M]"""
import io
import json
import sys
import time
import zipfile

import torch as th

from agents.model import compile_regions as cr, compile_trainer as ct
from agents.model.compile_gate_probe import per_param_grad_errors
from agents.training import learner_golden as LG
from agents.training.instrumented_ppo.micro_step import micro_step

M = "/home/goodlad/dev/gen3ai/models/"
STATES = {"oracle_fatal": M + "rb_x5ab_oracle_sp_s1001/final_model_canary_fatal.zip",
          "blob_1M": M + "rb_x5ab_blob_s1001/checkpoints/checkpoint_1000192_steps.zip"}
SP = "features_extractor.belief_head.species_head.weight"


def build(path):
    model = LG.build_learner()
    LG.load_buffer_into(model)
    sd = th.load(io.BytesIO(zipfile.ZipFile(path).read("policy.pth")), map_location="cpu")
    sd = {k: v for k, v in sd.items() if not k.startswith("ridealong.")}
    res = model.policy.load_state_dict(sd, strict=False)
    bad = [k for k in res.missing_keys if not k.startswith("ridealong.")]
    assert not bad and not res.unexpected_keys, (bad[:5], res.unexpected_keys[:5])
    return model


def _params(model):
    return ct.grad_parameters(model, model.policy.features_extractor)


def grads(model, fn, args, hook=False):
    names = [n for n, _ in _params(model)]
    params = [p for _, p in _params(model)]
    cap = {}
    hs = []
    if hook:
        mod = model.policy.features_extractor.belief_head.species_head
        hs.append(mod.register_forward_hook(lambda m, i, o: cap.__setitem__("x", i[0].detach())))
        hs.append(mod.register_full_backward_hook(lambda m, gi, go: cap.__setitem__("d", go[0].detach())))
    try:
        for p in params:
            p.grad = None
        out = fn(*args)
        out.loss.backward()
        g = th.cat([(p.grad if p.grad is not None else th.zeros_like(p)).detach().double().flatten()
                    for p in params])
        for p in params:
            p.grad = None
    finally:
        for h in hs:
            h.remove()
    return {"loss": float(out.loss.detach().double()), "grad": g,
            "grad_sizes": [p.numel() for p in params]}, names, cap


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _ctx(model, perturb):
    """``perturb``: False (live), True (every parameter: the fresh launch's rung), or "unmoved" (only the
    bit-exactly-zero parameters: `compile_regions.r1_rungs`' unmoved rung)."""
    if not perturb:
        return _null()
    from agents.model.parity_probe import PERTURB_LADDER, perturbed_parameters, rung_seed
    scale, k = PERTURB_LADDER[0]
    only = cr.unmoved_parameters(model) if perturb == "unmoved" else None
    return perturbed_parameters(model.policy, seed=rung_seed(k), scale=scale, only=only)


def _to64(a):
    if isinstance(a, dict):
        return {k: _to64(v) for k, v in a.items()}
    return a.double() if th.is_tensor(a) and a.is_floating_point() else a


def r64(path, B, perturb):
    import stable_baselines3.common.policies as _sbp
    m = build(path)
    with _ctx(m, perturb):          # perturb in fp32 (the same draw as the fp32 arms), then cast
        m.policy.double()
        args = tuple(_to64(a) for a in cr._r1_args(m, cr.r1_batch(m, B)))
        prev = th.get_default_dtype()
        th.set_default_dtype(th.float64)
        real_pre, real_float = _sbp.preprocess_obs, th.Tensor.float

        def pre64(obs, space, normalize_images=True):
            return _to64(real_pre(obs, space, normalize_images))
        _sbp.preprocess_obs = pre64
        th.Tensor.float = lambda self, *a, **k: self.to(th.get_default_dtype())
        try:
            m.policy.set_training_mode(True)
            res = grads(m, micro_step, args, hook=True)
        finally:
            th.Tensor.float = real_float
            _sbp.preprocess_obs = real_pre
            th.set_default_dtype(prev)
        m.policy.float()            # the context restores fp32 bytes
    return res


def rel(a, b, sizes):
    return dict(per_param_grad_errors(a.double(), b.double(), sizes, floor_frac=ct._PARAM_GRAD_FLOOR))


def main():
    out, B = sys.argv[1], int(sys.argv[2])
    do_compile = len(sys.argv) > 3 and sys.argv[3] == "compile"
    modes = [False, True] if len(sys.argv) <= 4 else [{"live": False, "all": True}.get(x, x)
                                                      for x in sys.argv[4].split(",")]
    states = STATES if len(sys.argv) <= 5 else {k: STATES[k] for k in sys.argv[5].split(",")}
    th.set_float32_matmul_precision("highest")
    for state, path in states.items():
        for perturb in modes:
            t0 = time.time()
            g64, names, cap = r64(path, B, perturb)
            m = build(path)
            with _ctx(m, perturb):
                m.policy.set_training_mode(True)
                args = cr._r1_args(m, cr.r1_batch(m, B))
                x32, _, _ = grads(m, micro_step, args)
                c32 = None
                if do_compile:
                    th._dynamo.reset()
                    comp = th.compile(micro_step, fullgraph=True, dynamic=False)
                    c32, _, _ = grads(m, comp, args)
                    th._dynamo.reset()
                wnorm = float(m.policy.features_extractor.belief_head.species_head.weight.norm())
            sizes = g64["grad_sizes"]
            norms64 = [float(x.norm()) for x in th.split(g64["grad"], sizes)]
            top = max(norms64)
            i = names.index(SP)
            x = cap["x"].reshape(-1, cap["x"].shape[-1])
            d = cap["d"].reshape(-1, cap["d"].shape[-1])
            gw = d.t() @ x
            gabs = d.abs().t() @ x.abs()
            R = {"x64": rel(x32["grad"], g64["grad"], sizes)}
            if c32 is not None:
                R["c64"] = rel(c32["grad"], g64["grad"], sizes)
                R["gate_cpu"] = rel(c32["grad"], x32["grad"], sizes)
            row = {"state": state, "perturbed": perturb, "B": B, "torch": th.__version__,
                   "species_head": {"grad_norm_fp64": norms64[i], "norm_frac": norms64[i] / top,
                                    "top_param": names[norms64.index(top)], "top_norm": top,
                                    "weight_norm": wnorm,
                                    "kappa_norm": float(gabs.norm() / max(float(gw.norm()), 1e-300)),
                                    "kappa_elem_median": float((gabs / gw.abs().clamp_min(1e-300)).median()),
                                    "terms_nonzero": int((d.abs().sum(-1) > 0).sum()),
                                    "terms_total": int(d.shape[0]),
                                    **{k: v.get(i) for k, v in R.items()}},
                   "summary": {k: {"max": max(v.values()), "argmax": names[max(v, key=v.get)], "n": len(v)}
                               for k, v in R.items()},
                   "loss": {"R64": g64["loss"], "X32": x32["loss"], "C32": c32["loss"] if c32 else None},
                   "secs": round(time.time() - t0, 1)}
            with open(out, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
