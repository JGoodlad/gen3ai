"""WHICH SDPA call site carries the E-config compiled-gradient error? (gpu_checks_endstate, 2026-10-09). GPU, lease.

    python site_probe.py --site {none,trunk,extra,hopb,all} [--cfg E] [--dump-inputs out.pt]

The launch-equivalent weight state (seed 42 + the gate's perturbed rung 0), the eager fp32 CUDA reference with
the DEFAULT SDPA backend everywhere, and a fresh `torch.compile(micro_step, fullgraph=True, dynamic=False)` in
which ONLY the named site's forward runs under `sdpa_kernel([SDPBackend.MATH])`:
  trunk = `BiasedEncoderLayer` (the two trunk rounds), extra = `IdentityInitRound` (`--trunk-layers 3`'s round),
  hopb  = `HiddenOppBeliefPool` (its decoder's self- and cross-attention, the X5 float key mask).
The site whose MATH restriction removes the disagreement is the one whose memory-efficient backward is off.
``--dump-inputs``: also records, per site, each SDPA call's mask statistics in the EAGER pass (fully-masked rows,
the mask's min / max, finite share, shape, stride, dtype, requires_grad).
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
from agents.model.team_transformer import BiasedEncoderLayer      # noqa: E402
from agents.model.trunk_depth import IdentityInitRound            # noqa: E402
from agents.model.pools import HiddenOppBeliefPool                # noqa: E402

SITES = {"trunk": [BiasedEncoderLayer], "extra": [IdentityInitRound], "hopb": [HiddenOppBeliefPool]}
SITES["all"] = SITES["trunk"] + SITES["extra"] + SITES["hopb"]


def math_only(cls):
    from torch.nn.attention import SDPBackend, sdpa_kernel
    orig = cls.forward

    def fwd(self, *a, **k):
        with sdpa_kernel([SDPBackend.MATH]):
            return orig(self, *a, **k)
    cls.forward = fwd
    return lambda: setattr(cls, "forward", orig)


def mask_stats(tag, store):
    real = th.nn.functional.scaled_dot_product_attention

    def wrapped(q, k, v, attn_mask=None, *a, **kw):
        if attn_mask is not None and attn_mask.dtype != th.bool:
            m = attn_mask.detach()
            rowmax = m.amax(-1)
            store.append({"site": tag, "q": list(q.shape), "mask_shape": list(m.shape), "stride": list(m.stride()),
                          "dtype": str(m.dtype), "requires_grad": bool(attn_mask.requires_grad),
                          "min": float(m.min()), "max": float(m.max()),
                          "rows_all_le_-1e8": int((rowmax <= -1e8).sum()), "rows": int(rowmax.numel()),
                          "finite_share": float(th.isfinite(m).float().mean())})
        elif attn_mask is not None:
            store.append({"site": tag, "bool_mask": list(attn_mask.shape),
                          "rows_fully_masked": int((~attn_mask).all(-1).sum())})
        return real(q, k, v, attn_mask=attn_mask, *a, **kw)
    return wrapped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="E")
    ap.add_argument("--site", required=True, choices=["none"] + sorted(SITES))
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--model-seed", type=int, default=42)
    ap.add_argument("--dump-inputs")
    a = ap.parse_args()
    NC.MODEL_SEED = a.model_seed
    th.set_float32_matmul_precision("highest")
    from agents.model.compile_cache import ensure_hermetic_cache
    ensure_hermetic_cache("gpu checks endstate site probe")
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
    if a.dump_inputs:
        import json
        store = []
        real = th.nn.functional.scaled_dot_product_attention
        th.nn.functional.scaled_dot_product_attention = mask_stats("any", store)
        try:
            with th.no_grad():
                micro_step(*args)
        finally:
            th.nn.functional.scaled_dot_product_attention = real
        json.dump(store, open(a.dump_inputs, "w"), indent=1)
        for s in store:
            print("SDPA call:", s, flush=True)
    _, ge, sizes = NC.N.grad_of(model.policy, micro_step, args)
    undo = [math_only(c) for c in SITES.get(a.site, [])]
    try:
        th._dynamo.reset()
        comp = th.compile(micro_step, fullgraph=True, dynamic=False)
        _, gc, _ = NC.N.grad_of(model.policy, comp, args)
    finally:
        for u in undo:
            u()
        th._dynamo.reset()
    gate = NC.N.rel(gc, ge, sizes)
    top = sorted(gate.items(), key=lambda x: -x[1])[:4]
    print(f"site={a.site} cfg={a.cfg} worst:", [(names[i].replace("features_extractor.", ""), f"{e:.2e}") for i, e in top],
          flush=True)


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        main()
