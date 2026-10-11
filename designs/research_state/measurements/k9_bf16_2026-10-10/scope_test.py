"""Design (b)'s probe: can the K9(b) probe forward be FORCED to fp32 while the policy carries the bf16 round
classes, and is today's fp32 gate then bit-for-bit untouched? (`README.md` item 3.) Measurement only: the two
candidate mechanisms are built HERE, never in ``src/``.

* **A. class-swap scope** — `set_trunk_precision(policy, "fp32")` for the probe, then back to ``"bf16"``
  (`common.trunk`).
* **B. a flag the bf16 forward reads** — measurement-local subclasses of the round classes whose forward skips
  the autocast region while a module-level flag is set (the production `Bf16*` classes read no flag).

Checks, on the probe's own forward (train mode, grad on, nothing saved) over a seeded real rollout:
1. under each mechanism, log pi and the full masked log-probs are BIT-IDENTICAL to a pure fp32 policy (a deep
   copy at ``"fp32"``) — and so are the tie margins (the gate's exclusion);
2. after the scope, the bf16 forward is bit-identical to before it (nothing leaked);
3. the COMPILED interaction (dynamo, the R1 hazard): a round compiled while bf16 (``fullgraph=True``, backend
   ``--backend``): (i) a call after a scope that ran only EAGER code adds no compile; (ii) a compiled call made
   INSIDE the scope — does dynamo's guard catch the swapped class / the flag (a recompile: loud under the
   K6 lock) or reuse the stale bf16 graph (silent)?

    python scope_test.py --out scope_result.json [--backend eager|aot_eager|inductor]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Any, Dict, Optional

import numpy as np
import torch as th

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

FORCE_FP32 = False


def _flag_class(base: type) -> type:
    def forward(self: Any, x: th.Tensor, bias: Optional[th.Tensor] = None) -> th.Tensor:
        if FORCE_FP32:
            return base.forward(self, x, bias)
        with th.autocast(device_type=x.device.type, dtype=th.bfloat16):
            out: th.Tensor = base.forward(self, x, bias)
        return out.float()
    return type(f"FlagBf16{base.__name__}", (base,), {"forward": forward, "__module__": __name__})


def main() -> int:
    global FORCE_FP32
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=64)
    ap.add_argument("--backend", default="eager")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    th.set_num_threads(a.threads)
    from agents.model.trunk_precision import _round_types, set_trunk_precision, trunk_precision_of
    from agents.training.rust_rollout.tie_margins import TieMargins

    model, env, _ = C.build(state="perturb05", weights_from=None, seed=1001, n_envs=a.n_envs, n_steps=a.n_steps,
                            run_seed=1001, device="cpu", t2_backend="eager", trunk="fp32")
    res: Dict[str, Any] = {"backend": a.backend, "torch": th.__version__}
    try:
        C.collect(model, env)
        R = C.rows(model)
        n = R["n"]
        pol = model.policy
        rk = lambda k: TieMargins(k, keep_calls=True)  # noqa: E731

        def probe() -> Dict[str, np.ndarray]:
            lp, full, md = C.forward(model, R, chunk=n, train=True, grad=True, mode_factory=rk)
            return {"lp": lp, "full": full, "m": C.margins_of(md, n)[0]}

        # the pure fp32 reference (today's probe), then the policy goes bf16
        C.K_release(pol)
        ref_pol = copy.deepcopy(pol)
        ref32 = probe()
        set_trunk_precision(pol, "bf16")
        bf = probe()

        def same(x: Dict[str, np.ndarray], y: Dict[str, np.ndarray]) -> Dict[str, bool]:
            return {k: bool(np.array_equal(x[k], y[k], equal_nan=True)) for k in x}

        res["bf16_differs_from_fp32"] = {k: not v for k, v in same(bf, ref32).items()}
        res["bf16_vs_fp32_max_abs_dlogp"] = float(np.max(np.abs(bf["lp"] - ref32["lp"])))
        # A: class-swap scope
        with C.trunk(pol, "fp32"):
            a_in = probe()
            res["A_precision_inside"] = trunk_precision_of(pol)
        res["A_precision_after"] = trunk_precision_of(pol)
        res["A_inside_equals_fp32"] = same(a_in, ref32)
        res["A_after_equals_bf16_before"] = same(probe(), bf)
        # B: the flag (measurement-local classes)
        flag_of = {base: _flag_class(base) for base in _round_types()}
        bf16_to_base = {v: k for k, v in _round_types().items()}
        for m in pol.modules():
            if type(m) in bf16_to_base:
                m.__class__ = flag_of[bf16_to_base[type(m)]]
        res["B_flag_off_equals_bf16"] = same(probe(), bf)
        FORCE_FP32 = True
        try:
            b_in = probe()
        finally:
            FORCE_FP32 = False
        res["B_flag_on_equals_fp32"] = same(b_in, ref32)
        res["B_after_equals_bf16"] = same(probe(), bf)
        # `set_trunk_precision` does not know a foreign subclass of a round class: it SKIPS it (and
        # `trunk_precision_of` does not count it) — recorded, then the production classes restored by hand
        res["set_trunk_precision_skips_foreign_subclass"] = set_trunk_precision(pol, "bf16") == 0
        res["trunk_precision_of_with_foreign_subclass"] = trunk_precision_of(pol)
        flag_to_base = {v: k for k, v in flag_of.items()}
        for m in pol.modules():
            if type(m) in flag_to_base:
                m.__class__ = _round_types()[flag_to_base[type(m)]]
        assert trunk_precision_of(pol) == "bf16"
        res["restored_equals_bf16"] = same(probe(), bf)

        # ---- the compiled interaction, on ONE trunk round at its real inputs
        import torch._dynamo as dynamo
        from agents.model.trunk_precision import Bf16BiasedEncoderLayer, Bf16IdentityInitRound
        rnd = next(m for m in pol.modules() if type(m) in (Bf16BiasedEncoderLayer, Bf16IdentityInitRound))
        cap: Dict[str, Any] = {}
        h = rnd.register_forward_pre_hook(lambda mod, args: cap.setdefault("args", tuple(
            t.detach().clone() if isinstance(t, th.Tensor) else t for t in args)))
        probe()
        h.remove()
        x, bias = (cap["args"] + (None,))[:2]
        rnd.eval()
        base_cls = bf16_to_base[type(rnd)]

        def run_compiled(fn: Any) -> th.Tensor:
            with th.no_grad():
                return fn(x, bias)

        def eager_at(cls: type) -> th.Tensor:
            was = rnd.__class__
            rnd.__class__ = cls
            try:
                with th.no_grad():
                    return rnd(x, bias)
            finally:
                rnd.__class__ = was

        out_bf16, out_fp32 = eager_at(type(rnd)), eager_at(base_cls)
        comp: Dict[str, Any] = {"eager_bf16_vs_fp32_max": float((out_bf16 - out_fp32).abs().max())}
        for mech in ("A_class_swap", "B_flag"):
            dynamo.reset()
            if mech == "B_flag":
                rnd.__class__ = flag_of[base_cls]
            else:
                rnd.__class__ = Bf16BiasedEncoderLayer if base_cls.__name__ == "BiasedEncoderLayer" else Bf16IdentityInitRound
            frames = []
            fn = th.compile(lambda xx, bb: rnd(xx, bb), backend=a.backend, fullgraph=True, dynamic=False)
            o1 = run_compiled(fn)
            frames.append(dynamo.utils.counters["stats"]["unique_graphs"])
            # (i) the scope runs only eager code, then the compiled call again
            if mech == "A_class_swap":
                rnd.__class__ = base_cls
                _ = eager_at(base_cls)
                rnd.__class__ = Bf16BiasedEncoderLayer if base_cls.__name__ == "BiasedEncoderLayer" else Bf16IdentityInitRound
            else:
                FORCE_FP32 = True
                with th.no_grad():
                    _ = rnd(x, bias)
                FORCE_FP32 = False
            o2 = run_compiled(fn)
            frames.append(dynamo.utils.counters["stats"]["unique_graphs"])
            # (ii) a compiled call INSIDE the scope
            if mech == "A_class_swap":
                rnd.__class__ = base_cls
                o3 = run_compiled(fn)
                rnd.__class__ = Bf16BiasedEncoderLayer if base_cls.__name__ == "BiasedEncoderLayer" else Bf16IdentityInitRound
            else:
                FORCE_FP32 = True
                o3 = run_compiled(fn)
                FORCE_FP32 = False
            frames.append(dynamo.utils.counters["stats"]["unique_graphs"])
            o4 = run_compiled(fn)
            frames.append(dynamo.utils.counters["stats"]["unique_graphs"])
            comp[mech] = {
                "graphs_after_first_call": frames[0],
                "graphs_after_eager_scope_then_call": frames[1],
                "graphs_after_compiled_call_inside_scope": frames[2],
                "graphs_after_return": frames[3],
                "compiled_bf16_equals_eager_bf16": bool(th.equal(o1, out_bf16)),
                "compiled_bf16_vs_eager_bf16_max": float((o1 - out_bf16).abs().max()),
                "after_eager_scope_equals_first": bool(th.equal(o2, o1)),
                "inside_scope_output_is_fp32_class": float((o3 - out_fp32).abs().max()),
                "inside_scope_output_vs_stale_bf16": float((o3 - o1).abs().max()),
                "after_return_equals_first": bool(th.equal(o4, o1)),
            }
        set_trunk_precision(pol, "bf16")
        res["compiled"] = comp
        del ref_pol
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
        print(json.dumps(res, indent=1))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
