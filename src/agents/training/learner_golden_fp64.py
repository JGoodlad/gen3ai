"""The K9 learner golden's fp64 REFERENCE (X5 U6; `designs/endstate/design_x5_belief_tokens.md` §6.3).

WHY. The golden pins what one fp32 update computes, EXACTLY — so it proves an update UNCHANGED, never
CORRECT. A re-bake is only trustworthy if the fp32 numbers it banks are also close to what the same
arithmetic gives at fp64: a precision bug (a catastrophic cancellation, an fp32-only overflow guard, a
mask written as a large finite number that rounds into a live value) moves fp32 away from fp64 by far more
than rounding, while a correct fp32 forward sits within rounding of it.

WHAT. At the golden's seeded INITIAL weights, on its committed buffer (intent labels aligned exactly as
`train()` aligns them), ONE learner micro-step over the rows — `micro_step.micro_step`, the very function
`train()` runs as region R1, so every loss term of the fold's steps 1–3a — computed twice:

* fp32: the learner as built;
* fp64: a deep copy cast to float64, run under `Fp64Mode` (every float32 tensor reaching a torch op is
  promoted to float64 and ``.float()`` means ``.double()``, so constants the forward builds in fp32 —
  ``obs.float()``, ``torch.tensor(..., dtype=float32)`` — are fp64 too).

and compared: every loss TERM (`TERM_ATOL` + `TERM_RTOL`·|t64|) and the gradient of each `KEY_GRADS`
parameter (relative L2 error ≤ `GRAD_RTOL`).

RULE 8 (deterministic). The forward SELECTS (the declared discrete sites, `selection_sites.py`); a row
whose smallest relative tie margin is under `consistency.FP32_TIE_EPS` could resolve a selection
differently at fp32 and fp64, so it is EXCLUDED from both computations (the margins come from the fp32
forward under `tie_margins.TieMargins`, the K9(b) recorder); the excluded count is recorded.

NOT independent code: the same source at two precisions. The independent fp64 numpy references of the
X5 construction (Σπ = k, the root-find, the bias) are `learner_golden_fixed_mass_test`'s own.
"""
from __future__ import annotations

import contextlib
import copy
from pathlib import Path
from typing import Any, Dict, Iterator, Tuple

import numpy as np
import torch as th
from torch.overrides import TorchFunctionMode

#: The parameters whose gradient the reference compares (fixed_mass's X5 groups + the shared heads the
#: loss reaches first). A name the learner does not have is skipped and recorded as absent.
KEY_GRADS: Tuple[str, ...] = (
    "features_extractor.hypothesis_builder.delta_out.weight",     # δ_θ (the presence BCE only)
    "features_extractor.hypothesis_builder.other_map.weight",     # OTHER's token
    "features_extractor.hypothesis_builder.hypothesis_marker",    # the hypothesis seats' marker
    "features_extractor.flat_intent_head.hidden.weight",          # the flat opponent pointer
    "features_extractor.belief_head.species_head.weight",         # the set BCE's per-slot logits
    "pointer_head.move_proj.weight",                              # the policy
    "features_extractor.win_head.net.3.weight",                   # the win-prob critic
)

#: DECLARED tolerances: |t32 - t64| <= TERM_ATOL + TERM_RTOL * |t64| per term, relative L2 <= GRAD_RTOL per
#: gradient. Each is >= 10x the largest fp32-vs-fp64 distance MEASURED on the fixed_mass golden, rounded up
#: to 1-2-5 (2026-10-04, torch 2.8.0+cu126, CPU, 1 thread; the recorded entry: 62 judged rows, 2 excluded):
#: terms <= 3.3e-7 absolute (the total loss) and <= 1.3e-6 relative on every term not at its null value (the
#: clipped surrogate is ~4e-9 at ratio 1 — the absolute bar governs it); gradients <= 1.5e-6 relative. (Trials
#: on the seed-17 arm buffer and the blob buffer read <= 4.1e-7 / 2.2e-6.)
TERM_ATOL = 5e-6
TERM_RTOL = 1e-5
GRAD_RTOL = 5e-5

_F32, _F64 = th.float32, th.float64


def _up(x: Any) -> Any:
    if isinstance(x, th.Tensor) and x.dtype == _F32:
        return x.double()
    if isinstance(x, (list, tuple)):
        return type(x)(_up(v) for v in x)
    if isinstance(x, dict):
        return {k: _up(v) for k, v in x.items()}
    if x is _F32:
        return _F64
    return x


class Fp64Mode(TorchFunctionMode):
    """Promote every float32 operand of every torch op to float64 (and ``dtype=float32`` to float64)."""

    def __torch_function__(self, func: Any, types: Any, args: Tuple[Any, ...] = (), kwargs: Any = None) -> Any:
        kwargs = kwargs or {}
        if func is th.Tensor.float:
            return args[0].double()
        return func(*_up(args), **_up(kwargs))


@contextlib.contextmanager
def _default_dtype(dt: th.dtype) -> Iterator[None]:
    """Set torch's default dtype for the block and RESTORE it (a leaked default dtype fails every later
    test — `utils.torch_state_guard`)."""
    prev = th.get_default_dtype()
    th.set_default_dtype(dt)
    try:
        yield
    finally:
        th.set_default_dtype(prev)


def _rows(model: Any, buffer: Path) -> Dict[str, Any]:
    """The buffer's rows as the learner reads them: loaded, intent labels aligned, flattened to [n]."""
    from agents.training import learner_golden as L

    L.load_buffer_into(model, buffer)
    model._align_opp_intent_labels()
    rb = model.rollout_buffer
    n = L.N_STEPS * L.N_ENVS

    def flat(a: np.ndarray) -> th.Tensor:
        return th.as_tensor(np.ascontiguousarray(np.asarray(a).reshape(n, *np.asarray(a).shape[2:])))

    return {"obs": {k: flat(v) for k, v in rb.observations.items()},
            "actions": flat(rb.actions).reshape(n, -1), "masks": flat(rb.action_masks).reshape(n, -1),
            "old_log_prob": flat(rb.log_probs).reshape(-1), "old_values": flat(rb.values).reshape(-1),
            "advantages": flat(rb.advantages).reshape(-1), "returns": flat(rb.returns).reshape(-1)}


def judged_rows(policy: Any, rows: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """``(judged [n] bool, margin [n])``: rule 8 — rows at a selection near-tie are excluded."""
    from agents.training.rust_rollout.consistency import FP32_TIE_EPS
    from agents.training.rust_rollout.tie_margins import selection_gaps

    margin, _site = selection_gaps(policy, rows["obs"], rows["actions"], rows["masks"].bool(), "cpu")
    return margin >= FP32_TIE_EPS, margin


def _step(model: Any, policy: Any, rows: Dict[str, Any], keep: th.Tensor) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
    from agents.training.instrumented_ppo.micro_step import micro_step

    st = model._micro_static(model._resolve_fold_flags())
    sel = {k: v[keep] for k, v in rows["obs"].items()}
    policy.set_training_mode(True)
    policy.zero_grad(set_to_none=True)
    out = micro_step(policy, sel, rows["actions"][keep], rows["masks"][keep].bool(), rows["old_log_prob"][keep],
                     rows["old_values"][keep], rows["advantages"][keep], rows["returns"][keep], st)
    out.loss.backward()
    terms = {k: float(v.detach()) for k, v in out.terms.items() if out.present.get(k, th.ones(())).item()}
    terms["loss"] = float(out.loss.detach())
    named = dict(policy.named_parameters())
    grads = {n: named[n].grad.detach().double().reshape(-1).numpy().copy()
             for n in KEY_GRADS if n in named and named[n].grad is not None}
    return terms, grads


def reference(model: Any, buffer: Path) -> Dict[str, Any]:
    """The fp32 and fp64 micro-step on ``model``'s (initial) weights over ``buffer``'s judged rows, the
    distances between them and the declared tolerances (module docs). ``model`` is consumed."""
    from agents.training import learner_golden as L

    with L._one_thread():
        rows = _rows(model, buffer)
        judged, margin = judged_rows(model.policy, rows)
        keep = th.as_tensor(judged)
        pol64 = copy.deepcopy(model.policy).double()   # BEFORE the fp32 step (its stash holds graph tensors)
        t32, g32 = _step(model, model.policy, rows, keep)
        with _default_dtype(th.float64), Fp64Mode():
            t64, g64 = _step(model, pol64, rows, keep)
    term_err = {k: abs(t32[k] - t64[k]) for k in t64 if k in t32}
    grad = {n: {"norm_fp64": float(np.linalg.norm(g64[n])),
                "rel_err": float(np.linalg.norm(g32[n] - g64[n]) / max(np.linalg.norm(g64[n]), 1e-300))}
            for n in g64 if n in g32}
    return {"rows_judged": int(judged.sum()), "rows_excluded": int((~judged).sum()),
            "min_margin": float(np.min(margin)),
            "terms_fp32": t32, "terms_fp64": t64, "term_abs_err": term_err,
            "grads": grad, "grads_absent": [n for n in KEY_GRADS if n not in grad],
            "tolerances": {"term_atol": TERM_ATOL, "term_rtol": TERM_RTOL, "grad_rtol": GRAD_RTOL}}


def violations(ref: Dict[str, Any]) -> list:
    """Every term / gradient outside the declared tolerances (empty = the fp32 golden agrees with fp64)."""
    out = []
    for k, v64 in ref["terms_fp64"].items():
        v32 = ref["terms_fp32"].get(k)
        if v32 is None or not (abs(v32 - v64) <= TERM_ATOL + TERM_RTOL * abs(v64)):
            out.append(f"term {k}: fp32 {v32!r} vs fp64 {v64!r}")
    for n, g in ref["grads"].items():
        if not (g["rel_err"] <= GRAD_RTOL):
            out.append(f"grad {n}: relative L2 error {g['rel_err']:.3g} > {GRAD_RTOL}")
    return out
