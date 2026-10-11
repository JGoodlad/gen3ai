"""THE TRUNK'S COMPUTE PRECISION — a declared bf16 region over the trunk rounds (T25 perf lever A,
`gen3_trunk_bf16_region_v1`; `designs/research_state/measurements/perf_phase_2026-10-10/PLAN.md`).

WHAT. `set_trunk_precision(module, "bf16")` makes every TRUNK ROUND under ``module`` (the post-LN
`BiasedEncoderLayer`s and the pre-LN `IdentityInitRound`s of `--trunk-layers 3/4`) run its forward under
`torch.autocast(<device>, torch.bfloat16)` and return float32. Nothing else changes precision:

* inside a round, autocast runs the Linears and the attention (`scaled_dot_product_attention`, its additive
  edge bias included) in bf16 with fp32 accumulation, and keeps LayerNorm, the residual adds that meet an fp32
  operand, and every reduction on its fp32 list in fp32 — so the residual stream between rounds is fp32;
* everything OUTSIDE the rounds — the embeddings, the damage op and every physics edge it computes, the
  belief / intent / win-prob heads, the pointer head, the masked log-softmax, the value and every loss — runs
  in float32 exactly as before. The DECISIONS (the policy logits and softmax, the value, the losses) are fp32
  by construction: the region ends at the round's output, cast back to fp32.

WHY ONLY THE ROUNDS. On the end-state graph the trunk rounds hold ~70 % of R1's matmul FLOPs (`module_flops.py`,
CPU, 2026-10-10); a region this small is one declared boundary per round instead of an audit of every head.

HOW. A CLASS SWAP (`m.__class__` = the round's bf16 subclass), not an instance-level forward: the subclass is a
module-level class, so `copy.deepcopy`, pickling and `isinstance` behave exactly as for the base class, the
`state_dict` is unchanged, and at ``"fp32"`` (the default, and the restore) the module is again exactly its base
class — the production graph is the same code object, byte for byte.

THE RUN FLAG (`--trunk-precision {fp32,bf16}`, `gen3_trunk_precision_flag_v1`, config v155; default fp32). It sets
the LEARNER's live policy only (`apply_run_precision`, from `apply_training_hparams`, before the compile installs R1),
so the training micro-step R1 — its forward, the backward through it, every probe that differentiates R1's
outputs — runs the bf16 rounds. Everything that must agree with the ROLLOUT stays fp32 (design (b),
`designs/research_state/measurements/perf_phase_2026-10-10/PLAN.md` §2):

* T2, the eval core and every snapshot build a FRESH policy from a state dict, so they are base classes (fp32);
* the K9(b) behaviour probe's forward (`rust_rollout.consistency.probe_forward`) and the compile gate's freshness
  read run under `fp32_scope`, so K9(b) compares fp32 with fp32 — its bars and teeth are unchanged;
* the R1 startup gate and the canary judge a bf16 R1 by THE bf16 PARITY RULE (`compile_regions`'s
  `_r1_verdict_bf16`): compiled bf16 against eager bf16 AND both against the fp32 eager reference, run under
  `fp32_scope`.

What a bf16 run gives up is measured, not assumed: PPO's epoch-0 ratio reads the bf16 learner against the fp32
behaviour (|Δ log π| p50 3e-5 to 5e-3 by state, clip fraction 0, CPU: `k9_bf16_2026-10-10/`), and the gradient carries bf16's
rounding — which is why adopting it needs its own strength A/B (PLAN.md §2d).
"""
from __future__ import annotations

import contextlib
from typing import Any, Dict, Iterator, List, Optional, Tuple, Type

import torch

TRUNK_PRECISIONS: Tuple[str, ...] = ("fp32", "bf16")


def _bf16_round_class(base: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """The bf16 subclass of trunk-round class ``base``: the base forward under bf16 autocast, fp32 out."""

    def forward(self: Any, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
            out: torch.Tensor = base.forward(self, x, bias)
        return out.float()

    return type(f"Bf16{base.__name__}", (base,), {
        "forward": forward, "__module__": __name__, "__qualname__": f"Bf16{base.__name__}",
        "_gen3_trunk_base": base,
        "__doc__": f"`{base.__name__}` run under bf16 autocast, float32 out (`trunk_precision`)."})


def _round_types() -> Dict[Type[torch.nn.Module], Type[torch.nn.Module]]:
    from agents.model.team_transformer import BiasedEncoderLayer
    from agents.model.trunk_depth import IdentityInitRound
    return {BiasedEncoderLayer: Bf16BiasedEncoderLayer, IdentityInitRound: Bf16IdentityInitRound}


def _make() -> Tuple[Type[torch.nn.Module], Type[torch.nn.Module]]:
    from agents.model.team_transformer import BiasedEncoderLayer
    from agents.model.trunk_depth import IdentityInitRound
    return _bf16_round_class(BiasedEncoderLayer), _bf16_round_class(IdentityInitRound)


# Module-level (picklable, deep-copyable) bf16 round classes.
Bf16BiasedEncoderLayer, Bf16IdentityInitRound = _make()


def set_trunk_precision(module: torch.nn.Module, precision: str) -> int:
    """Set every trunk round under ``module`` to ``precision`` (`TRUNK_PRECISIONS`); returns how many rounds
    were set. ``"fp32"`` restores the base classes (the default state). A compiled region that traced the
    module before this call is stale: set the precision BEFORE `compile_regions.install`."""
    if precision not in TRUNK_PRECISIONS:
        raise ValueError(f"trunk precision must be one of {TRUNK_PRECISIONS}, got {precision!r}")
    table = _round_types()
    bf16_to_base = {v: k for k, v in table.items()}
    n = 0
    for m in module.modules():
        cls = type(m)
        base = bf16_to_base.get(cls, cls)
        if base not in table:
            continue
        m.__class__ = table[base] if precision == "bf16" else base
        n += 1
    return n


def trunk_precision_of(module: torch.nn.Module) -> str:
    """``"bf16"`` when every trunk round under ``module`` is bf16, ``"fp32"`` when none is; a MIX raises."""
    table = _round_types()
    kinds = {("bf16" if type(m) in table.values() else "fp32")
             for m in module.modules() if type(m) in table or type(m) in table.values()}
    if len(kinds) > 1:
        raise RuntimeError("the trunk rounds are at MIXED precision — set_trunk_precision sets them all")
    return kinds.pop() if kinds else "fp32"


def apply_run_precision(model: Any, precision: Optional[str]) -> int:
    """The run flag's one application (`apply_training_hparams`): record ``model.trunk_precision`` and set the
    live policy's rounds. ``None`` (a namespace without the dest: the learner golden, a test) is fp32. Must run
    before `compile_regions.install` (the compiled R1 traces the classes it sees)."""
    prec = str(precision or "fp32")
    if prec not in TRUNK_PRECISIONS:
        raise ValueError(f"--trunk-precision must be one of {TRUNK_PRECISIONS}, got {prec!r}")
    model.trunk_precision = prec
    return set_trunk_precision(model.policy, prec)


@contextlib.contextmanager
def fp32_scope(module: torch.nn.Module) -> Iterator[None]:
    """Run the block with every trunk round under ``module`` at its fp32 BASE class, then put back EXACTLY the
    classes that were there (also on an exception). A no-op on an all-fp32 module. For the forwards that must
    match an fp32 counterpart — the K9(b) probe (T2 serves fp32), the bf16 parity rule's fp32 reference arm.

    Never call a compiled region inside it: the compiled R1 traced the bf16 classes, and a call under the base
    classes would be a new signature (after the lock, a FATAL). An EAGER call is safe: a class swap neither
    touches the dynamo cache nor any parameter, and the restore makes the compiled route's guards hold again."""
    table = _round_types()
    bf16_to_base = {v: k for k, v in table.items()}
    swapped: List[Tuple[torch.nn.Module, Type[torch.nn.Module]]] = []
    try:
        for m in module.modules():
            cls = type(m)
            if cls in bf16_to_base:
                swapped.append((m, cls))
                m.__class__ = bf16_to_base[cls]
        yield
    finally:
        for m, cls in swapped:
            m.__class__ = cls
