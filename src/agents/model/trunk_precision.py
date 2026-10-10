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

STATUS: a MEASUREMENT lever (no trainer flag). Adopting it needs (1) R1's parity rule for a bf16 region (compiled
bf16 vs eager bf16 differs by bf16 rounding, far above the fp32 bar; the matched-envelope rule against an fp64
reference is the candidate), (2) the K9(b) behaviour check's bar (the rollout's T2 forward is fp32, so epoch 0's
ratio is no longer 1 to 1e-6), and (3) its own strength A/B. PLAN.md carries the design.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Type

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
