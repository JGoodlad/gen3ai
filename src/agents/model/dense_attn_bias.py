"""`dense_attn_bias`: pin an SDPA attention bias to a DENSE (row-major) layout under Inductor (`gen3_dense_attn_bias_v1`).

**The defect it closes (F-ST-8, 2026-10-07).** CUDA's memory-efficient SDPA kernel requires the float
attention bias to have a dense LAST dim (`stride(-1) == 1`, "(*bias): last dimension must be contiguous").
Eager hands it one: the trunk bias is built `.contiguous()` and every `EdgeBias` write is an in-place slice
assignment. Under Inductor those writes are FUNCTIONALIZED into pointwise `add + copy` kernels whose output
layout Inductor is free to choose — and it chooses the layout of the dominant input, the head-innermost
`m.permute(0, 3, 1, 2)` of each family's `[B, R, C, 2H]` map output: strides `(n·n·H, 1, n·H, H)`.

Inductor's own SDPA stride constraint (`torch/_inductor/lowering.py::sdpa_constraint`, torch 2.8) is meant to
catch that. It first asks `is_aligned_realized_tensor(bias, 8)`; for a buffer whose layout is still FLEXIBLE
that check reads the provisional row-major strides, answers "aligned" whenever the key count `n` is a multiple
of 8, and passes the buffer through WITHOUT freezing its layout. The layout is decided later — head-innermost
— and the first layer's kernel receives it. When `n` is NOT a multiple of 8 the check fails and the
constraint emits a padded dense copy, which is why the legacy trunk (`n` = 62 under production `fixed_mass`)
never saw it and `--token-encoding static` (two more board tokens: `n` = 64) died in the T2 service's first
CUDA-graph build. The CPU `--debug` smokes run T2's EAGER backend and cannot see it, and a CPU compile takes
the constraint's `require_stride_order` branch, so the defect is CUDA × Inductor only.

**The fix.** Under compilation, the bias goes through Inductor's `inductor_force_stride_order` prim with
row-major strides: its lowering is `require_stride_order`, which FREEZES a flexible buffer row-major (no copy)
or copies a buffer already fixed in another order — so every consumer, SDPA included, reads a dense last dim
whatever `n` is. Values are unchanged (a layout is not a value). In EAGER it is `Tensor.contiguous()`, a
no-op on the trunk's already-contiguous bias — eager is byte-identical and costs nothing.

The prim ships with no autograd formula (Inductor only inserts it after AOT autograd), and the trunk bias
carries gradient (the edge-family maps train) — so this module registers the identity backward on it, once,
at import: the forward is a value-identity, its gradient is the incoming gradient unchanged. (An identity
`autograd.Function` around the prim does NOT suffice: AOT autograd traces through it into the prim.)"""
from __future__ import annotations

from typing import Any, List, Sequence

import torch


def _row_major_strides(shape: Sequence[Any]) -> List[Any]:
    strides: List[Any] = []
    acc: Any = 1
    for s in reversed(list(shape)):
        strides.append(acc)
        acc = acc * s
    return list(reversed(strides))


def _register_force_op() -> Any:
    """`prims.inductor_force_stride_order` with its identity backward — registered at IMPORT (a registration
    inside a dynamo trace is refused), once per process (`torch.library` refuses a second)."""
    import torch._inductor.inductor_prims  # noqa: F401  (registers prims.inductor_force_stride_order)

    op = torch.ops.prims.inductor_force_stride_order.default
    if not getattr(op, "_gen3_identity_backward", False):
        def _backward(ctx: Any, grad: torch.Tensor) -> Any:
            return grad, None

        torch.library.register_autograd("prims::inductor_force_stride_order", _backward)
        setattr(op, "_gen3_identity_backward", True)   # once per process: torch.library refuses a second
    return op


_FORCE_OP = _register_force_op()


def dense_attn_bias(bias: torch.Tensor) -> torch.Tensor:
    """`bias` with a guaranteed row-major (dense last dim) layout — pinned under `torch.compile` (see the
    module docstring), `.contiguous()` in eager. Call it on every float attention bias right before SDPA."""
    if torch.compiler.is_compiling():
        return _FORCE_OP(bias, _row_major_strides(bias.shape))  # type: ignore[no-any-return]
    return bias.contiguous()
