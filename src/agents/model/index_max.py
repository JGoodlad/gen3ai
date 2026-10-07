"""THE hard-max spelling of the policy forward: a maximum's VALUE selected by INDEX (gen3_fm_index_max_v1,
F-XC-4; every value-reduction max since the X5 version break, config v144, part 2 — architecture audit F6a).

``max_by_index(x, dim)`` is ``x.amax(dim)``'s value, taken as the element at the detached ``argmax`` — a
gather, so the value is bit-identical to ``amax``. The GRADIENT is what differs: ``amax``'s backward is
``grad · (x == amax) / Σ(x == amax)``, a float EQUALITY between ``x`` and the saved max. Under
``torch.compile`` the min-cut partitioner may RECOMPUTE a wide ``x`` in the backward kernel instead of saving
it, and Triton contracts ``a·b + c`` into FMAs differently in the two kernels, so the recomputed ``x`` can miss
the saved max by an ulp on every element: the tie count is 0 and the gradient is ``0/0 = NaN`` (measured on
X5's 400-wide incoming sweep on CUDA, 2026-10-05: the compiled R1 gradient NaN on 41 parameters;
``designs/research_state/measurements/x5_fxc4_nanfix_2026-10-05/``). Selecting by index makes the backward a
scatter at the saved index — no float comparison, so no recompute can break it.

THE DECLARED CONVENTION: on an EXACT tie the whole gradient goes to the FIRST maximal element (``argmax``'s
documented tie rule), where ``amax`` splits it evenly among the tied elements. Off ties the two gradients are
equal. The ``argmax`` is K9(b)'s ``MAX_VALUE`` EXACT site (`selection_sites`): its index is read ONLY by the
gather of its own operand, so which near-tied element wins never reaches log π.

A leaf module (no model import), so every forward module — the op's mixins included — imports it without a
cycle. `damage_op` re-exports it (`damage_op.max_by_index`)."""
from __future__ import annotations

import torch


def max_by_index(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """``x.amax(dim=dim, keepdim=keepdim)``'s VALUE, selected by index (module docstring)."""
    idx = x.detach().argmax(dim=dim, keepdim=True)      # K9(b): EXACT "MAX_VALUE" (selection_sites)
    out = torch.gather(x, dim, idx)
    return out if keepdim else out.squeeze(dim)


__all__ = ["max_by_index"]
