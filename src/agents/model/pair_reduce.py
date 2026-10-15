"""The CONTRACT-W reduction primitives over the opponent's believed-move axis (`design_pair_reduction.md` §3.1).

  Contract W — a weighted reducer emits ``α ∈ Δ^C``: ONE distribution over the believed-move axis, computed from the
  board and their moves ONLY — never per defender, never per channel. The reduced row is
  ``Σ_c α_c · cell[j, c, :]`` — a convex combination of HP fractions, which is an HP fraction (units survive).

What survives: the R1 rung ``alpha_belief_mean`` (α = w/Σw; read by `rust_sim/harness/g1_bakeoff.py`) and Contract
W's one line ``reduce_with_alpha``. The swappable reducer RUNGS (`PairReducer`, `build_pair_reducer`,
`pair_reduce_extra_dim`: R2 learned / DeepSets / multi) and the R0 contract-form helper were deleted with
`--value-threat-inject`, their only builder (legacy deletion U3): the op's ONE believed reduction is
`op_reduction.believed_reduce`.
"""
from __future__ import annotations

import torch

PAIR_REDUCE_EPS = 1e-8


def alpha_belief_mean(w: torch.Tensor) -> torch.Tensor:
    """R1 — the honest marginal: α = w/Σw. [B,C] → [B,C]; all-zero belief rows stay all-zero
    (no threat ⇒ no usage mass ⇒ reduced row 0, matching the legacy no-threat gate)."""
    total = w.sum(dim=-1, keepdim=True)
    return torch.where(total > PAIR_REDUCE_EPS, w / total.clamp_min(PAIR_REDUCE_EPS),
                       torch.zeros_like(w))


def reduce_with_alpha(alpha: torch.Tensor, cells: torch.Tensor) -> torch.Tensor:
    """Contract W's one line: ``Σ_c α_c · cell`` — [B,C] × [B,J,C,F] → [B,J,F]. α is
    defender-independent by SIGNATURE (no J axis exists on it): D3 is a shape error here."""
    return torch.einsum("bc,bjcf->bjf", alpha, cells)
