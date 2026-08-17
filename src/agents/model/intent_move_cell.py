"""G3 — the POLICY-side α consumer: c2 re-delivered through the pointer MOVE cell.

`gen3_intent_move_cell_v1` (`design_conditional_execution.md` §6 G3). The consequence families
(`c1`–`c5`, `x`) all read at the noise floor, and the doc's §0 hypothesis says why: they were routed
through an attention BIAS (a softmax-normalised ratio) instead of a per-action CELL (an absolute),
and they conditioned on the wrong distribution (presence `w` + hard max, never usage `α`). G3 is
the discriminating single-family arm: re-deliver ONE family — `c2`, the least-dead at 1.20% — with
both fixes at once. If it comes alive on a trained run, both halves of the hypothesis are confirmed
and the remaining mechanics (§3) are worth building; if it stays at zero, the consequence line is
dead for real.

## The derived per-move conditional (the doc's §1 operator applied to c2)

`c2` (`DamageOperator.pairwise_status_consequence`) prices "what LANDING our status move DOES" per
(our request slot m, opp mon j). The move cell is per-ACTION, so the opp-mon axis must resolve to
the mon our status move would actually hit: their ACTIVE when they stay. Per §1,
``E[my move m] = Σ_k α_k · f(m, k)`` with `f` a deterministic rules lookup:

* the channels whose value depends on WHICH move they click — burn's damage-control delta and
  sleep's suspended threat, both a hard `w·amax` over believed candidates in the edge cells —
  become the exact α-expectation over the SAME seat candidates α is a distribution over:
  ``e_burn_alpha = Σ_k α_k · (high_k(atk×0.5) − high_k(atk))`` (≤ 0) and
  ``e_slp_alpha = Σ_k α_k · (−high_k)``. Both use the **UNRENORMALIZED** move slice — the
  `IntentValueReduce` precedent: `f(m, SWITCH) = 0` is literally correct here (a switching active
  neither attacks us nor stands still to be statused), so a high α_SWITCH must shrink the term
  toward zero and renormalizing would assert they attacked;
* the k-independent consequences vs their active (paralysis Δ-outspeed, the residual tick, sleep's
  expected free turns) ride RAW, with the seat mass ``alpha_stay = Σ_k α_k`` emitted as its own
  decorrelated channel — §1's provide-the-facts convention (emit the probability and the
  consequence separately; the head forms the product). Their value given SWITCH is NOT zero (the
  status hits the β-weighted arrival), but pricing that branch needs `β`, which G3 deliberately
  does not consume — β is reserved for the class-B mechanics at build-order step 7;
* `land` is deliberately NOT duplicated: the move cell already carries the richer unified
  `_status_landing` p_land/known pair vs the same recipient.

Contract W holds: ONE α (softmaxed once from the published logits), computed from the board alone,
shared across every channel; no per-defender and no per-our-move dependence ever enters α itself.

## Delivery and alignment

The raw stack goes through a ZERO-INIT `Linear(_INTENT_MOVE_CELL_RAW, INTENT_MOVE_CELL_DIM)` and is
concatenated onto the existing 13-dim move cell, so at init the appended features are exactly 0 and
every action logit is untouched (`restore_identity_init` captures the projection by observation —
ledger M1). Alignment with α's seats is by construction — the operands are computed on the op's own
`last_topk_idx` candidate selection, the SAME axis `intent_axis_alignment_test` pins to α's seats —
and a width mismatch is a LOUD ValueError (the named `op move-order` bug class), never a broadcast.
"""
from __future__ import annotations

import torch

from agents.model.arch_constants import _INTENT_MOVE_CELL_RAW


class IntentMoveCell(torch.nn.Module):
    """`(published α logits, c2 operands) → the extra pointer-move-cell block [B, 4, out_dim]`.

    Zero-init projection, so ON-at-init contributes exactly zero to every logit. Seat-permutation
    invariant by construction: the only seat-indexed computation is `Σ_k α_k · f_k`, and a joint
    permutation of (α's seat logits, the per-candidate operand columns) leaves the sum unchanged.
    """

    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = int(out_dim)
        self.proj = torch.nn.Linear(_INTENT_MOVE_CELL_RAW, self.out_dim)
        # Zero-init: the pointer logits are bit-identical at step 0, so a measured effect on a
        # trained run is something the run LEARNED. `restore_identity_init` picks this up by
        # observation (ledger M1 — SB3's ortho pass would otherwise clobber it on policy build).
        torch.nn.init.zeros_(self.proj.weight)
        torch.nn.init.zeros_(self.proj.bias)

    def forward(self, alpha_logits: torch.Tensor, base: torch.Tensor,
                d_burn_k: torch.Tensor, d_slp_k: torch.Tensor,
                is_brn: torch.Tensor, is_slp: torch.Tensor) -> torch.Tensor:
        """`alpha_logits` [B,K+1] (last class = SWITCH) · `base` [B,4,4]
        ([is_status, d_their_outspeed, d_sched, e_slp_free] vs their active) · `d_burn_k`/`d_slp_k`
        [B,K] (per-seat-candidate consequence columns) · `is_brn`/`is_slp` [B,4] → [B,4,out_dim].

        Fails loud when K != the operand candidate width: α's seats and the op's top-K are the SAME
        axis (`intent_axis_alignment_test`), so a mismatch means one was reconfigured independently
        — silently broadcasting there would mis-weight every term while every shape check passed.
        """
        k = alpha_logits.shape[-1] - 1                       # last class is SWITCH
        if d_burn_k.shape[-1] != k or d_slp_k.shape[-1] != k:
            raise ValueError(
                f"alpha has {k} move seats but the c2 operands carry "
                f"{d_burn_k.shape[-1]}/{d_slp_k.shape[-1]} candidate columns. These must be the "
                "SAME axis (entity_topk_seats == damage_topk_k); a mismatch would pair each alpha "
                "weight with the wrong opponent move while every shape check still passed.")
        # UNRENORMALIZED move slice — see the module docstring: the missing SWITCH mass is the
        # correct statement that a switching opponent neither attacks nor receives the status.
        alpha = torch.softmax(alpha_logits.float(), dim=-1)[:, :k].to(base.dtype)     # [B,K]
        e_burn = (alpha * d_burn_k).sum(dim=-1, keepdim=True) * is_brn                # [B,4]
        e_slp = (alpha * d_slp_k).sum(dim=-1, keepdim=True) * is_slp                  # [B,4]
        # The decorrelated stay-mass channel, masked to status slots so a damage move's extra cell
        # row stays exactly zero (raw-channel zero; the shared zero-init bias is per-CHANNEL and
        # identical across slots either way).
        alpha_stay = alpha.sum(dim=-1, keepdim=True) * base[..., 0]                   # [B,4]
        cells = torch.stack([base[..., 0], base[..., 1], e_burn, base[..., 2],
                             e_slp, base[..., 3], alpha_stay], dim=-1)                # [B,4,7]
        return self.proj(cells)  # type: ignore[no-any-return]  # [B,4,out]
