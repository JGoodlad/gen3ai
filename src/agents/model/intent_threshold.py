"""gen3_intent_threshold_v1 — the α-weighted THRESHOLD operator (`p_thresh`), five mechanics at once.

`design_conditional_execution.md` §3.0 / build-order step 3. Five gen3 mechanics are the SAME
computation with a different threshold and comparison direction:

    p_thresh(τ, ⋛) = Σ_k α_k · 1[ damage(k, me) ⋛ τ ]

| mechanic       | τ            | direction | reading                                  |
|----------------|--------------|-----------|------------------------------------------|
| Focus Punch    | 0            | `>`       | any damage breaks it                     |
| Substitute     | 25% maxhp    | `<`       | the sub survives                         |
| Endure         | current HP   | `≥`       | I would otherwise die                    |
| Destiny Bond   | current HP   | `≥`       | same threshold, OPPOSITE valence         |
| Endeavor       | current HP   | `<`       | I survive to act                         |

The `τ = current HP` case is **`p_KO`** — the α-weighted P(I die this turn) — and it is exactly the
quantity ledger **H1**'s self-KO defect mis-values: the op has always computed a per-candidate
`ko_ramp`, but `_chan_max` collapses it to a hard max, so the critic priced "am I about to die"
from the single worst believed move instead of from the opponent's action DISTRIBUTION. α turns
that max into a calibrated probability. Same tensor, correct functional — no new physics: every
input here is a column of the op's `last_pair_cells` stash ([low, high, crit, ko, acc, is_phys]
per (defender, seat candidate), already oracle-gated by `damage_op_probe_fuzz_test`), gathered at
OUR ACTIVE. Focus Punch's immunity term falls out of the physics for free: an immune candidate has
eff 0 ⇒ high 0 ⇒ it does not break the punch.

## The threshold forms

* `p_ko`         = Σ_k α_k · ko_k            (ko_k is the op's accuracy-discounted modal P(KO))
* `p_fp_broken`  = Σ_k α_k · acc_k · 1[high_k > 0]     (a landing damaging move breaks the punch)
* `p_sub_broken` = Σ_k α_k · acc_k · ramp(high_k ≥ 25% maxhp)   (the op's own 15%-window KO ramp
                    shape, re-thresholded at the sub's HP — a threshold on the roll distribution
                    is not a function of its mean, which is why `max` could never produce it)

All three use the **UNRENORMALIZED** move slice of α (the `IntentValueReduce` precedent): the
missing SWITCH mass is the literally-correct statement that a switching opponent deals no damage
this turn, so a high α_SWITCH must shrink every threshold probability toward zero.

## Delivery — two consumers, two tiers, one producer

`threshold_probs` (pure, parameter-free) runs ONCE at the pointer stash (T2, where α first
exists); its outputs are consumed by BOTH heads:

* **`IntentThresholdMoveCell`** (T2) — per-request-slot channels through the pointer MOVE cell
  (the per-action-absolute channel measured to work: `d1` 12.17% / `d2` 19.25%), gated by which
  of the five mechanics the slot IS: `[is_fp·(1−p_fp_broken), is_sub·(1−p_sub_broken),
  is_endure·p_ko, is_dbond·p_ko, is_endeavor·(1−p_ko), p_ko]` — the probability and the mechanic
  gate are provided as facts (§1's convention); the head forms the products it needs. The last
  channel is the decorrelated `p_ko` context on every slot.
* **`IntentThresholdValue`** (T3) — `[p_ko, p_sub_broken, p_fp_broken]` appended to the CRITIC
  after the assembler (the `intent_value_reduce` placement). This half stands on its own whatever
  the G3 verdict says about the mechanic cells: H1 measured the critic over-valuing a healthy
  self-KO trade (dV ≈ +2.9 vs a −2.7 reward) because "am I about to die" reached it only as a max.

Both projections are ZERO-INIT (`restore_identity_init` captures them by observation — ledger M1),
so ON-at-init is bit-identical on both heads and any measured effect on a trained run is learned.
Seat-permutation invariant by construction: the only seat-indexed computation is `Σ_k α_k · f_k`.
"""
from __future__ import annotations

from typing import NamedTuple

import torch

from agents.model.arch_constants import (
    _INTENT_THRESH_RAW_MOVE, _INTENT_THRESH_RAW_VF,
)

# The five mechanics' move NUMS (gen3_data.moves, read 2026-08-16): focuspunch 264,
# substitute 164, endure 203, destinybond 194, endeavor 283. Resolved from the data facade at
# module import so a num drift is a loud failure here, not a silent gate that never fires.
def _mech_nums() -> torch.Tensor:
    from agents import gen3_data
    nums = []
    for mid in ("focuspunch", "substitute", "endure", "destinybond", "endeavor"):
        md = gen3_data.moves.get(mid)
        if md is None:
            raise ValueError(f"intent_threshold: move {mid!r} missing from gen3_data.moves — "
                             "the mechanic gate would silently never fire.")
        nums.append(int(md.num))
    return torch.tensor(nums, dtype=torch.long)


class ThresholdProbs(NamedTuple):
    """The α-contracted threshold probabilities, each [B, 1] — a typed tuple so the T2→T3
    stash (`_thresh_probs`) names its fields instead of relying on positional convention
    (`gen3_op_stashes_v1`, the extractor half). Still a tuple: `*probs` unpacking is unchanged."""
    p_ko: torch.Tensor
    p_sub_broken: torch.Tensor
    p_fp_broken: torch.Tensor


_SUB_HP_FRAC = 0.25          # gen3 Substitute HP = maxhp/4; `high` is a fraction of maxhp
_KO_RAMP_WINDOW = 0.15       # the op's own modal-roll KO-ramp window (see damage_op._rolls)


def threshold_probs(alpha_logits: torch.Tensor, pair_cells: torch.Tensor,
                    pair_gate: torch.Tensor, our_active_idx: torch.Tensor,
                    ) -> ThresholdProbs:
    """`(published α logits [B,K+1], op pair cells [B,6,K,6], gate [B,6,1], our_active [B])`
    → `(p_ko, p_sub_broken, p_fp_broken)`, each `[B, 1]`. Pure — no parameters, no state.

    Fails loud when K != the cell candidate width: α's seats and the op's top-K are the SAME axis
    (`intent_axis_alignment_test`), so a mismatch means one was reconfigured independently —
    silently broadcasting would pair each α weight with the wrong opponent move while every shape
    check still passed (the named `op move-order` bug class).
    """
    k = alpha_logits.shape[-1] - 1                                     # last class is SWITCH
    if pair_cells.shape[2] != k:
        raise ValueError(
            f"alpha has {k} move seats but the op stashed {pair_cells.shape[2]} candidate "
            "channels. These must be the SAME axis (entity_topk_seats == damage_topk_k); a "
            "mismatch would pair each alpha weight with the wrong opponent move while every "
            "shape check still passed.")
    B = pair_cells.shape[0]
    ar = torch.arange(B, device=pair_cells.device)
    cells = pair_cells[ar, our_active_idx]                             # [B,K,6] our ACTIVE's row
    gate = pair_gate[ar, our_active_idx]                               # [B,1] alive · has_opp
    high_k = cells[..., 1]                                             # [B,K] max-roll frac of maxhp
    ko_k = cells[..., 3]                                               # [B,K] acc-discounted P(KO)
    acc_k = cells[..., 4]                                              # [B,K]
    break_fp_k = acc_k * (high_k > 0).float()                          # any landing damage breaks it
    # The op's own ramp shape (clamp((dmg − τ)/(0.15·dmg), 0, 1)), re-thresholded at the sub's HP.
    break_sub_k = acc_k * torch.clamp(
        (high_k - _SUB_HP_FRAC) / (_KO_RAMP_WINDOW * high_k + 1e-6), 0.0, 1.0)
    # UNRENORMALIZED move slice — the missing SWITCH mass correctly reads "no damage this turn".
    alpha = torch.softmax(alpha_logits.float(), dim=-1)[:, :k].to(pair_cells.dtype)   # [B,K]
    p_ko = (alpha * ko_k).sum(dim=-1, keepdim=True) * gate             # [B,1]
    p_sub = (alpha * break_sub_k).sum(dim=-1, keepdim=True) * gate
    p_fp = (alpha * break_fp_k).sum(dim=-1, keepdim=True) * gate
    return ThresholdProbs(p_ko=p_ko, p_sub_broken=p_sub, p_fp_broken=p_fp)


class IntentThresholdMoveCell(torch.nn.Module):
    """`(threshold probs, our request-move ids) → the extra pointer-move-cell block [B,4,out]`.

    Zero-init projection ⇒ ON-at-init contributes exactly zero to every action logit."""

    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = int(out_dim)
        self.proj = torch.nn.Linear(_INTENT_THRESH_RAW_MOVE, self.out_dim)
        torch.nn.init.zeros_(self.proj.weight)
        torch.nn.init.zeros_(self.proj.bias)
        self.register_buffer("mech_nums", _mech_nums(), persistent=False)   # [5]

    def forward(self, p_ko: torch.Tensor, p_sub_broken: torch.Tensor,
                p_fp_broken: torch.Tensor, req_move_ids: torch.Tensor) -> torch.Tensor:
        """probs each [B,1] · `req_move_ids` [B,4] (move NUMS, 0 = empty slot) → [B,4,out]."""
        gates = (req_move_ids[..., None] == self.mech_nums).float()        # [B,4,5]
        is_fp, is_sub, is_endure, is_dbond, is_endv = gates.unbind(-1)     # each [B,4]
        raw = torch.stack([
            is_fp * (1.0 - p_fp_broken),         # P(the punch executes)
            is_sub * (1.0 - p_sub_broken),       # P(the sub survives their hit)
            is_endure * p_ko,                    # Endure pays only on the branch where I die
            is_dbond * p_ko,                     # same threshold, opposite valence — head decides
            is_endv * (1.0 - p_ko),              # P(I survive to act) — rises as the danger falls
            p_ko.expand_as(is_fp),               # decorrelated context on every slot
        ], dim=-1)                                                          # [B,4,6]
        return self.proj(raw)                                               # [B,4,out]


class IntentThresholdValue(torch.nn.Module):
    """`threshold probs → a vf-only additive block [B, out]`. Zero-init ⇒ OFF-at-init exact.

    The `p_KO` critic route — independently useful whatever the G3 verdict (ledger H1)."""

    def __init__(self, out_dim: int):
        super().__init__()
        self.out_dim = int(out_dim)
        self.proj = torch.nn.Linear(_INTENT_THRESH_RAW_VF, self.out_dim)
        torch.nn.init.zeros_(self.proj.weight)
        torch.nn.init.zeros_(self.proj.bias)

    def forward(self, p_ko: torch.Tensor, p_sub_broken: torch.Tensor,
                p_fp_broken: torch.Tensor) -> torch.Tensor:
        return self.proj(torch.cat([p_ko, p_sub_broken, p_fp_broken], dim=-1))   # [B,out]
