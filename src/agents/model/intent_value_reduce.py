"""Step 6 — the opponent-intent belief finally CONSUMED, not merely measured.

`gen3_intent_value_reduce_v1`. Until now `α` was a supervised readout: it predicted what the
opponent would click, that prediction was scored, and then nothing downstream used it. This routes
it into the critic.

## Why it is here and not where the design said

`design_tiered_belief.md` §4 specifies step 6 as *"α as the reduction's `how=` at the single
`_chan_max` call site"* and calls it "an A/B at THIS call site with no new plumbing". Two things
turned out to be false, and both are why this module exists instead:

1. **α cannot reach the operator.** The phase chain is `DamageOperator → EntityMoveSeats →
   TeamTransformer → CLSPool → α`. `α` is scored FROM the E4 seat tokens and the pooled board
   context, both computed *downstream of the op*, so it does not exist when the op reduces. That is
   a genuine cycle, not a plumbing gap — the same T2-object/T1-consumer shape as the species
   posterior (`t0_species.py`), and the tier contract is what makes it visible rather than subtle.
2. **`_chan_max` is not the single reduction site.** It has three call sites; the operator performs
   a dozen-plus raw `amax(dim=-1)` collapses of the believed-move axis inline. α-ifying only the
   three would leave the block *more* incoherent, which is precisely the defect
   `design_pair_reduction.md` §2 diagnoses ("nine independent maxima, so up to nine different
   opponent moves describe one defender").

So the operator is left **completely untouched** — its internals stay hard-max and byte-identical —
and ONE coherent α-weighted route is added beside them, over the same un-reduced cells the
pair-reducer already consumes. One distribution, shared across every channel, which is the property
the pair-reduction design actually wants and the existing block does not have.

## The marginalization is exact, and the SWITCH mass is why

`α ∈ Δ^(K+1)` over K believed move seats plus SWITCH. The expected incoming threat is

    E[threat_j] = Σ_k α_k · cell[j, k] + α_SWITCH · 0 = Σ_k α_k · cell[j, k]

so the move-seat slice is used **UNRENORMALIZED**. That is not a shortcut — renormalizing would be
the bug. If the opponent switches they deal no damage this turn, and the un-normalized slice carries
exactly that: a high `α_SWITCH` correctly shrinks the whole expected-threat row toward zero. Divide
by `Σ_k α_k` and you assert they attacked, discarding the single most decision-relevant thing α
knows.

## Delivery

The reduced rows land on the CRITIC only, through a zero-init projection, so at init the value
features are unchanged and the policy is untouched **at any weight** — `pi` never sees this tensor.
Chosen over the policy-side pointer cell first because it reuses the shipped `--value-threat-inject`
pattern and isolates *"does α-weighting beat hard-max"* from *"does a new pointer cell help"*.
"""
from typing import Optional

import torch

from agents.model.pair_reduce import reduce_with_alpha


class IntentValueReduce(torch.nn.Module):
    """`(α, un-reduced cells) → a vf-only additive term`. Zero-init, so OFF-at-init is exact.

    Equivariant in both axes by construction: `reduce_with_alpha`'s signature gives α no defender
    index (Contract W — the opponent chooses without seeing which of our mons it lands on), and the
    per-defender rows are flattened in a FIXED slot order that the caller keeps stable.
    """

    def __init__(self, n_mons: int, n_features: int, out_dim: int):
        super().__init__()
        self.in_dim = int(n_mons) * int(n_features)
        self.proj = torch.nn.Linear(self.in_dim, int(out_dim))
        # Zero-init: the critic's features are bit-identical at step 0, so any measured effect is
        # something the run LEARNED rather than an initialisation perturbation. `restore_identity_init`
        # picks this up by observation (ledger M1 — SB3's ortho pass would otherwise clobber it).
        torch.nn.init.zeros_(self.proj.weight)
        torch.nn.init.zeros_(self.proj.bias)

    def forward(self, alpha_logits: torch.Tensor, cells: torch.Tensor,
                gate: Optional[torch.Tensor] = None) -> torch.Tensor:
        """`alpha_logits` [B,K+1] · `cells` [B,J,C,F] · `gate` [B,J,1] → [B, out_dim].

        Fails loud when K != C. The two axes are the SAME opponent-move axis (gated by
        `intent_axis_alignment_test`: α's seats are the op's top-K, element-wise), so a width
        mismatch means one of them was reconfigured independently — and silently broadcasting or
        truncating there would mis-pair every term while every shape check still passed. That is
        this codebase's named `op move-order` bug class, so it gets an exception rather than a
        best-effort.
        """
        k = alpha_logits.shape[-1] - 1                       # last class is SWITCH
        c = cells.shape[2]
        if k != c:
            raise ValueError(
                f"alpha has {k} move seats but the op stashed {c} candidate channels. These must be "
                "the SAME axis (entity_topk_seats == damage_topk_k); a mismatch would pair each "
                "alpha weight with the wrong opponent move while every shape check still passed."
            )
        # UNRENORMALIZED move slice — see the module docstring: the missing SWITCH mass is the
        # correct statement that a switching opponent deals no damage this turn.
        alpha = torch.softmax(alpha_logits.float(), dim=-1)[:, :k].to(cells.dtype)   # [B,K]
        rows = reduce_with_alpha(alpha, cells)                                       # [B,J,F]
        if gate is not None:
            rows = rows * gate
        return self.proj(rows.flatten(start_dim=1))                                  # [B, out_dim]
