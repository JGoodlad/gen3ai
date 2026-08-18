"""gen3_pair_value_route_v1 — PV, the pair-VALUE critic route (`design_opponent_intent.md` §7a(2)).

Phase C of the conditional-mechanics substrate, and the half that is NOT the policy's. OA1/OA2 and
the two `pair_outcome` cells all deliver through `pointer_cells`, which is **policy-only** — the
critic reads pooled tokens and has never seen the unified outcome row in any per-entity form.

> **⚠️ THE C4 RE-ENTRY CONDITION — read before enabling, not after.**
> *Any α/β-critic route may be BUILT opt-in but its ENABLING owes the C4-style offline gate first.*
> Ledger row **C6** failed on 2026-08-17 with liveness PROVEN: all five v89 value routes trained off
> zero and `entity_pool` carried decisively (dV 6.28 = 110% of all-off), yet the critic's
> stall-loss over-confidence did not move (gen-13 confident-band gap +0.358, CI [0.23, 0.50]), and
> the delivery line was declared EXHAUSTED. `value_intent` was deleted with a registered re-entry
> condition, and this route inherits it verbatim. Building it is cheap and reversible; **enabling it
> without passing that gate is the thing C6 forbids.**

## The form, and why it is TOKEN CONTENT rather than a v89 seam route

`design_conditional_opponent_cells.md` §6 item 7 names two admissible critic routes: **7a**
generalized token-content injection, and **7b** PV proper (Shaw-style pair VALUES with k seed
queries). §7a(2) of `design_opponent_intent.md` specifies the shape actually taken here:

    send  Σ_k α_k · pair_in[k, j, :]  to the critic as TOKEN CONTENT on our mon j's token,
    pooled by `value_cls` — equivariant in BOTH axes, no seeds.

That is `value_threat_inject`'s mechanism (v64), which is the one enrichment pattern in this tree
that **demonstrably trains**: gen-12's own proof of the v89 dead-tail bug was that
`value_threat_proj` — the single `value_pooled` route — reached weight norm 0.117 while two vf-tail
routes sat bit-exact zero after 25M steps.

**The v89 `_value_pooled_routes` seam was considered and REJECTED, on structure rather than taste.**
A seam route yields one `[B, D_MODEL]` vector added AFTER pooling, so this route would have to
collapse the `J` axis itself — and the only equivariant collapse available is a sum, which is
exactly what destroys the signal: `Σ_j W·row_j` cannot tell *one mon about to lose 90% of its bar*
from *six mons losing 15% each*, and the first is a losing position while the second is a normal
turn. Token content does not collapse: the row rides the token that also carries the mon's
identity, HP and typing, and `value_cls`'s attention decides how much of the board each mon is
worth. Keeping the axis is the whole point (§2b.2 — *"you can only preserve an axis you have
output slots for"*), and here the tokens ARE the slots. Cost of the choice: this route is not
covered by the seam's gradient guard by construction, so `value_route_gradient_test` was EXTENDED
with a dedicated cell rather than left to inherit one.

## ⚠️ α here is the R1 rung, and that is ORDERING, not preference

`value_cls` pools at **T2, before the α/β heads are scored** (the chain is `TeamTransformer →
BeliefHead → CLSPool → α/β → …`). A token-content injection therefore CANNOT read the published α:
it does not exist yet. Two consequences, both stated rather than hidden:

* the route uses `pair_alpha(None, w_topk, seat_live)` — the shipped **R1 `belief_mean`** rung
  (`α := w/Σw`), unconditionally, **even when `--opp-intent` is on**. It is not a fallback that
  fires when a head is absent; it is the only distribution that exists at this point in the forward.
* the design pre-registers exactly this substitution (§7a(2): *"testable BEFORE α exists by
  substituting α := normalize(w)"*), which separates the **DELIVERY** claim — does a per-entity
  outcome absolute in the value pool help at all? — from the **DISTRIBUTION** claim. A null here
  indicts the route, not the belief.

`w` is a PRESENCE belief and α is a USAGE belief; they are NOT the same object and the R1 rung sums
to 1 where the published slice sums to `1 − α_SWITCH`. That difference is the substantive modelling
error `design_pair_reduction.md` names, and it is why this module says so in three places rather
than calling `w` an α.

**What the route delivers that `--value-threat-inject` does not.** Both inject a per-our-mon row on
the value pool's copy; the ROW is the difference. v64 sends the `pair_reduce` rung's 13-wide
DAMAGE-only summary. This sends `pair_in` — Phase A's unified 14-coordinate vector, whose last
eight are the six status identities, `neutralization` and `tempo_cost`. The critic has no other
route to any of them in a per-entity currency: incoming status reaches it only through the `s3`
edge family, i.e. as a softmax-normalised RATIO (`design_pair_reduction.md` §2.1). So the two flags
are not two spellings of one arm, and they are independently enableable for that reason.

## Contract compliance

Zero-init projection ⇒ ON-at-init is bit-identical on the critic; `restore_identity_init` captures
it BY OBSERVATION (ledger M1). **vf-ONLY structurally, at ANY weight**: the augmentation happens
inside `CLSPool` on a LOCAL copy that only `value_cls` reads, so `our_cls`, `our_active_refined` and
the pointer head all see the untouched tokens and `pi` is bit-identical for an arbitrary `W`.
Equivariant in both axes: α has no `J` axis by Contract W (invariant under permuting their moves),
the row rides our mon j's own token (equivariant under permuting ours), and attention pooling is
permutation-invariant over what it reads. ONE `Linear` shared over `j` — a per-slot projection
would reintroduce the positional dependence the entity re-home deleted.
"""
from __future__ import annotations

import torch


class PairValueInject(torch.nn.Module):
    """Project our mon j's α-reduced unified outcome row to `D_MODEL` and add it to mon j's token.

    Deliberately a sibling of `ValueThreatInject` rather than a reuse of it: the two carry different
    objects (a damage summary vs the unified outcome vector), they are independently enableable, and
    a shared class would give them one `state_dict` key to fight over. The forward is the same three
    lines because the MECHANISM is the shipped one — that is the point of choosing it.
    """

    def __init__(self, row_dim: int, d_model: int):
        super().__init__()
        if row_dim <= 0:
            raise ValueError(f"PairValueInject row_dim must be positive, got {row_dim}")
        self.row_dim = int(row_dim)
        self.proj = torch.nn.Linear(self.row_dim, d_model)
        torch.nn.init.zeros_(self.proj.weight)
        torch.nn.init.zeros_(self.proj.bias)

    def forward(self, our_team_out: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        """`our_team_out` [B, 6, D_MODEL] + `rows` [B, 6, row_dim] → augmented [B, 6, D_MODEL].

        Returns a NEW tensor; the caller keeps the unaugmented tokens for every policy-facing read,
        which is what makes "vf-only" a structural property rather than a convention.
        """
        if rows.shape[-1] != self.row_dim:
            raise ValueError(
                f"PairValueInject expected rows of width {self.row_dim}, got {rows.shape[-1]} — "
                "PAIR_OUTCOME_COORDS and this projection have drifted, which means the row's "
                "coordinate table changed without the route being rebuilt.")
        if rows.shape[:2] != our_team_out.shape[:2]:
            raise ValueError(
                f"PairValueInject shape mismatch: tokens {tuple(our_team_out.shape[:2])} vs rows "
                f"{tuple(rows.shape[:2])}")
        return our_team_out + self.proj(rows.to(our_team_out.dtype))  # type: ignore[no-any-return]
