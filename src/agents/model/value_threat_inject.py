"""VALUE THREAT INJECTION — the critic's per-entity magnitude route (gen3_value_threat_inject_v1).

`design_opponent_intent.md` §7a.2b. The critic's problem is NOT that it lacks a threat signal; it
is that every route carrying one is either a RATIO or a POOLED scalar:

  * the `d3` edge family writes an attention BIAS — softmax-normalised, so it can say *"this
    defender is threatened MORE than that one"* and structurally cannot say *"by 62% of max HP"*;
  * v61's `MultiSeedValueReadout` (now DELETED — see below) had k seeds each emitting a convex
    combination of the SAME six kv rows, and both pressures tried on them (VICReg repulsion,
    per-seed quantile assignment) left the differentiation ~1-DIMENSIONAL — measured centered PR
    0.846 / 0.835 at matched checkpoints, and gen-7's `out_effective_rank` reaching only 1.157
    against a ceiling of k=4. A shared readout constrains the component along its own weight
    vector and nothing orthogonal to it, so seed MULTIPLICITY was not the missing axis.

**This route WON the argument, and the alternative is gone.** The critic-route deletion wave
retired the seed readout on dV **0.0000 bit-exact across two consecutive end-of-run audits**, while
`threat` — this module — read **1.0686** (19.0% of the whole route joint) and had its registered
deadline discharged. Its docstring keeps the seed history because the *reason* is the durable part.

This module takes the third route the design names: put the magnitude in the critic's pool as TOKEN
CONTENT. For each of OUR mons `j`, the op's α-weighted incoming row `Σ_k α_k · pair_in[k, j, :]` is
projected to `D_MODEL` and ADDED to that mon's post-transformer token, and `value_cls` pools the
augmented set. An attention *value* carries a magnitude where an attention *bias* cannot (see
`designs/learning/marginalization_and_uncertainty.md` — the convex-combination primitive), so the
critic can finally read "this specific mon is about to lose 62% of its HP" per entity rather than a
board-level ratio.

**Equivariance (the property this route was chosen FOR, gated by V2).** The reduced row is a convex
combination over THEIR move axis with `α` shared across defenders (Contract W — `α` has no `J`
axis by signature), so it is invariant under permuting their moves; the row rides OUR mon `j`'s own
token, so it is equivariant under permuting our mons; and `value_cls` attention pooling is
permutation-INVARIANT over the tokens it reads. The composition is therefore invariant in both
axes — unlike the deleted head-concat, which was a flat 660-wide block whose meaning depended
entirely on slot order.

**vf-ONLY, and asserted as such (V1).** The augmentation happens INSIDE `CLSPool` on a local copy
that feeds only the `value_cls` pool. `our_cls`, `our_active_refined` and the pointer head all read
the UNAUGMENTED `our_team_out`, so `pi` is bit-identical for an ARBITRARY `W_inj` — not merely at
init. That is what makes this arm interpretable: a policy change cannot confound the critic result,
because the policy provably cannot see the injection.

`W_inj` is ZERO-INIT, so ON starts byte-identical to OFF. That guarantee is only real because
`Gen3FeaturesExtractor.restore_identity_init()` re-zeros it after SB3's ortho pass clobbers every
extractor `nn.Linear` (ledger M1 — the trap that silently falsified six shipped features' cold-start
contracts). The guard captures its set by OBSERVATION at the end of `__init__`, so this module is
protected automatically rather than by being added to a list.

**Scope, honestly.** v1 substitutes `α := normalize(w)` — the shipped R1 `belief_mean` rung, a
PRESENCE belief, where the design wants a supervised USAGE belief. That is deliberate: it separates
the DELIVERY claim (does a per-entity absolute in the value pool help at all?) from the
DISTRIBUTION claim (does a learned `α` beat `w`?), so a null here indicts the route and not the
belief. G1-FINAL's null does not predict this one — it tested aggregation over the POLICY's cells,
never a critic-side delivery change.
"""
from __future__ import annotations

import torch

# The rung `--value-threat-inject` forces on the op. `hard_max` (R0, production) builds no reducer
# at all and stashes nothing, so the inject would have no input; `belief_mean` is R1, parameter-free
# and defender-independent by construction (Contract W). Not a user-facing knob: the delivery claim
# is what this arm tests, and letting the rung vary would confound it.
VALUE_THREAT_INJECT_REDUCE_HOW = "belief_mean"


def value_threat_inject_dim() -> int:
    """Width of one reduced row under the forced rung — the projection's `in_features`.

    A function rather than a constant so the answer is DERIVED from the reducer's own arithmetic
    (`pair_reduce_extra_dim`) and the op's channel count, not restated here where it could drift.
    The imports are local because the caller (`Gen3FeaturesExtractor.__init__`) needs this width
    BEFORE the `DamageOperator` that owns those constants has been built.
    """
    from agents.model.damage_op import _PAIR_REDUCE_N_CHANNELS
    from agents.model.pair_reduce import pair_reduce_extra_dim
    return pair_reduce_extra_dim(VALUE_THREAT_INJECT_REDUCE_HOW, _PAIR_REDUCE_N_CHANNELS)


class ValueThreatInject(torch.nn.Module):
    """Project the op's per-our-mon reduced threat row to `D_MODEL` and add it to that mon's token.

    Shared over `j` — ONE `Linear` applied to all six rows — which is what makes the map equivariant
    rather than slot-indexed. A per-slot projection would reintroduce exactly the positional
    dependence the entity re-home deleted.
    """

    def __init__(self, extra_dim: int, d_model: int):
        super().__init__()
        if extra_dim <= 0:
            raise ValueError(f"ValueThreatInject extra_dim must be positive, got {extra_dim}")
        self.extra_dim = int(extra_dim)
        self.proj = torch.nn.Linear(self.extra_dim, d_model)
        # Zero-init ⇒ identity-at-init ⇒ ON is byte-identical to OFF at step 0. Re-zeroed after
        # SB3's ortho pass by `restore_identity_init` (captured by observation — see module docstring).
        torch.nn.init.zeros_(self.proj.weight)
        torch.nn.init.zeros_(self.proj.bias)

    def forward(self, our_team_out: torch.Tensor, threat_rows: torch.Tensor) -> torch.Tensor:
        """`our_team_out` [B, 6, D_MODEL] + reduced rows [B, 6, extra_dim] → augmented [B, 6, D_MODEL].

        Returns a NEW tensor; the caller keeps the unaugmented tokens for every policy-facing read.
        """
        if threat_rows.shape[-1] != self.extra_dim:
            raise ValueError(
                f"ValueThreatInject expected rows of width {self.extra_dim}, got "
                f"{threat_rows.shape[-1]} — the op's reducer and this projection disagree, which "
                "means `reduce_how` changed without rebuilding the injection."
            )
        if threat_rows.shape[:2] != our_team_out.shape[:2]:
            raise ValueError(
                f"ValueThreatInject shape mismatch: tokens {tuple(our_team_out.shape[:2])} vs rows "
                f"{tuple(threat_rows.shape[:2])}"
            )
        return our_team_out + self.proj(threat_rows.to(our_team_out.dtype))  # type: ignore[no-any-return]
