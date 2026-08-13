"""T0 RESOLVE — the team-composition species belief, produced BEFORE the physics reads it.

The gap this closes (`gen3_t0_species_prior_v1`). The model already forms a conditional belief about
which species occupies a hidden opponent slot: `BeliefHead.species_prior_logits` is a naive-Bayes
read over the revealed teammates, and it is genuinely informative — the difference between "the
average gen3ou mon" and "the fifth slot on a team that has already shown Tyranitar + Skarmory +
Cloyster". But `BeliefHead` is **T2 DECIDE** (post-transformer, a training-only side readout), while
the `DamageOperator` that prices unrevealed defenders is **T1 REASON** (pre-transformer). A T1
consumer cannot read a T2 output, so the physics fell back to `SPECIES_USAGE_PRIOR` — a static
gen3ou frequency table masked only by Species Clause. The belief and the physics were built
separately and never introduced.

Nothing here is new machinery. The naive-Bayes math is v69's, lifted verbatim into a free function
that `BeliefHead` now also calls, so the two cannot drift; this module only re-homes it to a tier
the op can read from. The op's `species_probs` argument has been sitting unused since
`gen3_unrevealed_outgoing_prior_v1` documented it as "the future-learned-belief seam".

**Why the prior alone, with no learned delta (v1).** A T0 read sees PRE-attention tokens, so a
learned delta here is strictly weaker than the one `BeliefHead` already gets at T2 — while the
*prior* half loses nothing at T0, because it never touched tokens in the first place. It reads only
`opp_species_ids` and `opp_believed_mask`. So the informative half moves down a tier for free and
the weak half is not built. A T0 delta remains available later; it would need its own compile gate
(see the shape note below).

**Shape is deliberate: `[B, n_species]`, not `[B, 6, n_species]`.** The belief is a property of the
opponent's TEAM, not of which hidden slot you ask about — v69's own docstring makes that argument,
and it means every slot's row would be identical anyway. Keeping it 2-D also lands exactly on the
static branch's contract, so this is a drop-in substitution that reuses the op's existing
broadcast-the-RESULTS path. That path exists because the `[B,6,S]` expand mis-vectorized under
Inductor CPU and took down gen-4's launch prewarm (`damage_op.py` `unrevealed_species_probs`); a
per-slot override re-enters that shape and must not be added without a compile gate.

Leak-safe by construction: the inputs are the revealed nums and the hidden-slot mask — the same two
tensors `MoveBelief.move_logits` takes. The true opponent team reaches the model only as a training
TARGET via `belief_target_slots`, and is not read here.
"""
from typing import Optional

import torch


def species_team_prior_logits(log_marginal: torch.Tensor,
                              log_lift: torch.Tensor,
                              opp_species_ids: torch.Tensor,
                              opp_believed_mask: "Optional[torch.Tensor]" = None) -> torch.Tensor:
    """`[B, n_species]` log P(species in a hidden slot | the revealed opponent team).

    The v69 naive-Bayes read, factored out of `BeliefHead.species_prior_logits` so that method and
    `T0SpeciesPrior` share ONE implementation rather than two copies that could drift:

        log P(s | R) ∝ log P(s) + Σ_{r ∈ R} log[ P(s | r) / P(s) ]

    `opp_species_ids` [B,6] national-dex nums (0 = the UNKNOWN sentinel a hidden slot carries);
    `opp_believed_mask` [B,6] bool, True where the slot is still hidden. A slot counts as EVIDENCE
    iff it is revealed AND carries a real num.

    The sum over the revealed set is `revealed_onehot @ log_lift.T` — `[B,S] @ [S,S]` — deliberately
    NOT the obvious `log_lift[:, ids]` gather, which would materialize a `[B,6,S]` intermediate
    (157 MB at the production minibatch of 16384) for the same answer.

    SPECIES CLAUSE is applied last and OVERRIDES the evidence: a species already on the board cannot
    also be hiding on the bench. Every entry stays FINITE (`SPECIES_CLAUSE_LOGIT`, never `-inf`), so
    the `log_softmax` cannot produce a NaN row and a floored candidate keeps a live gradient.

    Returns log-probabilities in the 2-D team-level shape. `BeliefHead` unsqueezes to `[B,1,S]` for
    its per-slot broadcast; the op consumes the 2-D form directly.
    """
    from agents.model.damage_tables import SPECIES_CLAUSE_LOGIT

    n_species = log_marginal.shape[0]
    ids = opp_species_ids.clamp(0, n_species - 1).long()               # [B,6] defensive index clamp
    revealed = ids > 0
    if opp_believed_mask is not None:
        revealed = revealed & (~opp_believed_mask.bool())
    # BRANCHLESS + SYNC-FREE one-hot over the revealed set (the `MoveBelief.move_logits` scatter
    # trick): a revealed slot writes 1.0 at its num (always >= 1); a hidden slot writes 0.0 at the
    # clamped index 0, which no revealed slot can occupy — so the two never collide. Species Clause
    # makes duplicate nums impossible, and `scatter_` is idempotent anyway, so a malformed duplicate
    # cannot double-count as evidence.
    onehot = torch.zeros(ids.shape[0], n_species,
                         dtype=log_marginal.dtype, device=log_marginal.device)
    onehot.scatter_(1, ids, revealed.to(log_marginal.dtype))            # [B, S]
    scores = log_marginal + onehot @ log_lift.t()                       # [B, S]
    scores = torch.where(onehot > 0, torch.full_like(scores, SPECIES_CLAUSE_LOGIT), scores)
    return torch.log_softmax(scores, dim=-1)                            # [B, S]


class T0SpeciesPrior(torch.nn.Module):
    """T0 RESOLVE: `[B, n_species]` P(species | revealed opponent team), for the T1 physics.

    Parameter-free — it owns only the two data-derived buffers, registered NON-persistent because
    they are recomputable from the committed pool artifact (the move prior's contract). So turning
    this flag on adds nothing to the `state_dict` and cannot shift an optimizer parameter position.
    """

    def __init__(self, n_species: int):
        super().__init__()
        from agents.model.damage_tables import build_species_cooccur_prior
        _log_marginal, _log_lift = build_species_cooccur_prior(n_species)
        self.register_buffer("species_prior_log_marginal", _log_marginal, persistent=False)
        self.register_buffer("species_prior_log_lift", _log_lift, persistent=False)

    def forward(self, opp_species_ids: torch.Tensor,
                opp_believed_mask: "Optional[torch.Tensor]" = None) -> torch.Tensor:
        """`opp_species_ids` [B,6] (OPPONENT side only) → `[B, n_species]` PROBABILITIES.

        ⚠️ `log_softmax(...).exp()`, never `torch.softmax` — the `gen3_species_posterior_spelling_v1`
        rule. `torch.softmax` over the last dim is the one op Inductor's CPU scheduler cannot codegen
        (`AssertionError: buf<N>` fusing the division), and it was the sole reason
        `--compile-extractor` ever needed `suppress_errors`. Mathematically identical, same
        max-subtraction stability.
        """
        return species_team_prior_logits(
            self.species_prior_log_marginal, self.species_prior_log_lift,
            opp_species_ids, opp_believed_mask).exp()
