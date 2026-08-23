"""The PRE-PROJECTION assembly — the static width arithmetic and the per-head concat.

Split out of `features_extractor.py` 2026-08-23 (one responsibility per file); that module
re-exports both names, so every historical import path still resolves. The two belong together
because they are the same statement twice: `compute_projection_widths` is the arithmetic
`ProjectionAssembler.forward` performs, and `projection_width_test.py` asserts a real forward's
measured widths equal it.
"""
from typing import Any, Dict, Optional, Tuple

import torch

from agents.model.arch_constants import D_MODEL
from agents.model.extractor_ctx import ExtractorContext

def compute_projection_widths(layout: Dict[str, Any], *,
                              opp_belief_cls_k: int = 0) -> Tuple[int, int]:
    """The `(pi, vf)` projection-input widths as STATIC ARITHMETIC (gen3_static_widths_v1).

    Mirrors `ProjectionAssembler.forward`'s concat exactly — this is the single place that
    arithmetic lives, and `projection_width_test.py` sweeps flag combos asserting a REAL
    forward's measured widths equal it (the old construction-time discovery forward,
    preserved as the verifier).

    **`vf` is now a CONSTANT** (`D_MODEL`). The critic-route deletion wave retired the whole
    post-assembler vf tail — the seed window, the hidden-opp belief's vf half and the
    `non_matchup_rest` vf concat — so `vf_combined IS value_pooled`, the same tensor the
    dist-head critic reads. That is the structural cure for the v89/M2 bug class rather than
    another instance of it: there is no longer a vf branch that `--value-from-dist` can orphan.
    Only TWO inputs still move `pi`: the layout (the `non_matchup_rest` scalar tail) and the
    hidden-opp belief pool (`opp_belief_cls_k` queries × D_MODEL, policy side only). Every
    other flag is width-neutral by construction: the v89 value routes inject ADDITIVELY into
    `value_pooled`, the intent cells widen the pointer stash (not pi/vf), and the token-stream
    enrichments change content, not shape.

    Pure — importable and unit-testable without building a model.
    """
    from agents.observation.schema import build_schema
    sl = build_schema(layout).slices()
    # The non-matchup scalar tail: global-env block + the 5 raw board scalars — everything
    # between the active contexts and the embedding-ID active_req_moves tail (ObsUnpack).
    non_matchup_rest = sl['reactive.active_req_moves'].start - sl['global_env'].start
    belief = opp_belief_cls_k * D_MODEL
    # pi: our_team_pooled + their_team_pooled + our_active_refined (D_MODEL each) + tail + belief.
    pi = 3 * D_MODEL + non_matchup_rest + belief
    # vf: value_pooled, and nothing else.
    vf = D_MODEL
    return pi, vf


class ProjectionAssembler(torch.nn.Module):
    """Assembles the pre-projection inputs for BOTH heads.

    Policy input: team pools + our active token + the non-matchup scalar tail (+ the hidden-opp
    belief when built). Value input: `value_pooled`, and nothing else.

    **THE VALUE TAIL IS GONE, and that is the point.** The critic-route deletion wave retired
    all three of its members on measured evidence — the v61 seed window (dV 0.0000 bit-exact on
    two consecutive end-of-run audits), the hidden-opp belief's vf half (0.0000, while its PI
    half flips 39.6% of argmaxes and therefore STAYS), and the `non_matchup_rest` vf concat
    (0.0000, its content substituting through the global token per C1). So `vf_combined IS
    value_pooled` — the very tensor `--value-from-dist`'s critic reads — which makes the v89/M2
    orphaned-branch bug class *unrepresentable* rather than merely fixed: there is no second vf
    path left for a critic parameterization to bypass. Every critic enrichment now goes through
    one of two declared seams, `_value_pooled_routes` (additive, gradient-guarded) or the two
    `CLSPool` token-content injections.

    gen3_ctx_dedup_v1: the per-side ENCODED active contexts (`active_ctx_encoder` on the raw
    58-dim boosts+volatiles blocks) are DELETED from both heads. They were duplicated delivery
    with a 1:1 entity-native replacement already live: the E2 injection scatters each side's
    FULL raw ctx block onto its ACTIVE mon's role token (`gen3_entity_rehome_v1`, pinned by
    `e2_ctx_injection_test.py`), and the global token carries both raw blocks as a second
    route. `non_matchup_rest` stays on the POLICY side: its only token route is the global
    token, which no pool reads directly, so the pi concat is still its one direct head path.
    """

    def __init__(self, layout: Dict[str, Any]):
        super().__init__()

    def forward(self, our_team_pooled: torch.Tensor, their_team_pooled: torch.Tensor,
                our_active_refined: torch.Tensor, value_pooled: torch.Tensor,
                ctx: ExtractorContext,
                hidden_opp_belief: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Concatenate the per-head pre-projection inputs → `(pi_combined, vf_combined)` [B, *].
        `hidden_opp_belief` [B, K*D_MODEL] feeds the POLICY head when built (its vf half was
        audited dead and deleted); the value half is `value_pooled` alone."""
        pi_parts = [our_team_pooled, their_team_pooled, our_active_refined,
                    ctx.non_matchup_rest]
        # gen3_no_concat_v1 (v61): THE OP HEAD-CONCAT IS DEAD. The 660-dim flat block no longer
        # enters either head — its measured end-state (gen-4, stratified, 53ef270): net policy
        # dependence +0.00%, all-edges-off ABOVE the concat arm on flips, and the critic's
        # magnitude content decodable without it (act_threat vf r² 0.418 concat-zeroed). The op
        # itself lives on: pointer cells (policy, lossless per-action), prefuse token injection,
        # the D/S/C/V/T/X edge cells, and `last_raw_block` for the probes; the critic reads it
        # through `--value-entity-pool`, which carries 97% of the critic's route dependence.
        # Hidden-opponent belief (flag-guarded; None when off) feeds the POLICY head — it reads
        # the threat over the hidden team. Appended last so the off-by-default block layout is
        # unchanged (`compute_projection_widths` sizes the projections; a new width-contributing
        # part must be added THERE too — the sweep test fails on any drift).
        if hidden_opp_belief is not None:
            pi_parts.append(hidden_opp_belief)
        pi_combined = torch.cat(pi_parts, dim=1)
        return pi_combined, value_pooled
