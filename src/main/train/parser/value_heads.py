"""Three neighbouring head sections, merged: the EVIDENTIAL BETA head, the TWIN HEADS +
SHADOW CRITIC, and the PER-ACTION Q WIN-PROB head.

Lifted VERBATIM out of the old single-file `parser.py` (lines 779-891); the flags
keep their original relative order, which is the order `--help` renders.
"""
import argparse

from main.train.parser.base import BoolFlag


def add_value_head_flags(parser: argparse.ArgumentParser) -> None:
    """Add this family's flags to `parser`, in their original order."""
    # --- THE EVIDENTIAL BETA HEAD (gen3_cf_evidential_head_v1). G0 convicted the scalar win-prob
    # head of RESOLUTION (within-decile true spread 0.11-0.36), not of an optimism offset. This head
    # cannot fix the blur — same input — but it can CONFESS it, as a Beta posterior whose width the
    # factory's priority sampler and the awareness stack can read. `--cf-evidential` is STRUCTURAL
    # (v98, version-gated, in flag_registry); the two coefficients are training-only.
    parser.add_argument("--cf-evidential", "--cf_evidential", dest="cf_evidential",
                        action=BoolFlag, default=None,
                        help="BUILD the evidential Beta head (α, β via softplus+1) off "
                             "value_pooled. STRUCTURAL and version-gated: its params are in the "
                             "state_dict, so a resume must match. Its input is detached "
                             "UNCONDITIONALLY — a pure supervised readout that feeds nothing "
                             "forward and is not even called by the forward, so OFF is "
                             "byte-identical and ON at coefficient 0 is bit-identical in pi/vf.")
    parser.add_argument("--cf-evidential-coef", "--cf_evidential_coef",
                        dest="cf_evidential_coef", type=float, default=None,
                        help="Weight on the evidential term: the Beta-Binomial MARGINAL "
                             "log-likelihood of the label's counts (p integrated out, the correct "
                             "evidential loss for count data), normalized per rollout like the "
                             "scalar term. Default 0.0 = OFF. Requires --cf-evidential. Watch "
                             "cf/evid_epistemic_std_mean (the confessed width — the headline) and "
                             "cf/evid_precision_mean (α+β, the claimed evidence).")
    parser.add_argument("--cf-evidential-reg", "--cf_evidential_reg",
                        dest="cf_evidential_reg", type=float, default=None,
                        help="Weight of the KL(Beta(α,β) ‖ Beta(1,1)) pull, RELATIVE to the NLL "
                             "(it rides inside --cf-evidential-coef, so coefficient 0 kills it "
                             "too). The standard evidential-overconfidence guard: nothing in the "
                             "likelihood bounds α+β on locally-consistent data, and an inflated "
                             "precision makes the width — the entire product — meaningless. "
                             "Default 1e-3.")
    # --- THE TWIN HEADS + THE SHADOW CRITIC (gen3_cf_twin_heads_v1, v99). The owner-authorized
    # amendment to the signed R1 pre-registration (ledger 2026-08-22 evening, "Three owner
    # sign-offs" item 3): the arm's primary comparison becomes a WITHIN-RUN paired head difference
    # instead of a run-vs-run one. Three win-prob heads on ONE trunk — A (control, on-policy BCE
    # only), B (+ the cf states with SINGLE-OUTCOME labels), C (+ the same states with TIGHT-MC
    # labels) — so B−A isolates coverage and C−B isolates pure variance reduction with every random
    # draw held identical. The two structural flags are version-gated; the coefficients are not.
    parser.add_argument("--cf-twin-heads", "--cf_twin_heads", dest="cf_twin_heads",
                        action=BoolFlag, default=None,
                        help="BUILD the TWIN win-prob heads B and C off value_pooled (the "
                             "within-run paired R1 comparison). STRUCTURAL and version-gated: "
                             "their params are in the state_dict, so a resume must match, and "
                             "because the forward never calls them nothing else would catch a "
                             "flip. Head-only ALWAYS in v1 (both read a DETACHED value_pooled), so "
                             "OFF is byte-identical and ON at coefficient 0 is bit-identical in "
                             "pi/vf. Requires --win-prob-mode read_only|shaping (head A must exist "
                             "for the twins to mirror its loss).")
    parser.add_argument("--cf-twin-coef", "--cf_twin_coef", dest="cf_twin_coef",
                        type=float, default=None,
                        help="Weight on BOTH twins' cf folds — ONE knob on purpose: B and C must "
                             "differ in their LABEL STREAM and in nothing else. B eats the row's "
                             "outcome_label at n=1, C eats its tight-MC label at n=R, through the "
                             "SAME per-rollout-normalized binomial NLL, so the two pull equally "
                             "hard and C−B reads label PRECISION rather than learning rate. Their "
                             "share of head A's own on-policy BCE rides --win-prob-coef, not this. "
                             "Default 0.0 = OFF (whole block skipped, byte-identical). Requires "
                             "--cf-twin-heads. Read cf/twin_b_coverage FIRST.")
    parser.add_argument("--cf-shadow-critic", "--cf_shadow_critic", dest="cf_shadow_critic",
                        action=BoolFlag, default=None,
                        help="BUILD the passive SHADOW CRITIC off value_pooled — a value twin "
                             "trained on tight-MC mc_return labels (the run's own shaped return). "
                             "It NEVER computes an advantage and NEVER enters GAE: it is the "
                             "staged promotion path for critic surgery (a critic route change owes "
                             "the C4 gate), so it accumulates evidence rather than changing the "
                             "critic. STRUCTURAL and version-gated; detached always; OFF "
                             "byte-identical, ON at coefficient 0 bit-identical in pi/vf.")
    parser.add_argument("--cf-shadow-coef", "--cf_shadow_coef", dest="cf_shadow_coef",
                        type=float, default=None,
                        help="Weight on the shadow critic's masked MSE against mc_return, computed "
                             "in the PopArt-normalized frame (the value loss's frame, so the "
                             "coefficient is scale-comparable with it). Default 0.0 = OFF. "
                             "Requires --cf-shadow-critic. THE METER is cf/shadow_shadow_vs_live_v "
                             "— the signed real-unit gap between the MC-grounded twin and the live "
                             "critic on the same states.")
    # --- THE PER-ACTION Q WIN-PROB HEAD (gen3_q_winprob_head_v1, v107; ai_v12 E5). Every value
    # readout in this tree evaluates a STATE, so "my win probability if I click Rock Slide" costs
    # eleven simulator re-rolls. This head amortizes that: one forward, eleven P(win|s,a), scored
    # from the pointer head's own action tokens. `--q-winprob-mode` is STRUCTURAL (version-gated,
    # in flag_registry); the two coefficients are training-only.
    parser.add_argument("--q-winprob-mode", "--q_winprob_mode", dest="q_winprob_mode",
                        choices=("none", "read_only"), default=None,
                        help="BUILD the PER-ACTION win-probability head over the pointer head's "
                             "action tokens (one shared zero-init readout ⇒ every Q logit is "
                             "exactly 0 at init ⇒ P=0.5 everywhere, an honest uninformative "
                             "start). STRUCTURAL and version-gated: its params are in the "
                             "state_dict and its ONLY output is a stash, so nothing downstream "
                             "would catch a flipped flag. 'none' (default) does not build it at "
                             "all — byte-for-byte the baseline. 'read_only' detaches EVERY input, "
                             "so pi/vf stay bit-identical at any coefficient; there is "
                             "deliberately no 'shaping' value, because a per-action readout "
                             "carrying a counterfactual label is a larger leak surface than a "
                             "per-state one and trunk exposure owes its own gate.")
    parser.add_argument("--q-winprob-coef", "--q_winprob_coef", dest="q_winprob_coef",
                        type=float, default=None,
                        help="Weight on the per-action COUNTERFACTUAL likelihood: the masked "
                             "binomial NLL over exactly those (state, action) pairs a label row's "
                             "`q_labels` covers, normalized per rollout like the scalar cf term. "
                             "Default 0.0 = OFF (the whole block is skipped). Requires "
                             "--q-winprob-mode read_only and a cf label buffer. Read "
                             "q_winprob/label_coverage FIRST — a producer shipping no per-action "
                             "labels trains this head on nothing while every other scalar looks "
                             "healthy.")
    parser.add_argument("--q-winprob-onpolicy-coef", "--q_winprob_onpolicy_coef",
                        dest="q_winprob_onpolicy_coef", type=float, default=None,
                        help="Weight on the WEAK on-policy fallback: the recorded battle's "
                             "realized outcome as a single-sample label for the ONE action that "
                             "was actually taken. Default 0.0 = OFF, and it should usually stay "
                             "there. ⚠️ BIAS: on-policy data labels 1 action out of 11 and the "
                             "measured preferred-alternative rate is p≈0.002, so this term teaches "
                             "the head only where the policy already goes — leaving it "
                             "confidently wrong on the never-tried moves, which is precisely the "
                             "starvation failure the counterfactual labels exist to avoid. It is "
                             "weighted SEPARATELY from --q-winprob-coef so the two can never be "
                             "confused in a run's provenance.")
