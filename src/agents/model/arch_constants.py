"""Architecture constants — the single source of truth for every weight-shape-relevant dim.

Relocated out of `features_extractor.py` (2026-08-01) so that `damage_op.py` can read them without
importing the extractor, which would be circular: the extractor imports `DamageOperator`. Nothing
else changed — `features_extractor` re-exports the whole block, so `from
agents.model.features_extractor import D_MODEL` (and every other historical import path) still
resolves, and `model_version.py` keeps reading them from there.

Change a dim HERE and nowhere else. See the root `CLAUDE.md` -> Model Versioning: a change to any of
these is weight-shape-relevant and `check_compatible` will (correctly) reject old checkpoints.
"""
ROLE_TOKEN_SIZE = 128
PROJECTION_DIM = 512
MOVE_NET_HIDDEN = [96, 32]        # [hidden, output] of shared move processor
# gen3_unified_move_system_v1: context-free per-move LATENT (MoveLatentEncoder) — a mechanics-grounded
# move identity (move/type embeddings ⊕ MOVE_ATTR), routed into the move network AND used as the
# similarity-grading target so Rock Slide ≈ Hidden Power Rock. Flag-gated (`move_latent`); OFF leaves the
# move network byte-identical.
MOVE_LATENT_HIDDEN = 64           # hidden width of the MoveLatentEncoder MLP
MOVE_LATENT_DIM = 32              # output dim of the per-move latent (grading is cosine in this space)
ROLE_ENCODER_HIDDEN = [256, 128]  # [hidden, output] of per-Pokémon role encoder
NET_ARCH = [512, 512]             # MLP policy layers (SB3 policy_kwargs["net_arch"])
# gen3_pointer_native_v1: the pointer action head's shared scorer hidden width (the ONLY action head —
# no flat action_net exists in this generation; see Gen3DualHeadMaskablePolicy._build).
POINTER_HIDDEN = 64               # hidden width of the pointer move/switch/struggle scorers

# Unified transformer hyperparameters. d_model matches ROLE_TOKEN_SIZE so team
# role tokens enter the transformer without a projection step.
D_MODEL = ROLE_TOKEN_SIZE         # 128
TRANSFORMER_N_LAYERS = 2
TRANSFORMER_N_HEADS = 4
TRANSFORMER_FFN_DIM = 256

# gen3_no_concat_v1 (v61, the gen-5 world): the multi-seed CRITIC readout that replaces the op
# head-concat's value window — k learned queries cross-attend over the op's per-our-mon rows
# (the `our_mon` arity-1 tensor), giving the critic readout MULTIPLICITY (P3 refuted width,
# never multiplicity). k*dim rides vf_parts only. Ships WITH the seeds/* TB collapse monitors
# (seed_diagnostics.py) and the pre-registered VICReg trigger.
VALUE_SEED_K = 4
VALUE_SEED_DIM = 64

# gen3_intent_value_reduce_v1 (step 6): the alpha-weighted threat term appended to the CRITIC's
# pre-projection features. `_INTENT_CELL_FEATURES` is the F axis of the operator's un-reduced
# `cells_pr` stack (low/high/crit/ko_ramp/acc/phys_mask) — change one and the other must follow.
_INTENT_CELL_FEATURES = 6

# gen3_unified_value_readout_v1 (v80, Stage-3 T3-DELIVER, design_unified_belief.md §3): ONE
# attention pool over the critic's ENTITY-ROW SET (the 12 post-transformer team tokens + the op's
# 6 per-our-mon incoming rows), the designed successor of the bolt-on vf routes (seed readout /
# threat-inject) the gen-11 critic-route audit adjudicates. K learned queries over rows projected
# to a shared UVR_DIM (with a per-SOURCE type embedding), zero-init output projection to
# UVR_OUT_DIM riding vf only. `_UVR_N_SOURCES` axes: 0=our mon token, 1=their mon token,
# 2=op incoming row.
# gen3_value_pooled_routes_v1 (v89): the value routes (intent_value_reduce, entity pool,
# threshold-vf, clock, intent) INJECT additively into value_pooled, so their output width
# is D_MODEL by definition — the old per-route width constants are deleted with the
# vf-tail concat they sized.
UVR_K = 4
UVR_DIM = 64
_UVR_N_SOURCES = 3
# gen3_unified_value_readout_v2 (v82, `value_entity_pool_full`): +source 3 = the refined
# GLOBAL token, +source 4 = the hidden-opp belief queries — the pool's COMPLETE row set (the
# one successor for every condemnable vf route). A separate flag/shape so v80-table
# checkpoints keep loading.
_UVR_N_SOURCES_FULL = 5

# gen3_intent_move_cell_v1 (G3, design_conditional_execution.md): the alpha-conditioned c2
# status-consequence channels appended to the pointer MOVE cell. `_INTENT_MOVE_CELL_RAW` is the
# raw channel stack [is_status, d_their_outspeed, e_burn_alpha, d_sched, e_slp_alpha,
# e_slp_free_turns, alpha_stay]; `INTENT_MOVE_CELL_DIM` is the zero-init projection's output width
# (what the pointer move scorer's in_features grow by when the flag is on).
INTENT_MOVE_CELL_DIM = 7
_INTENT_MOVE_CELL_RAW = 7

# gen3_intent_threshold_v1 (v84, design_conditional_execution.md §3.0 / build-order step 3): the
# α-weighted THRESHOLD operator `p_thresh(τ, ⋛) = Σ_k α_k · 1[damage(k, me) ⋛ τ]` — one
# contraction over the op's already-computed per-candidate cells that lands five mechanics at
# once (Focus Punch / Substitute / Endure / Destiny Bond / Endeavor) plus `p_KO`, the calibrated
# "am I about to die this turn" the critic previously inferred from a hard max (ledger H1).
# `_INTENT_THRESH_RAW_MOVE` is the per-request-slot raw stack
# [fp_execute, sub_survives, endure_pko, dbond_pko, endeavor_survive, p_ko_context];
# `_INTENT_THRESH_RAW_VF` is the critic's raw stack [p_ko, p_sub_broken, p_fp_broken].
# The DIM constants are the two zero-init projections' output widths.
INTENT_THRESH_MOVE_DIM = 6
_INTENT_THRESH_RAW_MOVE = 6
_INTENT_THRESH_RAW_VF = 3

# gen3_intent_conditional_v1 (v85, design_conditional_execution.md build steps 4+7): the
# remaining α-conditioned mechanic cells — Counter / Mirror Coat (the category test), flinch's
# missing (1−α_SWITCH) term, Explosion's execute/into-switch facts, and Pursuit's doubling
# trigger (the PORT-verified rule: the strike hits the DEPARTING mon at ×2 BP never-miss, so no
# β is needed — the design doc's β-weighted-arrival formula was wrong on the target).
# `_INTENT_COND_RAW` is the per-request-slot raw stack [counter_return, mc_return, cat_match,
# flinch_useful, boom_executes, boom_into_switch, pursuit_trigger, pursuit_bonus,
# protect_dmg_avoided, protect_p_success, protect_status_avoided, magiccoat_reflect] — the
# protect trio is build step 5 (§3.3): c4 carried the mechanical p_success and OMITTED the
# α-weighted quantity it multiplies; here both ride decorrelated, plus the α mass on status
# seats. magiccoat_reflect is step 6 (§3.12), built AFTER its G0 oracle resolved the
# reflectable set (measurements/gen3_magiccoat_reflectable_oracle.json: foe-targeting status
# bounces; side-targeting Spikes does NOT). boom_trade_ko is §3.1's β half — the FIRST
# forward-side β consumer: P(the detonation KOs its actual target), α_stay-weighted vs their
# active plus α_SWITCH·β-weighted vs the arrival (needs the outgoing matrix's per-(move, their
# mon) pko — hence intent_conditional requires damage_matrices_outgoing).
INTENT_COND_MOVE_DIM = 13
_INTENT_COND_RAW = 13

# gen3_pair_outcome_v1 (v93, design_opponent_intent.md §5.1 + design_pair_reduction.md §2.1/§9a):
# the UNIFIED PER-PAIR OUTCOME VECTOR — one `pair_in[k, j, :]` per (their believed move k, our mon
# j) carrying damage AND status AND neutralization AND tempo in the SAME vector, reduced by ONE
# shared α over the move axis and delivered to the pointer MOVE cell.
#
# `_PAIR_OUTCOME_DMG` is the op's existing pair-cell width (== `_INTENT_CELL_FEATURES`, the F axis
# of `last_pair_cells`: [low, high, crit, ko_ramp, acc, is_phys]); `_PAIR_OUTCOME_NEW` is the eight
# coordinates gen3_pair_outcome_v1 adds (6 per-identity status landing probabilities + neutralization
# + tempo_cost — the full table with each coordinate's §9a admission answer lives in
# `pair_outcome.py`). `_PAIR_OUTCOME_RAW` is their concatenation = the reduced row's width;
# `PAIR_OUTCOME_MOVE_DIM` is the zero-init projection's output width (what the pointer move
# scorer's in_features grow by when the flag is on).
_PAIR_OUTCOME_DMG = 6
_PAIR_OUTCOME_NEW = 8
_PAIR_OUTCOME_RAW = _PAIR_OUTCOME_DMG + _PAIR_OUTCOME_NEW      # 14
PAIR_OUTCOME_MOVE_DIM = _PAIR_OUTCOME_RAW

# gen3_pair_outcome_switch_v1 (v94, Phase B — design_pair_reduction.md §2.1's CANONICAL defect):
# the same α-reduced outcome row, per DEFENDER, at the pointer SWITCH cell. Phase A delivered our
# ACTIVE's row to the move cells as CONTEXT; §2.1's decision ("they will click Will-O-Wisp, so
# bring the Natural Cure mon") is made at the SWITCH logit, whose cell holds ten damage numbers,
# one speed number, two belief-mass numbers and NO status coordinate in any currency.
#
# `_PAIR_OUTCOME_SWITCH_EXTRA` is the one coordinate that is per-DEFENDER and is not part of the
# reduced row: `spin_denied` (the Pursuit mirror's defensive half — our Ghost candidate denies
# their believed Rapid Spin, priced by the hazard stake it preserves). The full table with each
# coordinate's §9a admission answer lives in `pair_outcome.py`.
_PAIR_OUTCOME_SWITCH_EXTRA = 1
_PAIR_OUTCOME_SWITCH_RAW = _PAIR_OUTCOME_RAW + _PAIR_OUTCOME_SWITCH_EXTRA   # 15
PAIR_OUTCOME_SWITCH_DIM = _PAIR_OUTCOME_SWITCH_RAW

# gen3_switch_branch_v1 (v94, Phase B — design_conditional_opponent_cells.md §2 "OA2, the
# SWITCH-BRANCH MOVE CELL", plus the owner's Rapid-Spin spinblock and the Protect α-conditioning):
# per-request-slot content for the branch in which the OPPONENT SWITCHES. Gen-3 is
# simultaneous-move, so P(they switch) is ONE scalar for the turn (§2.1) — but the CONSEQUENCE is
# per-move, because switches resolve first and our move lands on the ARRIVAL, which β names.
# `_SWITCH_BRANCH_RAW` is the coordinate count; the table with the §9a answers is in
# `switch_branch.py`.
_SWITCH_BRANCH_RAW = 9
SWITCH_BRANCH_MOVE_DIM = _SWITCH_BRANCH_RAW

# gen3_conditional_threat_v1 (v95, Phase C — design_conditional_opponent_cells.md §1 "OA1, the
# CONDITIONAL THREAT CELL"): the defensive-pivot coordinates the α-reduced outcome row structurally
# cannot carry — the accuracy-folded P(this mon dies) (§0.2(2)'s "precompute every nonlinearity of
# two numbers in the op"), the bulk-INDEPENDENT expected type multiplier, and the two §0.2(3)
# MARGINS (max roll and crit roll) that say by how much a saturated probability saturates. Delivered
# to the pointer SWITCH cell. §1.2's `λ`-weighted `w` is NOT built — `pair_alpha` is the shipped
# distribution and a second one would be a second α; the full substitution table and each
# coordinate's §9a admission answer are in `conditional_threat.py`.
_CONDITIONAL_THREAT_RAW = 4
CONDITIONAL_THREAT_SWITCH_DIM = _CONDITIONAL_THREAT_RAW

# gen3_pair_value_route_v1 (v95, Phase C — design_opponent_intent.md §7a(2), the equivariant CRITIC
# route): the α-reduced unified outcome row for our mon j, injected as TOKEN CONTENT on mon j's own
# token inside `CLSPool`, on the VALUE pool's copy only. Width == `_PAIR_OUTCOME_RAW`, because the
# object delivered IS Phase A's row — the point is that the critic has never had the status /
# neutralization / tempo currency in any per-entity form at all.
PAIR_VALUE_ROUTE_DIM = _PAIR_OUTCOME_RAW


# gen3_value_direct_routes_v1 (v87): two direct CRITIC routes appended at the vf tail, both
# zero-init. VALUE_CLOCK_DIM — the deadline clock's 3 raw scalars projected for the critic (the
# v67 clock fix was validated for exactly this reader, and the audit read its surviving
# indirect route as dead). VALUE_INTENT_DIM — the published α (K seats + SWITCH, belief-sorted
# canonical order) and β (6 team slots) posteriors as probabilities; the critic finally reads
# WHAT WE EXPECT THEM TO DO directly rather than only through the α-weighted physics cells.
