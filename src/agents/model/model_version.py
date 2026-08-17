from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

# Bump this whenever the ModelVersion schema changes (fields added/renamed/removed).
# Also add a migration case in _migrate_config().
#
# The per-version narrative (v3 -> v88: what each field means, which gate enforces it, and
# why) lives in designs/CHANGELOG.md under 'The MODEL_CONFIG_VERSION narrative' — moved
# there 2026-08-16; it is history, and this file keeps only the live machinery.
# v89 (gen3_value_pooled_routes_v1): the five value routes (intent_value_reduce v74,
#   value_entity_pool v80/82, intent_threshold's vf half v84, value_clock/value_intent v87)
#   INJECT into `value_pooled` instead of the post-assembler vf concat, which
#   `--value-from-dist` structurally bypassed (gen-12 proof: their zero-init projections
#   bit-exact ZERO after 25M steps). Route out-widths become D_MODEL and the vf concat
#   narrows for flag-ON configs, so a <v89 checkpoint recording ANY of them ON carries
#   shapes the surviving code cannot load — REFUSED (the v75 rule); OFF stamps forward.
MODEL_CONFIG_VERSION = 89

# The one-line effect of each `belief_grad_mode`, for the migration notice. Keyed by the SAME strings
# as `features_extractor.BELIEF_GRAD_MODES` (which owns the legal set + the ValueError); the two are
# pinned to agree by `belief_grad_mode_test.py::test_every_mode_has_a_migration_notice`, so a fourth
# mode cannot ship with a silently generic notice.
_BELIEF_GRAD_MODE_EFFECT = {
    "shaping": "the belief-aux gradient now SHAPES the shared trunk, and PPO trains the heads.",
    "detached": "the belief-aux gradient now STOPS at the heads (the trunk is stop-grad on the read).",
    "label_only": "the belief heads are now trained by their SUPERVISED LABELS ALONE — no policy/value "
                  "gradient reaches them (their outputs are published stop-grad to every consumer).",
}

# Change this when the neural architecture changes structurally in a way that makes
# weights from a different signature incompatible (e.g. adding LSTM, replacing attention).
# Same-family dim changes (role_token_size 128→256) don't need a new signature —
# check_compatible() catches those via the dim fields.
#
# The signature-by-signature history (v2 -> gen3_ctx_dedup_v1: what broke weight
# compatibility each time, and why) lives in designs/CHANGELOG.md under 'The
# ARCH_SIGNATURE narrative' — moved there 2026-08-16.
ARCH_SIGNATURE = "gen3_ctx_dedup_v1"

# The migration floor: the first MODEL_CONFIG_VERSION stamped with the current ARCH_SIGNATURE.
# Every `if version < N` migration branch with N <= this floor could only ever produce a config
# that the arch_signature gate — run by every consumer immediately after migration (snapshot.py's
# load paths, the fixed-opponent pool, train resume) — rejects anyway, so `_migrate_config`
# refuses pre-floor configs outright instead of walking dead branches.
# ⚠️ When ARCH_SIGNATURE next changes, raise this floor to the new signature's first stamped
# version IN THE SAME COMMIT (and append the pairing to SIGNATURE_FIRST_VERSION below) —
# migration_floor_test.py fails if the two drift apart.
MIGRATION_FLOOR = 76

# The signature → first-stamped-version pairing the floor is derived from. Append-only: add the
# new signature's row when it lands. migration_floor_test.py asserts
# MIGRATION_FLOOR == SIGNATURE_FIRST_VERSION[ARCH_SIGNATURE].
SIGNATURE_FIRST_VERSION = {
    "gen3_deadline_clock_v1": 67,
    "gen3_ctx_dedup_v1": 76,
}


class ModelVersionError(Exception):
    pass


@dataclass
class ModelVersion:
    # Schema and architecture identity
    config_version: int
    arch_signature: str

    # From state_encoder.get_layout()
    species_embedding_dim: int
    max_species: int
    move_embedding_dim: int
    max_moves: int
    item_embedding_dim: int
    max_items: int
    ability_embedding_dim: int
    max_abilities: int
    type_embedding_dim: int
    max_types: int
    total_dim: int
    active_context_dim: int

    # From features_extractor.py module constants
    role_token_size: int
    projection_dim: int
    move_net_hidden: List[int]
    role_encoder_hidden: List[int]
    n_history_turns: int

    # From policy_kwargs in train_rl_agent.py
    net_arch: List[int]

    # PPO value-loss coefficient (`--vf-coef`). Recorded for resume-immutability
    # (check_vf_coef), NOT a weight-shape field — see MODEL_CONFIG_VERSION v3 note and
    # its exclusion from _WEIGHT_FIELDS in check_compatible(). Defaults to the SB3 default
    # so versions built for a weight-shape-only check (current_model_version, the roundtrip
    # test) need not supply it.
    vf_coef: float = 0.5

    # Reward-config hparams (v4) — resume-immutable VALUE-meaning, NOT weight-shape. Default = the
    # single-variable run (material clutch-fix only; BIAS additive). Enforced via check_reward_config.
    bias_additivity: float = 1.0
    mat_alive_weight: float = 1.25
    bias_redesign: bool = False
    switch_bias_weight: float = 0.0   # v5: belief-risk-scaled stay-into-KO BIAS lever (default OFF)
    # v7: terminal reward for a DRAW / 250-turn timeout. -30.0 = the prior behavior (tie == decisive
    # loss). Resume-immutable VALUE-meaning (check_reward_config), excluded from the weight-shape check.
    draw_penalty: float = -30.0
    # v12: de-bias cleanup — zero audit-flagged distorting BIAS terms. Resume-immutable VALUE-meaning
    # (check_reward_config), excluded from the weight-shape check. False = the prior behavior.
    drop_redundant_bias: bool = False   # drop stall_tax + matchup_penalty (redundant w/ clock+draw / pbrs_belief)
    drop_switch_bias: bool = False      # drop the hand-coded switch-strategy subsidy family

    # v13/v14: end-state PBRS switches + the now-immutable no-progress penalty (Φ_progress's weight).
    # all_shaping_pbrs = "everything but stall"; stall_pbrs (v14) = the "stall" switch (Φ_progress).
    all_shaping_pbrs: bool = False
    stall_pbrs: bool = False
    no_progress_penalty: float = 0.15

    # v6 feature toggle (value-checked, not weight-shape): PopArt value-target normalization. The
    # value head's parameterization + buffers differ when on, so it cannot be toggled on a resume.
    # Defaulted (must follow the defaulted fields above) so weight-shape-only callers need not supply it.
    use_popart: bool = False

    # v8 behavioral toggle (value-checked, not weight-shape): keep the opponent's still-hidden party
    # ATTENDABLE in the transformer instead of key-masking unrevealed slots like fainted mons. Changes
    # the forward-pass mask (policy AND value), not any weight shape or the obs layout, so it lives in
    # config_version (not ARCH_SIGNATURE) and is checked in check_compatible — resuming with a
    # different value would silently change the masking the policy trained under.
    attend_unrevealed_opponents: bool = False

    # v9 structural toggle (weight-shape): hidden-opponent belief — `opp_belief_cls_k` distinct learned
    # query tokens (HiddenOppBeliefPool) that summarise the unrevealed opp party and feed both heads.
    # 0 = OFF (no module; reproduces the baseline arch byte-for-byte). k>0 ADDS the module + grows both
    # projection inputs by k*D_MODEL, so like use_popart it is gated in check_compatible — but as a plain
    # int every distinct value (incl. 0↔N) is a weight-shape mismatch, so NO conditional and NO
    # ARCH_SIGNATURE bump. k>0 requires attend_unrevealed_opponents (enforced at extractor-build time).
    opp_belief_cls_k: int = 0

    # v10 structural toggle (weight-shape): route our_active_refined (the active mon's refined token)
    # into the VALUE projection. The dual-head value readout (value_pooled) drops the active-mon view
    # the policy keeps; a probe found the critic predicts an incoming self-KO at AUC 0.79 vs the
    # policy's 0.90, which under-prices the V-tail. ON widens the value projection by D_MODEL; OFF
    # reproduces the baseline value head byte-for-byte, so like use_popart it lives in check_compatible
    # WITHOUT an ARCH_SIGNATURE bump.

    # v11 resume-immutable VALUE-meaning hparam (like vf_coef — NOT weight-shape): tail-weighted value
    # loss β. 0.0 = plain MSE; >0 blends the CVaR of the worst value misses into the loss. Changing it
    # mid-run silently shifts the value objective, so it is enforced ONLY on the training-resume path
    # via check_value_tail_weight (excluded from check_compatible, which gates frozen eval/pool/distill
    # opponents whose forward never touches it). Defaulted so weight-shape-only callers need not supply it.
    value_tail_weight: float = 0.0

    # v12 resume-immutable VALUE-meaning reward hparam (like draw_penalty — NOT weight-shape): the
    # decision-time-HP-scaled self-KO penalty weight (−w·hp on Explosion/Self-Destruct + we_fainted).
    # 0.0 = OFF (byte-identical). Enforced via check_reward_config; excluded from check_compatible.
    # Defaulted so weight-shape-only callers need not supply it.
    self_ko_hp_penalty: float = 0.0

    # v16 STRUCTURAL toggle (weight-shape via the BeliefHead + unknown-slot params): the in-place
    # hidden-opponent BELIEF AUX. ON fills un-revealed opp slots with distinct learned unknown-mon
    # tokens + builds a BeliefHead (species/moves aux logits) — a state_dict change, gated in
    # check_compatible like opp_belief_cls_k. OFF = baseline arch byte-for-byte (NO ARCH_SIGNATURE
    # bump). Requires attend_unrevealed_opponents (enforced at extractor build).
    opp_belief_slots: bool = False
    # v16 TRAINING-ONLY loss coefficient (like ent_coef — NOT weight-shape, NOT a resume FATAL): the
    # aux-loss weight on the belief CE+BCE. Affects only the loss magnitude, not the forward, so it is
    # recorded for provenance but EXCLUDED from check_compatible AND has no dedicated check_*. 0.0 = off.
    opp_belief_aux_coef: float = 0.0
    # v17 STRUCTURAL toggle (weight-shape via the MoveBelief module: a move head + a reinject Linear).
    # Predicts each opp slot's moveset and REINJECTS it into the token (flow-through). 'off' = no module
    # (baseline byte-for-byte); 'revealed'|'unrevealed'|'both' build it + change the forward (which slots are
    # enriched). Like attend_unrevealed_opponents the mode also changes the trained forward, so the
    # STRING is gated in check_compatible; OFF reproduces baseline (NO ARCH_SIGNATURE bump). Requires
    # attend_unrevealed_opponents.
    move_belief_mode: str = "off"
    # v17 TRAINING-ONLY loss coefficient for the move belief (like opp_belief_aux_coef). 0.0 = no aux.
    move_belief_coef: float = 0.0
    # v19 STRUCTURAL toggle (weight-shape via the DamageOperator's wider projections): the
    # differentiable GPU damage operator. ON consumes the move belief's predicted moves for the opp
    # active and appends a per-our-mon believed-damage block to BOTH heads (widening both projection
    # Linears), so like value_active_readout / opp_belief_slots it is gated in check_compatible. OFF =
    # baseline byte-for-byte (NO ARCH_SIGNATURE bump). Forward-only → no training-only coefficient.
    # Requires move_belief_mode in {revealed, both} (enforced at extractor build + CLI).
    damage_op: bool = False
    # (v31 `damage_reattend` is DELETED at v71 — see the v71 note above `MODEL_CONFIG_VERSION`.)
    # v20 FORWARD-BEHAVIOR toggle (NOT weight-shape, like attend_unrevealed_opponents): the unified
    # two-part move belief. ON fuses the Smogon move-frequency prior into the MoveBelief head as a
    # log-odds residual (+ pins revealed moves certain), so the stashed move-belief logits carry a
    # posterior (priors ⊕ prediction). The prior buffer is non-persistent (no new params → state_dict
    # identical), but the forward differs, so it is gated in check_compatible. Requires move_belief_mode
    # != off. OFF = the from-scratch head byte-for-byte (NO ARCH_SIGNATURE bump).
    move_prior_fusion: bool = False
    # (v32 `move_belief_prefuse` is DELETED at v71 — the PRE-transformer placement is unconditional.)
    # v49 FORWARD-BEHAVIOR (gen3_topk_candidates_v1): cap the op's incoming candidate sweep at the K
    # most-believed opponent moves (0 = full sweep, byte-identical). No tail bound — dropped mass is
    # dropped. No new params, so the state_dict is identical; only the forward's VALUES differ.
    damage_candidate_k: int = 0
    # v51 (gen3_pointer_native_v1): the v49 `pointer_head` bool is GONE — the pointer head is THE
    # action head, unconditionally (no flat action_net exists), so there is nothing to gate. The
    # ARCH_SIGNATURE bump carries the cross-era break; _migrate_config POPs the dead key.
    # (v50 `damage_op_prefuse` is DELETED at v71 — the PRE-transformer placement is unconditional, and
    # `prefuse_proj` is built whenever `damage_op` is on.)
    # v54 STRUCTURAL int (gen3_entity_move_seats_v1, the damage_topk_k gating pattern): the E4
    # threat-move SEAT count — the opp active's top-K believed candidate moves entering the trunk as
    # attention seats. >0 adds `entity_seats.threat_seat_proj` (state_dict) and K seats to every
    # attention pass (forward); 0 = E3-only (our 4 move seats, which are UNCONDITIONAL — their break
    # rides the ARCH_SIGNATURE bump, not this field). Requires damage_op + move_latent.
    entity_topk_seats: int = 0
    # consequence_topk (v59): the CONSEQUENCE kernels' believed-candidate axis (C1b/C2/C3's
    # k_cand + D4's k_bench — one knob, the coverage of the belief-weighted worst-case max).
    # FORWARD-BEHAVIOR int (no params — an internal reduction axis): gated in check_compatible
    # because a frozen opponent's forward changes with it. Pre-v59 checkpoints trained at 4.
    consequence_topk: int = 6
    # v57 STRUCTURAL bool (gen3_entity_tail_seats_v1, E5): the 6 tail-threat seats (adds
    # tail_proj + tail_marker to the state_dict and 6 seats to every attention pass).
    entity_tail_seats: bool = False
    # v56 STRUCTURAL str (gen3_edge_bias_trunk_v1): which edge families are delivered as attention
    # biases ("off" | "d" | comma list of d1,d3). A family adds its zero-init map (state_dict) and
    # its cells to every attention pass (forward). The layer swap itself is UNCONDITIONAL and rides
    # the ARCH_SIGNATURE, not this field.
    edge_bias_families: str = "off"
    # v21 FORWARD-BEHAVIOR toggle (NOT weight-shape, like attend_unrevealed_opponents): the
    # unified-architecture ablation. ON zeros the incoming-damage / OHKO obs block out of the model's
    # view (the block stays in the obs; the reward still reads it). State_dict identical; the forward
    # v22 STRUCTURAL toggle (weight-shape via the WinProbHead params): the tri-state auxiliary
    # win-probability head. 'none' = no module (baseline byte-for-byte). 'read_only'/'shaping' build a
    # WinProbHead (a side readout off value_pooled — NOT in pi/vf, so projection dims are unchanged; the
    # only state_dict delta is the head's params). Gated in check_compatible with a STRING compare:
    # 'none'↔head changes the state_dict AND read_only↔shaping is the user-chosen resume-IMMUTABLE mode
    # (flipping grad-flow mid-run is a silent training change), so ANY mismatch is FATAL. OFF reproduces
    # baseline byte-for-byte (NO ARCH_SIGNATURE bump).
    win_prob_mode: str = "none"
    # v22 TRAINING-ONLY loss coefficient for the win-prob head (like opp_belief_aux_coef). Scales the BCE
    # aux loss, affects no forward pass → recorded for provenance but NOT version-locked (resume-mutable,
    # inherited on a flagless resume). Default 1.0 (full weight when the mode is on; ignored when none).
    win_prob_coef: float = 1.0
    # v23 STRUCTURAL toggle (weight-shape, like damage_op): the OUTGOING per-move damage direction. ON makes
    # the DamageOperator ALSO emit the our-active→opp-active per-move block (request-slot aligned), widening
    # both projection Linears. Gated in check_compatible (bool); OFF = baseline byte-for-byte (NO
    # ARCH_SIGNATURE bump). Requires damage_op (the op must exist).
    damage_outgoing: bool = False
    # v23 FORWARD-BEHAVIOR float (NOT weight-shape, like move_prior_fusion), REDEFINED in v65
    # (gen3_unconditional_move_legality_v1): move LEGALITY is now UNCONDITIONAL — a move a species cannot
    # learn ALWAYS gets ~0 prior mass, with no flag and no opt-out (it is a correctness property, not a
    # feature). This float is now ONLY the LEGAL-BUT-UNOBSERVED base probability — the small liftable
    # floor a legal move with no recorded Smogon usage starts from. It no longer doubles as the on/off
    # switch, which is what let production's `--move-candidate-floor 0.0` silently disable legality
    # altogether. Must be >= damage_tables._MIN_PRIOR_FLOOR (1e-3); 0.0 is now REJECTED, so every
    # pre-v65 checkpoint (which recorded 0.0) fails this check LOUDLY rather than loading with a
    # different belief. Kept in check_compatible: it changes the belief the policy/value/op trained
    # under. The prior buffer is non-persistent so the state_dict is identical either way.
    # NOTE: the default MUST equal damage_tables._PRIOR_FLOOR — this module is deliberately stdlib-only
    # (no torch / no gen3_data import), so the literal is duplicated here and pinned by a test.
    move_candidate_floor: float = 0.02
    # v24 STRUCTURAL toggle (weight-shape, like damage_op): the context-free MoveLatentEncoder — a
    # mechanics-grounded per-move latent concatenated into the move-network input (widens it → state_dict
    # change). Gated in check_compatible (bool); OFF = baseline byte-for-byte (NO ARCH_SIGNATURE bump).
    move_latent: bool = False
    # v24 TRAINING-ONLY coefficient (like move_belief_coef, NOT version-locked): the move-belief LATENT
    # grading weight (cosine of the predicted move distribution's expected latent → true moveset mean
    # latent + VICReg). Recorded for provenance + flagless-resume read-back; reads the move_latent table.
    move_belief_latent_coef: float = 0.0
    # v25 STRUCTURAL toggle (like opp_belief_slots): the SpreadBelief module (predict+reinject the opp's
    # hidden spread). Gated in check_compatible; OFF byte-for-byte (no module, NO ARCH_SIGNATURE bump).
    spread_belief: bool = False
    # v40 STRUCTURAL toggle (gen3_nature_ev_belief_v1): swap the SpreadBelief's additive point-estimate head
    # for the NATURE/EV generative head (prior-fusion → compute the derived stat) to fix the largest-EV
    # over-estimate. Different SpreadBelief params → state_dict change → gated in check_compatible; requires
    # spread_belief; OFF byte-for-byte (the additive head, NO ARCH_SIGNATURE bump).
    spread_belief_nature: bool = False
    # v40 FORWARD-BEHAVIOR toggle (gen3_nature_ev_belief_v1, like move_prior_fusion): the DamageOperator
    # MARGINALISES the nonlinear P(KO)/damage over the believed nature distribution (compute-then-blend over the
    # top natures) instead of using E[nature_mult] — restores the ×1.1/×0.9 asymmetry in the threshold. No new
    # params (reads the head's nature posterior). Requires spread_belief_nature. Gated in check_compatible (a
    # mid-run flip feeds a different forward); OFF byte-for-byte.
    # v25 TRAINING-ONLY coefficient (NOT version-locked): the speed-supervision weight (masked BCE of the
    # believed P(outspeed) toward observed move order). Recorded for provenance + flagless-resume read-back.
    spread_belief_coef: float = 0.0
    # v29 STRUCTURAL toggle (weight-shape via the ValueDistHead params, like win_prob_mode): the
    # distributional VALUE readout — an interpretability side head off value_pooled emitting per-atom
    # return-distribution logits. 'none' = no module (baseline byte-for-byte, NOT in pi/vf so projection
    # dims are unchanged). Gated in check_compatible with a STRING compare (none↔head AND
    # read_only↔shaping — flipping grad-flow mid-run is a silent training change). OFF reproduces baseline
    # byte-for-byte (NO ARCH_SIGNATURE bump).
    value_dist_mode: str = "none"
    # v29 STRUCTURAL: the atom count — the head's output Linear width, so a mismatch is a weight-shape
    # change (gated in check_compatible with an unconditional int compare, like opp_belief_cls_k). 0 = off
    # (forced when mode == none).
    value_dist_bins: int = 0
    # v64 STRUCTURAL (gen3_value_threat_inject_v1): the critic-side magnitude route — one shared
    # zero-init Linear(reducer.extra_dim, D_MODEL) adding the op's alpha-weighted incoming row to
    # each of OUR mons' tokens on the VALUE POOL's copy only. Adding/removing changes the
    # state_dict, so it is gated in check_compatible with a bool compare. It also FORCES the op's
    # reduce_how from R0 hard_max to R1 belief_mean, which builds the (parameter-free) reducer —
    # another reason a flip cannot share weights. OFF (default) is byte-for-byte baseline; the
    # policy path is untouched at ANY value of the projection (NO ARCH_SIGNATURE bump).
    value_threat_inject: bool = False
    # v68 STRUCTURAL (gen3_opp_intent_v1): the ALPHA (their believed-move seats + SWITCH) and BETA
    # (which mon they bring) intent heads — two pointer scorers, so adding/removing them changes the
    # state_dict. Supervision-only (their input is detached), but STRUCTURAL all the same: a resume
    # that flips this would have no weights for them / orphan weights. OFF (default) builds neither.
    opp_intent: bool = False
    # v69 STRUCTURAL (gen3_species_prior_fusion_v1): fuse the TEAM-COMPOSITION species prior into
    # BeliefHead's species head — `species_logits = head_delta + log P(species | revealed opp mons)`,
    # from two NON-PERSISTENT co-occurrence buffers. The state_dict is UNCHANGED (no new params, the
    # buffers are recomputable), which is exactly why this needs an explicit gate: nothing else would
    # catch it. Flipping it mid-run RE-MEANS every species logit — a head trained as a from-scratch
    # predictor would suddenly be read as a delta on a prior it never saw, and vice versa — so it is
    # compared with a bool like move_prior_fusion. OFF (default) reproduces the from-scratch head
    # byte-for-byte (NO ARCH_SIGNATURE bump).
    species_prior_fusion: bool = False

    # v72 STRUCTURAL (gen3_t0_species_prior_v1): feed the T1 physics the model's OWN team-composition
    # species belief instead of the static `SPECIES_USAGE_PRIOR` frequency table. The belief itself is
    # v69's naive-Bayes read, re-homed to T0 so the DamageOperator (T1, pre-transformer) can consume
    # it — `BeliefHead` computes the same thing at T2 and the op could never reach it. Parameter-free
    # (two non-persistent buffers), so the state_dict is IDENTICAL either way; but every
    # unrevealed-defender damage number changes, so a resume may not flip it — hence the dedicated
    # check below rather than reliance on a shape mismatch that would never fire.
    t0_species_prior: bool = False

    # v73 (gen3_intent_grad_mode_v1): whether alpha/beta's gradient reaches the shared trunk.
    # "detached" = pure supervision (a null indicts the head); "shaping" = the intent objective also
    # shapes the representation. No weight shapes change, so nothing else would ever catch a flip —
    # and a flip mid-run silently changes WHAT THE TRUNK IS BEING TRAINED TO DO.
    opp_intent_grad_mode: str = "detached"

    # v74 STRUCTURAL (gen3_intent_value_reduce_v1, step 6): alpha-weighted expected incoming threat
    # appended to the critic's features. WIDENS vf_projection, so unlike the other intent toggles a
    # mismatch WOULD be shape-caught — the check is here anyway so the failure names the cause
    # instead of surfacing as an opaque size error deep in a load.
    intent_value_reduce: bool = False

    # v77 STRUCTURAL (gen3_intent_move_cell_v1, G3): the POLICY-side alpha consumer — the c2
    # status-consequence family re-delivered through the pointer MOVE cell, alpha-conditioned.
    # WIDENS the pointer move scorer's in_features (a state_dict change on the POLICY, not the
    # extractor), so a mismatch would be shape-caught — the check is here anyway so the failure
    # names the cause instead of surfacing as an opaque size error deep in a load.
    intent_move_cell: bool = False

    # v80 STRUCTURAL (gen3_unified_value_readout_v1, Stage-3 T3-DELIVER): the critic's unified
    # entity pool — K queries over the 12 team tokens + the op's incoming rows, zero-init output
    # riding vf only. WIDENS the value projection (a state_dict change), so a mismatch would be
    # shape-caught — the check is here anyway so the failure names the cause instead of
    # surfacing as an opaque size error deep in a load.
    value_entity_pool: bool = False
    # v82 STRUCTURAL (gen3_unified_value_readout_v2): the pool's COMPLETE row set (+global,
    # +belief queries). Grows source_emb 3→5 rows (state_dict shape), so it is its OWN field —
    # a v80-shape checkpoint (gen-12) stays loadable under full=False.
    value_entity_pool_full: bool = False

    # v83 STRUCTURAL (gen3_item_belief_v1): the hidden-item belief head (Smogon prior ⊕
    # zero-init delta; the op's p_cb unrevealed branch consumes its publication). Adds the
    # ItemBelief module — a state_dict change, so a mismatch would be shape-caught; the
    # check names the cause.
    item_belief: bool = False

    # v84 STRUCTURAL (gen3_intent_threshold_v1): the α-weighted threshold operator's two
    # zero-init projections (move cell + vf) — a state_dict change AND a pointer-cell/critic
    # width change, so a mismatch would be shape-caught; the check names the cause.
    intent_threshold: bool = False

    # v85 STRUCTURAL (gen3_intent_conditional_v1): the mechanic-cell projection — a state_dict
    # AND pointer-cell width change; shape-caught, the check names the cause.
    intent_conditional: bool = False

    # v86 STRUCTURAL (gen3_op_lean_forward_v1): drop_renders shrinks out_gain (state_dict
    # shape); believed_lean changes the d3 forward math (no shape — the version gate is the
    # ONLY thing that rejects a mismatched resume).
    op_drop_renders: bool = False
    op_believed_lean: bool = False

    # v87 STRUCTURAL (gen3_value_direct_routes_v1): both widen the value projection.
    value_clock: bool = False
    value_intent: bool = False

    # v81 STRUCTURAL (gen3_event_window_v1, Tier H-B): the event-seat consumer of the obs
    # event window. Builds EventSeats (kind/status embeddings + a projection + the marker) —
    # a state_dict change, so a mismatch would be shape-caught; the check names the cause.
    history_events: bool = False
    # v29 VALUE-MEANING support [vmin, vmax] (the return range the atoms span) — NOT weight-shape (the
    # atoms buffer is non-persistent), but the head's target/interpretation, so resume-IMMUTABLE and
    # enforced ONLY on the training-resume path via check_value_dist (like value_tail_weight), EXCLUDED
    # from check_compatible (a frozen opponent never reads the value-dist head).
    value_dist_vmin: float = 0.0
    value_dist_vmax: float = 0.0
    # v29 TRAINING-ONLY coefficient for the value-dist head's HL-Gauss CE (like win_prob_coef). Scales the
    # aux loss, affects no forward pass → recorded for provenance + flagless-resume read-back, NOT
    # version-locked. Default 1.0 (full weight when the mode is on; ignored when none).
    value_dist_coef: float = 1.0
    # v30 STRUCTURAL (gen3_unified_topk_incoming_v1): the DamageOperator's DISCRETE top-K incoming block —
    # K = the number of the opp active's most-believed candidate moves surfaced individually (each with its
    # move LATENT + per-pivot damage/status). 0 = off. out_dim (hence both projection in_features) scales
    # with K, so EVERY distinct value (incl. 0↔N) is a weight-shape change → gated in check_compatible with
    # an unconditional int compare (like opp_belief_cls_k / value_dist_bins). OFF (0) reproduces baseline
    # byte-for-byte (NO ARCH_SIGNATURE bump). Requires damage_op + move_latent (enforced at the extractor).
    damage_topk_k: int = 0
    # v32 STRUCTURAL (gen3_per_move_matrices_v1): the OUTGOING per-move DAMAGE MATRIX (our 4 moves × opp
    # active + REVEALED bench). Widens both projections via the op out_dim. OFF byte-for-byte (no module
    # output). Gated in check_compatible (bool, like damage_op). Requires damage_op.
    damage_matrices_outgoing: bool = False
    # v33 STRUCTURAL (gen3_per_move_matrices_v1): the INCOMING per-move DAMAGE MATRIX (enriched top-K —
    # per-move header + per-(our-mon, move) cell). Widens both projections via the op out_dim. OFF
    # byte-for-byte. Gated in check_compatible (bool, like damage_op). Requires damage_op + move_latent. It
    # REUSES damage_topk_k as its K (the matrix's width is gated by the existing damage_topk_k int). Since
    # gen3_op_block_trim_v1 deleted the lean top-K, this is the ONLY block that K sizes.
    damage_matrices_incoming: bool = False
    # v39 STRUCTURAL (gen3_per_move_matrices_v1): the TRANSPOSED outgoing matrix — our 6 MONS' 4 moves → the
    # opp ACTIVE (the switch-in offense read; the transpose of damage_matrices_outgoing). Widens both
    # projections via the op out_dim. OFF byte-for-byte (no module output). Gated in check_compatible (bool,
    # like damage_op). Requires damage_op.
    # v36 FORWARD-behavior (gen3_bidir_threat_trunk_v1): the UNCERTAINTY-AWARE P(outspeed) — divide the speed
    # gap by the believed speed std instead of a fixed scale. No new params (values only), gated bool.
    threat_prob_outspeed: bool = False
    # v38 STRUCTURAL + resume-IMMUTABLE tri-state (gen3_opp_hp_type_belief_v1, like win_prob_mode): the
    # opponent HIDDEN-POWER-TYPE belief + the typed-HP candidate fix. 'off' = legacy (the bare typeless
    # HP num-237 candidate out-ranked the 16 typed rows → the opp's Hidden Power read 0-damage/"immune").
    # 'prior' = the op masks the bare-237 + floors the typed-HP belief on the Smogon HP-type prior (a
    # FORWARD-behavior change, NO new params). 'learned' = ALSO build the HPTypeBelief head (prior ⊕ learned
    # gen3_typed_hp_belief_v1 (config v52): the v38 tri-state `hp_type_belief_mode` is GONE. The HP-type
    # head is now unconditional whenever there is a move belief (its "off" state was a correctness bug —
    # a typeless BP-0 candidate and a revealed HP priced as nonexistent — not an ablation), so there is
    # nothing left to gate. `_migrate_config` POPs the dead key.
    # gen3_hp_belief_ablation_v1 (config v53): HOW the 16 typed Hidden-Power channels are produced —
    # 'composed' (default: the presence x type factorisation, with the reveal constraint + the
    # moveset-exhaustion rule-out + effectiveness narrowing) or 'flat' (the ABLATION: the move head
    # predicts them independently, i.e. HP is treated like any other move). STRUCTURAL — 'composed'
    # builds HPTypeBelief and 'flat' does not, so it is a state_dict change as well as a forward one;
    # gated in check_compatible with a STRING compare, like win_prob_mode.
    hp_belief_mode: str = "composed"
    # TRAINING-ONLY coefficient (like move_belief_coef, NOT version-locked): the HP-type CE aux weight.
    # Recorded for provenance + flagless-resume read-back. Only meaningful under 'composed'.
    hp_type_belief_coef: float = 0.0
    # TRAINING-ONLY coefficient (gen3_item_belief_v1, NOT version-locked): the item CE aux weight.
    # Recorded for provenance + flagless-resume read-back. Only meaningful under item_belief=True.
    item_belief_coef: float = 0.0
    # gen3_belief_grad_mode_v1 (config v41): which gradient ARROW between the state-prediction belief
    # heads and the rest of the network is cut. THE TWO NON-DEFAULT MODES CUT OPPOSITE ARROWS — see
    # `Gen3FeaturesExtractor.__init__` for the four-route table:
    #   'shaping'    — nothing cut (the belief loss reshapes the trunk; PPO trains the heads).
    #   'detached'   — the heads READ a stop-grad trunk: no belief gradient reshapes the trunk.
    #   'label_only' — (gen3_belief_label_only_v1) the heads' outputs are PUBLISHED stop-grad to every
    #                  forward consumer: no POLICY/VALUE gradient reaches a belief head's parameters, so
    #                  the belief is trained by its supervised labels ALONE. The read stays LIVE, so the
    #                  label loss still shapes the trunk.
    # detach() is value-preserving → the FORWARD (eval/inference/frozen-opponent) is bit-identical in every
    # mode; only the TRAINING gradient differs. So it is a RESUME-IMMUTABLE training hparam (the vf_coef
    # class): recorded here, enforced ONLY on the training-resume path via check_belief_grad_mode, and
    # EXCLUDED from check_compatible / _WEIGHT_FIELDS (a frozen eval/pool/distill opponent's forward is
    # identical, so gating it would be a false rejection that breaks league play). NO ARCH_SIGNATURE bump.
    belief_grad_mode: str = "shaping"
    # gen3_dist_critic_v1 (config v45, Phase B): the distributional value head IS the critic — GAE /
    # bootstrap / deployment read E[Z] (policy._critic_value) and the HL-Gauss CE is the primary value
    # loss (vf_coef weight); the scalar value_net freezes as a fallback. No state_dict change (both heads
    # exist regardless) and the FORWARD's ACTION selection is unchanged (only the value output differs), so
    # a frozen eval/pool/distill opponent plays identically → RESUME-IMMUTABLE (the belief_grad_mode class):
    # recorded here, enforced ONLY on the training-resume path via check_value_from_dist, EXCLUDED from
    # check_compatible / _WEIGHT_FIELDS. NO ARCH_SIGNATURE bump. Requires value_dist_mode == 'shaping'.
    value_from_dist: bool = False
    # v43 STRUCTURAL + resume-IMMUTABLE tri-state (gen3_pubval_aux_v1, the win_prob_mode pattern): the
    # PUBLIC-information value aux head. 'none' = no module (baseline byte-for-byte). 'read_only'/'shaping'
    # build a PubValHead (side readout off value_pooled — NOT in pi/vf; the only state_dict delta is the
    # head's params) regressed toward the frozen human-replay-calibrated V_pub. Gated in check_compatible
    # with a STRING compare ('none'↔head = state_dict change; read_only↔shaping = the resume-immutable
    # grad-flow choice). OFF byte-for-byte (NO ARCH_SIGNATURE bump).
    # v43 TRAINING-ONLY loss coefficient for the pubval head (like win_prob_coef). Scales the soft-target
    # BCE aux loss, affects no forward pass → recorded for provenance but NOT version-locked
    # (resume-mutable, inherited on a flagless resume).

    @classmethod
    def from_layout_and_policy_kwargs(
        cls,
        layout: Dict[str, Any],
        policy_kwargs: Dict[str, Any],
        vf_coef: float = 0.5,
        reward_config: Any = None,               # duck-typed: read only via getattr(_, default)
        value_tail_weight: float = 0.0,
        opp_belief_aux_coef: float = 0.0,
        move_belief_coef: float = 0.0,
        win_prob_coef: float = 1.0,
        move_belief_latent_coef: float = 0.0,
        spread_belief_coef: float = 0.0,
        value_dist_coef: float = 1.0,
        hp_type_belief_coef: float = 0.0,
        item_belief_coef: float = 0.0,
    ) -> ModelVersion:
        from agents.model.features_extractor import (
            ROLE_TOKEN_SIZE,
            PROJECTION_DIM,
            MOVE_NET_HIDDEN,
            ROLE_ENCODER_HIDDEN,
            NET_ARCH,
            N_HISTORY_TURNS,
        )
        return cls(
            config_version=MODEL_CONFIG_VERSION,
            arch_signature=ARCH_SIGNATURE,
            species_embedding_dim=layout["species_embedding_dim"],
            max_species=layout["max_species"],
            move_embedding_dim=layout["move_embedding_dim"],
            max_moves=layout["max_moves"],
            item_embedding_dim=layout["item_embedding_dim"],
            max_items=layout["max_items"],
            ability_embedding_dim=layout["ability_embedding_dim"],
            max_abilities=layout["max_abilities"],
            type_embedding_dim=layout["type_embedding_dim"],
            max_types=layout["max_types"],
            total_dim=layout["total_dim"],
            active_context_dim=layout["active_context_dim"],
            role_token_size=ROLE_TOKEN_SIZE,
            projection_dim=PROJECTION_DIM,
            move_net_hidden=list(MOVE_NET_HIDDEN),
            role_encoder_hidden=list(ROLE_ENCODER_HIDDEN),
            n_history_turns=N_HISTORY_TURNS,
            net_arch=list(policy_kwargs.get("net_arch", NET_ARCH)),
            vf_coef=vf_coef,
            bias_additivity=float(getattr(reward_config, "bias_additivity", 1.0)),
            mat_alive_weight=float(getattr(reward_config, "mat_alive_weight", 1.25)),
            bias_redesign=bool(getattr(reward_config, "bias_redesign", False)),
            switch_bias_weight=float(getattr(reward_config, "switch_bias_weight", 0.0)),
            draw_penalty=float(getattr(reward_config, "draw_penalty", -30.0)),
            self_ko_hp_penalty=float(getattr(reward_config, "self_ko_hp_penalty", 0.0)),
            drop_redundant_bias=bool(getattr(reward_config, "drop_redundant_bias", False)),
            drop_switch_bias=bool(getattr(reward_config, "drop_switch_bias", False)),
            all_shaping_pbrs=bool(getattr(reward_config, "all_shaping_pbrs", False)),
            stall_pbrs=bool(getattr(reward_config, "stall_pbrs", False)),
            no_progress_penalty=float(getattr(reward_config, "no_progress_penalty", 0.15)),
            use_popart=bool(policy_kwargs.get("use_popart", False)),
            attend_unrevealed_opponents=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get(
                    "attend_unrevealed_opponents", False)
            ),
            opp_belief_cls_k=int(
                policy_kwargs.get("features_extractor_kwargs", {}).get("opp_belief_cls_k", 0)
            ),
            opp_belief_slots=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get("opp_belief_slots", False)
            ),
            move_belief_mode=str(
                policy_kwargs.get("features_extractor_kwargs", {}).get("move_belief_mode", "off")
            ),
            damage_op=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get("damage_op", False)
            ),
            damage_outgoing=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get("damage_outgoing", False)
            ),
            move_candidate_floor=float(
                policy_kwargs.get("features_extractor_kwargs", {}).get("move_candidate_floor", 0.02)
            ),
            move_latent=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get("move_latent", False)
            ),
            spread_belief=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get("spread_belief", False)
            ),
            spread_belief_nature=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get("spread_belief_nature", False)
            ),
            move_prior_fusion=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get("move_prior_fusion", False)
            ),
            damage_candidate_k=int(
                policy_kwargs.get("features_extractor_kwargs", {}).get("damage_candidate_k", 0)
            ),
            consequence_topk=int(
                policy_kwargs.get("features_extractor_kwargs", {}).get("consequence_topk", 6)
            ),
            entity_topk_seats=int(
                policy_kwargs.get("features_extractor_kwargs", {}).get("entity_topk_seats", 0)
            ),
            edge_bias_families=str(
                policy_kwargs.get("features_extractor_kwargs", {}).get("edge_bias_families", "off")
            ),
            entity_tail_seats=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get("entity_tail_seats", False)
            ),
            win_prob_mode=str(
                policy_kwargs.get("features_extractor_kwargs", {}).get("win_prob_mode", "none")
            ),
            value_dist_mode=str(
                policy_kwargs.get("features_extractor_kwargs", {}).get("value_dist_mode", "none")
            ),
            value_dist_bins=int(
                policy_kwargs.get("features_extractor_kwargs", {}).get("value_dist_bins", 0)
            ),
            value_threat_inject=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get("value_threat_inject", False)
            ),
            opp_intent=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get("opp_intent", False)
            ),
            t0_species_prior=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get("t0_species_prior", False)
            ),
            opp_intent_grad_mode=str(
                policy_kwargs.get("features_extractor_kwargs", {}).get(
                    "opp_intent_grad_mode", "detached")
            ),
            intent_value_reduce=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get(
                    "intent_value_reduce", False)
            ),
            intent_move_cell=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get(
                    "intent_move_cell", False)
            ),
            value_entity_pool=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get(
                    "value_entity_pool", False)
            ),
            history_events=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get(
                    "history_events", False)
            ),
            value_entity_pool_full=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get(
                    "value_entity_pool_full", False)
            ),
            item_belief=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get(
                    "item_belief", False)
            ),
            intent_threshold=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get(
                    "intent_threshold", False)
            ),
            intent_conditional=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get(
                    "intent_conditional", False)
            ),
            op_drop_renders=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get(
                    "op_drop_renders", False)
            ),
            op_believed_lean=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get(
                    "op_believed_lean", False)
            ),
            value_clock=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get(
                    "value_clock", False)
            ),
            value_intent=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get(
                    "value_intent", False)
            ),
            species_prior_fusion=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get("species_prior_fusion", False)
            ),
            value_dist_vmin=float(
                policy_kwargs.get("features_extractor_kwargs", {}).get("value_dist_vmin", 0.0)
            ),
            value_dist_vmax=float(
                policy_kwargs.get("features_extractor_kwargs", {}).get("value_dist_vmax", 0.0)
            ),
            damage_topk_k=int(
                policy_kwargs.get("features_extractor_kwargs", {}).get("damage_topk_k", 0)
            ),
            damage_matrices_outgoing=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get("damage_matrices_outgoing", False)
            ),
            damage_matrices_incoming=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get("damage_matrices_incoming", False)
            ),
            threat_prob_outspeed=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get("threat_prob_outspeed", False)
            ),
            hp_belief_mode=str(
                policy_kwargs.get("features_extractor_kwargs", {}).get("hp_belief_mode", "composed")
            ),
            belief_grad_mode=str(
                policy_kwargs.get("features_extractor_kwargs", {}).get("belief_grad_mode", "shaping")
            ),
            value_from_dist=bool(policy_kwargs.get("value_from_dist", False)),
            hp_type_belief_coef=float(hp_type_belief_coef),
            item_belief_coef=float(item_belief_coef),
            value_tail_weight=float(value_tail_weight),
            opp_belief_aux_coef=float(opp_belief_aux_coef),
            move_belief_coef=float(move_belief_coef),
            win_prob_coef=float(win_prob_coef),
            move_belief_latent_coef=float(move_belief_latent_coef),
            spread_belief_coef=float(spread_belief_coef),
            value_dist_coef=float(value_dist_coef),
        )

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_json_file(cls, path: str) -> ModelVersion:
        with open(path) as f:
            data = json.load(f)
        data = _migrate_config(data)
        return cls(**data)

    def check_compatible(self, saved: ModelVersion) -> None:
        """Raises ModelVersionError if saved is incompatible with self (current code).
        Call as: current_version.check_compatible(saved_version).
        """
        # Architecture family — hard stop if different
        if self.arch_signature != saved.arch_signature:
            raise ModelVersionError(
                f"Architecture family mismatch: saved='{saved.arch_signature}', "
                f"current='{self.arch_signature}'.\n"
                "These models use structurally different networks and cannot be loaded interchangeably.\n"
                "Start a fresh training run, or use subprocess isolation for league play."
            )

        # Weight-relevant fields — all must match exactly
        _WEIGHT_FIELDS = {
            "total_dim", "active_context_dim",
            "species_embedding_dim", "max_species",
            "move_embedding_dim", "max_moves",
            "item_embedding_dim", "max_items",
            "ability_embedding_dim", "max_abilities",
            "type_embedding_dim", "max_types",
            "role_token_size", "projection_dim",
            "move_net_hidden", "role_encoder_hidden",
            "n_history_turns",
            "net_arch",
        }
        current = asdict(self)
        saved_d = asdict(saved)
        mismatches = [
            f"  {k}: saved={saved_d[k]!r}, current={current[k]!r}"
            for k in sorted(_WEIGHT_FIELDS)
            if current[k] != saved_d.get(k)
        ]
        if mismatches:
            raise ModelVersionError(
                "Model weight-shape mismatch — cannot load saved model with current architecture.\n"
                "Mismatched fields:\n" + "\n".join(mismatches) + "\n\n"
                "Fix: restore matching constants, or start a fresh training run."
            )

        # Feature toggle — value-checked (not weight-shape) but STRUCTURAL: PopArt adds value-head
        # buffers + normalized output, so loading a use_popart mismatch breaks the state_dict on
        # EVERY load. Unlike vf_coef / reward-config (value-meaning, resume-only) it lives here in
        # check_compatible (gates eval / pool / distill loads too), with a dedicated message.
        if self.use_popart != saved.use_popart:
            raise ModelVersionError(
                f"PopArt mismatch: saved={saved.use_popart}, current={self.use_popart}.\n"
                "PopArt changes the value head's parameterization (normalized output + running "
                "mu/sigma buffers), so it cannot be toggled on a resumed model.\n"
                "Resume with the matching --use-popart setting, or start a fresh training run."
            )

        # Behavioral toggle — value-checked (not weight-shape): unmasking the opponent's hidden
        # party changes the transformer's key_padding_mask (policy AND value forward). The state_dict
        # is identical either way, but a resume that flips it would feed the policy a different mask
        # than it trained under. Lives here (gates resume) with a dedicated message; same-run
        # pool/sentinel/distill snapshots carry the same value so they pass trivially.
        if self.attend_unrevealed_opponents != saved.attend_unrevealed_opponents:
            raise ModelVersionError(
                f"attend_unrevealed_opponents mismatch: saved={saved.attend_unrevealed_opponents}, "
                f"current={self.attend_unrevealed_opponents}.\n"
                "Unmasking the opponent's hidden party changes the transformer mask the policy was "
                "trained under, so it cannot be toggled on a resumed model.\n"
                "Resume with the matching --attend-unrevealed-opponents setting, or start a fresh run."
            )

        # Structural toggle — like use_popart it changes the state_dict (k>0 adds HiddenOppBeliefPool +
        # widens both projection Linears by k*D_MODEL), so a mismatch breaks the load. As a plain int,
        # EVERY distinct value is a weight-shape change (incl. 0↔N = adding/removing the module), so one
        # unconditional comparison gates it — no separate on/off field.
        if self.opp_belief_cls_k != saved.opp_belief_cls_k:
            raise ModelVersionError(
                f"opp_belief_cls_k mismatch: saved={saved.opp_belief_cls_k}, current={self.opp_belief_cls_k}.\n"
                "The number of hidden-opponent belief query tokens (0 = off) sets the projection width, "
                "so it is a weight-shape parameter and cannot change on an existing model.\n"
                f"Resume with --opp-belief-cls-k {saved.opp_belief_cls_k}, or start a fresh training run."
            )

        # Structural toggle — adds our_active_refined to the value projection (widens it by D_MODEL), so
        # a mismatch breaks the value head's state_dict. Like use_popart it gates EVERY load.

        # Structural toggle — ON adds the BeliefHead + per-slot unknown-mon embeddings to the
        # state_dict (the in-place hidden-opponent belief). Like use_popart it gates EVERY load; the
        # training-only opp_belief_aux_coef is deliberately NOT checked (it touches no forward pass).
        if self.opp_belief_slots != saved.opp_belief_slots:
            raise ModelVersionError(
                f"opp_belief_slots mismatch: saved={saved.opp_belief_slots}, current={self.opp_belief_slots}.\n"
                "The hidden-opponent belief-aux module (learned unknown-mon slot tokens + BeliefHead) "
                "changes the state_dict, so it cannot be toggled on an existing model.\n"
                "Resume with the matching --opp-belief-aux-coef setting, or start a fresh training run."
            )

        # Structural toggle — the MoveBelief module (move head + reinject Linear) is in the state_dict
        # AND its mode changes the trained forward (which slots are enriched). Gated as a STRING, every
        # load; the training-only move_belief_coef is NOT checked.
        if self.move_belief_mode != saved.move_belief_mode:
            raise ModelVersionError(
                f"move_belief_mode mismatch: saved={saved.move_belief_mode!r}, current={self.move_belief_mode!r}.\n"
                "The move-belief module (predict+reinject the opp moveset) changes the state_dict and the "
                "forward, so the mode cannot change on an existing model.\n"
                "Resume with the matching --move-belief-mode setting, or start a fresh training run."
            )

        # Structural toggle — the DamageOperator appends a believed-damage block to BOTH projection
        # inputs, so toggling it changes both projection Linears' shapes. Like value_active_readout it
        # gates EVERY load with a dedicated bool compare; OFF = baseline byte-for-byte.
        if self.damage_op != saved.damage_op:
            raise ModelVersionError(
                f"damage_op mismatch: saved={saved.damage_op}, current={self.damage_op}.\n"
                "The differentiable damage operator widens both projection heads, so it changes the "
                "state_dict and cannot be toggled on an existing model.\n"
                "Resume with the matching --damage-op setting, or start a fresh training run."
            )

        # Structural toggle (weight-shape, like damage_op): the OUTGOING per-move block widens both
        # projection Linears, so toggling it changes the state_dict.
        if self.damage_outgoing != saved.damage_outgoing:
            raise ModelVersionError(
                f"damage_outgoing mismatch: saved={saved.damage_outgoing}, current={self.damage_outgoing}.\n"
                "The outgoing per-move damage block widens both projection heads, so it changes the "
                "state_dict and cannot be toggled on an existing model.\n"
                "Resume with the matching --unified-damage setting, or start a fresh training run."
            )

        # v24 STRUCTURAL toggle (weight-shape, like damage_op): the MoveLatentEncoder widens the
        # move-network input, so toggling it changes the state_dict.
        if self.move_latent != saved.move_latent:
            raise ModelVersionError(
                f"move_latent mismatch: saved={saved.move_latent}, current={self.move_latent}.\n"
                "The MoveLatentEncoder concatenates a per-move latent into the move network, so it changes "
                "the state_dict and cannot be toggled on an existing model.\n"
                "Resume with the matching --move-latent setting, or start a fresh training run."
            )

        # v25 STRUCTURAL toggle (like opp_belief_slots): the SpreadBelief module adds params, so toggling
        # it changes the state_dict.
        if self.spread_belief != saved.spread_belief:
            raise ModelVersionError(
                f"spread_belief mismatch: saved={saved.spread_belief}, current={self.spread_belief}.\n"
                "The SpreadBelief module (the hidden-spread belief head) changes the state_dict and cannot "
                "be toggled on an existing model.\n"
                "Resume with the matching --spread-belief setting, or start a fresh training run."
            )

        # v40 STRUCTURAL toggle (gen3_nature_ev_belief_v1): the nature/EV generative head has DIFFERENT
        # SpreadBelief params (nature_head + ev_head vs the additive stat_head), so toggling it changes the
        # state_dict.
        if self.spread_belief_nature != saved.spread_belief_nature:
            raise ModelVersionError(
                f"spread_belief_nature mismatch: saved={saved.spread_belief_nature}, "
                f"current={self.spread_belief_nature}.\n"
                "The nature/EV generative head reparameterises SpreadBelief (its params differ from the "
                "additive head), so it changes the state_dict and cannot be toggled on an existing model.\n"
                "Resume with the matching --spread-belief-nature setting, or start a fresh training run."
            )

        # v40 FORWARD-BEHAVIOR toggle (gen3_nature_ev_belief_v1, like move_prior_fusion): no new params, but a
        # mid-run flip feeds the op a different (marginalised vs mean-field) forward.

        # v25 FORWARD-BEHAVIOR toggles (like mask_incoming_damage_obs): each zeros a now-subsumed obs region
        # from the model's view → a different forward the policy/value trained under (state_dict identical).

        # Forward-behavior float (no weight-shape change, like move_prior_fusion): the LEGAL-BUT-UNOBSERVED
        # base of the move prior. Legality itself is unconditional (v65) and not comparable — only the
        # height of this floor is a choice, and changing it changes the belief the policy/value/op trained
        # under. A pre-v65 checkpoint recorded 0.0 (which used to mean "legality OFF") and will land here
        # against the current 0.02 default: that rejection is the POINT, not a bug to migrate away.
        # NOTE (MIGRATION_FLOOR): every pre-v65 config is now refused at _migrate_config's floor
        # before it can reach this check, so the saved==0.0 message below is reachable only from a
        # hand-built v67+ config (the extractor itself refuses floors below _MIN_PRIOR_FLOOR).
        # Kept as defence in depth — behavior deliberately unchanged.
        if self.move_candidate_floor != saved.move_candidate_floor:
            raise ModelVersionError(
                f"move_candidate_floor mismatch: saved={saved.move_candidate_floor}, "
                f"current={self.move_candidate_floor}.\n"
                "This is the legal-but-unobserved base of the move prior; changing it changes the belief "
                "the policy trained under, so it cannot be changed on a resumed model.\n"
                + (
                    "saved=0.0 predates v65 (gen3_unconditional_move_legality_v1), where 0.0 meant "
                    "'no legality gate' — a prior that gave phantom mass to moves a species cannot "
                    "learn. That is no longer representable: legality is unconditional now. This "
                    "checkpoint cannot be resumed; start a fresh training run.\n"
                    if saved.move_candidate_floor == 0.0 else
                    "Resume with the matching --move-candidate-floor, or start a fresh training run.\n"
                )
            )

        # Forward-behavior toggle (no weight-shape change, like attend_unrevealed_opponents): fusing the
        # move prior changes the belief the policy/value/damage-op trained under, so a resume that flips
        # it would feed a different forward. The state_dict is identical either way (the prior buffer is
        # non-persistent), so this is a train/eval-consistency gate, not a loadability one.
        if self.move_prior_fusion != saved.move_prior_fusion:
            raise ModelVersionError(
                f"move_prior_fusion mismatch: saved={saved.move_prior_fusion}, current={self.move_prior_fusion}.\n"
                "Fusing the Smogon move prior into the belief changes the forward the policy trained "
                "under, so it cannot be toggled on a resumed model.\n"
                "Resume with the matching --move-prior-fusion setting, or start a fresh training run."
            )

        # (v32 `move_belief_prefuse` and v50 `damage_op_prefuse` had gates here. Both are DELETED at
        # v71: the PRE-transformer placement is the only one, so there is no longer a second forward to
        # be inconsistent with. A saved config that recorded either at a non-production value is
        # REFUSED by the v71 migration, which is louder than this gate was.)

        # v49 gen3_topk_candidates_v1 — FORWARD-BEHAVIOR (no weight-shape change): truncating the
        # op's candidate axis changes the damage the policy/value trained under. Unconditional int
        # compare (the `damage_topk_k` pattern), so 0<->K and K<->M both fail.
        if self.damage_candidate_k != saved.damage_candidate_k:
            raise ModelVersionError(
                f"damage_candidate_k mismatch: saved={saved.damage_candidate_k}, "
                f"current={self.damage_candidate_k}.\n"
                "Capping the DamageOperator's candidate sweep changes the incoming damage the model "
                "trained under, so it cannot be toggled on a saved model.\n"
                "Load with the matching --damage-candidate-k, or start a fresh training run."
            )

        # v51 gen3_pointer_native_v1: the pointer head is unconditional (no gate) — a pre-generation
        # checkpoint fails the ARCH_SIGNATURE family check above, which is the intended loud break.

        # v54 gen3_entity_move_seats_v1 — STRUCTURAL int (the damage_topk_k pattern): >0 adds
        # `threat_seat_proj` (state_dict) and K threat seats to every attention pass (forward), so
        # 0<->K and K<->M both fail. The unconditional E3 seats ride the ARCH_SIGNATURE, not this.
        if self.consequence_topk != saved.consequence_topk:
            raise ValueError(
                f"consequence_topk mismatch: saved={saved.consequence_topk}, "
                f"current={self.consequence_topk} — the consequence kernels' candidate axis is a "
                "forward-behavior toggle (the worst-case max covers a different candidate set); "
                "load with matching --consequence-topk."
            )
        if self.entity_topk_seats != saved.entity_topk_seats:
            raise ModelVersionError(
                f"entity_topk_seats mismatch: saved={saved.entity_topk_seats}, "
                f"current={self.entity_topk_seats}.\n"
                "The E4 threat-move seats add a projection (threat_seat_proj) and change every "
                "attention pass's seat count, so the weights are not interchangeable.\n"
                "Load with the matching --entity-topk-seats, or start a fresh training run."
            )

        # v56 gen3_edge_bias_trunk_v1 — STRUCTURAL str (the win_prob_mode pattern): a family adds its
        # zero-init bias map (state_dict) and its cells to every attention pass (forward), so any
        # mismatch — off<->on or a different family set — fails.
        if self.edge_bias_families != saved.edge_bias_families:
            raise ModelVersionError(
                f"edge_bias_families mismatch: saved={saved.edge_bias_families!r}, "
                f"current={self.edge_bias_families!r}.\n"
                "The edge-bias families add per-family map parameters and change the attention "
                "biases the model trained under, so the weights are not interchangeable.\n"
                "Load with the matching --edge-bias-families, or start a fresh training run."
            )

        # v57 gen3_entity_tail_seats_v1 — STRUCTURAL bool: adds tail_proj/tail_marker (state_dict)
        # and 6 seats to every attention pass (forward).
        if self.entity_tail_seats != saved.entity_tail_seats:
            raise ModelVersionError(
                f"entity_tail_seats mismatch: saved={saved.entity_tail_seats}, "
                f"current={self.entity_tail_seats}.\n"
                "The E5 tail-threat seats add parameters and change every attention pass's seat "
                "count, so the weights are not interchangeable.\n"
                "Load with the matching --entity-tail-seats, or start a fresh training run."
            )

        # Structural + resume-IMMUTABLE toggle — gated as a STRING so BOTH 'none'↔head (a state_dict
        # change: the WinProbHead params) AND read_only↔shaping (same params, but flipping the trunk
        # gradient flow mid-run is a silent training change the user chose to forbid) FATAL on a
        # mismatch. Like move_belief_mode it gates EVERY load; same-run pool/sentinel snapshots carry the
        # identical mode so they pass trivially. The training-only win_prob_coef is NOT checked.
        # gen3_hp_belief_ablation_v1 (v53, like win_prob_mode): 'composed' builds HPTypeBelief and
        # 'flat' does not (a state_dict change), and the two produce different typed-HP posteriors (a
        # forward change). A STRING compare gates both. The training-only hp_type_belief_coef is NOT
        # checked.
        if self.hp_belief_mode != saved.hp_belief_mode:
            raise ModelVersionError(
                f"hp_belief_mode mismatch: saved={saved.hp_belief_mode!r}, "
                f"current={self.hp_belief_mode!r}.\n"
                "How the opponent's typed Hidden-Power belief is produced is fixed for a run's "
                "lifetime: 'composed' adds the HPTypeBelief head and the presence x type "
                "factorisation, 'flat' predicts the 16 typed channels independently.\n"
                "Resume with the matching --hp-belief-mode, or start a fresh training run."
            )
        if self.win_prob_mode != saved.win_prob_mode:
            raise ModelVersionError(
                f"win_prob_mode mismatch: saved={saved.win_prob_mode!r}, current={self.win_prob_mode!r}.\n"
                "The win-probability head is fixed for a run's lifetime: adding/removing it changes the "
                "state_dict, and switching read_only↔shaping flips whether its loss shapes the shared "
                "trunk (a silent mid-run training change).\n"
                "Resume with the matching --win-prob-mode setting, or start a fresh training run."
            )


        # v29 distributional VALUE head (like win_prob_mode): the MODE gates none↔head (the
        # ValueDistHead params) AND read_only↔shaping (grad-flow); the BIN COUNT is the head's output
        # Linear width. Both are weight-shape/forward changes → FATAL on a resume mismatch. The support
        # (vmin/vmax) is value-meaning → resume-only check_value_dist, not here.
        # (gen3_seed_quantile_v1's `seed_quantile` gate is DELETED at v78 with the head itself.)
        # gen3_value_threat_inject_v1 (v64): the injection projection is a state_dict-changing
        # module, AND the flag switches the op's reducer on, so a flip is doubly incompatible.
        if self.value_threat_inject != saved.value_threat_inject:
            raise ModelVersionError(
                f"value_threat_inject mismatch: saved={saved.value_threat_inject}, "
                f"current={self.value_threat_inject}.\n"
                "The critic's threat-injection projection is fixed for a run's lifetime: adding or "
                "removing it changes the state_dict, and the flag also switches the DamageOperator's "
                "pair reduction from hard_max to belief_mean — so a mid-run flip would change what "
                "the critic reads AND which modules exist.\n"
                "Resume with the matching --value-threat-inject setting, or start a fresh training run."
            )
        # gen3_opp_intent_v1 (v68): the alpha/beta pointer heads are state_dict-changing modules.
        if self.opp_intent != saved.opp_intent:
            raise ModelVersionError(
                f"opp_intent mismatch: saved={saved.opp_intent}, current={self.opp_intent}.\n"
                "The opponent-intent heads are fixed for a run's lifetime: adding or removing them "
                "changes the state_dict.\n"
                "Resume with the matching --opp-intent-coef setting, or start a fresh training run."
            )
        # gen3_t0_species_prior_v1 (v72): the state_dict is IDENTICAL either way (the co-occurrence
        # tables are non-persistent buffers and the module has no parameters), so — exactly as with
        # species_prior_fusion below — this compare is the ONLY thing that can reject a mid-run flip.
        # Nothing about the shapes would ever complain, while every unrevealed-defender damage number
        # the policy and critic were trained against would silently change under them.
        # gen3_intent_grad_mode_v1 (v73): flipping this mid-run changes what the shared trunk is
        # being trained to do, with no shape anywhere to notice.
        if self.intent_value_reduce != saved.intent_value_reduce:
            raise ModelVersionError(
                f"intent_value_reduce mismatch: saved={saved.intent_value_reduce}, "
                f"current={self.intent_value_reduce}.\n"
                "Step 6 widens the critic's pre-projection features, so it is fixed for a run's "
                "lifetime.\nResume with the matching --intent-value-reduce, or start a fresh run."
            )
        # gen3_intent_move_cell_v1 (v77): widens the pointer move scorer's in_features (a policy
        # state_dict change), so a mismatch would be shape-caught — this names the cause instead.
        if self.intent_move_cell != saved.intent_move_cell:
            raise ModelVersionError(
                f"intent_move_cell mismatch: saved={saved.intent_move_cell}, "
                f"current={self.intent_move_cell}.\n"
                "The G3 alpha-conditioned c2 move-cell channels widen the pointer move scorer, "
                "so the flag is fixed for a run's lifetime.\n"
                "Resume with the matching --intent-move-cell, or start a fresh run."
            )
        # gen3_unified_value_readout_v1 (v80): widens the value projection (a state_dict change),
        # so a mismatch would be shape-caught — this names the cause instead.
        if self.value_entity_pool != saved.value_entity_pool:
            raise ModelVersionError(
                f"value_entity_pool mismatch: saved={saved.value_entity_pool}, "
                f"current={self.value_entity_pool}.\n"
                "The unified critic entity pool widens the value projection, so the flag is "
                "fixed for a run's lifetime.\n"
                "Resume with the matching --value-entity-pool, or start a fresh run."
            )
        # gen3_unified_value_readout_v2 (v82): grows the pool's source table (state_dict).
        if self.value_entity_pool_full != saved.value_entity_pool_full:
            raise ModelVersionError(
                f"value_entity_pool_full mismatch: saved={saved.value_entity_pool_full}, "
                f"current={self.value_entity_pool_full}.\n"
                "The full row set grows the pool's source-embedding table, so the flag is "
                "fixed for a run's lifetime.\n"
                "Resume with the matching --value-entity-pool-full, or start a fresh run."
            )
        # gen3_item_belief_v1 (v83): builds the ItemBelief module (a state_dict change).
        if self.item_belief != saved.item_belief:
            raise ModelVersionError(
                f"item_belief mismatch: saved={saved.item_belief}, "
                f"current={self.item_belief}.\n"
                "The item-belief head adds trunk modules, so the flag is fixed for a run's "
                "lifetime.\n"
                "Resume with the matching --item-belief, or start a fresh run."
            )
        # gen3_intent_threshold_v1 (v84): two zero-init projections + width changes (state_dict).
        if self.intent_threshold != saved.intent_threshold:
            raise ModelVersionError(
                f"intent_threshold mismatch: saved={saved.intent_threshold}, "
                f"current={self.intent_threshold}.\n"
                "The threshold operator widens the pointer move cell and the critic, so the "
                "flag is fixed for a run's lifetime.\n"
                "Resume with the matching --intent-threshold, or start a fresh run."
            )
        # gen3_intent_conditional_v1 (v85): a zero-init projection + a pointer-cell width change.
        if self.intent_conditional != saved.intent_conditional:
            raise ModelVersionError(
                f"intent_conditional mismatch: saved={saved.intent_conditional}, "
                f"current={self.intent_conditional}.\n"
                "The mechanic cells widen the pointer move cell, so the flag is fixed for a "
                "run's lifetime.\n"
                "Resume with the matching --intent-conditional, or start a fresh run."
            )
        # gen3_op_lean_forward_v1 (v86): out_gain shape / d3 forward math.
        if self.op_drop_renders != saved.op_drop_renders:
            raise ModelVersionError(
                f"op_drop_renders mismatch: saved={saved.op_drop_renders}, "
                f"current={self.op_drop_renders}.\n"
                "The lean forward block shrinks out_gain, so the flag is fixed for a run's "
                "lifetime.\nResume with the matching --op-drop-renders, or start a fresh run."
            )
        if self.op_believed_lean != saved.op_believed_lean:
            raise ModelVersionError(
                f"op_believed_lean mismatch: saved={saved.op_believed_lean}, "
                f"current={self.op_believed_lean}.\n"
                "The believed-lean d3 physics are a forward-math change with no shape, so this "
                "gate is the ONLY thing that rejects a mismatched resume.\n"
                "Resume with the matching --op-believed-lean, or start a fresh run."
            )
        # gen3_value_direct_routes_v1 (v87): both widen the value projection (state_dict).
        if self.value_clock != saved.value_clock:
            raise ModelVersionError(
                f"value_clock mismatch: saved={saved.value_clock}, "
                f"current={self.value_clock}.\n"
                "The direct clock route widens the value projection, so the flag is fixed for "
                "a run's lifetime.\nResume with the matching --value-clock, or start a fresh run."
            )
        if self.value_intent != saved.value_intent:
            raise ModelVersionError(
                f"value_intent mismatch: saved={saved.value_intent}, "
                f"current={self.value_intent}.\n"
                "The direct intent route widens the value projection, so the flag is fixed for "
                "a run's lifetime.\nResume with the matching --value-intent, or start a fresh run."
            )
        # gen3_event_window_v1 (v81): builds the EventSeats consumer (a state_dict change).
        if self.history_events != saved.history_events:
            raise ModelVersionError(
                f"history_events mismatch: saved={saved.history_events}, "
                f"current={self.history_events}.\n"
                "The H-B event seats add trunk modules, so the flag is fixed for a run's "
                "lifetime.\nResume with the matching --history-events, or start a fresh run."
            )
        if self.opp_intent_grad_mode != saved.opp_intent_grad_mode:
            raise ModelVersionError(
                f"opp_intent_grad_mode mismatch: saved={saved.opp_intent_grad_mode!r}, "
                f"current={self.opp_intent_grad_mode!r}.\n"
                "Whether the opponent-intent objective shapes the trunk is fixed for a run's "
                "lifetime.\nResume with the matching --opp-intent-grad-mode, or start a fresh run."
            )
        if self.t0_species_prior != saved.t0_species_prior:
            raise ModelVersionError(
                f"t0_species_prior mismatch: saved={saved.t0_species_prior}, "
                f"current={self.t0_species_prior}.\n"
                "The T0 species belief is fixed for a run's lifetime: it decides whether the physics "
                "prices an unrevealed opponent from the model's own team-composition belief or from "
                "the static gen3ou usage prior. Flipping it re-means every damage number against a "
                "hidden slot.\n"
                "Resume with the matching --t0-species-prior setting, or start a fresh training run."
            )
        # gen3_species_prior_fusion_v1 (v69): the state_dict is IDENTICAL either way (the co-occurrence
        # tables are non-persistent buffers), so this compare is the ONLY thing standing between a
        # resume and a silently re-meant species head — ON reads the head's output as a DELTA on the
        # team-composition prior, OFF reads the same numbers as the whole prediction.
        if self.species_prior_fusion != saved.species_prior_fusion:
            raise ModelVersionError(
                f"species_prior_fusion mismatch: saved={saved.species_prior_fusion}, "
                f"current={self.species_prior_fusion}.\n"
                "The species belief's prior fusion is fixed for a run's lifetime: flipping it changes "
                "what the species head's output MEANS (delta-on-prior vs. the full prediction), and "
                "nothing in the weights would catch it.\n"
                "Resume with the matching --species-prior-fusion setting, or start a fresh training run."
            )
        if self.value_dist_mode != saved.value_dist_mode:
            raise ModelVersionError(
                f"value_dist_mode mismatch: saved={saved.value_dist_mode!r}, current={self.value_dist_mode!r}.\n"
                "The distributional value head is fixed for a run's lifetime: adding/removing it changes "
                "the state_dict, and switching read_only↔shaping flips whether its loss shapes the shared "
                "trunk (a silent mid-run training change).\n"
                "Resume with the matching --value-dist-mode setting, or start a fresh training run."
            )
        if self.value_dist_bins != saved.value_dist_bins:
            raise ModelVersionError(
                f"value_dist_bins mismatch: saved={saved.value_dist_bins}, current={self.value_dist_bins}.\n"
                "The atom count is the value-dist head's output width — a different N is a weight-shape "
                "change.\n"
                "Resume with the matching --value-dist-bins setting, or start a fresh training run."
            )
        # gen3_unified_topk_incoming_v1 (v30): the discrete incoming move-space K scales the
        # DamageOperator out_dim → both projection in_features. Every distinct K (incl. 0↔N = adding/
        # removing the block) is a weight-shape change → a single unconditional int compare gates it
        # (like opp_belief_cls_k / value_dist_bins).
        if self.damage_topk_k != saved.damage_topk_k:
            raise ModelVersionError(
                f"damage_topk_k mismatch: saved={saved.damage_topk_k}, current={self.damage_topk_k}.\n"
                "The top-K incoming block's K is the number of opp moves surfaced — it scales the damage "
                "operator's output (hence both projection widths), so any change is a weight-shape "
                "mismatch.\n"
                "Resume with the matching --damage-topk setting, or start a fresh training run."
            )
        # gen3_per_move_matrices_v1 (v32): the outgoing per-move damage matrix widens the op out_dim → both
        # projection in_features. Toggling it is a weight-shape change (like damage_op).
        if self.damage_matrices_outgoing != saved.damage_matrices_outgoing:
            raise ModelVersionError(
                f"damage_matrices_outgoing mismatch: saved={saved.damage_matrices_outgoing}, "
                f"current={self.damage_matrices_outgoing}.\n"
                "The outgoing per-move damage matrix widens the damage operator's output (hence both "
                "projection widths), so toggling it is incompatible with a saved checkpoint.\n"
                "Resume with the matching --damage-matrices setting, or start a fresh training run."
            )
        # gen3_per_move_matrices_v1 (v33): the incoming per-move matrix widens the op out_dim → both
        # projection in_features (and supersedes topk). Toggling it is a weight-shape change (like damage_op).
        if self.damage_matrices_incoming != saved.damage_matrices_incoming:
            raise ModelVersionError(
                f"damage_matrices_incoming mismatch: saved={saved.damage_matrices_incoming}, "
                f"current={self.damage_matrices_incoming}.\n"
                "The incoming per-move damage matrix widens the damage operator's output (hence both "
                "projection widths), so toggling it is incompatible with a saved checkpoint.\n"
                "Resume with the matching --damage-matrices setting, or start a fresh training run."
            )
        # gen3_bidir_threat_trunk_v1 (v36): the uncertainty-aware P(outspeed) is a version-gated
        # forward-behavior toggle — fresh-only.
        if self.threat_prob_outspeed != saved.threat_prob_outspeed:
            raise ModelVersionError(
                f"threat_prob_outspeed mismatch: saved={saved.threat_prob_outspeed}, "
                f"current={self.threat_prob_outspeed}.\n"
                "It changes the P(outspeed) forward (uncertainty-aware scale), a version-checked "
                "forward-behavior change. Resume with the matching flag, or start a fresh run."
            )

    def check_opponent_compatible(self, foreign: "ModelVersion") -> None:
        """Gate for loading a frozen model from ANOTHER run as an inference-only OPPONENT
        (a "stable opponent"). Call as: ``current_version.check_opponent_compatible(foreign)``.

        A stable opponent is a pure ``observation -> action`` function: it consumes the obs the
        LIVE encoder produces and emits an action index that crosses into the shared battle. So the
        ONLY axis that must match is the OBSERVATION FAMILY — and ``arch_signature`` is the proxy
        for it: any obs-layout/meaning change bumps the signature, so equal signatures guarantee the
        same obs layout. (It ALSO bumps on pure network-structure refactors, making this stricter
        than strictly necessary — but in a safe direction, and same-arch ⟹ identical net sizes, so
        the foreign zip rebuilds its extractor at shapes matching its own weights with no further
        check needed. If an obs-identical-but-model-refactored opponent is ever wanted, split a
        dedicated ``obs_signature`` out of ``arch_signature`` and gate on that instead.)

        Deliberately DISTINCT from ``check_compatible`` (which gates the trainee's own resume + the
        self-play pool/sentinels, where every ``_WEIGHT_FIELD`` AND ``use_popart`` must match): an
        opponent never shares weights with the trainee and never reads its value head, so
        ``use_popart`` / ``vf_coef`` / the reward-config hparams are all irrelevant to its forward
        and are deliberately NOT checked here.
        """
        if self.arch_signature != foreign.arch_signature:
            raise ModelVersionError(
                f"Stable opponent architecture-family mismatch: "
                f"opponent='{foreign.arch_signature}', current='{self.arch_signature}'.\n"
                "A stable opponent must share the live run's arch_signature — i.e. the SAME "
                "observation layout (a different signature means the live encoder cannot feed it).\n"
                "Use an opponent trained at the current architecture, or start the new run at the "
                "opponent's architecture."
            )
        # Defensive: same arch_signature already implies these match, but a hand-edited config
        # could lie — and feeding the opponent a wrong-width obs would be a silent-garbage bug.
        for field in ("total_dim", "active_context_dim"):
            cur, opp = getattr(self, field), getattr(foreign, field)
            if cur != opp:
                raise ModelVersionError(
                    f"Stable opponent {field} mismatch: opponent={opp}, current={cur} "
                    "(arch_signature matched — the opponent's model_config.json looks hand-edited)."
                )

    def check_vf_coef(self, requested: float) -> None:
        """Raise ModelVersionError if `requested` (the resume `--vf-coef`) differs from this
        saved config's vf_coef.

        Call as: saved_version.check_vf_coef(args.vf_coef).

        vf_coef is a training-loss coefficient, not a weight-shape concern, so it is
        deliberately NOT part of check_compatible() — that gates EVERY checkpoint load,
        including the frozen eval / self-play-pool / distill opponents, where vf_coef is
        irrelevant (the forward pass is identical regardless of it). This check is invoked
        ONLY on the training-resume path: silently changing the value head's gradient scale
        mid-run would let a forgotten/typo'd flag drift training, so a resume with a
        different value is a hard error rather than a quiet change.
        """
        if not math.isclose(self.vf_coef, requested, rel_tol=1e-9, abs_tol=1e-12):
            raise ModelVersionError(
                f"vf_coef mismatch: saved={self.vf_coef!r}, requested={requested!r}.\n"
                "The PPO value-loss coefficient is fixed for the lifetime of a run — changing it on "
                "resume silently alters the value head's gradient scale.\n"
                f"Fix: resume with --vf-coef {self.vf_coef!r}, or start a fresh training run to use "
                f"{requested!r}."
            )

    def check_belief_grad_mode(self, requested: str, allow_change: bool = False) -> None:
        """Raise ModelVersionError if `requested` (the resume `--belief-grad-mode`) differs from this
        saved config's belief_grad_mode. Call as: saved_version.check_belief_grad_mode(args.belief_grad_mode).

        gen3_belief_grad_mode_v1: detach() is value-preserving, so the FORWARD (eval / inference / a frozen
        pool / distill opponent) is bit-identical regardless of the mode — only the TRAINING gradient (does
        the belief reshape the trunk) differs. So, like vf_coef, it is EXCLUDED from check_compatible (gating
        a frozen opponent on it would be a false rejection that breaks self-play) and enforced ONLY on the
        training-resume path: flipping shaping↔detached mid-run silently changes whether the belief
        gradient shapes the shared trunk, so a drift is a hard error rather than a quiet change.

        ``allow_change=True`` (--allow-belief-grad-mode-change) is the INTENTIONAL-migration escape hatch:
        because detach() is value-preserving, flipping the mode on a converged checkpoint is weight-safe —
        the gate exists to prevent ACCIDENTAL drift, not because the transition is unsound. A permitted
        mismatch prints a loud notice; the next checkpoint save records the new mode, so the flag is only
        needed once per migration (the staged shaping-flip experiment, next_run_plan item 5)."""
        if self.belief_grad_mode != requested:
            if allow_change:
                print(
                    f"[ModelVersion] NOTICE: belief_grad_mode MIGRATION {self.belief_grad_mode!r} -> "
                    f"{requested!r} (--allow-belief-grad-mode-change). Forward is bit-identical; "
                    + _BELIEF_GRAD_MODE_EFFECT.get(requested, "the belief gradient routing changed.")
                    + " The next checkpoint save records the new mode."
                )
                return
            raise ModelVersionError(
                f"belief_grad_mode mismatch: saved={self.belief_grad_mode!r}, requested={requested!r}.\n"
                "Whether the belief heads reshape the shared trunk is fixed for a run's lifetime — flipping "
                "it on resume silently changes the training signal.\n"
                f"Fix: resume with --belief-grad-mode {self.belief_grad_mode}, pass "
                "--allow-belief-grad-mode-change for an intentional migration, or start a fresh run."
            )

    def check_value_from_dist(self, requested: bool, allow_change: bool = False) -> None:
        """Raise ModelVersionError if `requested` (the resume `--value-from-dist`) differs from this
        saved config's value_from_dist. gen3_dist_critic_v1 (Phase B): swapping the GAE/bootstrap value
        source between the scalar value_net and the distributional E[Z] silently changes the training
        objective, so — like belief_grad_mode/vf_coef — a mid-run drift is a hard error, enforced ONLY on
        the training-resume path (a frozen opponent's ACTION selection is unchanged, so it's EXCLUDED from
        check_compatible). ``allow_change=True`` (--allow-value-from-dist-change) is the intentional
        warm-start-migration hatch (the offline probe confirmed E[Z]≈V, so the swap is near-seamless);
        it prints a loud notice and the next save records the new mode."""
        if bool(self.value_from_dist) != bool(requested):
            if allow_change:
                print(
                    f"[ModelVersion] NOTICE: value_from_dist MIGRATION {self.value_from_dist} -> {requested} "
                    "(--allow-value-from-dist-change). The GAE/bootstrap critic is now "
                    + ("the distributional E[Z] (scalar value_net frozen as fallback)." if requested
                       else "the scalar value_net.")
                    + " The next checkpoint save records the new mode."
                )
                return
            raise ModelVersionError(
                f"value_from_dist mismatch: saved={self.value_from_dist}, requested={requested}.\n"
                "Whether the critic is the scalar value_net or the distributional E[Z] is fixed for a run's "
                "lifetime — flipping it on resume silently changes the value objective + GAE source.\n"
                f"Fix: resume with --value-from-dist={self.value_from_dist}, pass "
                "--allow-value-from-dist-change for the intentional Phase-B migration, or start a fresh run."
            )

    def check_value_tail_weight(self, requested: float) -> None:
        """Raise ModelVersionError if `requested` (the resume `--value-tail-weight`) differs from this
        saved config's value_tail_weight. Call as: saved_version.check_value_tail_weight(args...).

        Same treatment as check_vf_coef: a value-loss hparam (the CVaR-blend weight), not weight-shape,
        so it is EXCLUDED from check_compatible (frozen eval/pool/distill opponents never run the value
        loss) and enforced ONLY on the training-resume path. Changing it mid-run silently reshapes the
        value objective (how hard the critic chases its tail), so a drift is a hard error."""
        if not math.isclose(self.value_tail_weight, requested, rel_tol=1e-9, abs_tol=1e-12):
            raise ModelVersionError(
                f"value_tail_weight mismatch: saved={self.value_tail_weight!r}, requested={requested!r}.\n"
                "The tail-weighted value-loss β is fixed for a run's lifetime — changing it on resume "
                "silently reshapes the value objective.\n"
                f"Fix: resume with --value-tail-weight {self.value_tail_weight!r}, or start a fresh run."
            )

    def check_value_dist(self, vmin: float, vmax: float) -> None:
        """Raise ModelVersionError if the resume `--value-dist-vmin/--value-dist-vmax` differ from this
        saved config's support. Call as: saved_version.check_value_dist(args.value_dist_vmin, ...).

        Same treatment as check_value_tail_weight: the atom support is VALUE-meaning (it is what the
        head's logits are read against — the loss target and the prober's atoms→return mapping), not
        weight-shape (the atoms buffer is non-persistent), so it is EXCLUDED from check_compatible
        (frozen eval/pool/distill opponents never read the value-dist head) and enforced ONLY on the
        training-resume path. Shifting the support mid-run silently re-targets the head."""
        problems = []
        if not math.isclose(self.value_dist_vmin, vmin, rel_tol=1e-9, abs_tol=1e-12):
            problems.append(f"vmin saved={self.value_dist_vmin!r} requested={vmin!r}")
        if not math.isclose(self.value_dist_vmax, vmax, rel_tol=1e-9, abs_tol=1e-12):
            problems.append(f"vmax saved={self.value_dist_vmax!r} requested={vmax!r}")
        if problems:
            raise ModelVersionError(
                "value_dist support mismatch: " + "; ".join(problems) + ".\n"
                "The distributional value head's atom support is fixed for a run's lifetime — changing "
                "it on resume silently re-targets the head.\n"
                f"Fix: resume with --value-dist-vmin {self.value_dist_vmin!r} --value-dist-vmax "
                f"{self.value_dist_vmax!r}, or start a fresh run."
            )

    def check_reward_config(self, reward_config: Any) -> None:
        """Raise ModelVersionError if the resume `reward_config` differs from this saved config's
        reward hparams (bias_additivity / mat_alive_weight / bias_redesign). Like check_vf_coef:
        these are VALUE-meaning (changing them mid-run silently shifts the reward), NOT weight-shape,
        so they are enforced ONLY on the training-resume path and excluded from check_compatible().
        Call as: saved_version.check_reward_config(args_reward_config)."""
        req_ba = float(getattr(reward_config, "bias_additivity", 1.0))
        req_maw = float(getattr(reward_config, "mat_alive_weight", 1.25))
        req_br = bool(getattr(reward_config, "bias_redesign", False))
        req_sbw = float(getattr(reward_config, "switch_bias_weight", 0.0))
        req_dp = float(getattr(reward_config, "draw_penalty", -30.0))
        req_skp = float(getattr(reward_config, "self_ko_hp_penalty", 0.0))
        req_drb = bool(getattr(reward_config, "drop_redundant_bias", False))
        req_dsb = bool(getattr(reward_config, "drop_switch_bias", False))
        req_asp = bool(getattr(reward_config, "all_shaping_pbrs", False))
        req_sp = bool(getattr(reward_config, "stall_pbrs", False))
        req_npp = float(getattr(reward_config, "no_progress_penalty", 0.15))
        problems = []
        if not math.isclose(self.bias_additivity, req_ba, rel_tol=1e-9, abs_tol=1e-12):
            problems.append(f"  bias_additivity: saved={self.bias_additivity!r}, requested={req_ba!r}")
        if not math.isclose(self.mat_alive_weight, req_maw, rel_tol=1e-9, abs_tol=1e-12):
            problems.append(f"  mat_alive_weight: saved={self.mat_alive_weight!r}, requested={req_maw!r}")
        if self.bias_redesign != req_br:
            problems.append(f"  bias_redesign: saved={self.bias_redesign!r}, requested={req_br!r}")
        if not math.isclose(self.switch_bias_weight, req_sbw, rel_tol=1e-9, abs_tol=1e-12):
            problems.append(f"  switch_bias_weight: saved={self.switch_bias_weight!r}, requested={req_sbw!r}")
        if not math.isclose(self.draw_penalty, req_dp, rel_tol=1e-9, abs_tol=1e-12):
            problems.append(f"  draw_penalty: saved={self.draw_penalty!r}, requested={req_dp!r}")
        if not math.isclose(self.self_ko_hp_penalty, req_skp, rel_tol=1e-9, abs_tol=1e-12):
            problems.append(f"  self_ko_hp_penalty: saved={self.self_ko_hp_penalty!r}, requested={req_skp!r}")
        if self.drop_redundant_bias != req_drb:
            problems.append(f"  drop_redundant_bias: saved={self.drop_redundant_bias!r}, requested={req_drb!r}")
        if self.drop_switch_bias != req_dsb:
            problems.append(f"  drop_switch_bias: saved={self.drop_switch_bias!r}, requested={req_dsb!r}")
        if self.all_shaping_pbrs != req_asp:
            problems.append(f"  all_shaping_pbrs: saved={self.all_shaping_pbrs!r}, requested={req_asp!r}")
        if self.stall_pbrs != req_sp:
            problems.append(f"  stall_pbrs: saved={self.stall_pbrs!r}, requested={req_sp!r}")
        if not math.isclose(self.no_progress_penalty, req_npp, rel_tol=1e-9, abs_tol=1e-12):
            problems.append(f"  no_progress_penalty: saved={self.no_progress_penalty!r}, requested={req_npp!r}")
        if problems:
            raise ModelVersionError(
                "Reward-config mismatch on resume — these hparams are fixed for a run's lifetime "
                "(changing them silently shifts the reward / objective):\n" + "\n".join(problems) +
                "\n\nFix: resume with the saved values, or start a fresh run."
            )


def _migrate_config(data: dict) -> dict:
    """Apply incremental forward-migrations to bring an old config up to the current schema.

    Configs older than MIGRATION_FLOOR (v67 — the first version stamped with the current
    ARCH_SIGNATURE) are REFUSED outright rather than migrated: every loader that consumes a
    migrated config gates on `arch_signature` immediately afterwards (snapshot.py's load paths,
    the fixed-opponent pool, train resume), so a pre-floor migration could only ever produce a
    config that check_compatible rejects a moment later. The v2..v66 branches that used to live
    here were therefore dead code; the history they recorded is preserved in the comment block
    below (and, in narrative form, in the version-history comments above MODEL_CONFIG_VERSION
    and in designs/CHANGELOG.md).
    """
    version = data.get("config_version", 1)
    if version < MIGRATION_FLOOR:
        raise ModelVersionError(
            f"config_version {version} is a PRE-GENERATION checkpoint: it predates "
            f"v{MIGRATION_FLOOR}, the first MODEL_CONFIG_VERSION stamped with the current "
            f"ARCH_SIGNATURE ({ARCH_SIGNATURE!r}). A checkpoint from an earlier generation "
            "cannot be loaded by this code — its weights were trained against an architecture "
            "this codebase no longer contains, and no config migration can bridge that.\n"
            "To re-probe it, use the git_hash recorded in the checkpoint's own metadata.json "
            "(git checkout <git_hash> and probe from there — the prober prints exactly this "
            "diagnosis, with the hash, for archived runs)."
        )
    # ----------------------------------------------------------------------------------------
    # PRE-FLOOR MIGRATION HISTORY (v2–v66) — documentation, not code. The executable branches
    # were deleted when MIGRATION_FLOOR landed. What each one injected (setdefault) or removed
    # (POP), verbatim from the deleted code:
    #   v2:  n_history_turns=1 (old models used a single TurnDelta).
    #   v3:  vf_coef=0.5 (the SB3 default every pre-flag run trained with).
    #   v4:  reward-config hparams — bias_additivity=1.0, mat_alive_weight=1.25,
    #        bias_redesign=False (pre-flag runs used the single-variable defaults).
    #   v5:  switch_bias_weight=0.0 (the switch-bias lever; absent = OFF).
    #   v6:  use_popart=False.
    #   v7:  draw_penalty=-30.0 (old runs scored a tie/timeout as a decisive loss).
    #   v8:  attend_unrevealed_opponents=False (old models key-masked unrevealed opp slots).
    #   v9:  opp_belief_cls_k=0 (no belief module); POPped the interim never-shipped
    #        `opp_belief_cls` bool.
    #   v10: value_active_readout=False (old value heads did not read the active-mon token).
    #   v11: value_tail_weight=0.0 (plain MSE value loss).
    #   v12: self_ko_hp_penalty=0.0 (the symmetric material PBRS priced a healthy
    #        Explosion/Self-Destruct trade at ~0).
    #   v13: drop_redundant_bias=False, drop_switch_bias=False (old runs kept every BIAS term).
    #   v14: all_shaping_pbrs=False, no_progress_penalty=0.15 (end-state PBRS switch).
    #   v15: stall_pbrs=False (the companion switch).
    #   v16: opp_belief_slots=False, opp_belief_aux_coef=0.0 (in-place hidden-opp belief-aux).
    #   v17: move_belief_mode="off", move_belief_coef=0.0 (no MoveBelief module).
    #   v18: opp_belief_latent=False, opp_belief_latent_coef=0.0 (BeliefHead latent predictor).
    #   v19: damage_op=False (no DamageOperator).
    #   v20: move_prior_fusion=False (no unified-move-belief prior fusion).
    #   v21: stamp — the incoming-damage-obs ablation toggle (its field went at v48).
    #   v22: win_prob_mode="none", win_prob_coef=1.0 (the tri-state win-probability head).
    #   v23: damage_outgoing=False, move_candidate_floor=0.0 (incoming-only op + the un-gated
    #        0.02-floor-less legacy prior).
    #   v24: move_latent=False, move_belief_latent_coef=0.0 (MoveLatentEncoder).
    #   v25: spread_belief=False, spread_belief_coef=0.0.
    #   v26: stamp — gen3_unified_op_physics_v1 (op physics parity: boosts/burn/weather/para/
    #        fixed-damage, intrinsic to damage_op; values-only).
    #   v27: stamp — gen3_unified_status_landing_v1: the op's OUTGOING direction gained the
    #        per-OUR-move status-landing block, Leech Seed's Grass immunity, Sleep Clause and
    #        Substitute-blocks-status. Intrinsic to damage_outgoing (out_dim grew, so a v26
    #        damage_outgoing checkpoint failed the SB3 load_state_dict projection in_features —
    #        the projection input dim is runtime-discovered, not a ModelVersion field).
    #   v28: stamp — gen3_unified_choice_band_v1: the op priced Choice Band (our known CB ×1.5
    #        physical Atk deterministically OUTGOING; a DECORRELATED CB-conditional physical tail
    #        [phys_high_cb + P(OHKO|CB)] + shared p_cb INCOMING — OHKO is a nonlinear threshold a
    #        mean-field ×(1+0.5·p_cb) would blur). Intrinsic to damage_op.
    #   v29: value_dist_mode="none", value_dist_bins=0, value_dist_vmin=0.0, value_dist_vmax=0.0,
    #        value_dist_coef=1.0 (the distributional value head, Phase A).
    #   v30: damage_topk_k=0 (gen3_unified_topk_incoming_v1 — the discrete top-K incoming block;
    #        STRUCTURAL int gated in check_compatible).
    #   v31: stamp — `damage_reattend` added (deleted at v71; no setdefault, so an ABSENT key
    #        stayed absent and the v71 judge only saw RECORDED values).
    #   v32: stamp — `move_belief_prefuse` added (same v71 treatment).
    #   v33: stamp — gen3_iterative_damage_v1: `damage_refine_rounds` (deleted at v70).
    #   v34: damage_matrices_outgoing=False (gen3_per_move_matrices_v1, our 4 moves × opp active
    #        + revealed bench).
    #   v35: damage_matrices_incoming=False (the incoming per-move matrix over the enriched
    #        top-K; reuses damage_topk_k as its K).
    #   v36: threat_prob_outspeed=False (gen3_bidir_threat_trunk_v1; its siblings
    #        threat_refine_outgoing / threat_unrevealed_outgoing were deleted at v70).
    #   v37: stamp — gen3_status_trunk_v1: `threat_status_refine` (deleted at v70).
    #   v38: hp_type_belief_coef=0.0 (gen3_opp_hp_type_belief_v1; the v38 mode key was POPped by
    #        the v52 migration).
    #   v39: damage_matrices_outgoing_all=False (the TRANSPOSED outgoing matrix — our 6 mons'
    #        moves → opp active).
    #   v40: spread_belief_nature=False (the SpreadBelief nature/EV generative head).
    #   v41: belief_grad_mode="shaping" (gen3_belief_grad_mode_v1; detach() is value-preserving,
    #        so 'shaping' reproduced the v40 forward AND backward byte-for-byte).
    #   v42: stamp — turn-history depth cut N_HISTORY_TURNS 10 → 7 (obs 3469 → 2992); the
    #        obs-dim weight-field check carried the rejection.
    #   v43: pubval_mode="none", pubval_coef=0.0 (gen3_pubval_aux_v1).
    #   v44: zarch_film="off", zarch_dim=0, zarch_recon_coef=0.0, zarch_vicreg_coef=0.0
    #        (gen3_zarch_film_v1).
    #   v45: value_from_dist=False (gen3_dist_critic_v1 Phase B; resume-immutable via
    #        check_value_from_dist, excluded from check_compatible).
    #   v46: zarch_lut="off", zarch_lut_teams=0 (gen3_zarch_lut_v1).
    #   v47: stamp — gen3_belief_single_compute_v1: `move_belief_single_compute` (deleted at v70).
    #   v48: gen3_cpu_damage_deleted_v1 — POPped the three `--unified-obs` ablation fields
    #        (mask_incoming_damage_obs, mask_active_move_scalars_obs, mask_move_effects_obs)
    #        along with the obs blocks they masked (obs 2992 → 2889). POP rather than setdefault:
    #        `from_json_file` does `cls(**data)`, so a stale key would TypeError before the clear
    #        obs-dim rejection.
    #   v49: damage_candidate_k=0 (the op candidate cap; the `pointer_head` bool it also added
    #        was deleted at v51).
    #   v50: stamp — `damage_op_prefuse` added (deleted at v71).
    #   v51: gen3_pointer_native_v1 — POPped `pointer_head` (the pointer head became THE action
    #        head, unconditionally; POP for the v48 reason).
    #   v52: gen3_typed_hp_belief_v1 — POPped `hp_type_belief_mode` (the opponent's Hidden Power
    #        is composed into the 16 typed move-nums inside the belief, so the tri-state had
    #        nothing left to gate; the belief's forward math changed while projection widths did
    #        not, so the ARCH_SIGNATURE bump was the only rejection).
    #   v53: hp_belief_mode="composed" (gen3_hp_belief_ablation_v1; 'flat' is the opt-in
    #        ablation, 'composed' reproduced the v52 forward exactly).
    #   v54: entity_topk_seats=0 (gen3_entity_move_seats_v1; E3 rode the ARCH_SIGNATURE).
    #   v55: stamp — gen3_op_block_trim_v1 (the op's output shrank by 28 dims; the deleted lean
    #        top-K block re-meant `damage_topk_k`; signature carried the break).
    #   v56: edge_bias_families="off" (gen3_edge_bias_trunk_v1; the layer swap itself rode the
    #        ARCH_SIGNATURE).
    #   v57: entity_tail_seats=False (gen3_entity_tail_seats_v1).
    #   v58: stamp — the SpD-as-speed GIGO fix (pairwise_speed/pairwise_boost read stat index 4
    #        as "speed"; values-only — a pre-v58 checkpoint's v_map/c1_map trained against the
    #        buggy feature).
    #   v59: consequence_topk=4 (pre-v59 models trained with the hardcoded k_cand/k_bench = 4).
    #   v60: stamp — gen3_entity_rehome_v1 (signature carried the break).
    #   v61: stamp — gen3_no_concat_v1 (the head-concat deletion + the multi-seed critic
    #        readout; signature carried the break).
    #   v62: value_seed_vicreg_coef=0.0 (gen3_seed_vicreg_v1; resume-immutable, the vf_coef
    #        class).
    #   v63: seed_quantile=False (gen3_seed_quantile_v1; structural, one shared Linear).
    #   v64: value_threat_inject=False (gen3_value_threat_inject_v1; structural, one shared
    #        zero-init Linear on the value pool's copy of our tokens).
    #   v65: stamp — gen3_unconditional_move_legality_v1. `move_candidate_floor` was deliberately
    #        left EXACTLY as recorded (0.0 on every pre-v65 run): 0.0 used to mean "legality
    #        OFF", is no longer a valid value, and check_compatible rejects it against the new
    #        0.02 default with a dedicated message rather than silently migrating 0.0 → 0.02
    #        (which would load a checkpoint whose policy trained on a prior that gave phantom
    #        mass to unlearnable moves).
    #   v66: gen3_nature_marginalize_removed_v1 — POPped `spread_belief_nature_marginalize`.
    #        The kernel (`DamageOperator._nature_marg_ko`) was DELETED, not defaulted off:
    #        measured on gen-8's own checkpoint over 1,075,200 ALIVE (defender, candidate)
    #        cells, |ΔP(KO)| vs mean-field was p50/p90/p95 = 0.00000, p99 = 0.00047, mean
    #        0.0003, and only 0.39% of cells moved by >0.02 — the nature posterior is peaked
    #        (top-1 mass 0.75, entropy 0.64 of 3.22 nats), so integrating over it ≈ evaluating
    #        at its mode (ledger K1's shape: sound theory, absent magnitude). It also computed a
    #        WRONG answer on empty slots: `dmg.clamp(min=eps)` turned zero damage into 1e-6, and
    #        with cur_hp also 0 the ramp gave 1e-6/(0.15e-6+1e-6) = 0.8696 — a spurious 87% KO on
    #        a slot with nothing in it (gated downstream, believed harmless, but a trap). The
    #        KEPT half is `--spread-belief-nature`, the generative nature/EV head — the actual
    #        correctness fix (it makes the largest-EV order-statistic bias structurally
    #        unrepresentable), and why `belief/spread_largest_bias` closed -26 → -12.8 on gen-8.
    #   v67: stamp — gen3_deadline_clock_v1: the obs CLOCK group 1 → 3 scalars (GLOBAL_ENV_DIM
    #        18 → 20, obs 2667 → 2669). No model_config field — the break is in the obs width +
    #        weight shapes, which ARCH_SIGNATURE carries. The first version of the
    #        gen3_deadline_clock_v1 generation (the gen-8/gen-9 world).
    #   v68: gen3_opp_intent_v1 — opp_intent=False (the alpha/beta intent heads).
    #   v69: gen3_species_prior_fusion_v1 — species_prior_fusion=False.
    #   v70: gen3_refine_loop_removed_v1 — POPped the refine loop's five dead fields
    #        (damage_refine_rounds, threat_refine_outgoing, threat_unrevealed_outgoing,
    #        threat_status_refine, move_belief_single_compute — 0-rounds/unreachable/INERT in
    #        every production config; the expected-latent outgoing math itself was re-homed,
    #        unconditional, by gen3_unrevealed_outgoing_prior_v1).
    #   v71: gen3_tiered_pipeline_v1 — the PRE-transformer placement became the only one;
    #        POPped move_belief_prefuse/damage_op_prefuse/damage_reattend, REFUSING (not
    #        defaulting) any config that recorded the deleted POST placement, because that
    #        toggle changed no weight shape and nothing downstream could catch a silent pop.
    #   v72: gen3_t0_species_prior_v1 — t0_species_prior=False.
    #   v73: gen3_intent_grad_mode_v1 — opp_intent_grad_mode="detached".
    #   v74: gen3_intent_value_reduce_v1 — intent_value_reduce=False.
    #   v75: SimSiam latent belief DELETED — POPped opp_belief_latent/opp_belief_latent_coef,
    #        REFUSING a config that recorded opp_belief_latent=True (it carried PARAMETERS the
    #        live extractor has no home for; the zip-kwargs sanitizer keeps that judgment).
    # ----------------------------------------------------------------------------------------
    if version < 77:
        # gen3_intent_move_cell_v1: the G3 alpha-conditioned c2 move-cell channels. Post-floor
        # (the floor stays 76 — OFF is byte-identical, so the flag alone gates it), legal to
        # migrate: absent means the run predates the flag, i.e. OFF.
        data.setdefault("intent_move_cell", False)
        data["config_version"] = 77
    if version < 78:
        # gen3_flag_surface_p1_v1: the zarch family + the seed-pressure pair are DELETED. Two of the
        # eight keys are JUDGED rather than popped — they built MODULES, so a config that recorded
        # them ON names a state_dict the live extractor cannot place, and popping would turn that
        # into an opaque "unexpected key" deep inside SB3's load (the v75 opp_belief_latent
        # precedent). Every gen-8/9/10 production config recorded both OFF, so this refuses only a
        # checkpoint from one of the closed research arms.
        for judged, dead_value in (("zarch_film", "off"), ("seed_quantile", False)):
            recorded = data.get(judged, dead_value)
            if recorded != dead_value:
                raise ModelVersionError(
                    f"{judged}={recorded!r} is no longer supported (gen3_flag_surface_p1_v1, config "
                    f"v78): the module behind it is DELETED, so this checkpoint's weights include "
                    f"parameters the current extractor has no home for.\n"
                    + ("The z_arch/FiLM conditioning line is closed — the free-per-team-code (LUT) "
                       "arm moved the N=20 multi-team ceiling by +0.024, CI [-0.016, +0.064], and "
                       "the orthogonal 2x2 measured team COUNT dominating conditioning.\n"
                       if judged == "zarch_film" else
                       "Both seed-differentiation pressures capped at ~1-D of k=4 "
                       "(out_effective_rank 1.157 with crossing_rate 0.000), so the shared readout, "
                       "not the coefficient, was the binding constraint.\n")
                    + "To re-read this checkpoint, use the git_hash in its own metadata.json."
                )
        for dead in ("zarch_film", "zarch_dim", "zarch_lut", "zarch_lut_teams",
                     "zarch_recon_coef", "zarch_vicreg_coef",
                     "seed_quantile", "value_seed_vicreg_coef"):
            data.pop(dead, None)      # POP, not setdefault: `cls(**data)` TypeErrors on a stale key
        data["config_version"] = 78
    if version < 79:
        # gen3_pair_history_v1 (stamp only, the v67 pattern): the H-A obs widening carries the
        # break in total_dim + the widened encoder shapes (weight-field-caught); no new config
        # field, and the "h" edge family rides the recorded edge_bias_families string.
        data["config_version"] = 79
    if version < 80:
        # gen3_unified_value_readout_v1: post-floor flag-gated module — absent means the run
        # predates the flag, i.e. OFF (the v77 intent_move_cell pattern).
        data.setdefault("value_entity_pool", False)
        data["config_version"] = 80
    if version < 81:
        # gen3_event_window_v1: the H-B obs widening is weight-field-caught (total_dim); the
        # consumer flag is post-floor — absent means the run predates it, i.e. OFF.
        data.setdefault("history_events", False)
        data["config_version"] = 81
    if version < 82:
        # gen3_unified_value_readout_v2: post-floor flag-gated variant — absent means OFF.
        data.setdefault("value_entity_pool_full", False)
        data["config_version"] = 82
    if version < 83:
        # gen3_item_belief_v1: post-floor flag-gated head — absent means OFF.
        data.setdefault("item_belief", False)
        data["config_version"] = 83
    if version < 84:
        # gen3_intent_threshold_v1: post-floor flag-gated operator — absent means OFF.
        data.setdefault("intent_threshold", False)
        data["config_version"] = 84
    if version < 85:
        # gen3_intent_conditional_v1: post-floor flag-gated cells — absent means OFF.
        data.setdefault("intent_conditional", False)
        data["config_version"] = 85
    if version < 86:
        # gen3_op_lean_forward_v1: post-floor flag-gated pair — absent means OFF.
        data.setdefault("op_drop_renders", False)
        data.setdefault("op_believed_lean", False)
        data["config_version"] = 86
    if version < 87:
        # gen3_value_direct_routes_v1: post-floor flag-gated pair — absent means OFF.
        data.setdefault("value_clock", False)
        data.setdefault("value_intent", False)
        data["config_version"] = 87
    # v88 (gen3_dead_flag_purge_v1) — runs for EVERY version (the keys must leave the config
    # whatever vintage wrote them). A recorded ON value named parameters/widths the surviving
    # code cannot rebuild ⇒ refuse (the v75 rule); OFF pops silently.
    for _dead, _ok in (("value_active_readout", False),
                       ("damage_matrices_outgoing_all", False),
                       ("pubval_mode", "none")):
        if _dead in data:
            _rec = data.pop(_dead)
            _bad = (_rec != _ok if isinstance(_ok, str) else bool(_rec) is not _ok)
            if _bad:
                raise ModelVersionError(
                    f"{_dead}={_rec!r} is no longer supported (gen3_dead_flag_purge_v1): the "
                    f"only supported value is {_ok!r}. This checkpoint trained under a forward "
                    "that no longer exists; re-read it from the git_hash in its metadata.json.")
    data.pop("pubval_coef", None)
    if version < 88:
        data["config_version"] = 88
    # v89 (gen3_value_pooled_routes_v1) — the value routes moved from the vf-tail concat into
    # `value_pooled`, changing their projection shapes and the vf concat width. A <v89 config
    # with any of them ON recorded a forward that no longer exists ⇒ refuse with the re-read
    # diagnosis; OFF stamps forward (byte-identical — the routes built nothing).
    if version < 89:
        for _rt in ("intent_value_reduce", "value_entity_pool", "intent_threshold",
                    "value_clock", "value_intent"):
            if data.get(_rt):
                raise ModelVersionError(
                    f"{_rt}=True at config_version {version} is no longer loadable "
                    "(gen3_value_pooled_routes_v1): the route was re-homed from the vf-tail "
                    "concat into value_pooled, so its recorded projection shapes no longer "
                    "exist. This checkpoint trained under a forward the current code cannot "
                    "rebuild; re-read it from the git_hash in its metadata.json.")
        data["config_version"] = 89
    return data
