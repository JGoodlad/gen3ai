"""`ModelVersionFields` -- the dataclass field block, and nothing else.

Split out so the FIELDS (what a run records) and the GATES (what a mismatch costs) can be read
apart. `ModelVersion` in `spec.py` is this class plus the three check/construct mixins; every
mixin inherits from here, so each one genuinely has the fields it reads and needs no protocol.

The field DECLARATION ORDER is load-bearing -- `dataclasses.fields()` order is the constructor's
positional order and `asdict()`'s key order -- so this block is a verbatim move.
"""
from __future__ import annotations   # field annotations stay STRINGS, as in the pre-split file

from dataclasses import dataclass
from typing import List


@dataclass
class ModelVersionFields:
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
    # v7: terminal reward for a DRAW / 250-turn timeout. Resume-immutable VALUE-meaning
    # (check_reward_config), excluded from the weight-shape check. The default tracks
    # RewardConfig.draw_penalty (flipped -30.0 -> -35.0, owner decision 2026-08-18) so a version
    # built with no reward_config records what a default run actually trains with.
    draw_penalty: float = -35.0
    # v12: de-bias cleanup — zero audit-flagged distorting BIAS terms. Resume-immutable VALUE-meaning
    # (check_reward_config), excluded from the weight-shape check. False = the prior behavior.
    drop_redundant_bias: bool = False   # drop stall_tax + matchup_penalty (redundant w/ clock+draw / pbrs_belief)
    drop_switch_bias: bool = False      # drop the hand-coded switch-strategy subsidy family

    # v13/v14: end-state PBRS switches + the now-immutable no-progress penalty (Φ_progress's weight).
    # all_shaping_pbrs = "everything but stall"; stall_pbrs (v14) = the "stall" switch (Φ_progress).
    # `all_shaping_pbrs` defaults TRUE, tracking RewardConfig (owner decision 2026-08-18 — the
    # validated ai_v8 composition). Every config at/above MIGRATION_FLOOR records the key explicitly,
    # so this default is reached only by a ModelVersion built with no reward_config at all; keeping
    # it in step with RewardConfig is what stops such a version recording a composition no run uses.
    all_shaping_pbrs: bool = True
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

    # v93 STRUCTURAL (gen3_pair_outcome_v1): the unified outcome vector's zero-init move-cell
    # projection — a state_dict AND pointer-move-cell width change, so a mismatch would be
    # shape-caught; the check names the cause.
    pair_outcome_cell: bool = False

    # v94 STRUCTURAL (gen3_pair_outcome_switch_v1 / gen3_switch_branch_v1, substrate Phase B):
    # `pair_outcome_switch` is the FIRST widener of the pointer SWITCH cell (state_dict AND
    # switch-cell width); `switch_branch_cell` widens the MOVE cell with OA2 + the spinblock +
    # Protect's attack mass. Both shape-caught; the checks below name the cause.
    pair_outcome_switch: bool = False
    switch_branch_cell: bool = False

    # v95 STRUCTURAL (gen3_conditional_threat_v1 / gen3_pair_value_route_v1, substrate Phase C):
    # `conditional_threat_cell` is OA1 — the SECOND widener of the pointer SWITCH cell (state_dict
    # AND switch-cell width). `pair_value_route` is PV — one zero-init D_MODEL projection inside
    # CLSPool (state_dict only; it injects ADDITIVELY, so no projection width moves and the version
    # gate is the ONLY thing that rejects a mismatched resume). Both shape/gate-caught below.
    conditional_threat_cell: bool = False
    pair_value_route: bool = False

    # v86 STRUCTURAL (gen3_op_lean_forward_v1): drop_renders shrinks out_gain (state_dict
    # shape); believed_lean changes the d3 forward math (no shape — the version gate is the
    # ONLY thing that rejects a mismatched resume).
    op_drop_renders: bool = False
    op_believed_lean: bool = False

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
    # v90 TRAINING-ONLY coefficient (gen3_td_consistency_aux_v1, NOT version-locked): the weight of the
    # Bellman-residual consistency term (V(s_t) − r_t − γ·V(s_{t+1}))² over contiguous rollout pairs.
    # 0.0 = OFF (loss byte-identical). Scales a loss and touches no forward pass, so it is the
    # opp_belief_aux_coef class: recorded here for PROVENANCE and for flagless-resume read-back
    # (`_resolve` reads this field), never compared by check_compatible or any check_*.
    td_aux_coef: float = 0.0
    # v102 TRAINING-ONLY coefficient (gen3_policy_grad_coef_v1, NOT version-locked): the weight on the PPO
    # policy-gradient term itself — `policy_grad_coef * policy_loss` in the loss fold, scaling ONLY the
    # clipped surrogate (never entropy, never the value term, never an aux). 1.0 = the upstream
    # expression, byte-identical (the unscaled tensor is used); 0.0 = the pure-distill/aux phase
    # (arm F). The td_aux_coef class exactly: it scales a loss, touches no forward pass, so it is
    # recorded here for PROVENANCE and for flagless-resume read-back (`_resolve` reads this field)
    # and is never compared by check_compatible or any check_*.
    policy_grad_coef: float = 1.0
    # gen3_intent_label_bot_weight_v1 (config v97): per-sample weight on the opponent-intent
    # (alpha/beta) label rows whose opponent was a heuristic BOT (`opp_class == 0`); every other
    # class stays 1.0. 1.0 = OFF (the unweighted cross_entropy call is taken unchanged, so the loss
    # is bit-identical). The td_aux_coef class exactly: it scales a loss, touches no forward pass,
    # so it is recorded for PROVENANCE and for flagless-resume read-back (`_resolve` reads this
    # field) and is never compared by check_compatible or any check_*.
    intent_label_bot_weight: float = 1.0
    # ---- gen3_cf_coef_provenance_v1 (config v100) — THE COUNTERFACTUAL COEFFICIENT FAMILY -------
    # Ten TRAINING-only knobs, ONE family. Each shapes a LOSS computed in the PPO step; none is
    # read by the extractor forward, none changes a weight shape ⇒ the td_aux_coef class exactly:
    # recorded for PROVENANCE + flagless-resume read-back (`_resolve` reads these fields), NEVER
    # compared by check_compatible. Not registry rows — that registry declares EXTRACTOR toggles
    # and none of these builds a module. The v100 header comment carries the full why.
    cf_records: bool = False
    cf_records_keep: int = 512
    cf_winprob_coef: float = 0.0
    cf_head_only: bool = True
    cf_label_lag_steps: int = 150_000
    cf_label_likelihood: str = "binomial"
    cf_evidential_coef: float = 0.0
    cf_evidential_reg: float = 1e-3
    cf_twin_coef: float = 0.0
    cf_shadow_coef: float = 0.0
    # ---- gen3_capacity_telemetry_v1 (config v101) — LIVE CAPACITY TELEMETRY --------------------
    # Four TRAINING-only diagnostic knobs (the plasticity canary / half-batch trunk cosine /
    # feature velocity). They are the td_aux_coef class and then some: td_aux_coef at least scales
    # a LOSS, while these fold nothing into `loss` and write no `.grad` at all, so the policy's
    # parameter updates are bit-identical on or off. Recorded for PROVENANCE + flagless-resume
    # read-back (`_resolve` reads these fields), NEVER compared by check_compatible — a frozen
    # eval/pool/distill opponent runs no train step, so gating it on a train-step diagnostic would
    # be a false rejection. Not registry rows: the registry declares EXTRACTOR toggles, and the
    # canary head is owned by the PPO object, not by the extractor.
    capacity_telemetry: bool = False
    canary_reset_steps: int = 1_000_000
    capacity_cosine_every: int = 50
    capacity_velocity_every: int = 50
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
    # v98 STRUCTURAL bool (gen3_cf_evidential_head_v1, the win_prob_mode pattern): the EVIDENTIAL Beta
    # readout over P(win|state) off value_pooled — the counterfactual factory's uncertainty confession.
    # False = no module (baseline byte-for-byte; it is never in pi/vf and never even called by the
    # forward, so the projection dims AND the forward outputs are unchanged). True builds a
    # CfEvidentialHead, whose params ARE the state_dict delta — so it is gated in check_compatible with
    # a bool compare. There is no read_only/shaping split by design: the head's input is detached
    # UNCONDITIONALLY, so no coefficient can make it shape the trunk. NO ARCH_SIGNATURE bump.
    cf_evidential: bool = False
    # v99 STRUCTURAL bools (gen3_cf_twin_heads_v1, the v98 pattern twice over). `cf_twin_heads`
    # builds the two extra `WinProbHead`s (B = coverage arm, C = tight-MC arm) that make the R1
    # comparison a WITHIN-RUN paired head difference instead of a run-vs-run one;
    # `cf_shadow_critic` builds the passive `ShadowValueHead` trained on tight-MC `mc_return`
    # labels — the staged promotion path for critic surgery, which never computes an advantage and
    # never enters GAE. Neither is called by the forward, so False is byte-for-byte the baseline and
    # True is bit-identical in pi/vf; the params are the entire delta. NO ARCH_SIGNATURE bump.
    cf_twin_heads: bool = False
    cf_shadow_critic: bool = False
