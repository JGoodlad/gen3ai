from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List

# Bump this whenever the ModelVersion schema changes (fields added/renamed/removed).
# Also add a migration case in _migrate_config().
#
# v3: added `vf_coef` — the PPO value-loss coefficient, recorded so a training resume
#   with a different `--vf-coef` is a hard error (changing the value head's gradient
#   scale mid-run is a silent training change). It is NOT weight-shape-relevant, so it
#   is deliberately EXCLUDED from check_compatible()'s universal load-check (which gates
#   frozen eval / self-play-pool / distill opponents too, where vf_coef is irrelevant);
#   it is enforced only on the training-resume path via check_vf_coef(). Old configs
#   migrate to the SB3 default 0.5 (= the value every pre-flag run was trained with).
#
# v4: added the reward-config hparams — `bias_additivity` (--bias-additivity, the per-run
#   BIAS additive↔telescoping knob), `mat_alive_weight` (--mat-alive-weight, the material-PBRS
#   per-mon-alive weight), and `bias_redesign` (--bias-redesign, the staged no-progress-clock +
#   reframe enable). Like vf_coef, these are resume-immutable VALUE-meaning hparams (changing them
#   mid-run silently shifts the reward) but NOT weight-shape — enforced only on the training-resume
#   path via check_reward_config(), excluded from check_compatible(). Old configs migrate to the
#   defaults (the single-variable run: 1.0 / 1.25 / False).
#
# v5: added `switch_bias_weight` (--switch-bias-weight, the belief-risk-scaled stay-into-KO BIAS lever
#   for the under-switch pathology; design_reward_switching.md §7). Same resume-immutable VALUE-meaning
#   treatment as the v4 reward hparams (folded into check_reward_config, excluded from
#   check_compatible). Old configs migrate to 0.0 (OFF = the lever absent, behavior unchanged).
#
# v6: added `use_popart` (PopArt value-target normalization toggle). Unlike the v3-v5 VALUE-meaning
#   hparams, PopArt changes the value head's STRUCTURE (normalized output + mu/sigma buffers), so it
#   is enforced in check_compatible() (gates EVERY load), not the resume-only path. Old configs
#   default False (no PopArt).
#
# v7: added `draw_penalty` (--draw-penalty, the terminal reward for a DRAW / 250-turn timeout). Same
#   resume-immutable VALUE-meaning treatment as the v4-v5 reward hparams (folded into
#   check_reward_config, excluded from check_compatible). Old configs migrate to -30.0 (== a decisive
#   loss = the prior behavior, where a tie scored -VICTORY_VALUE).
#
# v8: added `attend_unrevealed_opponents` (--attend-unrevealed-opponents). A BEHAVIORAL toggle that
#   keeps the opponent's still-hidden party attendable in the transformer instead of key-masking it.
#   Like v6/use_popart it changes the forward pass (the mask, policy AND value) rather than a reward
#   meaning, so it is enforced in check_compatible(); but unlike PopArt it leaves the state_dict
#   identical (no weight-shape / ARCH_SIGNATURE change). Old configs default False (baseline masking).
#
# v9: added `opp_belief_cls_k` (--opp-belief-cls-k). A STRUCTURAL toggle: k distinct learned query
#   tokens (HiddenOppBeliefPool) summarise the unrevealed opp party and feed both heads. 0 = off; k>0
#   changes the state_dict (adds the module + widens both projection Linears by k*D_MODEL). Like
#   v6/use_popart it is enforced in check_compatible() — but as a plain int every distinct value (incl.
#   0↔N) is a weight-shape mismatch, so a single unconditional compare gates it. OFF (k=0) reproduces the
#   baseline arch byte-for-byte → NO ARCH_SIGNATURE bump. k>0 requires attend_unrevealed_opponents
#   (enforced at extractor build). Old configs default to 0.
#
# v10: added `value_active_readout` (--value-active-readout). A STRUCTURAL toggle: route the active
#   mon's refined token into the VALUE projection (the dual-head readout drops it; a probe found the
#   critic predicts an incoming self-KO at AUC 0.79 vs the policy's 0.90). ON widens the value
#   projection by D_MODEL; like v6/use_popart it is enforced in check_compatible(). OFF reproduces the
#   baseline value head byte-for-byte → NO ARCH_SIGNATURE bump. Old configs default False.
#
# v11: added `value_tail_weight` (--value-tail-weight). A resume-immutable VALUE-meaning hparam (like
#   vf_coef, NOT weight-shape): the tail-weighted value-loss β (CVaR-blend of the worst value misses).
#   0.0 = plain MSE. Enforced ONLY on the training-resume path via check_value_tail_weight, EXCLUDED
#   from check_compatible (a frozen opponent's forward never runs the value loss). No ARCH_SIGNATURE
#   bump (network/obs unchanged). Old configs migrate to 0.0.
#
# v12: added `self_ko_hp_penalty` (--self-ko-hp-penalty). A resume-immutable VALUE-meaning reward
#   hparam (like draw_penalty): a decision-time-HP-scaled penalty (−w·hp) for self-KOing a mon via
#   Explosion/Self-Destruct. The symmetric material PBRS prices a healthy 1-for-1 trade at ~0, so the
#   critic learns to value a full-HP self-KO POSITIVELY and the policy throws away healthy mons; this
#   restores a negative signal (scaled by HP, so legitimate low-HP sac-for-KO is spared). 0.0 = OFF.
#   Enforced via check_reward_config, EXCLUDED from check_compatible. No ARCH_SIGNATURE bump. Old
#   configs migrate to 0.0.
#
# v13: added `drop_redundant_bias` + `drop_switch_bias` (--drop-redundant-bias / --drop-switch-bias).
#   Two resume-immutable VALUE-meaning bools (like the v4-v7 reward hparams): the de-bias cleanup that
#   ZEROES audit-flagged distorting BIAS terms. `drop_redundant_bias` removes stall_tax + matchup_penalty
#   (redundant with the no-progress clock + --draw-penalty / pbrs_belief); `drop_switch_bias` removes the
#   hand-coded switch-strategy subsidy (switch_base / switch_bouncing_tax / escape_threat_switch / se_switch
#   / pivot_* / sleep_in / sleep_out). Folded into check_reward_config, EXCLUDED from check_compatible. No
#   ARCH_SIGNATURE bump (reward-value only). Old configs migrate to False (== the prior behavior).
#
# v14: added `all_shaping_pbrs` (--all-shaping-pbrs, "everything but stall": folds Φ_hazard/Φ_boost/
#   Φ_opp_boosts + Φ_status and ZEROES every BIAS term EXCEPT the anti-stall tilt `no_progress_tax`, so
#   all non-stall shaping is policy-invariant; the bad turn-ramp `stall_tax` is zeroed) and made
#   `no_progress_penalty` resume-immutable (it is now Φ_progress's weight). Resume-immutable VALUE-meaning
#   (check_reward_config), EXCLUDED from check_compatible, NO ARCH_SIGNATURE bump. Old configs migrate to
#   all_shaping_pbrs=False / no_progress_penalty=0.15 (== the prior behavior).
#
# v15: added `stall_pbrs` (--stall-pbrs, the "stall" companion switch: folds Φ_progress and zeroes
#   `no_progress_tax`+`stall_tax`, so the anti-stall signal is policy-invariant too). Run --all-shaping-pbrs
#   WITH --stall-pbrs for a fully-PBRS reward (whole BIAS class zero); without it, keep the no_progress
#   stall tilt as the single acknowledged BIAS. Same resume-immutable VALUE-meaning treatment
#   (check_reward_config), EXCLUDED from check_compatible, NO ARCH_SIGNATURE bump. Old configs migrate to
#   stall_pbrs=False (== the prior behavior).
#
# v16: added `opp_belief_slots` (the hidden-opponent BELIEF-AUX arch toggle) + `opp_belief_aux_coef`
#   (its training-only loss weight). opp_belief_slots is STRUCTURAL like opp_belief_cls_k / use_popart:
#   ON fills the un-revealed opp team slots with distinct learned unknown-mon tokens (refined in-lineup
#   by the transformer) and builds a BeliefHead emitting species/moves aux logits — a state_dict change,
#   so it is gated in check_compatible() with a dedicated bool compare. Requires attend_unrevealed_opponents
#   (enforced at extractor build). OFF reproduces the baseline arch byte-for-byte → NO ARCH_SIGNATURE bump.
#   opp_belief_aux_coef is a TRAINING-ONLY coefficient (like ent_coef): it scales the aux loss, affects no
#   forward pass, so it is recorded for provenance but NOT version-locked (NOT in check_compatible / any
#   check_*; a resume may change it freely). Old configs migrate to opp_belief_slots=False / coef=0.0.
#
# v17: added `move_belief_mode` (the move-prediction REINJECTION arch toggle: off|revealed|unrevealed|both) +
#   `move_belief_coef` (its training-only loss weight). move_belief_mode is STRUCTURAL like opp_belief_slots:
#   any value != "off" builds a MoveBelief module that predicts each opp mon's moveset, soft-embeds it
#   (sigmoid(logits) @ move_embedding) and ADDS the projection back onto the opp token BEFORE the CLS pools
#   — so the predicted moves flow through to both heads. The mode selects which slots get enriched
#   (revealed = seen mons, unrevealed = believed slots, both). It changes the state_dict (a new Linear head +
#   reinjection projection + LayerNorm), so it is gated in check_compatible() with a string compare. Requires
#   attend_unrevealed_opponents (enforced at extractor build). OFF reproduces the baseline arch byte-for-byte
#   → NO ARCH_SIGNATURE bump. move_belief_coef is a TRAINING-ONLY coefficient (like opp_belief_aux_coef):
#   it scales the move-belief supervised loss, affects no forward pass, so it is recorded for provenance but
#   NOT version-locked. Old configs migrate to move_belief_mode="off" / move_belief_coef=0.0.
# v18: added `opp_belief_latent` (the LATENT-belief arch toggle) + `opp_belief_latent_coef` (its training-only
#   loss weight). opp_belief_latent is STRUCTURAL like opp_belief_slots: ON adds an asymmetric SimSiam
#   predictor to BeliefHead that maps each believed slot's refined token into the pokemon_encoder role-token
#   space, where a cosine loss regresses it toward the STOP-GRAD encoder role-token of the TRUE hidden mon
#   (graded identity supervision the hard species CE can't give). It changes the state_dict (the predictor
#   MLP), so it is gated in check_compatible() with a bool compare. Requires opp_belief_slots (the believed
#   slots + BeliefHead it attaches to). OFF reproduces the baseline arch byte-for-byte → NO ARCH_SIGNATURE
#   bump. opp_belief_latent_coef is a TRAINING-ONLY coefficient (like opp_belief_aux_coef): it scales the
#   latent cosine+VICReg loss, affects no forward pass, recorded for provenance but NOT version-locked. Old
#   configs migrate to opp_belief_latent=False / opp_belief_latent_coef=0.0.
#
# v19: added `damage_op` (the differentiable GPU damage operator arch toggle). STRUCTURAL like
#   value_active_readout / opp_belief_slots: ON builds a `DamageOperator` that consumes the move
#   belief's PREDICTED moves for the opp active and emits a per-our-mon believed-move incoming-damage
#   block appended to BOTH projection heads, so it WIDENS both projection inputs (a state_dict change)
#   — gated in check_compatible() with a dedicated bool compare. The operator's lookup tables are
#   non-persistent buffers (fixed physics from data/), so the only state_dict deltas are the wider
#   projections. OFF reproduces the baseline arch byte-for-byte → NO ARCH_SIGNATURE bump. Hard-requires
#   move_belief_mode in {revealed, both} (the op reads the opp-active's predicted logits, only
#   supervised for a revealed mon) — enforced at extractor build + the CLI. It is forward-only (no new
#   labels / no loss term), so there is no training-only coefficient. Old configs migrate to False.
#
# v20: added `move_prior_fusion` (the unified two-part move belief). FORWARD-BEHAVIOR toggle like
#   `attend_unrevealed_opponents` (NOT weight-shape): the MoveBelief head's output becomes a learned
#   log-odds DELTA fused with the Smogon move-frequency prior — `posterior = prior_logit(species) +
#   head_delta`, revealed moves pinned certain — so the stashed move-belief logits (read by the damage
#   op + the BCE loss) carry a proper POSTERIOR (priors ⊕ prediction unified). The prior buffer is a
#   non-persistent lookup, no new params → state_dict byte-identical either way, but the forward differs
#   when ON, so (like attend_unrevealed_opponents / damage_op) it is gated in check_compatible — a resume
#   that flips it would feed a different belief. Requires move_belief_mode != off (enforced at extractor
#   build + CLI). OFF reproduces the from-scratch head byte-for-byte → NO ARCH_SIGNATURE bump. Old configs
#   migrate to False.
#
# v21: added `mask_incoming_damage_obs` (the unified-architecture ABLATION toggle). FORWARD-BEHAVIOR
#   toggle like attend_unrevealed_opponents (NOT weight-shape): ON zeros the 51-dim incoming-damage /
#   OHKO obs block out of the model's view (the block STAYS in the obs at a fixed dim; the reward PBRS
#   still reads the belief from live_view). Lets the unified DamageOperator's learned belief->damage
#   REPLACE the CPU usage-prior collapse for the MODEL, A/B-ably, without deleting any code. State_dict
#   byte-identical (just zeros an obs slice), but the forward differs, so it is gated in check_compatible.
#   OFF = baseline byte-for-byte (NO ARCH_SIGNATURE bump). Old configs migrate to False.
#
# v22: added `win_prob_mode` (the tri-state auxiliary WIN-PROBABILITY head: none|read_only|shaping) +
#   `win_prob_coef` (its training-only loss weight). win_prob_mode is the STRUCTURAL toggle: 'none' = no
#   module (baseline byte-for-byte); 'read_only'/'shaping' build a `WinProbHead` (a side readout off
#   value_pooled, NOT in pi/vf so projection dims are unchanged — the only state_dict delta is the head's
#   own params). It is gated in check_compatible with a STRING compare so that BOTH 'none'↔head (a
#   state_dict change) AND read_only↔shaping (same params, but the user-chosen resume-IMMUTABLE mode — a
#   mid-run grad-flow flip is a silent training change) are FATAL on a resume mismatch. OFF reproduces the
#   baseline arch byte-for-byte → NO ARCH_SIGNATURE bump. win_prob_coef is a TRAINING-ONLY coefficient
#   (like opp_belief_aux_coef): it scales the BCE aux loss, affects no forward pass, so it is recorded for
#   provenance but NOT version-locked (a resume may change it freely, and a flagless resume inherits it).
#   Old configs migrate to win_prob_mode="none" / win_prob_coef=1.0.
# v23: added `damage_outgoing` (the OUTGOING per-move damage direction of the unified DamageOperator) +
#   `move_candidate_floor` (the learnset + rarity-cap move-prior gate). damage_outgoing is STRUCTURAL like
#   damage_op (the per-move outgoing block widens BOTH projection heads), gated in check_compatible with a
#   bool compare; OFF = baseline byte-for-byte (NO ARCH_SIGNATURE bump), requires damage_op. move_candidate_floor
#   is a FORWARD-BEHAVIOR float like move_prior_fusion: 0.0 = OFF (legacy 0.02-floor prior, byte-identical),
#   >0 enables the learnset-legality + <floor rarity prune on the move prior (a different belief → gated in
#   check_compatible; the prior buffer is non-persistent so the state_dict is identical either way). Old
#   configs migrate to damage_outgoing=False / move_candidate_floor=0.0.
# v24: gen3_unified_move_system_v1. Added `move_latent` (the context-free MoveLatentEncoder arch toggle:
#   a mechanics-grounded per-move latent concatenated into the move network — STRUCTURAL like damage_op,
#   it WIDENS the move-network input → state_dict change, gated in check_compatible; OFF = baseline
#   byte-for-byte, NO ARCH_SIGNATURE bump) + `move_belief_latent_coef` (its training-only latent-grading
#   loss weight: cosine of the predicted move distribution's expected latent toward the true moveset's mean
#   latent so Rock Slide ≈ Hidden Power Rock — NOT version-locked, like move_belief_coef). ALSO in v24 the
#   DamageOperator's effect block is enriched with per-status SECONDARY probabilities (incoming + per-move
#   outgoing, Serene Grace / Shield Dust) — intrinsic to `damage_op` (no separate flag), so a v23
#   damage_op checkpoint won't load into v24 (the op's output dim grew); damage_op OFF stays byte-identical.
#   Old configs migrate to move_latent=False / move_belief_latent_coef=0.0.
# v25: gen3_unified_spread_belief_v1 + the disable-redundant-obs master flag. (1) `spread_belief` (the THIRD
#   belief leg — predicts the opp's hidden SPREAD = 5 derived stats per slot, reinjected into the opp token,
#   consumed by the DamageOperator to REPLACE its hand-coded de-timid/neutral opp-spread constants; STRUCTURAL
#   like opp_belief_slots — adds the SpreadBelief module, gated in check_compatible, OFF byte-identical, NO
#   ARCH_SIGNATURE bump) + `spread_belief_coef` (its training-only speed-supervision loss weight, NOT
#   version-locked). (2) the disable-redundant obs masks `mask_active_move_scalars_obs` +
#   `mask_move_effects_obs` (FORWARD-BEHAVIOR like mask_incoming_damage_obs — zero a now-GPU-subsumed obs
#   region from the model's view; the master --unified-obs flips all three). (3) the DamageOperator op effects
#   are further unified (MOVE_EFFECT_FLAGS folded into MOVE_ATTR; fixed-damage moves type-gated) — intrinsic to
#   damage_op. Old configs migrate every new field to False/0.0.
# v26: gen3_unified_op_physics_v1 — the DamageOperator reaches PARITY with the CPU incoming_damage block it
#   (optionally) masks, so --unified-obs no longer regresses the model's damage understanding. INTRINSIC to
#   damage_op (no new field): the op now applies stat-stage BOOSTS (offense/defence/speed, both directions —
#   a +2 sweeper's Atk doubles), BURN (½ physical Atk), WEATHER (rain ×1.5 Water/×0.5 Fire; sun the reverse),
#   PARALYSIS (×0.25 speed), and FIXED-DAMAGE moves (Seismic Toss/Night Shade = level HP, type-immunity-gated
#   — 0 vs Ghost). Values-only (no dims/state_dict change → no new check_compatible field); the version bump
#   marks it. Counter/Mirror Coat (return-damage) is deferred.
# v27: gen3_unified_status_landing_v1 — the op's OUTGOING direction gains a per-OUR-move STATUS-LANDING block
#   (8 dims: P(a dedicated status move lands vs THIS opponent) + a `known` bit per move) — the GPU home for
#   the masked move-effect block's `status_will_land`, so --mask-move-effects-obs no longer drops that signal.
#   It folds accuracy × per-MOVE type immunity (Thunder Wave→Ground, Toxic/Poison→Steel/Poison, Will-O-Wisp
#   →Fire, **+ Leech Seed→Grass**, the v26-deferred item) × ability immunity (revealed→exact, else the Smogon
#   ability-prior marginal) × already-statused (majors) × **Sleep Clause** (a 2nd inflicted sleep fails; a
#   Rest self-sleep does NOT consume the cap) × **Substitute** (a Sub blocks every status move incl. Leech
#   Seed). The gen3 rules are imported from gen3_mechanics (one source); Shield Dust is N/A here (it only
#   scales SECONDARY effects, never a primary status move). INTRINSIC to damage_outgoing (no new field) — it
#   grows the outgoing output dim, so a v26 damage_outgoing checkpoint won't load (the SB3 load_state_dict
#   shape mismatch on the projection Linear in_features — the runtime-discovered projection dim is NOT a
#   ModelVersion field, so check_compatible passes). OFF (no damage_outgoing) byte-identical; no
#   ARCH_SIGNATURE bump. Bare version marker.
# v28: gen3_unified_choice_band_v1 — the op prices CHOICE BAND (×1.5 physical Atk + move-lock; the dominant
#   damage-relevant gen3 item). OUTGOING: our own CB (item known) ×1.5 our physical Atk DETERMINISTICALLY
#   (values-only). INCOMING: a CB-CONDITIONAL physical tail per our 6 mons — `phys_high_cb` (max-roll with
#   the ×1.5) + `pko_cb` (P(OHKO | CB)) — plus a shared `p_cb` scalar (P(opp active holds CB) from
#   `SPECIES_CB_PRIOR`, the Smogon item usage prior, collapsed to 1.0/0.0 once the held/consumed item is
#   revealed). DECORRELATED from the modal (no-CB) line so the head weights them — OHKO is a nonlinear
#   threshold a mean-field ×(1+0.5·p_cb) would blur (same provide-the-fact rationale as the crit-split). The
#   ×1.5 is applied at the Atk-STAT level (so core=k·A+2's +2 floor isn't boosted) in BOTH directions. The
#   move-lock + the ChoiceBandTracker's move-lock DISPROOF are a documented follow-up. INTRINSIC to damage_op
#   (the incoming CB block grows the incoming output dim → a v27 damage_op checkpoint won't load, SB3
#   load_state_dict in_features mismatch); OFF (no damage_op) byte-identical; no ARCH_SIGNATURE bump. Marker.
# v29: added the distributional VALUE head (Phase A interpretability side readout) — `value_dist_mode`
#   (none|read_only|shaping STRUCTURAL toggle, like win_prob_mode) + `value_dist_bins` (the atom count =
#   the head's output Linear width, weight-shape like opp_belief_cls_k), both gated in check_compatible;
#   and the value-meaning support `value_dist_vmin`/`value_dist_vmax` (resume-only check_value_dist, like
#   value_tail_weight). A SIDE readout off value_pooled (NOT in pi/vf → projection dims unchanged), so
#   OFF (mode none) is baseline byte-for-byte — NO ARCH_SIGNATURE bump. Old configs migrate to
#   value_dist_mode="none" / bins=0 / vmin=vmax=0.0. Design: designs/ai_v6/design_distributional_value_critic.md.
# v30: gen3_unified_topk_incoming_v1 — the DamageOperator's DISCRETE top-K incoming move-space block.
#   `damage_topk_k` (int, 0 = off) = the number of the opp ACTIVE's most-believed CANDIDATE moves surfaced
#   INDIVIDUALLY (vs the worst-case `_chan_max` collapse that loses WHICH move it is). Per top-K move: its
#   move LATENT identity (gathered from the MoveLatentEncoder — DIFFERENTIABLE → sharpens the latent) +
#   belief weight (DIFFERENTIABLE → sharpens the move belief) + accuracy + is_phys, then per OUR mon
#   [high-roll, P(KO), status_lands] — the discrete-move + per-pivot (incl. damage-immunity 0 AND
#   status-immunity 0, e.g. Thunder-Wave→Ground) read that makes "anticipate the move / pick the safe
#   switch" decidable. Added ALONGSIDE the worst-case summary. K scales out_dim (hence both projection
#   in_features) → STRUCTURAL int gated in check_compatible (like opp_belief_cls_k); OFF (0) byte-for-byte
#   (NO ARCH_SIGNATURE bump). Requires damage_op + move_latent. A v29 damage_op checkpoint won't load into a
#   topk-ON op (projection in_features mismatch). Design: designs/ai_v6/design_topk_incoming_moves.md.
# v31: added `damage_reattend` (gen3_damage_reattend_v1) — re-attend the team tokens to the computed
#   DamageOperator physics, then re-derive the CLS pools, so the policy/value DECISION path (incl. the
#   switch logits) reads damage-contextualised summaries (today the damage block is a post-pool concat that
#   no attention sees). STRUCTURAL toggle like opp_belief_slots (adds a damage→token projection + LayerNorm
#   + one TransformerEncoder layer; re-pooling preserves the pooled shapes ⇒ projection WIDTHS unchanged),
#   gated in check_compatible (bool); OFF byte-for-byte (NO ARCH_SIGNATURE bump). Requires damage_op.
# v32: added `move_belief_prefuse` (gen3_move_prefuse_v1) — move the MoveBelief reinjection from
#   POST-transformer to PRE-transformer, so the predicted opp moves co-refine with the species/team belief
#   through the 2 attention layers instead of being grafted on afterwards. FORWARD-BEHAVIOR toggle like
#   move_prior_fusion (same MoveBelief params → state_dict identical; only the call timing differs), gated
#   in check_compatible (bool); OFF byte-for-byte (NO ARCH_SIGNATURE bump). Requires move_belief_mode != off.
# v33: gen3_iterative_damage_v1 — ITERATIVE damage refinement. `damage_refine_rounds` (int, 0 = off) is the
#   number of transformer layers (capped by TRANSFORMER_N_LAYERS in effect) before which the DamageOperator's
#   LEAN discrete incoming threat is recomputed from the CURRENT (being-enriched) opp tokens and injected back
#   onto our-mon tokens via a `refine_proj` Linear (zero-init → identity-at-init) — so each layer attends over
#   physics derived from the freshest move belief (physics-in-the-loop), and the per-round read sharpens the
#   move-belief head. STRUCTURAL: 0 builds no module (baseline forward byte-for-byte, NO ARCH_SIGNATURE bump);
#   N>0 builds refine_proj (its SHAPE is N-independent — weight-tied across rounds) and changes the forward, so
#   EVERY distinct value (0↔N a state_dict change; N↔M a forward-behavior change) is gated in check_compatible
#   with an unconditional int compare (like opp_belief_cls_k). Requires damage_op (→ the op physics + a
#   move_belief to re-read). Old configs migrate to 0. Design: designs/ai_v6/design_iterative_damage_refinement.md.
# v34: gen3_per_move_matrices_v1 — the OUTGOING per-move DAMAGE MATRIX. `damage_matrices_outgoing` (bool, off)
#   makes the DamageOperator ALSO emit our 4 moves × the opp's 6 mons (active + REVEALED bench) — per (move,
#   opp mon) [low,high,crit,pko,type_mult] + a per-opp-mon revealed bit — the bench extension of the single-
#   active outgoing block (price a KO on a switch-in). Unrevealed opp slots zeroed (belief-driven = TODO).
#   STRUCTURAL toggle like damage_op (widens both projection in_features); gated in check_compatible (bool);
#   OFF byte-for-byte (NO ARCH_SIGNATURE bump). Requires damage_op. Design: designs/ai_v6/design_per_move_damage_matrices.md.
# v35: gen3_per_move_matrices_v1 — the INCOMING per-move DAMAGE MATRIX. `damage_matrices_incoming` (bool, off)
#   makes the DamageOperator emit the ENRICHED top-K block: per opp-active move a header [latent, belief, acc,
#   is_phys, EXPLICIT effect bits(6), secondary chances(10)] + per (OUR mon, move) cell [low,high,crit,pko,
#   type_mult,status_lands] — the un-collapsed evolution of the v30 top-K + the deleted p_effect/p_sec maxes.
#   REUSES damage_topk_k as its K (one knob, try 4/5/6). Since gen3_op_block_trim_v1 deleted the lean top-K
#   block it superseded, this is the ONLY block K sizes; requires damage_op + move_latent.
#   STRUCTURAL toggle like damage_op (widens both projections via the op out_dim); gated in check_compatible
#   (bool); OFF byte-for-byte (NO ARCH_SIGNATURE bump). Design: designs/ai_v6/design_per_move_damage_matrices.md.
# v36: gen3_bidir_threat_trunk_v1 — the BIDIRECTIONAL in-trunk threat field. `threat_refine_outgoing` (bool)
#   adds a zero-init `outgoing_proj` that injects a per-opp-mon OUTGOING-threat residual onto the OPP tokens
#   via the SAME between-layers refine loop (symmetric to the incoming refine; STRUCTURAL — a saved weight).
#   `threat_unrevealed_outgoing` (bool) prices that residual's UNREVEALED columns via the EXPECTED-LATENT
#   defender — marginalize the move-belief's P(species) through SPECIES_EXP_MULT (type chart × expected
#   ability immunity) + SPECIES_SPREAD_PRIOR (E[bulk]), with P(KO) NULLED (forward toggle, no new params).
#   `threat_prob_outspeed` (bool) makes P(outspeed) UNCERTAINTY-AWARE (÷ believed speed std, not a fixed
#   scale; forward toggle). All three OFF byte-for-byte (NO ARCH_SIGNATURE bump). threat_refine_outgoing
#   requires damage_op + damage_refine_rounds>0; threat_unrevealed_outgoing requires threat_refine_outgoing
#   (+ a belief head for P(species)). Design: designs/ai_v6/design_bidirectional_threat_trunk.md.
# v37: gen3_status_trunk_v1 — STATUS-LANDING into the trunk (the last CPU-obs deprecation gap).
#   `threat_status_refine` (bool) adds two zero-init Linears riding the refine loop: status_in_proj (incoming
#   "will I be statused" onto OUR tokens, from the opp active's believed status moves) + status_out_proj
#   (outgoing "can I status this opp mon" onto OPP tokens, revealed-gated, from our active's status moves),
#   each a per-defender [P(major), P(immobilize=para/frz/slp)] computed by reusing the v27 status-landing
#   physics (type × ability × already-statused × Sleep-Clause × Substitute). Status immunity is a computed
#   MECHANICS fact (the same class as type effectiveness) — handed over, not learned across non-local tokens.
#   STRUCTURAL (saved weights); OFF byte-for-byte (NO ARCH_SIGNATURE bump). Requires damage_op +
#   damage_refine_rounds>0. Completes the FULL --unified-obs deprecation (the A/B is the arbiter). Design:
#   designs/ai_v6/design_bidirectional_threat_trunk.md.
# v39: gen3_per_move_matrices_v1 — the TRANSPOSED outgoing matrix. `damage_matrices_outgoing_all` (bool, off)
#   makes the DamageOperator ALSO emit our 6 MONS' 4 moves → the opp ACTIVE — per (attacker mon, move)
#   [low,high,crit,pko] + a per-attacker p_outspeed + an alive bit. The TRANSPOSE of v34's
#   damage_matrices_outgoing (our active's 4 moves × the opp's 6 mons): here the ATTACKER axis is our 6 mons,
#   the defender is the opp ACTIVE only. On a FORCED SWITCH our active is fainted → the single-active outgoing
#   block zeroes, so the policy picks switch-ins BLIND to offense; this prices every candidate switch-in. The
#   ACTIVE row reproduces _outgoing_block byte-for-byte (parity); bench rows reuse the SAME _rolls physics with
#   NEUTRAL boosts (gen3 resets on switch). STRUCTURAL toggle like damage_op (widens both projection
#   in_features via the op out_dim); gated in check_compatible (bool); OFF byte-for-byte (NO ARCH_SIGNATURE
#   bump). Requires damage_op. Design: designs/ai_v6/design_per_move_damage_matrices.md.
# v43: gen3_pubval_aux_v1 — the PUBLIC-information value aux head. `pubval_mode` (none|read_only|shaping,
#   the win_prob_mode pattern) builds a PubValHead off value_pooled, regressed toward the FROZEN
#   human-replay-calibrated public value V_pub (agents.training.pubval + data/gen3_pubval.json — 164k rated
#   gen3ou games, the value-INDEPENDENT exogenous signal; dense per-step, so the trunk sees WHEN the game
#   swung). SIDE readout (never in pi/vf, never in GAE); the target rides a training-only `pubval_target`
#   obs key computed env-side from PUBLIC state only. STRUCTURAL + resume-IMMUTABLE STRING gate like
#   win_prob_mode ('none'↔head = state_dict; read_only↔shaping = grad-flow); OFF byte-for-byte (NO
#   ARCH_SIGNATURE bump). `pubval_coef` training-only. Design: designs/ai_v8/design_public_info_value.md.
# v44: gen3_zarch_film_v1 — the team-archetype latent z_arch + head FiLM (the amortization-gap STORAGE
#   fix: per-team gradients modulate different rank-z subspaces instead of cancelling in the shared
#   heads — designs/learning/amortization_gap_and_conditioning.md). `zarch_film` (off|heads) builds a
#   ZArchEncoder (a TEAM-STATIC, permutation-invariant DeepSets code over OUR team's INVARIANT facts:
#   species ⊕ item ⊕ ability ⊕ moves ⊕ spread, detached embedding reads — zero trunk interference) +
#   two ZERO-INIT FiLM generators applied post-projection pre-ReLU per root head (identity-at-init ⇒
#   ON starts byte-identical). `zarch_dim` (int) is the latent width = the FiLM conditioning rank —
#   the generators' in_features, so every distinct value is a weight-shape mismatch (unconditional int
#   compare, the value_dist_bins pattern). STRING + INT gated in check_compatible; OFF (off/0) builds
#   no modules = baseline byte-for-byte (NO ARCH_SIGNATURE bump). `zarch_recon_coef` (species multi-hot
#   reconstruction BCE — the anti-collapse anchor) + `zarch_vicreg_coef` (per-dim variance floor) are
#   TRAINING-ONLY loss coefs (recorded for provenance + flagless-resume read-back, NOT version-locked).
# v46: gen3_zarch_lut_v1 — the per-team LUT on top of z_arch. `zarch_lut` (off|add|only) adds an
#   Embedding[n_teams+1, zarch_dim] (row 0 = unknown, ZERO-init; rows 1..N random-init so the per-team
#   codes are LARGE and ~orthogonal from step 0) + a LayerNorm, and the team is resolved from the
#   OBSERVATION by a sorted species(6) ⊕ moves(24) signature (agents.model.team_signature) — so NO
#   env/eval/prober/frozen-opponent plumbing changes. `zarch_lut_teams` (int) is the table height, a
#   weight-shape field (unconditional int compare, the zarch_dim pattern). It exists to test whether
#   the measured multi-team exploiter ceiling (N=1/3/10 distil cleanly, N=20 stalls) is a
#   conditioning-SIGNAL limit: the DeepSets z is COMPOSITIONAL, so z-similar teams sit at z̄ + a tiny
#   ε and the FiLM generator's gradient is proportional to that residual (ill-conditioned); a free
#   code removes exactly that limit. 'add' = LN(z_deepsets + code) keeps composition (an unmatched
#   team hits the zero row ⇒ exactly the DeepSets z); 'only' = LN(code), the sharpest ablation.
#   STRING + INT gated in check_compatible; OFF byte-for-byte (NO ARCH_SIGNATURE bump).
# v47: added `move_belief_single_compute` (gen3_belief_single_compute_v1) — compute the move belief
#   EXACTLY ONCE per forward (pre-attention) and FREEZE it. Under prefuse the belief is predicted +
#   reinjected before the transformer, but the between-layers refine callback then RE-READ move_logits
#   off the reinjected tokens: the belief was computed twice and the physics consumed a different
#   posterior than the one attention was handed. ON, the refine kernels reuse the stashed
#   pre-transformer logits ⇒ belief ONCE → physics ONCE → N attention layers that CANNOT revise it
#   (the frozen-belief arm of the iterative-refinement A/B; also one fewer head pass per forward).
#   FORWARD-BEHAVIOR toggle like move_belief_prefuse (same MoveBelief params → state_dict identical;
#   only which posterior the refine kernels read differs), gated in check_compatible (bool); OFF
#   byte-for-byte (NO ARCH_SIGNATURE bump). Requires move_belief_prefuse.
# v49: added `damage_candidate_k` (gen3_topk_candidates_v1) + `pointer_head` (gen3_pointer_head_v1).
#   `damage_candidate_k` (int, 0 = the full ~400-wide sweep) caps the DamageOperator's INCOMING
#   candidate axis at the K most-believed opponent moves, NO tail bound — the truncated mass is
#   dropped. FORWARD-BEHAVIOR (no new params; the per-candidate args just get narrower), gated with an
#   unconditional int compare like `damage_topk_k`; 0 is byte-identical.
#   `pointer_head` (bool) was the DELTA pointer head — a zero-init additive term on the flat head's
#   logits. REMOVED at v51: the pointer head became THE head (see below).
# v50: added `damage_op_prefuse` (gen3_damage_op_prefuse_v1) — ONE damage computation per forward,
#   PRE-attention. The op ran TWICE: a LEAN `discrete_*` recompute inside the between-layers refine
#   loop (×`damage_refine_rounds`) plus the FULL 835-dim block after the transformer. At B=1 on CPU —
#   the PFSP frozen-opponent regime that sits on the rollout critical path — the two together are ~75%
#   of a dispatch-bound 6.45 ms forward (the attention layers themselves are 0.27 ms). ON, the spread +
#   HP-type beliefs and the FULL op all run on the PRE-transformer role tokens, the per-OUR-mon incoming
#   rows are injected onto our tokens via a zero-init `prefuse_proj` (the refine_proj convention), and
#   the SAME full block is concatenated to both heads — so the P1 head-concat dependency is preserved at
#   full width, just sourced from a pre-attention belief. Mutually exclusive with damage_refine_rounds>0
#   (the loop is what it replaces); requires damage_op + move_belief_prefuse. The justification is CPU
#   cost; the "attention reasons over full-fidelity physics" story is secondary and, per K9/K10/K10a,
#   unlikely to pay on its own. STRUCTURAL (adds prefuse_proj) → bool compare in check_compatible; OFF
#   byte-identical (NO ARCH_SIGNATURE bump).
# v51: gen3_pointer_native_v1 — the FRESH-GENERATION pointer-native action head. The flat positional
#   `action_net` is DELETED (replaced in Gen3DualHeadMaskablePolicy._build by a raising stub) and the
#   `PointerNativeActionHead` is THE action head, unconditionally: move logit k from the REQUEST-slot-k
#   move token ⊕ its op cells [low,high,crit,pko,p_land,known,sec×10], switch logit j from our-team
#   token j ⊕ its incoming row + CB tail + OAX attacker row, struggle from the latent_pi context —
#   position-EQUIVARIANT (one shared scorer per entity; the sorted-vs-request ordering bug class is
#   unrepresentable at the logits). The v49 `pointer_head` FIELD is removed (POPped in
#   _migrate_config); no gate exists because there is no off state. Cross-era break carried by the
#   ARCH_SIGNATURE bump (see gen3_pointer_native_v1 below).
# v52: gen3_typed_hp_belief_v1 — the v38 tri-state `hp_type_belief_mode` FIELD is DELETED (POPped in
#   _migrate_config): the HP-type head is UNCONDITIONAL under a move belief, the presence×type
#   composition moved into `HPTypeBelief.compose_typed_hp` beside the move head, and every consumer
#   reads a posterior that carries HP at the 16 real typed nums 355-370 with the bare 237 hard-off.
#   Forward math changed with unchanged projection widths → the cross-era break rides the
#   ARCH_SIGNATURE bump (see gen3_typed_hp_belief_v1 below). `hp_type_belief_coef` stays training-only.
# v53: added `hp_belief_mode` (gen3_hp_belief_ablation_v1) — 'composed' (default, the v52 forward
#   byte-for-byte: HPTypeBelief + the presence×type factorisation + the two certain-fact eliminations)
#   vs 'flat' (the ABLATION: no HPTypeBelief head; the multi-label move head predicts the 16 typed
#   channels INDEPENDENTLY off their own per-typed Smogon priors — both arms still mask the bare 237
#   via the shared `mask_typeless_hp`). STRUCTURAL ('flat' drops a module) → STRING compare in
#   check_compatible (the win_prob_mode pattern); default byte-identical (NO ARCH_SIGNATURE bump).
# v54: gen3_entity_move_seats_v1 — Stage 1 of the entity generation (the roadmap's move-tokens-into-
#   the-body slice): MOVE tokens become first-class attention SEATS in the unified trunk, appended
#   after the global token. E3 (unconditional): our active's 4 request-ordered move tokens, projected
#   32 → d_model — the pointer head now reads the REFINED seats (post-attention, d_model-wide;
#   `move_seat_proj` + the token-type table growing 4 → 6 + the head's wider `move_proj` are the
#   unconditional state_dict changes → the `ARCH_SIGNATURE` bump below carries the break). E4
#   (`entity_topk_seats` int, 0 = off): the opp active's top-K believed threat-move seats — the op's
#   `refine_candidates(k=K)` candidate definition (belief-weighted, typed-HP-scattered) gathered as
#   `[latent ⊕ w ⊕ acc ⊕ is_phys]` per seat; adds `threat_seat_proj` (STRUCTURAL int, the
#   `damage_topk_k` gating pattern; requires damage_op_prefuse + move_latent). NO edges yet (Stage 2).
# v55: gen3_op_block_trim_v1 — NO new field; the DamageOperator's output SHRINKS by 28 dims and its
#   `damage_topk_k` knob changes meaning, so the version marks the break (the ARCH_SIGNATURE bump below
#   is what actually rejects an older checkpoint). Deleted, on the ledger-P1 per-block dependence
#   ablation: the opp-active-level believed-EFFECT scalars (6 dims, 1.2% of the zero-whole-op ceiling),
#   the opp-active per-STATUS incoming SECONDARY scalars (10 dims, 0.1% — the single most INERT channel
#   in the operator), and the OUTGOING per-move slp/psn/tox secondary columns (12 dims = 4 moves × 3,
#   structural zeros: gen3 has NO damaging move that inflicts sleep, and the psn/tox carriers appear on
#   1 / 0 of the 773 pool teams). Also deleted: `_topk_block`, the v30 LEAN top-K block — a strict
#   subset of the v35 `_incoming_matrix` that already suppressed it, which the same cProfile measured at
#   **0 calls per forward** in the production build. `damage_topk_k` now means "the incoming matrix's
#   K"; K>0 without `damage_matrices_incoming` is a hard error (never a silent empty block).
#   INDEPENDENT of v54's entity seats — the two touch disjoint machinery (seats enter the TRUNK; this
#   trims the op's HEAD-CONCAT output), so they compose and only the signature is shared.
# v56: gen3_edge_bias_trunk_v1 — Stage 2 of the entity generation (physics as attention EDGES):
#   the trunk's encoder stack becomes the BIASED clone (`BiasedEncoderLayer` — same math, but
#   attention takes an additive per-pair per-head float bias; the key-pad mask rides the same
#   tensor), an UNCONDITIONAL state_dict change (layer keys `in_proj.*` vs `self_attn.in_proj_*`)
#   → the ARCH_SIGNATURE bump below carries it. `edge_bias_families` (str, "off" default) gates the
#   FAMILIES delivered — each through a ZERO-INIT Linear(cell → 2·n_heads) map (identity at init):
#   D1 our-move→opp-mon (the v34 outgoing-matrix kernel), D2 our-mon→opp-ACTIVE (the v39 switch-in
#   kernel, move-collapsed, one-hot column), D3 threat-seat→our-mon (the pre-collapse incoming
#   kernel at the E4 candidate selection), S1 our-status-move→opp-mon + S3 threat-seat→our-mon
#   (the v27/v37 status-landing kernels' per_pair branches). "d" is the FROZEN d1,d3 alias (a saved
#   config never silently grows maps); new families are explicit comma-list only. Growing the VALID
#   family set is not a version bump — the string gate catches any mismatch. The op head-concat is
#   NOT deleted (deprecation playbook: home first, ablation audit before deletion).
# v57: added `entity_tail_seats` (gen3_entity_tail_seats_v1, E5) — 6 per-opp-mon TAIL-THREAT seats
#   summarizing the beyond-top-K belief mass every candidate consumer truncates ([p_tail, worst_phys,
#   worst_spec, revealed] → tail_proj + a learned tail_marker; NO new token-type row, deliberately —
#   growing the type table would break loading in-generation checkpoints into newer code). STRUCTURAL
#   bool (adds tail_proj + tail_marker + 6 seats per forward); OFF byte-identical; requires
#   damage_op_prefuse + entity_topk_seats>0 (the tail is defined relative to the E4 truncation).
# v58 is a STAMP (no field, no migration — the v26/v55 convention): the SpD-as-speed GIGO fix
# in pairwise_speed/pairwise_boost (the V/C1 kernels read stat index 4 = Special Defense as
# "speed"; both trained generations' V edge priced bulk). VALUES-only forward-math change: a
# pre-v58 checkpoint still LOADS, but its v_map/c1_map trained against the buggy feature — its
# V-edge inputs shift under fixed code (documented, accepted: gen-3 retrains under true physics).
# v60 is the gen3_entity_rehome_v1 STAMP (Stage-3 re-home; the ARCH_SIGNATURE carries the break —
# obs dim, POKEMON_FULL_DIM and the move/role net widths all move, so no migration is possible).
# v61 is the gen3_no_concat_v1 STAMP — the op head-concat deletion + the multi-seed critic
# readout (the gen-5 world; the signature carries the break).
# v62: added `value_seed_vicreg_coef` (gen3_seed_vicreg_v1) — the VICReg variance+covariance floor on
#   the MultiSeedValueReadout seed OUTPUTS (agents/model/seed_vicreg.py), built because the
#   pre-registered trigger in seed_diagnostics.py FIRED on gen-5 (seeds/out_effective_rank 1.0
#   sustained 13M+ steps — full seed collapse). Resume-immutable VALUE-meaning hparam (the
#   vf_coef class, NOT weight-shape): enforced only on the training-resume path via
#   check_value_seed_vicreg; excluded from check_compatible/_WEIGHT_FIELDS (a frozen opponent's
#   forward never touches it). 0.0 = OFF (loss byte-identical).
# v65: gen3_unconditional_move_legality_v1 — move-belief LEGALITY is now UNCONDITIONAL. A move a
#   species physically CANNOT LEARN always carries ~zero prior mass; there is no flag and no opt-out,
#   because it is a correctness property rather than a feature. `learnset_gate` is DELETED from
#   `damage_tables.build_move_prior_logits`, and `move_candidate_floor` (which used to double as the
#   on/off switch via `floor > 0.0`) is demoted to what its name says: the LEGAL-BUT-UNOBSERVED base
#   probability, default 0.02 (was 0.0). STAMP-ONLY migration — no new field. Nothing can be toggled,
#   so there is nothing to record; the version stamp exists to say "a pre-v65 checkpoint was trained
#   on a prior that gave phantom mass to unlearnable moves". Pre-v65 configs recorded
#   move_candidate_floor=0.0, which check_compatible now rejects against the 0.02 default — deliberate,
#   loud, and NOT migrated up (rewriting 0.0→0.02 would let an incompatible belief load silently).
#   NO ARCH_SIGNATURE bump: the prior buffer is non-persistent and unchanged in shape, and every
#   floor > 0 config produces a bit-identical buffer before and after (only floor == 0.0 changes).
# v67 is the gen3_deadline_clock_v1 STAMP — the obs CLOCK group goes 1 → 3 scalars (log-elapsed +
#   remaining-linear + log-remaining), so GLOBAL_ENV_DIM 18 → 20 and the obs 2667 → 2669, and the
#   move/global context widths that read `_gl['clock']['dim']` move with it. Obs width + weight
#   shapes change together, so no migration is possible — the ARCH_SIGNATURE carries the break.
#   Motivation (measured on 14/14 timeout losses at ai_v9_09 step 16M): the critic reported a
#   POSITIVE V on the last decision before a −30 forfeit in 13 of 14 games (mean +9.33, mean
#   terminal TD surprise −39.3) and was RISING into the forfeit in 10 of 14. The single
#   log-ELAPSED scalar gave the last 20 turns 1.5% of its range; log-REMAINING gives them 55.1%
#   (37×), which is the link TD must fit FIRST before it can bootstrap value back down a
#   200-turn episode.
# v70: gen3_refine_loop_removed_v1 — the between-layers refine loop is DELETED, and with it FIVE
#   fields: `damage_refine_rounds`, `threat_refine_outgoing`, `threat_unrevealed_outgoing`,
#   `threat_status_refine`, `move_belief_single_compute`. The loop was 0 rounds in production, and
#   0 rounds is exactly what made the three `threat_*` flags UNREACHABLE — each hard-requires
#   damage_refine_rounds>0, which is itself mutually exclusive with `damage_op_prefuse` (the
#   production placement), so setting any of them RAISED at extractor build.
#   `move_belief_single_compute` only chose which posterior the refine callback re-read; with no
#   callback it was INERT (production recorded it True and it did nothing).
#   The expected-latent OUTGOING math is NOT deleted — it was re-homed onto the live outgoing kernel
#   at a usage prior (gen3_unrevealed_outgoing_prior_v1) and runs unconditionally there.
#   Forward BIT-IDENTICAL on the production config; NO ARCH_SIGNATURE bump (no module built, no
#   weight shape moved, no forward value changed). Stale keys are POPped by the v70 migration.
# v71: gen3_tiered_pipeline_v1 — the TIER ORDER becomes the ONLY order. `move_belief_prefuse` and
#   `damage_op_prefuse` selected between a PRE- and a POST-transformer placement for the move belief
#   and for the spread/HP-type + DamageOperator group; production ran PRE for both. The POST call
#   sites are DELETED and the PRE placement is unconditional, so the two flags no longer select
#   anything and are removed. `damage_reattend` (v31) goes with them: it re-attended the physics onto
#   the team tokens AFTER the pools, which is a compensation for computing the physics post-attention
#   — the thing this step removes — and it was off in production.
#   `prefuse_proj` is now built whenever `damage_op` is on (it was gated on `damage_op_prefuse`,
#   which required `damage_op`), so the production state_dict is UNCHANGED.
#   Forward BIT-IDENTICAL on the production config; NO ARCH_SIGNATURE bump.
#   ⚠️ This BREAKS every non-prefuse config BY DESIGN, and the v71 migration REFUSES them with a
#   clear error rather than popping the key — `move_belief_prefuse` changed no weight shape, so a
#   silent pop would load a post-ordering checkpoint into a pre-ordering forward with nothing
#   downstream able to notice.
# v75: the SimSiam LATENT belief is DELETED — `opp_belief_latent` and `opp_belief_latent_coef` go,
#   along with the predictor MLP, the `belief_target_slots` training-only obs key that fed it, and
#   the env work that built that key every decision. It was a side readout: the latent never
#   entered pi or vf, so removing it changes no forward value on any config that had it off.
#   It cost ~13% of the train step (marginal +341 ms at the production batch, measured against a
#   +349 ms `opp_belief_cls_k=6` that DOES feed both projections), and its own role-geometry probe
#   concluded decodable != helps. Predicting the opponent's unrevealed mons is untouched: the
#   species CE, the moves BCE and the T0 species prior all remain.
#   ⚠️ The v75 migration REFUSED a config that recorded `opp_belief_latent=True` — unlike v71's
#   forward-only flags, this one carried PARAMETERS, so such a checkpoint's state_dict holds keys
#   the live extractor has no home for. (That branch is now below the v76 floor — the blanket
#   pre-generation refusal subsumes it; the zip-kwargs sanitizer keeps the per-field judgment.)
# v76 (gen3_ctx_dedup_v1): the per-side ENCODED active contexts are DELETED from both projection
#   heads — `ProjectionAssembler.active_ctx_encoder` no longer exists and both projection input
#   widths shrink by 2·32. The content was duplicated delivery with a 1:1 entity-native
#   replacement already live (the E2 injection scatters each side's FULL raw ctx block onto its
#   active token; the global token is a second route). The `active_ctx_hidden` ModelVersion
#   field goes with the module (no migration branch needed: the floor rises to 76 with the
#   signature, so no pre-v76 config is ever migrated). state_dict changes → the signature
#   carries the break; fresh lineage.
# v77 STRUCTURAL (gen3_intent_move_cell_v1, the G3 gate of design_conditional_execution.md):
#   `intent_move_cell` — the POLICY-side alpha consumer. The c2 status-consequence family is
#   re-delivered through the pointer MOVE cell as a per-action ABSOLUTE, alpha-conditioned: the
#   burn/sleep consequence channels become UNRENORMALIZED alpha-expectations over the op's own
#   top-K seat candidates (`f(m, SWITCH)=0` is exact — a switching active neither attacks nor
#   receives the status), the k-independent c2 columns ride raw vs the opp ACTIVE with the seat
#   mass `alpha_stay` as a decorrelated channel. ON widens the pointer move scorer's in_features
#   by INTENT_MOVE_CELL_DIM through a zero-init projection (identity at init, M1-guarded); OFF
#   builds no module and is byte-identical → NO ARCH_SIGNATURE bump. Requires opp_intent +
#   damage_op (+ damage_topk_k>0 at runtime, fail-loud). Old configs migrate to False.
# v78: gen3_flag_surface_p1_v1 — the TIER-1 flag-surface cleanup. EIGHT fields are DELETED with the
#   modules behind them; nothing is added.
#   (1) The ZARCH family — `zarch_film`, `zarch_dim`, `zarch_lut`, `zarch_lut_teams`,
#       `zarch_recon_coef`, `zarch_vicreg_coef` — goes with `ZArchEncoder`, the two FiLM generators,
#       the per-team LUT Embedding + its `team_signature` roster table, and `attach_zarch_lut`. The
#       line it existed to test is CLOSED and the result was NULL twice over: the LUT arm — a FREE
#       per-team code, i.e. the sharpest possible removal of the conditioning-signal limit — moved the
#       N=20 multi-team ceiling by +0.024 with CI [-0.016, +0.064], and the orthogonal 2x2 measured
#       COUNT (N 20->10, +0.077 SIG) dominating CONDITIONING (+0.027 n.s.). Every gen-8/9/10 run
#       recorded it OFF, so deleting it changes no production forward.
#   (2) The SEED-PRESSURE pair — `seed_quantile` (v63) + `value_seed_vicreg_coef` (v62) — goes with
#       `seed_quantile.py` and `seed_vicreg.py`. BOTH cap at ~1-D differentiation of the k=4 value
#       seeds and the two measurements meet in the middle: gen-6's VICReg satisfied every term with
#       out_effective_rank 1.05 (three seeds identical, one breakaway), and gen-7's quantile arm drove
#       crossing_rate to 0.000 with out_effective_rank 1.157 of 4. A SHARED readout can only constrain
#       each seed along its own weight vector, so no coefficient reaches the orthogonal directions —
#       multiplicity is not the missing axis. `seed_diagnostics.py` (the MEASUREMENT) stays.
#   MIGRATION: POP for a config that recorded them OFF; REFUSE one that recorded zarch_film != 'off'
#   or seed_quantile=True, on the v75 principle — those carried PARAMETERS the live extractor has no
#   home for, so a silent pop would load a state_dict with keys nothing can place.
#   `value_seed_vicreg_coef` and the two zarch coefs are training-only, so any value pops silently.
#   NO ARCH_SIGNATURE bump and the MIGRATION FLOOR stays 76: every deleted module was OFF in
#   production, so the production forward AND state_dict are bit-identical across this change
#   (verified: same state_dict keys, max|delta| 0.0 on pi/vf under designs/production_config.json).
#   Also in v78, with no field consequence: `--use-showdown-bridge` (the deprecated `--use-bridge=node`
#   alias) is deleted and `--use-bridge` now DEFAULTS to `rust`; and three settled toggles are DEMOTED
#   to the config_only tier (`attend_unrevealed_opponents` frozen ON, `value_active_readout` and
#   `damage_matrices_outgoing_all` frozen OFF) — their FIELDS and check_compatible gates are
#   deliberately UNCHANGED, because a demotion removes the SELECT role only. See
#   `agents.model.flag_registry` and designs/flag_registry.md.
# v79 (gen3_pair_history_v1, Tier H-A of design_history_entity.md): the COMPILED history tier.
#   Obs 2669 → 2921: per-mon LAST-ACTION fields (POKEMON_FULL_DIM 116 → 122 — the embedded
#   last-move id is manifest-routed, its raw column zeroed at the slice) + the 180-dim
#   pair-history block h[i,j] (6×6×5 tendency counters, EpisodeTracker-folded from PUBLIC
#   events, log-saturated). New edge family "h" (obs-fed, zero-init, mon×mon) joins the
#   edge_bias_families vocabulary — NOT in the production string, so the family is opt-in;
#   the obs widening is unconditional (retrain-class). No new ModelVersion field and NO
#   ARCH_SIGNATURE bump (the recency precedent): total_dim + the widened role-encoder shapes
#   are weight-field-caught, and the family rides the recorded edge_bias_families string.
# v80 (gen3_unified_value_readout_v1, Stage-3 T3-DELIVER of design_unified_belief.md §3): the
#   critic's UNIFIED ENTITY POOL — `value_entity_pool`, opt-in. K learned queries attention-pool
#   the critic's entity-row set (the 12 post-transformer team tokens + the op's 6 per-our-mon
#   incoming rows, per-source type embeddings, explicit NaN-safe softmax) through a ZERO-INIT
#   output projection appended to vf ONLY (the intent_value_reduce placement: pi untouched at
#   any weight). The designed SUCCESSOR contract of the bolt-on vf routes (seed readout /
#   threat-inject) the gen-11 critic_route_audit adjudicates — built so a condemned route has a
#   replacement the next generation can enable in the same config. OFF builds nothing
#   (byte-identical baseline; no ARCH_SIGNATURE bump); ON widens the value projection
#   (weight-field-caught).
# v81 (gen3_event_window_v1, Tier H-B of design_history_entity.md): the EVENT-TOKEN history
#   window. Obs 2921 → 3529: a 32×19 typed event-record block (last-N decision-relevant events —
#   move/switch_in/faint/status/boost/item/hazard/switch_rejected — with attributed hp_delta,
#   outcome/crit/effectiveness, we_first, log-saturated recency, forced-window phase tag),
#   folded by `EventWindowTracker` from PUBLIC events (seq-idempotent, the H-A machinery).
#   The obs widening is unconditional (retrain-class, weight-field-caught via total_dim); the
#   CONSUMER — `history_events`, the event SEATS joining the trunk with per-type projections +
#   the recency embedding — is opt-in (OFF builds nothing, byte-identical). v1 trims, recorded:
#   no faint-cause multi-hot, no item/hazard content ids, SETBOOST/CLEARBOOST skipped.
# v82 (gen3_unified_value_readout_v2): `value_entity_pool_full` — the entity pool's COMPLETE
#   row set (+the refined GLOBAL token, +the hidden-opp belief queries; sources 3 and 4). Its
#   own field because the source-embedding table grows 3→5 (a state_dict shape), keeping
#   v80-shape checkpoints (gen-12's) loadable under full=False. With this, every vf route the
#   critic_route_audit can condemn has ONE successor: the pool.
# v83 (gen3_item_belief_v1): `item_belief` — a learned posterior over each opp slot's HIDDEN
#   item (Smogon per-species item-usage prior ⊕ zero-init trunk delta; cold start == prior
#   exactly). Supervised as the BeliefBank's SEVENTH row (CE vs the privileged true item num
#   at revealed slots, --item-belief-coef). The op's Choice-Band-conditional tail consumes
#   P(CB) from the PUBLISHED posterior at the unrevealed branch, replacing the static
#   SPECIES_CB_PRIOR scalar there (revealed branch unchanged: exactness stays 0/1). Adds the
#   ItemBelief module (state_dict), so STRUCTURAL, version-checked, own flag.
# v84 (gen3_intent_threshold_v1): `intent_threshold` — the α-weighted THRESHOLD operator
#   `p_thresh(τ,⋛) = Σ_k α_k·1[damage(k,me) ⋛ τ]` (design_conditional_execution.md §3.0, build
#   step 3). One contraction over the op's existing per-candidate pair cells lands FIVE
#   mechanics at once through the pointer MOVE cell (Focus Punch executes / Sub survives /
#   Endure·p_KO / Destiny Bond·p_KO / Endeavor survives-to-act) and produces `p_KO` — the
#   calibrated "am I about to die this turn" — for the CRITIC (the ledger-H1 payoff: the
#   critic previously inferred it from _chan_max's hard max). Two zero-init projections
#   (state_dict): the move-cell block widens the pointer move cell by INTENT_THRESH_MOVE_DIM;
#   the vf block appends INTENT_THRESH_VF_DIM after the entity pool. STRUCTURAL, own flag.
# v85 (gen3_intent_conditional_v1): `intent_conditional` — the REMAINING α-conditioned mechanic
#   cells (design build steps 4+7): Counter / Mirror Coat as the α-weighted CATEGORY sums
#   ("the purest read-the-opponent moves in gen3 — literally unplayable without an intent
#   model"), flinch's missing (1−α_SWITCH) conditioning, Explosion's p_executes
#   (1 − Σα·is_protect) + into-switch mass (the H1 companion facts), and Pursuit's ×2
#   never-miss doubling trigger — CORRECTED against the rust port: the strike hits the
#   DEPARTING mon, not a β-weighted arrival, so no β enters. One zero-init projection widens
#   the pointer move cell by INTENT_COND_MOVE_DIM. STRUCTURAL, own flag.
# v86 (gen3_op_lean_forward_v1, design_op_tensors step 3): TWO flags. `op_drop_renders` — the
#   op's flat forward block loses its three RENDER regions (outgoing matrix / incoming matrix /
#   OAX), which have had no forward consumer since gen3_no_concat_v1; the matrices' SELECTION
#   machinery still runs and every consumer value survives as a typed stash, so out_dim (and
#   out_gain — a state_dict shape) shrink while every surviving offset is unchanged (renders
#   always appended last). `op_believed_lean` — the lean d3 physics (`_incoming_rolls`) price
#   the attacker from the BELIEVED spread instead of the legacy de-timid fiction (the B-spread
#   correctness fix at the last de-timid site the edges read); forward-math change, no shape.
# v87 (gen3_value_direct_routes_v1): two direct CRITIC routes, both zero-init vf-tail appends.
#   `value_clock` — the v67 deadline clock's 3 raw scalars get the explicit critic route the
#   fix was validated for (its surviving indirect route, the nmr concat, was audited dead).
#   `value_intent` — the published α/β posteriors AS DISTRIBUTIONS (α over its K belief-sorted
#   seats + SWITCH, β over the 6 slots): α previously reached vf only as a weighting inside
#   intent_value_reduce's physics cells and β not at all — the block was ORDERING (T2 heads vs
#   the assembler), which the post-assembler tail dissolves. Both widen the value projection
#   (state_dict), so mismatches are shape-caught; the checks name the cause.
MODEL_CONFIG_VERSION = 87

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
# v2 (gen3_unified_v2): turn-history TurnDelta slot expanded to 88 dims —
#   actor / target / switch_to species IDs (×6), boost deltas (×14), phase flag,
#   target_hp_delta, per-slot HP-level vectors, target-status onehots (×14, at
#   move-fire time, for Flash Fire-vs-frozen and sleep-talker reads). The history
#   embedding now reaches the species_embedding table for the first time, a new
#   wire that's not weight-compatible with v1 even if total_dim coincidentally
#   matched.
#
# v3 (gen3_abilities_v1): per-Pokémon ability block expanded 2 → 3 dims
#   ([ability1_id, ability2_id, known_flag]). For unrevealed opp slots the two
#   dex-possible Gen 3 abilities are written so the model has prior knowledge
#   (e.g. Snorlax = Immunity OR Thick Fat) instead of a flat zero. The role
#   encoder embeds BOTH ability IDs through the existing ability_embedding
#   table — a wire that didn't exist in v2. POKEMON_FULL_DIM 97 → 98, total
#   obs dim 2414 → 2426.
#
# v4 (gen3_abilities_v2): ability block grows to 4 dims with an inserted
#   `dominance` scalar — the Smogon-observed probability of ability1.
#   Layout becomes [ability1_id, ability2_id, dominance, known]. Priors are
#   now sourced from data/pokemon/gen3_ability_priors.json (top-2 by Smogon
#   usage), replacing the dex-slot-order approach from v3. POKEMON_FULL_DIM
#   98 → 99, total obs dim 2426 → 2438. The role encoder picks up the
#   dominance scalar as a passthrough float alongside the two ability
#   embeddings.
#
# v5 (gen3_move_outcome_v1): each turn-history TurnDelta slot gains move-outcome
#   reporting — our/opp move-outcome onehots (hit/miss/fail, ×6), our/opp crit
#   bits (×2), and the |cant| reason onehot widens 5 → 11 (recharge/taunt/
#   disable/imprison/truant/nopp added, with "move:"/"ability:" prefix
#   normalization). These are pass-through scalars routed through the existing
#   history embedding, inserted before the species-ID tail. TURN_DELTA_DIM
#   88 → 108 (+12 from the wider cant onehot, +8 from outcome/crit); total obs
#   dim shifts by N_HISTORY_TURNS × 20. Not weight-compatible with v4 — the
#   history projection input width changed.
#
# v6 (gen3_modular_v1): pure structural refactor — forward_internal decomposed
#   into phase nn.Modules (Embeddings / ObsUnpack / PokemonEncoder /
#   TeamTransformer / CLSPool / ProjectionAssembler). The math, dims, and outputs
#   are byte-identical to v5, but state_dict keys are now phase-prefixed
#   (e.g. move_network.* → pokemon_encoder.move_network.*, our_cls →
#   cls_pool.our_cls). Old checkpoints are intentionally incompatible so they
#   fail with a clean arch-family error instead of an SB3 strict-load KeyError.
#
# v7 (gen3_dual_value_v1): value-dedicated CLS readout (H4 / Option C). CLSPool
#   gains a third learned query (`value_cls`) that attends over all 12 team
#   tokens to produce a global value summary; ProjectionAssembler now emits a
#   (pi_combined, vf_combined) pair, and the root extractor has a second
#   projection head (`value_pre_norm` + `value_projection`). `forward` returns a
#   (pi_features, vf_features) tuple consumed by the new
#   `Gen3DualHeadMaskablePolicy`. The transformer body stays shared; only the
#   readout + projection + critic mlp branch are now independent. New weights and
#   a tuple-returning forward make this incompatible with v6 checkpoints.
#
# v8 (gen3_live_state_v1): the active-context + global-env blocks are re-sourced from
#   the event-sourced LiveView and substantially enriched (retrain-class). Active
#   context grows 23 → 55: the volatile block goes from a hand-picked 9 to the full
#   source-derived gen3 set (VOLATILE_DIM=41, crash-don't-drop, perish/stockpile
#   counters normalised) — recovering ~30 dropped volatiles (Disable/Encore/Taunt/
#   Destiny Bond/Curse/Yawn/Flash Fire/partial-trap/…). Global env grows 13 → 18:
#   weather is event-sourced with cause-aware permanence + turns-remaining (ability
#   weather = permanent, move weather = 5-turn countdown — read from the |-weather|
#   protocol, never guessed), the dead gen4+ weather slot is dropped, and per-side
#   Safeguard + Mist are added alongside Reflect/Light Screen. The weather feature the
#   extractor broadcasts into per-mon move context widens 6 → 7. Obs dim 2734 → 2823;
#   the global-token / active-ctx projection input widths all shift. Not weight-
#   compatible with v7.
#
# v9 (gen3_own_spread_v1): the own-team spread block (per-mon IVs/EVs/nature, 18 dims ×6
#   slots) now carries REAL data instead of constant fallbacks. gen3ou has no team preview,
#   so poke-env's apply_teambuilder_team (which matches the empty team-preview list) never
#   attached the spread, and own Pokemon.ivs/evs/nature stayed None — the spread block had
#   been emitting a constant vector (IVs all-31, EVs all-0, neutral nature) for every own mon,
#   i.e. zero signal. Fixed in the poke-env fork: Battle.parse_request now calls
#   backfill_teambuilder_spread() after building the team from the request, matching the
#   declared teambuilder team by species and filling in IVs/EVs/nature (spread only — it does
#   not re-run the full _update_from_teambuilder, so request-derived moves/PP/stats are
#   untouched). The obs spread block + LiveView read mon.ivs as before, now populated. Obs DIM
#   is unchanged (still 2823) — only the spread VALUES change — but the meaning of those dims
#   changes, so this is retrain-class: old checkpoints must not silently load.
#
# v10 (gen3_turn_delta_v2): TurnDelta is now folded from the event log (Step 4 of
#   the event-sourced battle migration). New per-decision-window fields: an 8-dim
#   faint-cause multi-hot per side (attack/hazard/weather/status/recoil/selfko/
#   leechseed/other), and our_attempted_move_id (the move we pressed, preserved even
#   when it never fired — freeze/sleep/flinch/cant/KO-before-act). attempted_switch_to
#   is NOT encoded (a pressed switch always executes, so it == switch_to); faint counts
#   live on the dataclass for reward but aren't encoded (redundant with the faint flags
#   + cause popcount). The cant one-hot switches to the authoritative gen3_effects vocab
#   (slp/frz/par/flinch/recharge/attract/disable/taunt/imprison/focuspunch/nopp/truant),
#   crash-don't-drop. Volatiles added to the active-context block: doomdesire/futuresight
#   (`-start` future-move volatiles) + the 11 gen3 ability-activation volatiles (Immunity/
#   Synchronize/Oblivious/Insomnia/Limber/OwnTempo/ShedSkin/StickyHold/SuctionCups/
#   VitalSpirit/MagmaArmor — poke-env's -activate path records them as effects; MagmaArmor
#   required adding Effect.MAGMA_ARMOR to the fork's enum); the event-log fuzz's per-decision
#   check + training smoke caught doomdesire/immunity. Ability activations now ALSO reveal
#   the opponent's ability persistently (abstract_battle -activate handler sets mon.ability
#   when None → per-mon ability block flips known=1), so the 11 ability-activation volatiles
#   COLLAPSE to one shared `ability_activated` slot (identity is in the ability block; the
#   volatile is just a hint to go look). VOLATILE_DIM 41 → 44. TurnDelta also folds STATUS
#   TRANSITIONS from the event log: our/opp status_applied + status_cured (4 × 7-dim
#   onehots) — the per-turn event (e.g. Lum Berry curing Toxic to enable a Dragon Dance),
#   distinct from the current-status snapshot; the cause-identity stays in the item/ability
#   block. Plus our/opp item-used BITS (2) marking an item was consumed/removed this window
#   (just a bit — the WHICH is in the per-mon item block, parity with ability_activated).
#   The embedded-ID positions are no longer hardcoded in the extractor: a single
#   TURN_DELTA_EMBEDDED_IDS manifest (in turn_delta_encoder) drives both the encoder
#   layout and features_extractor.embed_delta_slot (11 embedded IDs: 3 move + 2 type +
#   6 species). TURN_DELTA_DIM = 157, obs dim 2823 → 3299. Builds on v9 (own-team spread
#   backfill carries through). Not weight-compatible with v9.
#
# v11 (gen3_turn_delta_v3): turn-history window correctness fix. `prev_N_delta_vecs` was
#   folding each of the N history slots over `events_since(cursor)` — i.e. that turn's
#   cursor THROUGH NOW (no upper bound) — so every slot but the most-recent reported the
#   *latest* turn's event-derived fields (move/outcome/boosts/status/faint-cause), and the
#   per-step cost was O(N²). Now each slot folds exactly its own decision window
#   (`events_between(cursors[-1-i], cursors[-i])`; end=None for the most-recent). Obs dim is
#   unchanged (3299) — only the turn-history values change (older slots now carry their own
#   turn) — so this is retrain-class, not weight-shape-incompatible.
#
# v12 (gen3_trapping_signals_v1): route the three trapping signals into the model so it can
#   learn the hidden-information trap read (Arena Trap / Shadow Tag / Magnet Pull / Mean Look).
#   (1) + (2) two new reactive obs bits from the server-authoritative LegalActions snapshot —
#   trapped (confirmed cannot switch; redundant with the mask but explicit) and maybe_trapped
#   (the opponent MIGHT trap us; switches stay legal, so this is the only way the model can see
#   the risk before attempting a blind pivot and eating a rejection). They sit before the
#   matchups in the reactive block, so the extractor picks them up in non_matchup_rest;
#   REACTIVE_DIM 300 -> 302. (3) the rejected pivot becomes a first-class history event: a new
#   EventKind.CHOICE_REJECTED is recorded out-of-band (poke-env intercepts |error|[Unavailable
#   choice] before parse_message, so a duck-typed hook in _handle_battle_message calls
#   Gen3Battle.record_choice_rejected), TurnView folds it (attempted_rejected), TurnDelta gains
#   attempted_switch_rejected + the restored attempted_switch_to, and each TurnDelta slot gains
#   2 dims — an attempted_switch_rejected bit + the embedded attempted-switch species id
#   (manifest entry #12). TURN_DELTA_DIM 157 -> 159. Obs dim 3299 -> 3321 (+2 reactive +
#   N_HISTORY_TURNS x 2 history). Builds on v11. Not weight-compatible with v11.
# gen3_item_num_fix_v1: the per-Pokémon item id is now the true item-dex `num` (from data/, via
#   the gen3_data facade), not Showdown's `spritenum` as before. Obs dim unchanged (3321) and the
#   item embedding table size is unchanged (max_items=600 still covers the new max, 499), but the
#   item id -> item meaning is re-mapped for every item, so item embeddings learned under the old
#   ids are semantically invalid. Re-meaning an obs block is retrain-class. Builds on
#   gen3_trapping_signals_v1; not weight-compatible with it.
#
# gen3_move_effects_v1: action-aligned per-move EFFECT features in the reactive block. The only
#   per-move signals that previously reached the policy head in REQUEST (action) order were base
#   power and the type multiplier — so for status/utility moves (power 0, neutral multiplier) every
#   option looked identical at the head, and the model could not tell a setup move from a heal from
#   a wasted Toxic (it clicked immune Toxic into Poison-types for many turns). Now each of the 4
#   request-order move slots carries 9 flags — is_boost, is_heal, is_protect, is_phaze, is_hazard,
#   inflicts_status, status_will_land, pp_fraction, status_will_land_known. Static flags are derived
#   in the acquisition tool
#   from the field Showdown keys each mechanic on (flags.heal, volatileStatus, forceSwitch,
#   sideCondition, primary `status`, declarative self-positive boosts) PLUS a curated callback
#   override for Belly Drum (onHit-only boost); Curse's type-conditional setup is resolved live in
#   the encoder. status_will_land is a PRIOR-WEIGHTED probability in [0,1] (priors first, then
#   confirmation — same ability-distribution path as the matchup cells): 0 on a certain block
#   (type immunity / already statused / Substitute), else 1 − P(ability blocks the status) over the
#   opponent's Smogon ability prior, collapsing to 0/1 once the ability is revealed; the trailing
#   status_will_land_known bit flags confirmed-vs-prior with the SAME predicate the per-mon ability
#   block's `known` flag uses (revealed ability OR a type-certain hard block), so the policy can
#   tell a confirmed outcome from a prior estimate — parity with how abilities are routed. The block
#   sits before the matchups, so the extractor picks it up in non_matchup_rest → both policy and
#   value projection input widths grow (auto-discovered). REACTIVE_DIM 302 → 338; obs dim 3321 → 3357.
#   Builds on gen3_item_num_fix_v1; not weight-compatible with it.
# gen3_incoming_damage_v1: per-our-mon INCOMING-DAMAGE / OHKO BELIEF block (incoming_damage.py +
#   gen3_{move,spread,item}_priors): for the opp active vs each of our 6 mons, the phys/spec
#   expected-damage-fraction + mode-max P(KO) (gen3 damage formula + fixed-damage branch
#   [Seismic Toss/Night Shade/…] + Reflect/Screen/Sub/burn/weather modifiers + roll→P(KO), over the
#   usage-prior belief: revealed∪prior moves, offensive-tail stat) + P(outspeed) over the Speed
#   distribution, then 3 opp recovery scalars (Suicune-Rest discriminator). Sits after move-effects,
#   before the matchups → flows to both heads via non_matchup_rest (auto-discovered widths).
#   REACTIVE_DIM 338 → 371; obs dim 3357 → 3390. Builds on gen3_move_effects_v1; not weight-compatible.
# gen3_incoming_damage_v2: re-calibrates the incoming-damage / OHKO belief VALUES (same 33-dim block,
#   same obs dim 3390 — only the numbers change, so it's retrain-class, not weight-shape). Two
#   complementary belief-value fixes for the calibration tail found on run_20260606_204351 (17% of
#   direct-hit deaths read P(KO)<0.25): (1) P(KO) was too timid on near-OHKOs — the offensive-stat
#   tail percentile is raised 0.85→0.95 (the KO magnitude rides the tail; expected-damage
#   re-normalises to the mean, so the chip belief is unchanged) AND a gen3 critical-hit term
#   (_CRIT_P=1/16, ×2, screen-ignoring) is folded into P(KO), so a hit that only KOs on a strong set
#   or a crit reads a calibrated risk instead of ~0; (2) the candidate set is widened so the killing
#   move is no longer silently absent — a revealed bare Hidden Power (dex BP 0) expands into per-type
#   candidates (~70 BP, typed from the HP tracker's narrowed distribution / Smogon HP prior),
#   variable-power Return/Frustration (dex BP 0) are priced at 102 BP, and the prior floor/cap widen
#   (0.12→0.05, 4→6 per channel) so a low-usage super-effective coverage move survives into the pool
#   (the per-defender max over p_in_set·P(KO) is the real type-effectiveness gate). The HP tracker is
#   now threaded into the incoming-damage encoder. Not weight-compatible with v1 (the belief values a
#   reload would read are different → old critic readings of the block are invalid).
# gen3_markovian_progress_v1: adds the turns_since_progress reactive scalar (vec[14]) — the
#   log-saturated no-progress clock (design_markovian_reward_and_features.md §5.1), an
#   EpisodeTracker-owned cross-turn counter threaded into encode() like the HP tracker.
#   REACTIVE_SCALAR_DIM 14 → 15 → REACTIVE_DIM 371 → 372, obs dim 3390 → 3391. The scalar is
#   present in every run (the clock always tracks it for the obs); the no-progress PENALTY +
#   the obs-keyed reward reframes are gated on the reward's bias_redesign flag, so the
#   single-variable material-clutch-fix run and the bias-redesign run share one architecture.
#   The reward redesign also folds the material spine into a PBRS Φ_mat and renames the belief
#   PBRS field (pbrs_material → pbrs_belief); those are reward-VALUE changes (retrain-class) that
#   need no further arch bump. Not weight-compatible with gen3_incoming_damage_v2 (obs dim +1).
# gen3_incoming_crit_split_v1: SPLITS the incoming-damage belief's P(KO) into a modal no-crit line +
#   a per-channel crit-risk DELTA (crit-inclusive − no-crit ∈ [0, _CRIT_P]), and adds a per-mon
#   threat-PROVENANCE scalar (the dominant KO threat's p_in_set: 1.0 = a REVEALED move, <1.0 = a
#   usage-prior GUESS, 0.0 = no candidate can KO). Motivation: the model over-weighted uncontrollable
#   crit RNG (it should optimise EXPECTED value over the modal line, with crit as a priced tail) and had
#   no signal for how much of a threat is KNOWN vs guessed — both validated as gaps by the
#   representation-probe harness (the rep barely encodes damage spread). The crit risk is exposed as the
#   DELTA (not the near-redundant absolute crit-inclusive line, which is ≤6% above no-crit and gets
#   buried after standardization). INCOMING_PER_MON 5 → 8 → INCOMING_DMG_DIM 33 → 51 → REACTIVE_DIM
#   372 → 390, obs dim 3391 → 3409. Crit was ALREADY computed (folded into P(KO) since v2); this unblends
#   it as a delta + adds provenance, so the underlying numbers are unchanged — but the block layout/width
#   differ, so it is not weight-compatible with gen3_markovian_progress_v1.
# gen3_move_slot_align_v1: FIXES a per-move obs misalignment (GIGO). The active-move features in
#   reactive.py (base power vec[0:4], type multiplier vec[4:8], the 36-dim move-effect block) were
#   filled by iterating `battle.available_moves`, which poke-env builds with DISABLED moves dropped
#   (`available_moves_from_request`). The action mask / mapper index `legal.move_slots` (request-slot
#   order, disabled KEPT) → so under a disabled non-last slot (Disable / Taunt / Imprison / 0-PP) every
#   per-move feature shifted out of alignment with its action logit (feature slot k described a
#   DIFFERENT move than action 6+k), and the trailing slot kept the np.ones(4)/4 default — which decodes
#   to a phantom 4× super-effective KO threat on a legal action. Now the loop iterates request-slot
#   order via `_request_slot_moves` (disabled kept, typed-HP preserved) and the unwritten-slot default
#   is the neutral 0.25 (1×). Same dims (obs 3409 unchanged), VALUES only on the disabled-slot /
#   <4-move / no-opp-active cases — so it is retrain-class (not weight-shape), not byte-compatible with
#   gen3_incoming_crit_split_v1. The common all-moves-available decision is byte-identical.
# gen3_protect_odds_v1: adds TWO reactive scalars (vec[15] our active, vec[16] opp active) — P(a
#   Protect/Detect/Endure succeeds NOW) under the gen3 floored-doubling stall rule. Showdown's gen3
#   format inherits the stall condition through gen4 → gen5 (NOT the base data/conditions.ts *3): the
#   counter starts at 2 and DOUBLES each consecutive successful stall move (gen5), capped at 8 (gen4
#   counterMax → "the chance does not fall below 1/8") → 100/50/25/12.5 then a 12.5% floor. Sourced from
#   each active mon's `LivePokemon.protect_counter` (poke-env's consecutive-successful-stall counter,
#   reset on switch/faint/non-stall move/failed roll) via the LiveView read-model — never raw poke-env.
#   The model had no other view of the stall counter (poke-env doesn't enumerate the 'stall' volatile,
#   and turn-history saliency decays before a chain can be counted). Public for both sides (the opp's
#   counter derives entirely from their revealed move stream → no leak). REACTIVE_SCALAR_DIM 15 → 17 →
#   REACTIVE_DIM 390 → 392, obs dim 3409 → 3411. Verified: protect_success_prob_fuzz_test.py (encoded
#   scalar == the gen3-correct prob for the live protect_counter, + the empirical % match). Not
#   weight-compatible with gen3_move_slot_align_v1 (obs dim +2).
# gen3_status_cure_moves_v1: ADDS two static per-move EFFECT bits to the action-aligned move-effect
#   block — cures_self_status (Refresh clears the user's own status) and cures_team_status (Heal Bell /
#   Aromatherapy clear the whole party's). Motivation (prober-verified on ai_v6_01): the policy head
#   had NO per-move signal that a move CLEARS status — Refresh read as an inert move (base power 0, all
#   effect flags 0), so the head routed its own status onto Recover/switch (intervention: removing a
#   Toxic moved P(recover)/switch by ~11pp each but P(refresh) by ~1.5pp) and under-used the cure
#   (~1.4% when badly poisoned). The cure lives in an onHit callback (invisible declaratively), so the
#   bits are a curated override in the acquisition tool (like Belly Drum) and read against the per-mon
#   status one-hots the head already sees — provide the fact, let it learn. MOVE_EFFECT_FEATURES 9 → 11
#   → MOVE_EFFECTS_DIM 36 → 44 → REACTIVE_DIM 392 → 400, obs dim 3411 → 3419 (stacks on gen3_protect_odds_v1).
#   Not weight-compatible (move-effect block widened); the non-cure obs values are otherwise unchanged.
# gen3_sleep_wake_belief_v1: ADDS a 3-dim per-mon SLEEP WAKE belief block to each team slot (after the
#   HP block) — [sleep_is_deterministic (1.0 = Rest fixed-duration source), p_wake (COMPUTED P(wake on the
#   next move attempt) over the verified gen3 sleep tables: opp time=random(2,6)∈{2,3,4,5}, Rest time=3,
#   Early Bird halves; marginalised over the opp's Smogon Early-Bird prior, collapsing to exact 0/1 for our
#   own mon or a revealed opp), sleep_counter_reliable (0.0 once a Sleep Talk / Snore turn has corrupted
#   poke-env's +3-noisy counter)]. Motivation: poke-env exposes only Status.SLP + a noisy turn counter — NOT
#   the rolled duration, remaining time, or the source move — so a policy reading the raw counter must LEARN
#   the gen3 sleep RNG and cannot tell deterministic Rest from a random opp-sleep at the same counter. We
#   COMPUTE the wake odds (provide-the-fact) and read the Rest source from our event log's [from] clause
#   (poke-env discards it). Mechanics research + adversarial re-simulation: the four P(wake) tables were
#   re-derived bit-for-bit; Sleep Talk +3 counter-noise empirically confirmed → the reliability bit instead
#   of reconstructing Showdown's skippedTime switch refund. Fuzz-calibrated vs the real sim RNG. POKEMON_VECTOR_DIM
#   106 → 109 → POKEMON_FULL_DIM 107 → 110 (+3 per slot × 12), obs dim 3419 → 3455. Stacks on the same
#   unshipped change as the status-cure bits; not weight-compatible (per-mon slot widened).
# gen3_wish_reserve_v1: RESERVES two reactive scalars (vec[17] our side, vec[18] opp side) for a future
#   pending-Wish "floating heal" signal — NOT wired (the encoder leaves both 0.0). Reserved now so wiring
#   Wish later (a Wish queued for a side heals the mon switched in at the end of the next turn) is a
#   VALUES-only change with NO obs-dim / ARCH bump. REACTIVE_SCALAR_DIM 17 → 19 → REACTIVE_DIM 400 → 402,
#   obs dim 3455 → 3457. Pure placeholder: with the dims at 0 the obs is byte-identical to
#   gen3_sleep_wake_belief_v1 EXCEPT for the two reserved zeros + the shifted move-effect/incoming/matchup
#   offsets, so it is retrain-class (weight-shape) but carries no new information until Wish is wired.
# gen3_wish_wired_v1: WIRES the reserved wish_floating scalars (vec[17] our side, vec[18] opp side) with
#   the pending-Wish "floating heal" signal — a VALUES-only change (same obs dim 3457, no shape change), so
#   a gen3_wish_reserve_v1 checkpoint is retrain-class-incompatible only in the two dims' values. gen3 Wish
#   (INHERITS the gen4 condition, NOT base): heals the RECIPIENT's floor(maxhp/2) at the END of the turn
#   AFTER cast, SLOT-keyed (survives faint / Roar-phaze / switch / self-KO — the slot's occupant at resolve
#   is healed; gen3 sends replacements in mid-turn before residuals), duration 2, double-Wish on an occupied
#   slot FAILS, full-HP resolve is silent. poke-env tracks NONE of this → reconstructed from our event log
#   (observation/wish_belief.py): pending for a side iff it successfully cast Wish last turn (double-Wish-
#   aware). Because the heal is the RECIPIENT's maxhp/2, the heal fraction is ALWAYS ≈0.5 — so the encoded
#   value is a flat WISH_HEAL_FRACTION (0.5) when pending, 0.0 else: no max-HP read, no GIGO. Fuzz-calibrated
#   vs the real sim (the |-heal|[from] move: Wish resolve confirms the pending signal fired the turn before).
# gen3_rest_loop_stall_v1: RE-MEANS the turns_since_progress no-progress-clock scalar (vec[14],
#   gen3_markovian_progress_v1) — a REST-LOOP (our active Rested earlier this episode, woke, and re-Rested
#   without Sleep Talk) is now classified a NO_OP (stalled) instead of a free defensive heal, so it ADVANCES
#   the clock (obs) and CHARGES no_progress_tax (reward, when the clock charge is active — bias_redesign /
#   all_shaping_pbrs) like any other wheel-spin. A Sleep-Talk mon (legitimate act-while-asleep loop) and a
#   WINNING residual rest-stall (Toxic/Leech chipping the opp down → caught by _is_progress first) stay
#   exempt. VALUES-only on rest-loop turns (same obs dim 3457, no shape change) — but it re-means an obs
#   feature, so it is retrain-class: an old checkpoint won't load (loud arch-family error), which is correct
#   since it was trained with the prior clock semantics. (progress_clock.py: the heal-grace bypass.)
#   This rest-loop signature ALSO covers a SECOND no-progress-clock (vec[14]) refinement authored alongside
#   it and folded in WITHOUT its own signature bump (owner decision — a values-only clock change; the live
#   ARCH below has since moved on for unrelated reasons): a self-status-cure move (Refresh) used with NO
#   status to cure (`cures_self_status` + `our_status_cured is None`) is a NO_OP charged BEFORE the progress
#   check (a definitional-no-op short-circuit, like capped Spikes), so even a WINNING residual (our Leech
#   Seed / Toxic chipping the opp net-down) can't launder it into "progress" — the Refresh-spam-while-seeded
#   stall. (progress_clock.py: _is_wasted_self_cure short-circuit.)
# gen3_op_move_align_v1: FIXES a DamageOperator OUTGOING move-order bug. The op's per-move OUTGOING blocks
#   (_outgoing_block v23, _status_landing v27, _outgoing_matrix v34) emit one feature group per OUR move and
#   the POLICY head reads group k as action 6+k (request order) — but they READ ctx.all_move_ids[our_active],
#   the per-mon obs block, which is SORTED-BY-ID (the role token concatenates the 4 move encodings, so its
#   value is order-sensitive and the block can't be reordered). Sorted-by-id ≠ request order in ~96% of
#   decisions, so the outgoing tie-break / status-landing / switch-in-KO matrix were positionally misaligned
#   with the actions they inform (an under-`--unified-obs` correctness bug, since the CPU per-move blocks are
#   masked there). The FIX adds a request-ordered OUR-ACTIVE obs slice (reactive.py `active_req_moves`:
#   [move_num ×4, resolved_type_id ×4, legal_now ×4], from legal.move_slots, the same source the action mask +
#   move-effect block use) → ctx.our_active_req_move_{ids,type_ids,legal}; the 3 op methods read THAT (request
#   order) + gate with the current-decision legality (was the prev-turn, sorted-by-id move_mask). The v36/v37
#   refine OUTGOING methods (discrete_outgoing*) max-pool over our moves → order-invariant, left unchanged.
#   A STRUCTURAL/SHAPE change: REACTIVE_DIM 402 → 414, obs dim 3457 → 3469. Old checkpoints fail loudly (the
#   total_dim weight-shape check AND the arch-family signature), which is correct — they were trained with the
#   misaligned op. Guarded so it can't silently recur: move_alignment_fuzz_test asserts the obs slice IS in
#   legal.move_slots order, and damage_op_test asserts the op's outgoing slot k uses request-slot k.
# gen3_typed_hidden_power_ids_v1: gives each TYPED Hidden Power its OWN distinct move num so OUR side's
#   HP is represented by the move embedding itself, not a soft-type-blend workaround — a VALUES-only obs
#   change (same obs dim 3469 — it stacks on gen3_op_move_align_v1's reactive-block widening; NO
#   weight-shape change: the typed nums 355-370 are previously-unused rows
#   in the move embedding, max_moves=400). KNOWN→DISTINCT, UNKNOWN→TYPELESS+BELIEF:
#   - data/pokemon/gen3_moves.json: bare `hiddenpower` stays num 237; the 16 typed variants get distinct
#     nums 355-370 (deterministic, alphabetical — tools/pokemon_data_extractor/sync.py `_HP_TYPE_NUMS`).
#   - OUR side (type known): the obs move-id channel + the damage-op per-num tables (BP/type/attr/latent)
#     now carry the distinct num & real type, so our HP is a normal typed move (the feature extractor's
#     `is_hp_slot == 237` no longer matches it → it skips the hp_probs soft-type blend) and our OUTGOING
#     HP is priced correctly (was BP-0/type-0 before). The turn-history `our_move` also folds the distinct
#     num (via LegalActions.own_hp_typed_id). This SUPERSEDES the gen3_own_hp_typed_history_v1 hp_probs
#     one-hot workaround (reverted — own-HP hp_probs stays all-zero, correct since the blend is opp-only).
#   - OPPONENT side (type unrevealed — Gen3 never reveals it): the protocol gives bare `hiddenpower` → 237;
#     ALL opp-belief machinery stays on 237 — the HP tracker, the hp_probs soft-type blend, the damage-op
#     237→16-typed-candidate expansion, AND the move-belief PRIOR + LABELS (damage_tables._belief_num and
#     gen3_env._move_num fold every typed-HP usage/label back onto 237, so the opp-HP belief mass is NOT
#     scattered to 355-370). This known/unknown boundary is the load-bearing invariant (fuzzed by
#     move_id_decode_fuzz_test + hidden_power_typed_obs_fuzz_test). Design:
#     designs/ai_v6/design_typed_hidden_power_ids.md.
#   gen3_opp_hp_typed_candidates_v1: the DamageOperator now treats the OPPONENT's Hidden Power as 16
#     ORDINARY typed-move candidates at the distinct dex nums 355-370 (real BP/type from the typed-HP data
#     above) instead of a synthetic appended-16 block; the bare typeless 237 (BP 0) is the masked presence
#     token, and the per-type HP belief (mode off=obs / prior / learned) is scattered onto 355-370. A
#     FORWARD-MATH change to the op (the obs is unchanged + the op out_dim/projection widths are unchanged,
#     so it's not caught by shape checks) → bump ARCH_SIGNATURE so a pre-unification damage_op checkpoint
#     fails loud rather than silently computing the old HP candidates. The HP-type belief + the (v2) token
#     reinjection ride the existing `hp_type_belief_mode` (config v38).
#   gen3_pointer_native_v1 (the FRESH-GENERATION reset, designs/ai_v9/design_pointer_action_head.md §0):
#     the flat positional action head is DELETED — `Gen3DualHeadMaskablePolicy._build` replaces SB3's
#     `action_net` Linear with a raising stub and the `PointerNativeActionHead` scores every action from
#     the token of the entity it selects (move logit k ← the REQUEST-slot-k move token ⊕ its op cells;
#     switch logit j ← our-team token j ⊕ its incoming/OAX cells; struggle ← the latent_pi context) —
#     position-EQUIVARIANT by construction. The state_dict changes shape (no `action_net.*` Linear, new
#     `pointer_head.*` keys) AND the forward changes for every model, unconditionally (no flag), so the
#     signature carries the cross-era break: every pre-generation checkpoint fails the family check loud.
#     No old checkpoint is resumed/warm-forked across this boundary (owner decision, 2026-08-03); pools
#     and opponents are re-seeded from the new lineage.
#   gen3_typed_hp_belief_v1 (config v52 — stacks on gen3_pointer_native_v1): the model never reasons
#     over a typeless Hidden Power again. The presence×type composition `P(HP_t) = presence · P(type)`
#     moves into `HPTypeBelief.compose_typed_hp`, right beside the move-belief head, so the posterior
#     EVERY consumer reads (damage op, top-K, move BCE, latent grading, token reinjection, prober)
#     carries HP at the 16 real typed nums 355-370 with the bare BP-0 num 237 driven hard-off (a finite
#     -30 logit; 237 survives only as the internal PRESENCE channel). Supersedes the
#     gen3_opp_hp_typed_candidates_v1 op-side scatter above: the v38 tri-state `hp_type_belief_mode` is
#     DELETED (its 'off' state was a correctness bug — a revealed HP priced as nonexistent), the head is
#     unconditional under a move belief, the belief LABELS use the true typed num, and the op is a plain
#     consumer (no hp_type_fix / SPECIES_HP_PRIOR). Forward math changed with out_dim + projection
#     widths UNCHANGED → nothing shape-based catches it, so the signature carries the break.
#   gen3_entity_move_seats_v1 (config v54, Stage 1 of the entity generation — the roadmap's move-tokens
#     slice; stacks on gen3_typed_hp_belief_v1): move tokens become first-class attention SEATS in the
#     trunk, and the pointer head reads the REFINED E3 seats. UNCONDITIONAL state_dict changes for
#     every model (the token-type embedding table grows 4 → 6, `entity_seats.move_seat_proj` is new,
#     the pointer head's `move_proj` widens 32+cells → d_model+cells) plus an unconditional forward
#     change (4+ new seats in every attention pass) — no off state, so the signature carries the break
#     exactly like the v51 bump. The within-generation knob is `entity_topk_seats` (E4 threat seats),
#     gated in check_compatible. No pre-v54 checkpoint was ever trained (the generation's bumps all
#     landed same-day), so nothing is stranded.
#   gen3_op_block_trim_v1 (config v55 — stacks on gen3_entity_move_seats_v1): the DamageOperator sheds
#     its three least-used output families and one dead code path, on the ledger-P1 per-block dependence
#     ablation. OUT: the opp-active believed-EFFECT scalars (6 dims, 1.2%), the opp-active per-STATUS
#     incoming SECONDARY scalars (10 dims, 0.1% — INERT), the OUTGOING slp/psn/tox per-move secondary
#     columns (12 dims, structural zeros on the whole team pool), and the v30 LEAN `_topk_block` (0
#     calls/forward — a strict subset of the v35 incoming matrix, which suppressed it). Net −28 op dims
#     off BOTH projection heads, and the unmasked-belief `w` read leaves the forward entirely. The
#     projection widths DO change, so a stale checkpoint would fail on a state_dict shape mismatch —
#     the signature bump is what turns that into a clear arch error instead. Orthogonal to v54's entity
#     seats (trunk) — this trims the op's head-concat output — so the two compose; only the signature,
#     which is one shared string, had to be sequenced.
#   gen3_edge_bias_trunk_v1 (config v56, Stage 2 of the entity generation): the encoder stack is
#     the biased-attention clone — state_dict keys change for every model (no off state), so the
#     signature carries the break like v51/v54/v55. The within-generation knob is `edge_bias_families`
#     (which families are delivered; zero-init maps ⇒ ON starts identical to OFF).
#   gen3_entity_rehome_v1 (config v60, Stage 3 of the entity generation): the flat obs's DERIVED
#     blocks are deleted and every raw fact re-homed to its entity — the 288-dim CPU matchup
#     matrices and 6 of the 11 reactive scalars are GONE (obs 2925 → 2667), protect_odds /
#     trapped / maybe_trapped ride the per-mon slots (POKEMON_FULL_DIM 113 → 116), and the
#     PokemonEncoder's move/role nets narrow (matchup + validity + struggle inputs deleted).
#     Weight shapes AND obs meaning change together — a fresh-lineage break (gen-4).
#   gen3_no_concat_v1 (config v61, the gen-5 world): THE OP HEAD-CONCAT IS DEAD — the 660-dim
#     flat block no longer enters either projection (pi 1131→471); the critic's replacement
#     window is the multi-seed readout (MultiSeedValueReadout, k=4×64 over the op's per-our-mon
#     rows, vf-only, with the seeds/* TB collapse contract). Executed on the gen-4 stratified
#     evidence (53ef270): net policy dependence +0.00%, flips half of the acceptance clause met
#     by training, act_threat decodable concat-zeroed. state_dict changes (projection widths +
#     the new module) → the signature carries the break; fresh lineage (gen-5).
#   gen3_deadline_clock_v1 (config v67): the obs CLOCK group is 3 scalars, not 1 — log-ELAPSED
#     (opening structure) plus remaining-LINEAR and log-REMAINING (deadline structure). The old
#     single log-elapsed scalar put 58.6% of its range on turns 1–50 and 4.0% on turns 200–250,
#     i.e. it had almost no resolution at the forfeit cap the trainee actually loses on. Obs
#     2667 → 2669 (GLOBAL_ENV_DIM 18 → 20) and the move/global context projections widen with
#     `_gl['clock']['dim']` → state_dict changes; fresh lineage.
#   gen3_ctx_dedup_v1 (config v76): the assembler's per-side encoded active contexts are DELETED
#     from both heads (duplicated delivery — the E2 injection + the global token already carry
#     the full raw ctx blocks into the trunk). `active_ctx_encoder` state_dict keys removed,
#     both projection inputs narrow by 64 → the signature carries the break; fresh lineage.
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
    value_active_readout: bool = False

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
    damage_matrices_outgoing_all: bool = False
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
    pubval_mode: str = "none"
    # v43 TRAINING-ONLY loss coefficient for the pubval head (like win_prob_coef). Scales the soft-target
    # BCE aux loss, affects no forward pass → recorded for provenance but NOT version-locked
    # (resume-mutable, inherited on a flagless resume).
    pubval_coef: float = 0.0

    @classmethod
    def from_layout_and_policy_kwargs(
        cls,
        layout: Dict[str, Any],
        policy_kwargs: Dict[str, Any],
        vf_coef: float = 0.5,
        reward_config=None,
        value_tail_weight: float = 0.0,
        opp_belief_aux_coef: float = 0.0,
        move_belief_coef: float = 0.0,
        win_prob_coef: float = 1.0,
        move_belief_latent_coef: float = 0.0,
        spread_belief_coef: float = 0.0,
        value_dist_coef: float = 1.0,
        hp_type_belief_coef: float = 0.0,
        item_belief_coef: float = 0.0,
        pubval_coef: float = 0.0,
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
            value_active_readout=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get("value_active_readout", False)
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
            damage_matrices_outgoing_all=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get("damage_matrices_outgoing_all", False)
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
            pubval_mode=str(
                policy_kwargs.get("features_extractor_kwargs", {}).get("pubval_mode", "none")
            ),
            pubval_coef=float(pubval_coef),
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
        if self.value_active_readout != saved.value_active_readout:
            raise ModelVersionError(
                f"value_active_readout mismatch: saved={saved.value_active_readout}, "
                f"current={self.value_active_readout}.\n"
                "Routing the active-mon readout into the value head widens the value projection, so it "
                "changes the state_dict and cannot be toggled on an existing model.\n"
                "Resume with the matching --value-active-readout setting, or start a fresh training run."
            )

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

        # v43 PUBLIC-information value aux head (gen3_pubval_aux_v1, like win_prob_mode): STRING-gated so
        # BOTH 'none'↔head (the PubValHead params, a state_dict change) AND read_only↔shaping (the
        # resume-immutable trunk-gradient choice) FATAL on a mismatch. The training-only pubval_coef is
        # NOT checked.
        if self.pubval_mode != saved.pubval_mode:
            raise ModelVersionError(
                f"pubval_mode mismatch: saved={saved.pubval_mode!r}, current={self.pubval_mode!r}.\n"
                "The public-value aux head is fixed for a run's lifetime: adding/removing it changes the "
                "state_dict, and switching read_only↔shaping flips whether its loss shapes the shared "
                "trunk (a silent mid-run training change).\n"
                "Resume with the matching --pubval-mode setting, or start a fresh training run."
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
        # gen3_per_move_matrices_v1 (v39): the TRANSPOSED outgoing matrix (our 6 mons' moves → opp active)
        # widens the op out_dim → both projection in_features. Toggling it is a weight-shape change (like damage_op).
        if self.damage_matrices_outgoing_all != saved.damage_matrices_outgoing_all:
            raise ModelVersionError(
                f"damage_matrices_outgoing_all mismatch: saved={saved.damage_matrices_outgoing_all}, "
                f"current={self.damage_matrices_outgoing_all}.\n"
                "The transposed outgoing per-move damage matrix (our 6 mons' moves → opp active) widens the "
                "damage operator's output (hence both projection widths), so toggling it is incompatible with "
                "a saved checkpoint.\n"
                "Resume with the matching --damage-matrices-outgoing-all setting, or start a fresh training run."
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

    def check_reward_config(self, reward_config) -> None:
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
    return data
