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
MODEL_CONFIG_VERSION = 23

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
ARCH_SIGNATURE = "gen3_wish_wired_v1"


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
    active_ctx_hidden: List[int]
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
    # v18 STRUCTURAL toggle (weight-shape via the BeliefHead latent predictor MLP): the LATENT belief.
    # ON adds an asymmetric SimSiam predictor that regresses each believed slot's refined token toward
    # the stop-grad pokemon_encoder role-token of the true hidden mon (graded identity supervision).
    # A state_dict change, gated in check_compatible like opp_belief_slots; OFF = baseline byte-for-byte
    # (NO ARCH_SIGNATURE bump). Requires opp_belief_slots (the BeliefHead it attaches to).
    opp_belief_latent: bool = False
    # v18 TRAINING-ONLY loss coefficient for the latent belief (like opp_belief_aux_coef). 0.0 = no aux.
    opp_belief_latent_coef: float = 0.0
    # v19 STRUCTURAL toggle (weight-shape via the DamageOperator's wider projections): the
    # differentiable GPU damage operator. ON consumes the move belief's predicted moves for the opp
    # active and appends a per-our-mon believed-damage block to BOTH heads (widening both projection
    # Linears), so like value_active_readout / opp_belief_slots it is gated in check_compatible. OFF =
    # baseline byte-for-byte (NO ARCH_SIGNATURE bump). Forward-only → no training-only coefficient.
    # Requires move_belief_mode in {revealed, both} (enforced at extractor build + CLI).
    damage_op: bool = False
    # v20 FORWARD-BEHAVIOR toggle (NOT weight-shape, like attend_unrevealed_opponents): the unified
    # two-part move belief. ON fuses the Smogon move-frequency prior into the MoveBelief head as a
    # log-odds residual (+ pins revealed moves certain), so the stashed move-belief logits carry a
    # posterior (priors ⊕ prediction). The prior buffer is non-persistent (no new params → state_dict
    # identical), but the forward differs, so it is gated in check_compatible. Requires move_belief_mode
    # != off. OFF = the from-scratch head byte-for-byte (NO ARCH_SIGNATURE bump).
    move_prior_fusion: bool = False
    # v21 FORWARD-BEHAVIOR toggle (NOT weight-shape, like attend_unrevealed_opponents): the
    # unified-architecture ablation. ON zeros the incoming-damage / OHKO obs block out of the model's
    # view (the block stays in the obs; the reward still reads it). State_dict identical; the forward
    # differs (a zeroed obs slice), so it is gated in check_compatible. OFF = baseline byte-for-byte.
    mask_incoming_damage_obs: bool = False
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
    # v23 FORWARD-BEHAVIOR float (NOT weight-shape, like move_prior_fusion): the LEGALITY-only move-prior
    # gate. 0.0 = OFF (legacy flat 0.02-floor prior, byte-identical); >0 drives moves a species CANNOT learn
    # to ~0 (impossible) while legal moves keep their true usage (rare-but-liftable, never pruned) and a
    # legal-unobserved move gets this small floor base — a different belief, gated in check_compatible. The
    # prior buffer is non-persistent so the state_dict is identical either way.
    move_candidate_floor: float = 0.0

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
        opp_belief_latent_coef: float = 0.0,
        win_prob_coef: float = 1.0,
    ) -> ModelVersion:
        from agents.model.features_extractor import (
            ROLE_TOKEN_SIZE,
            PROJECTION_DIM,
            MOVE_NET_HIDDEN,
            ROLE_ENCODER_HIDDEN,
            ACTIVE_CTX_HIDDEN,
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
            active_ctx_hidden=list(ACTIVE_CTX_HIDDEN),
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
            opp_belief_latent=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get("opp_belief_latent", False)
            ),
            damage_op=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get("damage_op", False)
            ),
            damage_outgoing=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get("damage_outgoing", False)
            ),
            move_candidate_floor=float(
                policy_kwargs.get("features_extractor_kwargs", {}).get("move_candidate_floor", 0.0)
            ),
            move_prior_fusion=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get("move_prior_fusion", False)
            ),
            mask_incoming_damage_obs=bool(
                policy_kwargs.get("features_extractor_kwargs", {}).get("mask_incoming_damage_obs", False)
            ),
            win_prob_mode=str(
                policy_kwargs.get("features_extractor_kwargs", {}).get("win_prob_mode", "none")
            ),
            value_tail_weight=float(value_tail_weight),
            opp_belief_aux_coef=float(opp_belief_aux_coef),
            move_belief_coef=float(move_belief_coef),
            opp_belief_latent_coef=float(opp_belief_latent_coef),
            win_prob_coef=float(win_prob_coef),
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
            "move_net_hidden", "role_encoder_hidden", "active_ctx_hidden",
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

        # Structural toggle — ON adds the BeliefHead latent predictor MLP to the state_dict (the
        # latent-belief escalation). Like opp_belief_slots it gates EVERY load; the training-only
        # opp_belief_latent_coef is deliberately NOT checked (it touches no forward pass).
        if self.opp_belief_latent != saved.opp_belief_latent:
            raise ModelVersionError(
                f"opp_belief_latent mismatch: saved={saved.opp_belief_latent}, current={self.opp_belief_latent}.\n"
                "The latent-belief predictor (the SimSiam head on the BeliefHead) changes the state_dict, "
                "so it cannot be toggled on an existing model.\n"
                "Resume with the matching --opp-belief-latent-coef setting, or start a fresh training run."
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

        # Forward-behavior toggle (no weight-shape change, like move_prior_fusion): the learnset + rarity
        # gate produces a different move prior → a different belief the policy/value/op trained under.
        if self.move_candidate_floor != saved.move_candidate_floor:
            raise ModelVersionError(
                f"move_candidate_floor mismatch: saved={saved.move_candidate_floor}, "
                f"current={self.move_candidate_floor}.\n"
                "The learnset + rarity-cap move-prior gate changes the belief the policy trained under, so "
                "it cannot be changed on a resumed model.\n"
                "Resume with the matching --move-candidate-floor, or start a fresh training run."
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

        # Forward-behavior toggle (no weight-shape change): ablating the incoming-damage obs block
        # changes the input the policy/value trained on, so a resume flip would feed a different forward.
        if self.mask_incoming_damage_obs != saved.mask_incoming_damage_obs:
            raise ModelVersionError(
                f"mask_incoming_damage_obs mismatch: saved={saved.mask_incoming_damage_obs}, "
                f"current={self.mask_incoming_damage_obs}.\n"
                "Zeroing the incoming-damage obs block out of the model's view changes the forward the "
                "policy trained under, so it cannot be toggled on a resumed model.\n"
                "Resume with the matching --mask-incoming-damage-obs setting, or start a fresh training run."
            )

        # Structural + resume-IMMUTABLE toggle — gated as a STRING so BOTH 'none'↔head (a state_dict
        # change: the WinProbHead params) AND read_only↔shaping (same params, but flipping the trunk
        # gradient flow mid-run is a silent training change the user chose to forbid) FATAL on a
        # mismatch. Like move_belief_mode it gates EVERY load; same-run pool/sentinel snapshots carry the
        # identical mode so they pass trivially. The training-only win_prob_coef is NOT checked.
        if self.win_prob_mode != saved.win_prob_mode:
            raise ModelVersionError(
                f"win_prob_mode mismatch: saved={saved.win_prob_mode!r}, current={self.win_prob_mode!r}.\n"
                "The win-probability head is fixed for a run's lifetime: adding/removing it changes the "
                "state_dict, and switching read_only↔shaping flips whether its loss shapes the shared "
                "trunk (a silent mid-run training change).\n"
                "Resume with the matching --win-prob-mode setting, or start a fresh training run."
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
    """Apply incremental forward-migrations to bring an old config up to the current schema."""
    version = data.get("config_version", 1)
    if version < 2:
        # v2: added n_history_turns. Old models used a single TurnDelta (N=1).
        data.setdefault("n_history_turns", 1)
        data["config_version"] = 2
    if version < 3:
        # v3: added vf_coef. Every pre-flag run trained with the SB3 default 0.5.
        data.setdefault("vf_coef", 0.5)
        data["config_version"] = 3
    if version < 4:
        # v4: added reward-config hparams. Pre-flag runs used the single-variable defaults.
        data.setdefault("bias_additivity", 1.0)
        data.setdefault("mat_alive_weight", 1.25)
        data.setdefault("bias_redesign", False)
        data["config_version"] = 4
    if version < 5:
        # v5: added the switch-bias lever. Pre-flag runs had it absent (OFF).
        data.setdefault("switch_bias_weight", 0.0)
        data["config_version"] = 5
    if version < 6:
        # v6: added use_popart. Old models did not use PopArt value normalization.
        data.setdefault("use_popart", False)
        data["config_version"] = 6
    if version < 7:
        # v7: added draw_penalty. Old runs scored a tie/timeout as a decisive loss (-VICTORY_VALUE).
        data.setdefault("draw_penalty", -30.0)
        data["config_version"] = 7
    if version < 8:
        # v8: added attend_unrevealed_opponents. Old models key-masked unrevealed opp slots.
        data.setdefault("attend_unrevealed_opponents", False)
        data["config_version"] = 8
    if version < 9:
        # v9: added the hidden-opponent belief toggle (k=0 = no belief module). Old models had none.
        data.setdefault("opp_belief_cls_k", 0)
        data.pop("opp_belief_cls", None)  # never shipped; drop the interim bool if a dev config has it
        data["config_version"] = 9
    if version < 10:
        # v10: added value_active_readout. Old models' value head did not read the active-mon token.
        data.setdefault("value_active_readout", False)
        data["config_version"] = 10
    if version < 11:
        # v11: added value_tail_weight. Old runs used a plain MSE value loss (β=0).
        data.setdefault("value_tail_weight", 0.0)
        data["config_version"] = 11
    if version < 12:
        # v12: added self_ko_hp_penalty. Old runs had no self-KO penalty (the symmetric material PBRS
        # priced a healthy Explosion/Self-Destruct trade at ~0).
        data.setdefault("self_ko_hp_penalty", 0.0)
        data["config_version"] = 12
    if version < 13:
        # v13: added the de-bias cleanup flags. Old runs kept every BIAS term (== False).
        data.setdefault("drop_redundant_bias", False)
        data.setdefault("drop_switch_bias", False)
        data["config_version"] = 13
    if version < 14:
        # v14: end-state PBRS switch (off) + no_progress_penalty now recorded (default 0.15 = prior).
        data.setdefault("all_shaping_pbrs", False)
        data.setdefault("no_progress_penalty", 0.15)
        data["config_version"] = 14
    if version < 15:
        # v15: added the `stall_pbrs` companion switch (off = prior behavior).
        data.setdefault("stall_pbrs", False)
        data["config_version"] = 15
    if version < 16:
        # v16: added the in-place hidden-opponent belief-aux toggle (off) + its training-only loss
        # coefficient (0.0). Old models had neither module nor aux loss.
        data.setdefault("opp_belief_slots", False)
        data.setdefault("opp_belief_aux_coef", 0.0)
        data["config_version"] = 16
    if version < 17:
        # v17: added the move-belief reinjection toggle (off) + its training-only loss coefficient (0.0).
        # Old models had no MoveBelief module and no move-belief loss.
        data.setdefault("move_belief_mode", "off")
        data.setdefault("move_belief_coef", 0.0)
        data["config_version"] = 17
    if version < 18:
        # v18: added the latent-belief toggle (off) + its training-only loss coefficient (0.0).
        # Old models had no BeliefHead latent predictor and no latent loss.
        data.setdefault("opp_belief_latent", False)
        data.setdefault("opp_belief_latent_coef", 0.0)
        data["config_version"] = 18
    if version < 19:
        # v19: added the differentiable damage-operator toggle (off). Old models had no DamageOperator.
        data.setdefault("damage_op", False)
        data["config_version"] = 19
    if version < 20:
        # v20: added the unified-move-belief prior-fusion toggle (off). Old models had no prior fusion.
        data.setdefault("move_prior_fusion", False)
        data["config_version"] = 20
    if version < 21:
        # v21: added the incoming-damage-obs ablation toggle (off). Old models always saw the obs block.
        data.setdefault("mask_incoming_damage_obs", False)
        data["config_version"] = 21
    if version < 22:
        # v22: added the tri-state win-probability head (off) + its training-only loss coef (1.0).
        # Old models had no WinProbHead and no win-prob loss.
        data.setdefault("win_prob_mode", "none")
        data.setdefault("win_prob_coef", 1.0)
        data["config_version"] = 22
    if version < 23:
        # v23: added the outgoing per-move damage direction (off) + the learnset/rarity move-prior gate
        # (0.0 = legacy floor). Old models had incoming-only and the un-gated 0.02-floor prior.
        data.setdefault("damage_outgoing", False)
        data.setdefault("move_candidate_floor", 0.0)
        data["config_version"] = 23
    return data
