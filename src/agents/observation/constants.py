# Sub-dimensions
SPECIES_ID_DIM = 1 # ID for embedding
ITEM_ID_DIM = 1    # Only the first slot is the ID; was 16 (wasted)
ITEM_KNOWN_DIM = 1
ITEM_CONSUMED_DIM = 1  # 1.0 when item was consumed this battle (Berry, Trick, etc.)
COMBINED_TYPES_DIM = 2  # Two type IDs only; was 8 (6 placeholders removed)
ABILITY_SLOT_DIM = 2    # [ability1_id, ability2_id]: revealed ability fills slot 1 when
                        # known=1; the species' top-2 Smogon-observed abilities fill
                        # both slots when known=0 (opp not yet revealed), sorted by usage
                        # so ability1 is the favorite.
ABILITY_DOMINANCE_DIM = 1  # Smogon prob(ability1) ∈ [0, 1]; forced to 1.0 when known=1
ABILITY_KNOWN_DIM = 1
MOVE_SLOT_DIM = 11
MOVES_KNOWN_DIM = 0
CONDITION_DIM = 7 # None, BRN, PAR, SLP, FRZ, PSN, TOX
STATS_DIM = 6 # HP, Atk, Def, SpA, SpD, Spe

# Active Context Sub-dimensions (temporal block removed — was unimplemented)
BOOSTS_DIM = 14
# Full gen3 volatile set (source-derived, crash-don't-drop). VOLATILES_DIM mirrors
# gen3_effects.VOLATILE_DIM — defined there as the single source of truth so the obs
# layout and the encoder can never disagree. (Was a hand-set 9 that dropped ~30 real
# volatiles — Disable/Encore/Taunt/Destiny Bond/Curse/Yawn/traps/…)
from agents.observation.gen3_effects import VOLATILE_DIM as _VOLATILE_DIM
VOLATILES_DIM = _VOLATILE_DIM

# Internal Pokémon Vector Offsets
# Species: 7 (1 ID + 6 Stats)
# Items: ITEM_ID_DIM(1) + ITEM_KNOWN_DIM(1) + ITEM_CONSUMED_DIM(1) = 3
# Types: COMBINED_TYPES_DIM(2) = 2
# Abilities: ABILITY_SLOT_DIM(2) + ABILITY_DOMINANCE_DIM(1) + ABILITY_KNOWN_DIM(1) = 4
# Condition: CONDITION_DIM(7) = 7
# Moves: 4 * MOVE_SLOT_DIM(11) = 44
# HP: 1   Total: 68
POKEMON_SPECIES_OFFSET = 0
POKEMON_ITEMS_OFFSET = 7   # 0 + 7
POKEMON_TYPES_OFFSET = 10  # 7 + 3
POKEMON_ABILITIES_OFFSET = 12  # 10 + 2
POKEMON_CONDITION_OFFSET = 16  # 12 + 4 (ability1 + ability2 + dominance + known)
POKEMON_MOVES_OFFSET = 23  # 16 + 7
POKEMON_HP_OFFSET = 67          # 23 + 44
POKEMON_SPECIES_KNOWN_OFFSET = 68  # 67 + 1 (HP); 1.0 when slot is populated, 0.0 when absent
POKEMON_COUNTER_OFFSET = 69    # 68 + 1 (species_known): sleep_ctr, toxic_ctr
POKEMON_COUNTER_DIM = 2        # sleep turn count (norm), toxic turn count (norm)
# Spread block (18 dims): IVs (6) + EVs (6) + spread_known (1) + nature modifiers (5)
# Stat order for IVs/EVs: [health_pts, atk, def, spa, spd, spe]
# Stat order for nature modifiers: [atk, def, spa, spd, spe] (HP is never nature-modified)
# spread_known=1.0 own team (IVs/EVs/nature are real values), 0.0 opponent (all zeros = padding)
POKEMON_SPREAD_OFFSET = 71     # 69 + 2 (status counters)
POKEMON_SPREAD_DIM = 18        # 6 IVs + 6 EVs + 1 flag + 5 nature modifiers
# Hidden Power candidate block (per opp slot; all-zero for our team slots).
# hp_revealed flag (1) lets the model distinguish "HP not yet seen" (whole block zero)
# from "HP seen but type ambiguous" (one or more prob entries non-zero, flag = 1).
# hp_type_probs (16) is the tracker's per-species candidate distribution in
# HIDDEN_POWER_TYPE_ORDER (alphabetical: Bug, Dark, Dragon, Electric, Fighting, Fire,
# Flying, Ghost, Grass, Ground, Ice, Poison, Psychic, Rock, Steel, Water).
POKEMON_HP_BLOCK_OFFSET = 89   # 71 + 18 (spread end)
POKEMON_HP_REVEALED_OFFSET = 89
POKEMON_HP_PROBS_OFFSET = 90
POKEMON_HP_BLOCK_DIM = 17      # 1 hp_revealed flag + 16 candidate-type probs
# Sleep WAKE belief (gen3_sleep_wake_belief_v1): 3 dims for an asleep mon, else all zeros —
# [sleep_is_deterministic (1.0 = Rest fixed-duration), p_wake (computed P(wake next move attempt)
# over the verified gen3 sleep tables, marginalising the opp Early-Bird prior), sleep_counter_reliable
# (0.0 once a Sleep Talk / Snore turn has corrupted the poke-env counter)]. poke-env exposes only the
# noisy counter + Status.SLP; this hands the model the COMPUTED wake odds + the Rest source it can't
# otherwise see (the [from] move clause, read from our event log). See observation/sleep_belief.py.
POKEMON_SLEEP_BELIEF_OFFSET = 106  # 89 + 17 (HP block end)
POKEMON_SLEEP_BELIEF_DIM = 3
POKEMON_VECTOR_DIM = 109       # 71 + 18 (spread) + 17 (HP block) + 3 (sleep belief)
POKEMON_FULL_DIM = 110         # 109 + 1 (active flag appended by state_encoder)

# Active context: boosts(14) + volatiles(VOLATILES_DIM)
ACTIVE_CONTEXT_DIM = BOOSTS_DIM + VOLATILES_DIM

# Global env (event-sourced via LiveView, gen3ou-relevant only):
#   weather one-hot (5: none/sun/rain/sand/hail — gen4+ slots dropped) +
#   weather_permanent (1) + weather_turns_remaining (1) +
#   spikes ×2 + log-turn (1) +
#   per-side screens/safeguard/mist: reflect ×2 + light_screen ×2 + safeguard ×2 + mist ×2
WEATHER_ONEHOT_DIM = 5
GLOBAL_ENV_DIM = WEATHER_ONEHOT_DIM + 2 + 2 + 1 + 8  # = 18

MATCHUP_DIM = 288 # (6*4*6) for Our vs Their + (6*4*6) for Their vs Our
# 19 scalar reactive dims lead the block: move power(4) + multiplier(4) +
# fainted(2) + active_status(1) + forced_struggle(1) + trapped(1) + maybe_trapped(1) +
# turns_since_progress(1) + protect_odds ×2 (our active, opp active) + wish_floating ×2 (our, opp).
# trapped/maybe_trapped are the gen3_trapping_signals_v1 additions; turns_since_progress (vec[14],
# gen3_markovian_progress_v1) is the log-saturated no-progress clock (design §5.1) — an
# EpisodeTracker-owned cross-turn counter (NOT LiveView), threaded into encode() like the HP tracker, so
# obs and the no-progress reward key on ONE value. protect_odds (vec[15]/vec[16], gen3_protect_odds_v1)
# is P(a Protect/Detect/Endure succeeds NOW) for each active mon — the gen3 floored-doubling odds
# (100/50/25/12.5, floor 1/8) the model can't otherwise see (the 'stall' counter isn't enumerated by
# poke-env's volatiles, and history saliency decays before a chain can be counted). Sourced from the
# LiveView's per-mon protect_counter; public for both sides (the opp's counter derives entirely from
# their revealed move stream → no leak). **wish_floating (vec[17] our side, vec[18] opp side,
# gen3_wish_wired_v1) is the pending-Wish "floating heal" signal — WISH_HEAL_FRACTION (≈0.5, the gen3
# recipient-maxhp/2 heal) when a Wish cast last turn will heal the slot mon at the END of this turn
# (slot-keyed, reconstructed from the event log since poke-env doesn't track it), else 0.0. See
# observation/wish_belief.py.**
REACTIVE_SCALAR_DIM = 19

# gen3_move_effects_v1 / gen3_status_cure_moves_v1: action-aligned per-move EFFECT flags. For
# each of the 4 request-order move slots (so feature slot k lines up with action logit 6+k), 11
# features: is_boost, is_heal, is_protect, is_phaze, is_hazard, inflicts_status,
# status_will_land, pp_fraction, status_will_land_known, cures_self_status, cures_team_status.
# Status / utility moves are otherwise indistinguishable at the policy head (base power 0 + neutral
# type multiplier for all of them), so the model could not tell a setup move from a heal from a
# wasted Toxic — nor that a move CLEARS status. status_will_land is a prior-weighted probability;
# status_will_land_known is the prior-vs-confirmed flag (routed with the SAME predicate as the
# per-mon ability block's `known` bit — see reactive.py). cures_self_status (Refresh) /
# cures_team_status (Heal Bell, Aromatherapy) are static curated facts (gen3_status_cure_moves_v1)
# the head reads alongside the per-mon status one-hots to value a cure (a verified gap: the head
# routed its own status onto Recover/switch but never onto the cure move). These sit AFTER the 15
# scalars and BEFORE the matchups, so the extractor picks them up in `non_matchup_rest`
# automatically (the matchup offset is read from the layout, never hardcoded).
MOVE_EFFECT_FEATURES = 11
MOVE_EFFECTS_DIM = 4 * MOVE_EFFECT_FEATURES                       # 44 (N_MOVE_SLOTS=4 × 11)

NUM_POKEMON = 12
TEAM_SIZE = 6

# incoming_damage (gen3_incoming_crit_split): per-our-mon incoming-KO BELIEF (opp active → our
# active + 5 bench). Per mon INCOMING_PER_MON features [phys_expdmg_frac, spec_expdmg_frac,
# phys_pko_nocrit, spec_pko_nocrit, phys_crit_delta, spec_crit_delta, p_outspeed, threat_revealed],
# then INCOMING_RECOVERY_DIM opp-active scalars [recovery_rate, cures_status(P rest),
# recovery_known]. P(KO) is the modal no-crit line; the crit risk is the DELTA (crit-inclusive −
# no-crit ∈ [0, _CRIT_P]) — a decorrelated "crit tax" feature, not the near-redundant absolute crit
# line. Plus a provenance scalar (revealed move vs usage-prior guess). PER_MON / RECOVERY are owned by incoming_damage.py
# (the math core that emits the block) and imported here so the layout and the encoder can never
# disagree — same pattern as VOLATILE_DIM above. Sits AFTER move-effects and BEFORE the matchups, so
# the feature extractor picks the whole block up in `non_matchup_rest` (→ both heads + global token)
# automatically — the matchup offset is read from get_layout(), never hardcoded.
from agents.observation.incoming_damage import (
    PER_MON as INCOMING_PER_MON, RECOVERY as INCOMING_RECOVERY_DIM,
)
INCOMING_DMG_DIM = TEAM_SIZE * INCOMING_PER_MON + INCOMING_RECOVERY_DIM  # 51
INCOMING_DMG_OFFSET = REACTIVE_SCALAR_DIM + MOVE_EFFECTS_DIM             # 63 (within the reactive block)

REACTIVE_MATCHUP_OFFSET = REACTIVE_SCALAR_DIM + MOVE_EFFECTS_DIM + INCOMING_DMG_DIM  # 114

# gen3_op_move_align_v1: the OUR-ACTIVE mon's 4 moves in REQUEST-slot order (so slot k ↔ action
# logit 6+k) — [move_num ×4, resolved_type_id ×4, legal_now ×4]. The DamageOperator's OUTGOING
# per-move blocks (_outgoing_block / _status_landing / _outgoing_matrix) READ THIS so their per-move
# output aligns with the action order, instead of the per-mon block's sorted-by-id order. The per-mon
# move block (all_move_ids) stays sorted-by-id on purpose — it feeds the role token, whose value is
# order-sensitive (the 4 move encodings are concatenated), so it can't be reordered without changing
# the network. `move_num` is the dex num (HP → 237 regardless of type); `resolved_type_id` is the
# TypeEncoder index (our own Hidden Power is typed); `legal_now` is the CURRENT-decision choosability
# (`not legal.move_slots[k].disabled`, the exact action-mask move-bit), in request order — strictly
# fresher than the prev-turn / sorted-by-id `prev_mask` the op used to gate with. These are embedding
# IDs (not scalars), so the block sits AFTER the matchups: existing offsets (incl. the matchup offset
# `non_matchup_rest` stops at) are UNDISTURBED, and ObsUnpack slices it explicitly into ctx (it never
# enters the raw-scalar projection path). Retrain-class (obs dim grows; ARCH gen3_op_move_align_v1).
ACTIVE_REQ_MOVES_PER = 4                                                 # request slots (== N_MOVE_SLOTS)
ACTIVE_REQ_MOVES_DIM = 3 * ACTIVE_REQ_MOVES_PER                          # 12 = ids(4) + type_ids(4) + legal(4)
ACTIVE_REQ_MOVES_OFFSET = REACTIVE_MATCHUP_OFFSET + MATCHUP_DIM          # 402 — after the two matchup matrices

REACTIVE_DIM = (REACTIVE_SCALAR_DIM + MOVE_EFFECTS_DIM + INCOMING_DMG_DIM
                + MATCHUP_DIM + ACTIVE_REQ_MOVES_DIM)                    # 414

# Top-level Offsets — all derived from the named constants (only the constants are load-bearing;
# these comments are the post-gen3_markovian_progress_v1 values: base dim = 1790, full obs = 3391).
OFFSET_OUR_TEAM = 0
OFFSET_OPP_TEAM = 6 * POKEMON_FULL_DIM                     # 642
OFFSET_CONTEXT = 2 * OFFSET_OPP_TEAM                       # 1284
OFFSET_GLOBAL = OFFSET_CONTEXT + (2 * ACTIVE_CONTEXT_DIM)  # 1400
OFFSET_REACTIVE = OFFSET_GLOBAL + GLOBAL_ENV_DIM            # 1418

# Max values for normalization
MAX_TURNS = 250
MAX_SPIKES = 3
MAX_STATS = 255
MAX_SPECIES_ID = 387
MAX_MOVE_ID = 371
MAX_ITEM_ID = 600
MAX_ABILITY_ID = 100
MAX_PP = 64  # Splash (40 base) * 8//5 = 64; covers all Gen 3 moves with full PP Ups
TRACE_INTERVAL = 15
