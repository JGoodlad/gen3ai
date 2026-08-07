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
# E9 step 1 (gen3_entity_recency_v1, roadmap §3.9): per-mon RECENCY — [turns_since_seen,
# turns_since_acted, turns_since_was_hit], log-saturated over a 10-turn cap (the
# turns_since_progress convention), BOTH sides (all three derive from observed protocol
# events — public, no leak). Sourced from the EpisodeTracker-owned RecencyTracker (the same
# per-decision event window the TurnDelta fold reads); a never-tracked mon reads 1.0 (max
# staleness — the honest default for an unrevealed slot).
POKEMON_RECENCY_OFFSET = 109   # 106 + 3 (sleep-belief end)
POKEMON_RECENCY_DIM = 3
POKEMON_VECTOR_DIM = 112       # 71 + 18 (spread) + 17 (HP block) + 3 (sleep) + 3 (recency)
POKEMON_FULL_DIM = 113         # 112 + 1 (active flag appended by state_encoder)

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
# 11 scalar reactive dims lead the block:
# fainted(2) + active_status(1) + forced_struggle(1) + trapped(1) + maybe_trapped(1) +
# turns_since_progress(1) + protect_odds ×2 (our active, opp active) + wish_floating ×2 (our, opp).
# Indices below are the CURRENT ones (post gen3_cpu_damage_deleted_v1, which removed the 8
# active-move scalars that used to occupy vec[0:8] and shifted everything down by 8):
#   vec[0]=fainted_ours vec[1]=fainted_opp vec[2]=active_status vec[3]=forced_struggle
#   vec[4]=trapped vec[5]=maybe_trapped vec[6]=turns_since_progress
#   vec[7]/vec[8]=protect_odds (our/opp)  vec[9]/vec[10]=wish_floating (our/opp)
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
# gen3_cpu_damage_deleted_v1: the 8 ACTIVE-MOVE SCALARS (4 base-power + 4 type-multiplier) that used
# to head this block are GONE. They were the primitive CPU damage signal, fully subsumed by the
# DamageOperator's OUTGOING per-move block (`--damage-outgoing`, request-ordered so slot k ↔ action
# 6+k, with the real gen3 physics rather than bp/200 and mult/4). `--unified-obs` had merely MASKED
# them from the model while the CPU still computed them every decision; this deletes the producer.
# Scalar indices therefore shift down by 8 (the old vec[8] fainted-count is now vec[0]).
REACTIVE_SCALAR_DIM = 11

# gen3_cpu_damage_deleted_v1: the 44-dim action-aligned MOVE-EFFECT block (4 slots x 11 flags:
# is_boost/is_heal/is_protect/is_phaze/is_hazard/inflicts_status/status_will_land/pp_fraction/
# status_will_land_known/cures_self_status/cures_team_status) is GONE. GPU homes, all live:
#   * the static per-move mechanics  -> MoveLatentEncoder's MOVE_ATTR latent (--move-latent, v24)
#   * board-conditional status_will_land -> DamageOperator._status_landing (--damage-outgoing, v27)
#     and, in-trunk, discrete_{in,out}going_status (--threat-status-refine, v37)
#   * pp_fraction -> the per-mon move slot (unchanged)
# `--unified-obs` had merely MASKED this from the model while the CPU rebuilt all 44 dims per
# decision; this deletes the producer. Honest residual: the Rest-cure coarsening noted in the v37
# deprecation audit.

NUM_POKEMON = 12
TEAM_SIZE = 6

# gen3_cpu_damage_deleted_v1: the 51-dim per-our-mon INCOMING-DAMAGE / OHKO belief block is GONE
# from the observation. The GPU DamageOperator computes the same physics from the LEARNED move
# belief (rather than this block's FIXED usage prior) and feeds both heads -- that was the whole
# point of --damage-op. `--unified-obs` had merely MASKED it while the CPU still ran the ~6
# per-channel threat computations every decision.
# NOTE: `agents.observation.incoming_damage` (the math core) STAYS -- the reward PBRS
# (reward_manager.py) and the prober both import it. Only the obs WRITE is removed.

REACTIVE_MATCHUP_OFFSET = REACTIVE_SCALAR_DIM                            # 11

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
ACTIVE_REQ_MOVES_OFFSET = REACTIVE_MATCHUP_OFFSET + MATCHUP_DIM          # 299 — after the two matchup matrices

REACTIVE_DIM = REACTIVE_SCALAR_DIM + MATCHUP_DIM + ACTIVE_REQ_MOVES_DIM  # 311

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
