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
# gen3_entity_rehome_v1 (Stage 3): the PROTECT-SUCCESS odds moved from the deleted reactive
# scalars (old vec[7]/vec[8], ACTIVES ONLY) onto every mon token — P(a Protect/Detect/Endure by
# THIS mon succeeds now) from its LiveView protect_counter (gen3 floored doubling 100/50/25/12.5,
# floor 1/8). A benched mon's counter is 0 → odds 1.0, which is TRUE (the stall counter resets on
# switch) — the fact now rides the entity instead of a global slot.
POKEMON_PROTECT_OFFSET = 112   # 109 + 3 (recency end)
POKEMON_VECTOR_DIM = 113       # 71 + 18 (spread) + 17 (HP block) + 3 (sleep) + 3 (recency) + 1 (protect)
# state_encoder appends per-slot: the two OUR-side trapping bits, then the active flag — active
# stays the LAST dim of the slot ON PURPOSE (the model's `hp_and_active[:, :, -1]` convention is
# load-bearing across ObsUnpack / DamageOperator / entity seats; a reorder here would silently
# repoint every one of those reads). The trapping bits are gen3_entity_rehome_v1 — old reactive
# vec[4]/vec[5], now on the trapped ENTITY: nonzero only at our ACTIVE slot; server-authoritative
# LegalActions facts, so they stay obs bits rather than becoming a T-edge belief.
POKEMON_TRAPPED_OFFSET = POKEMON_VECTOR_DIM               # 113 (our active only; 0 elsewhere)
POKEMON_MAYBE_TRAPPED_OFFSET = POKEMON_VECTOR_DIM + 1     # 114 (our active only; 0 elsewhere)
POKEMON_ACTIVE_OFFSET = POKEMON_VECTOR_DIM + 2            # 115 — LAST in the slot (see above)
POKEMON_FULL_DIM = POKEMON_VECTOR_DIM + 3                 # 116 = 113 + trapped + maybe_trapped + active

# Active context: boosts(14) + volatiles(VOLATILES_DIM)
ACTIVE_CONTEXT_DIM = BOOSTS_DIM + VOLATILES_DIM

# Global env (event-sourced via LiveView, gen3ou-relevant only):
#   weather one-hot (5: none/sun/rain/sand/hail — gen4+ slots dropped) +
#   weather_permanent (1) + weather_turns_remaining (1) +
#   spikes ×2 + CLOCK (3) +
#   per-side screens/safeguard/mist: reflect ×2 + light_screen ×2 + safeguard ×2 + mist ×2
#
# gen3_deadline_clock_v1 — the CLOCK group is 3 scalars, not 1. It used to be the single
# log-ELAPSED scalar `log(1+turn)/log(1+MAX_TURNS)`, which is the right shape for "how far into
# the game am I" (opening structure) and the WRONG shape for "how close am I to the forfeit".
# Measured on that encoding: turns 1-50 occupy 58.6% of the feature's range while turns 200-250
# — the deadline window — occupy 4.0%, and the last 20 turns just 1.5%; per-turn sensitivity at
# turn 249 is 125x lower than at turn 1. The trainee FORFEITS at MAX_TURNS (see StallConfig,
# which now reads this constant), so value near the cap is a function of turns REMAINING, and a
# critic cannot price a deadline it has no resolution on. Both remaining forms are provided as
# RAW FACTS (linear + log-remaining) alongside the elapsed scalar; which one matters is the
# model's to learn.
WEATHER_ONEHOT_DIM = 5
CLOCK_DIM = 3            # [log_elapsed, remaining_linear, log_remaining]
GLOBAL_ENV_DIM = WEATHER_ONEHOT_DIM + 2 + 2 + CLOCK_DIM + 8  # = 20

# gen3_entity_rehome_v1 (Stage 3, the re-home): the two 144-dim MATCHUP MATRICES ARE GONE from
# the observation — the roadmap §3 Stage-3 delete, executed after the gen-3 40M audit confirmed
# the D/V edge families as a trained, load-bearing SUPERSET (D-cells carry
# [low, high, crit, pko, type_mult, revealed] per pair from real gen3 physics + the learned move
# belief, vs these matrices' bare type_mult/4 from a fixed prior; gen-3 audit: d2 16.3% flips).
# The ~35% of obs-build CPU they cost (the whole reactive matchup loop —
# effective_multiplier_by_types was the #1 tottime line at 202 calls/encode) is the Stage-3
# refund. The 11 reactive scalars are re-homed or deleted:
#   * protect_odds ×2      → the PER-MON slot (POKEMON_PROTECT_OFFSET — every mon, not 2 actives)
#   * trapped/maybe_trapped → the OUR-ACTIVE mon slot (POKEMON_TRAPPED_OFFSET / _MAYBE_)
#   * active_status        → DELETED (byte-redundant with the active mon's condition one-hot)
#   * forced_struggle      → DELETED (derivable: active_req_moves legal ×4 all-zero on a
#                            move-request ⇔ struggle is the only move — and the action mask
#                            carries the authoritative bit)
#   * fainted ×2, turns_since_progress, wish_floating ×2 → the 5 BOARD scalars below (E7/E6-class
#     side/board facts; wish is slot-keyed side state, the clock is EpisodeTracker cross-turn
#     state — same sources and same one-value-with-reward contract as before):
#   vec[0]=fainted_ours vec[1]=fainted_opp vec[2]=turns_since_progress
#   vec[3]=wish_floating_our vec[4]=wish_floating_opp
REACTIVE_SCALAR_DIM = 5

# gen3_cpu_damage_deleted_v1: the 44-dim action-aligned MOVE-EFFECT block (4 slots x 11 flags:
# is_boost/is_heal/is_protect/is_phaze/is_hazard/inflicts_status/status_will_land/pp_fraction/
# status_will_land_known/cures_self_status/cures_team_status) is GONE. GPU homes, all live:
#   * the static per-move mechanics  -> MoveLatentEncoder's MOVE_ATTR latent (--move-latent, v24)
#   * board-conditional status_will_land -> DamageOperator._status_landing (--damage-outgoing, v27)
#     and, as the S1/S3 edge families, discrete_{in,out}going_status
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

# gen3_entity_rehome_v1: REACTIVE_MATCHUP_OFFSET / MATCHUP_DIM are DELETED with the matrices.

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
ACTIVE_REQ_MOVES_OFFSET = REACTIVE_SCALAR_DIM                            # 5 — right after the board scalars

REACTIVE_DIM = REACTIVE_SCALAR_DIM + ACTIVE_REQ_MOVES_DIM                # 17

# Top-level Offsets — all derived from the named constants (only the constants are load-bearing;
# these comments are the post-gen3_markovian_progress_v1 values: base dim = 1790, full obs = 3391).
OFFSET_OUR_TEAM = 0
OFFSET_OPP_TEAM = 6 * POKEMON_FULL_DIM                     # 642
OFFSET_CONTEXT = 2 * OFFSET_OPP_TEAM                       # 1284
OFFSET_GLOBAL = OFFSET_CONTEXT + (2 * ACTIVE_CONTEXT_DIM)  # 1400
OFFSET_REACTIVE = OFFSET_GLOBAL + GLOBAL_ENV_DIM            # 1418

# Max values for normalization.
# MAX_TURNS is ALSO the forfeit deadline: `StallConfig.threshold` defaults to it (training/stall.py
# imports it) so the obs clock and the turn at which the trainee actually forfeits cannot drift
# apart. Pinned by `global_env_test.py::test_max_turns_is_the_forfeit_deadline`.
MAX_TURNS = 250
MAX_SPIKES = 3
MAX_STATS = 255
MAX_SPECIES_ID = 387
MAX_MOVE_ID = 371
MAX_ITEM_ID = 600
MAX_ABILITY_ID = 100
MAX_PP = 64  # Splash (40 base) * 8//5 = 64; covers all Gen 3 moves with full PP Ups
TRACE_INTERVAL = 15
