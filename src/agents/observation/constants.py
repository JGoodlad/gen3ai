# Sub-dimensions
SPECIES_ID_DIM = 1 # ID for embedding
ITEM_ID_DIM = 1    # Only the first slot is the ID; was 16 (wasted)
ITEM_KNOWN_DIM = 1
ITEM_CONSUMED_DIM = 1  # 1.0 when item was consumed this battle (Berry, Trick, etc.)
COMBINED_TYPES_DIM = 2  # Two type IDs only; was 8 (6 placeholders removed)
ABILITY_SLOT_DIM = 2    # [ability1_id, ability2_id]: revealed ability fills slot 1 when
                        # known=1; the species' two dex-possible abilities fill both slots
                        # when known=0 (opp not yet revealed)
ABILITY_KNOWN_DIM = 1
MOVE_SLOT_DIM = 9
MOVES_KNOWN_DIM = 0
CONDITION_DIM = 7 # None, BRN, PAR, SLP, FRZ, PSN, TOX
STATS_DIM = 6 # HP, Atk, Def, SpA, SpD, Spe

# Active Context Sub-dimensions (temporal block removed — was unimplemented)
BOOSTS_DIM = 14
VOLATILES_DIM = 9

# Internal Pokémon Vector Offsets
# Species: 7 (1 ID + 6 Stats)
# Items: ITEM_ID_DIM(1) + ITEM_KNOWN_DIM(1) + ITEM_CONSUMED_DIM(1) = 3
# Types: COMBINED_TYPES_DIM(2) = 2
# Abilities: ABILITY_SLOT_DIM(2) + ABILITY_KNOWN_DIM(1) = 3
# Condition: CONDITION_DIM(7) = 7
# Moves: 4 * MOVE_SLOT_DIM(9) = 36
# HP: 1   Total: 59
POKEMON_SPECIES_OFFSET = 0
POKEMON_ITEMS_OFFSET = 7   # 0 + 7
POKEMON_TYPES_OFFSET = 10  # 7 + 3
POKEMON_ABILITIES_OFFSET = 12  # 10 + 2
POKEMON_CONDITION_OFFSET = 15  # 12 + 3 (ability1 + ability2 + known)
POKEMON_MOVES_OFFSET = 22  # 15 + 7
POKEMON_HP_OFFSET = 58          # 22 + 36
POKEMON_SPECIES_KNOWN_OFFSET = 59  # 58 + 1 (HP); 1.0 when slot is populated, 0.0 when absent
POKEMON_COUNTER_OFFSET = 60    # 59 + 1 (species_known): sleep_ctr, toxic_ctr
POKEMON_COUNTER_DIM = 2        # sleep turn count (norm), toxic turn count (norm)
# Spread block (18 dims): IVs (6) + EVs (6) + spread_known (1) + nature modifiers (5)
# Stat order for IVs/EVs: [health_pts, atk, def, spa, spd, spe]
# Stat order for nature modifiers: [atk, def, spa, spd, spe] (HP is never nature-modified)
# spread_known=1.0 own team (IVs/EVs/nature are real values), 0.0 opponent (all zeros = padding)
POKEMON_SPREAD_OFFSET = 62     # 60 + 2 (status counters)
POKEMON_SPREAD_DIM = 18        # 6 IVs + 6 EVs + 1 flag + 5 nature modifiers
# Hidden Power candidate block (per opp slot; all-zero for our team slots).
# hp_revealed flag (1) lets the model distinguish "HP not yet seen" (whole block zero)
# from "HP seen but type ambiguous" (one or more prob entries non-zero, flag = 1).
# hp_type_probs (16) is the tracker's per-species candidate distribution in
# HIDDEN_POWER_TYPE_ORDER (alphabetical: Bug, Dark, Dragon, Electric, Fighting, Fire,
# Flying, Ghost, Grass, Ground, Ice, Poison, Psychic, Rock, Steel, Water).
POKEMON_HP_BLOCK_OFFSET = 80   # 62 + 18 (spread end)
POKEMON_HP_REVEALED_OFFSET = 80
POKEMON_HP_PROBS_OFFSET = 81
POKEMON_HP_BLOCK_DIM = 17      # 1 hp_revealed flag + 16 candidate-type probs
POKEMON_VECTOR_DIM = 97        # 62 + 18 (spread) + 17 (HP block)
POKEMON_FULL_DIM = 98          # 97 + 1 (active flag appended by state_encoder)

# Active context: boosts(14) + volatiles(9) = 23
ACTIVE_CONTEXT_DIM = 23

# Global env: weather(6) + spikes(2) + turn(1) + our_reflect(1) + our_light_screen(1) + opp_reflect(1) + opp_light_screen(1) = 13
GLOBAL_ENV_DIM = 13

MATCHUP_DIM = 288 # (6*4*6) for Our vs Their + (6*4*6) for Their vs Our
REACTIVE_DIM = 12 + MATCHUP_DIM  # 300 (removed duplicate hp+spikes: 4 dims)

# Top-level Offsets (Base dim = OFFSET_REACTIVE + REACTIVE_DIM = 1535)
NUM_POKEMON = 12
TEAM_SIZE = 6
OFFSET_OUR_TEAM = 0
OFFSET_OPP_TEAM = 6 * POKEMON_FULL_DIM                     # 588
OFFSET_CONTEXT = 2 * OFFSET_OPP_TEAM                       # 1176
OFFSET_GLOBAL = OFFSET_CONTEXT + (2 * ACTIVE_CONTEXT_DIM)  # 1222
OFFSET_REACTIVE = OFFSET_GLOBAL + GLOBAL_ENV_DIM            # 1235

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
