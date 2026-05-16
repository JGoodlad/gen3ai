# Sub-dimensions
SPECIES_ID_DIM = 1 # ID for embedding
ITEM_ID_DIM = 1    # Only the first slot is the ID; was 16 (wasted)
ITEM_KNOWN_DIM = 1
COMBINED_TYPES_DIM = 2  # Two type IDs only; was 8 (6 placeholders removed)
ABILITY_SLOT_DIM = 1    # ID only; was 8 (possible-ability slots removed)
ABILITY_KNOWN_DIM = 1
MOVE_SLOT_DIM = 9
MOVES_KNOWN_DIM = 0
CONDITION_DIM = 7 # None, BRN, PAR, SLP, FRZ, PSN, TOX
STATS_DIM = 6 # HP, Atk, Def, SpA, SpD, Spe

# Active Context Sub-dimensions (temporal block removed — was unimplemented)
BOOSTS_DIM = 14
VOLATILES_DIM = 8

# Internal Pokémon Vector Offsets
# Species: 7 (1 ID + 6 Stats)
# Items: ITEM_ID_DIM(1) + ITEM_KNOWN_DIM(1) = 2
# Types: COMBINED_TYPES_DIM(2) = 2
# Abilities: ABILITY_SLOT_DIM(1) + ABILITY_KNOWN_DIM(1) = 2
# Condition: CONDITION_DIM(7) = 7
# Moves: 4 * MOVE_SLOT_DIM(9) = 36
# HP: 1   Total: 57
POKEMON_SPECIES_OFFSET = 0
POKEMON_ITEMS_OFFSET = 7   # 0 + 7
POKEMON_TYPES_OFFSET = 9   # 7 + 2
POKEMON_ABILITIES_OFFSET = 11  # 9 + 2
POKEMON_CONDITION_OFFSET = 13  # 11 + 2
POKEMON_MOVES_OFFSET = 20  # 13 + 7
POKEMON_HP_OFFSET = 56     # 20 + 36
POKEMON_VECTOR_DIM = 57    # 56 + 1
POKEMON_FULL_DIM = 58      # 57 + 1 (active flag)

# Active context: boosts(14) + volatiles(8) = 22
ACTIVE_CONTEXT_DIM = 22

# Global env: weather(6) + spikes(2) + turn(1) + our_reflect(1) + our_light_screen(1) + opp_reflect(1) + opp_light_screen(1) = 13
GLOBAL_ENV_DIM = 13

MATCHUP_DIM = 288 # (6*4*6) for Our vs Their + (6*4*6) for Their vs Our
REACTIVE_DIM = 12 + MATCHUP_DIM  # 300 (removed duplicate hp+spikes: 4 dims)

# Top-level Offsets (Total Dim: 1021)
NUM_POKEMON = 12
TEAM_SIZE = 6
OFFSET_OUR_TEAM = 0
OFFSET_OPP_TEAM = 6 * POKEMON_FULL_DIM   # 330
OFFSET_CONTEXT = 2 * OFFSET_OPP_TEAM     # 660
OFFSET_GLOBAL = OFFSET_CONTEXT + (2 * ACTIVE_CONTEXT_DIM)  # 704
OFFSET_REACTIVE = OFFSET_GLOBAL + GLOBAL_ENV_DIM  # 717

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
