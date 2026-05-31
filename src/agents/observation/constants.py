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
POKEMON_VECTOR_DIM = 106       # 71 + 18 (spread) + 17 (HP block)
POKEMON_FULL_DIM = 107         # 106 + 1 (active flag appended by state_encoder)

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
REACTIVE_DIM = 12 + MATCHUP_DIM  # 300 (removed duplicate hp+spikes: 4 dims)

# Top-level Offsets (Base dim = OFFSET_REACTIVE + REACTIVE_DIM = 1547)
NUM_POKEMON = 12
TEAM_SIZE = 6
OFFSET_OUR_TEAM = 0
OFFSET_OPP_TEAM = 6 * POKEMON_FULL_DIM                     # 594
OFFSET_CONTEXT = 2 * OFFSET_OPP_TEAM                       # 1188
OFFSET_GLOBAL = OFFSET_CONTEXT + (2 * ACTIVE_CONTEXT_DIM)  # 1234
OFFSET_REACTIVE = OFFSET_GLOBAL + GLOBAL_ENV_DIM            # 1247

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
