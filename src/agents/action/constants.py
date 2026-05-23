"""Action space constants for Gen 3 OU: 11 discrete actions."""

ACTION_SPACE_SIZE = 11

# Switches: team slots 0-5
SWITCH_START = 0
SWITCH_END = 6      # exclusive; action < SWITCH_END is a switch
N_SWITCH_SLOTS = 6

# Moves: request slots 0-3, mapped to actions 6-9
MOVE_START = 6
MOVE_END = 10       # exclusive; action < MOVE_END is a move
N_MOVE_SLOTS = 4

# Struggle: dedicated slot
STRUGGLE = 10
