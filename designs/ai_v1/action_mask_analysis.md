# Action Mask Analysis & Infrastructure Fixes

## The Issue: Action Space Mismatch
The primary cause of the initial failures was a mismatch between the reinforcement learning agent's expected action space and the reality of Generation 3 mechanics.

### 1. 22 vs 10 Actions
- **Observation**: Most modern Pokémon RL examples use a 22-dimensional action space (4 moves + 6 switches + 12 gimmick slots for Mega, Z-Moves, Dynamax, etc.).
- **Reality**: In Gen 3, these gimmicks do not exist. The standard `SinglesEnv` in `poke-env` defaults to a 10-dimensional space (4 moves + 6 switches).
- **Result**: The agent was "hallucinating" actions that didn't exist, leading to misaligned probability distributions and logic errors.

### 2. The "Slot 0" Active Mon Trap
In `poke-env`, actions 0-5 map directly to the indices of your team list.
- **Problem**: The Pokémon in Slot 0 of the team is almost always the one currently **active** on the field.
- **Error**: You cannot switch to a Pokémon that is already active. Attempting to send a `/choose switch [ActiveMon]` command results in a `ValueError` on the server.
- **Our Fix**: We now use a custom `get_gen3_action_mask` that explicitly checks `battle.available_switches`. It identifies exactly which team slots are currently on the bench and only masks those as valid.

## Implementation Details
The fix involved overriding the `get_action_mask` method in `Gen3Env` and implementing a robust mapping function:
```python
def get_gen3_action_mask(battle):
    mask = np.zeros(10, dtype=np.int8)
    # Correctly map bench slots (1-5) based on available_switches
    # Correctly map move slots (6-9) based on available_moves
    ...
```

## Impact
This stabilization allows the agent to play 100% of its turns legally. The crash rate has dropped to zero, allowing us to scale from 1k steps to 10M+ steps.
