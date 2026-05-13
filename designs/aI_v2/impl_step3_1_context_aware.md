# Implementation Plan: Step 3.1 - Strategic Context Injection

This plan outlines the "Contextual Upgrade" to the architecture, focusing on making the model's 1v1 reasoning state-aware by injecting HP, Turn count, Weather, Spikes, and Game Phase into the core feature extraction pipeline.

## Motivation: From Dictionary to Prioritization Engine

In the previous architecture, the network evaluated moves and Pokémon in a vacuum ("Static Knowledge"). A Blissey with 10% HP was indistinguishable from a Blissey with 100% HP during the initial move-selection phase.

By injecting **Multi-Context features** directly into the feature extractors, we transition the model from a "Dictionary" (knowing what a move is) to a **"Strategic Prioritization Engine"**:

*   **HP Awareness**: Learns that healing moves are utility-zero at 100% HP.
*   **Temporal Awareness (Turn)**: Learns that investment moves (Spikes/Hazards) have high decay and should be prioritized early.
*   **Phase Awareness (Fainted Counts)**: Differentiates between "Early Game" scouting and "Late Game" cleaning.
*   **Environmental Awareness (Weather & Spikes)**: Accounts for move accuracy shifts and hazard-war strategy.

## Goal
Enable the agent to perform fine-grained threat assessment by allowing the **Move Processor** and **Role Encoder** to pre-calculate the utility of actions based on the dynamic battle state.

## Implementation Details

### 1. Phase & Environment Move Processing (Step 1 Upgrade)
Modify the `MoveNetwork` to accept a 12-dimension context vector:
- **HP (1)**: Current Pokémon health fraction.
- **Turn (1)**: Battle clock (log-scaled).
- **Weather (6)**: One-hot encoding of current weather.
- **Fainted Counts (2)**: Game Phase (How many are gone?).
- **Spikes (2)**: Hazard presence (Us/Them).

### 2. Context-Aware Role Encoding (Step 2 Upgrade)
Modify the `RoleEncoder` to accept the 11-dimension global context (Turn, Weather, Fainted Counts, Spikes).
- **Dynamic Roles**: A Pokémon's role changes from "Wall" to "Sacrifice" based on phase and health.
- **Environmental Roles**: Shifts roles for "Hazard Setters" when Spikes are already maxed out.

## Technical Architecture

### `src/agents/model/features_extractor.py`

#### Updated Dimensions
```python
# Move Network now takes 12 extra context dims
move_input_dim = 39 + 12 = 51

# Role Encoder now takes 11 extra context dims
role_input_dim = 226 + 11 = 237
```

#### Context Fusion Logic
1.  **Extract Context**: Pull Weather, Spikes, and Turn from the Global block, and Fainted Counts from the Reactive block.
2.  **Broadcast Context**: Every move slot and every Pokémon slot receives the shared global context.
3.  **Local Context**: Moves additionally receive the specific HP of the Pokémon they belong to.

## Verification Plan
1.  **Behavioral Audit**: Verify through turn logs that Blissey no longer prioritizes `Softboiled` at 100% HP.
2.  **Hazard Logic**: Verify that hazard-setting moves show decreased utility when Spikes are already at maximum (3).
3.  **Win Rate Impact**: Phase-awareness should improve the model's performance in the 1v1 endgame where "stay-in" decisions are critical.
