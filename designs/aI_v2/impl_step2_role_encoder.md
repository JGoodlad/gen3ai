# Implementation Plan: Step 2 - Pokémon Role Encoder

This plan focuses on implementing the **Pokémon Role Encoder**, which fuses Species, Stats, and Moves into a single semantic "Role Token" for each Pokémon.

## Goal
Pass each of the 12 Pokémon through a shared `RoleEncoder` MLP to create a condensed 128-dimension representation (Token) per Pokémon.

## Proposed Changes

### `src/agents/model/features_extractor.py`

#### 1. Initialize `RoleEncoder` in `__init__`
Add the network that will create the Pokémon tokens.
```python
# In __init__, after move_network
# pokemon_input_dim is the size of the stitched pokemon_enriched vector
self.role_encoder = torch.nn.Sequential(
    torch.nn.Linear(pokemon_input_dim, 256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 128) # Final Role Token size
)
```

#### 2. Refactor `forward_internal`
- **Encode Roles:** After `pokemon_enriched` is stitched (including the processed moves from Step 1):
    - Reshape `pokemon_enriched` to `[Batch * 12, pokemon_input_dim]`.
    - Pass through `self.role_encoder`.
    - Reshape to `[Batch, 12, 128]`.
- **Final Concatenation:** Flatten the 12 tokens (`[Batch, 1536]`) and concatenate with `remaining_part` (global context).

## Verification Plan
1. **Dimension Stability:** Verify `projection_input_dim` is now `(12 * 128) + context_dim`.
2. **Feature Quality:** Check if `explained_variance` begins to stabilize faster than the flat v1 model.
