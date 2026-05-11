# Goal Description
We are implementing Step 1 (Shared Move Processor) and Step 2 (Pokémon Role Encoder) from the architectural roadmap. This refactor will update `Gen3FeaturesExtractor` to process moves universally and fuse Pokémon features into dense "Role Tokens", setting the foundation for the Attention layer. We will explicitly preserve the static ordering of the observation array to avoid state-flickering bugs.

> [!NOTE]
> This plan focuses exclusively on `features_extractor.py`. We are not changing `state_encoder.py` or the observation space size, so this will not break any existing checkpoints or environment wrappers.

## User Review Required
Please review the proposed dimensions for the new sub-networks.
- `MoveNetwork`: Will project each move down to 32 dimensions. Since there are 4 moves, this results in 128 dimensions dedicated to moves per Pokémon.
- `PokemonRoleEncoder`: Will compress the entire `[Species + Stats + 128_dim_moves + Items + Abilities]` vector (currently ~284 dims) down to a **128-dimension Role Token**.

## Open Questions
- Do you want to keep the final output dimension of the feature extractor at 512, or should we adjust it now that the parameter count is changing? (I propose keeping it at 512 for stability with PPO's default MLP).

## Proposed Changes

### `src/agents/model/`
---

#### [MODIFY] [features_extractor.py](file:///home/goodlad/dev/gen3ai/src/agents/model/features_extractor.py)
**1. Initialize New Networks in `__init__`:**
We will define two new small MLPs:
- `self.move_network = nn.Sequential(nn.Linear(move_input_dim, 64), nn.ReLU(), nn.Linear(64, 32))`
- `self.role_encoder = nn.Sequential(nn.Linear(pokemon_input_dim, 256), nn.ReLU(), nn.Linear(256, 128))`

**2. Refactor `forward_internal` - Shared Move Processing:**
Instead of flattening embeddings directly into the Pokémon vector:
- Concatenate the embedded move ID, embedded move type, power, accuracy, and known flags for each slot.
- Reshape the tensor to `[Batch * 12_Pokemon * 4_Moves, move_feature_size]`.
- Pass it through `self.move_network`.
- Reshape back to `[Batch, 12_Pokemon, 4_Moves * 32_dims]`.
- *Why:* This forces the network to use the exact same logic to evaluate a move regardless of which Pokémon or slot holds it.

**3. Refactor `forward_internal` - Pokémon Role Encoding:**
- Stitch the Pokémon vector together just like before (Species, Stats, Item, Abilities, and our newly processed Moves).
- Instead of flattening all 12 Pokémon together immediately, reshape to `[Batch * 12_Pokemon, pokemon_feature_size]`.
- Pass it through `self.role_encoder`.
- Reshape back to `[Batch, 12_Pokemon, 128_dims]`.
- *Why:* This compresses the raw stats and moves into a semantic "Role Token" (e.g., Special Wallbreaker). 

**4. Preserving Static Ordering (Nuance for Step 3 Prep):**
- We will explicitly document and enforce in the code that the resulting `[Batch, 12, 128]` tensor **remains strictly in its original slot order** (Indices 0-5 are Our Slots 1-6, Indices 6-11 are Their Slots 1-6). 
- We do not sort or slice based on the active Pokémon at this stage. The `hp_and_active` flags are fed into the `role_encoder`, meaning the resulting Role Token inherently "knows" if it is currently on the field or benched, setting up the Attention mechanism (Step 3) to dynamically query the active token without scrambling the array.

**5. Final Assembly:**
- Flatten the 12 Role Tokens: `[Batch, 12 * 128]`.
- Concatenate with the global/context vectors.
- Pass through the final `self.projection` (to 512 dims).

## Verification Plan

### Automated Tests
- Run `python3 src/main/train_rl_agent.py --debug --steps 100` to ensure the new forward pass executes without shape mismatches or tensor errors.
- Verify the Deep Trace still functions correctly (since we aren't changing the `state_encoder`, trace descriptions should remain perfect).

### Manual Verification
- Observe the TensorBoard `explained_variance` in the first 50k steps. It may start slightly lower than a loaded checkpoint, but we should see it climb quickly due to the generalized move weights.
