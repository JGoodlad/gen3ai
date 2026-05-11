# Implementation Plan: Step 1 - Shared Move Processor

This plan focuses on implementing the **Shared Move Processor**, the first step in our architectural evolution. This change forces the model to learn a generalized understanding of moves by routing every move through the same neural weights.

## Goal
Replace the current independent move slot processing with a shared `MoveNetwork` that processes each of the 48 move slots (12 Pokémon * 4 moves) identically.

## Proposed Changes

### `src/agents/model/features_extractor.py`

#### 1. Initialize `MoveNetwork` in `__init__`
Add a small sub-network that will process individual move features.
```python
# In __init__, after embeddings
move_dim = (
    layout['move_embedding_dim'] + 
    layout['type_embedding_dim'] + 
    6 # Power, Secondary, Recoil, and 3 extra remnants
    + 1 # Known flag
)
self.move_network = torch.nn.Sequential(
    torch.nn.Linear(move_dim, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 32) # Projects each move to a 32-dim latent space
)
```

#### 2. Refactor `forward_internal`
Modify the stitching logic to pass moves through the network.
- **Gather Move Features:** Collect `embedded_moves`, `embedded_move_types`, `all_move_remnants`, and `known_flags`.
- **Reshape & Process:** 
    - Reshape to `[Batch * 12 * 4, move_dim]`.
    - Pass through `self.move_network`.
- **Stitch Back:** Reshape the result back to `[Batch, 12, 128]` (4 moves * 32 dims) and replace the raw move concatenations in `pokemon_enriched`.

## Verification Plan
1. **Shape Check:** Run training with `--steps 100` to verify tensor dimensions match.
2. **Parameter Count:** Verify that `self.projection_input_dim` has decreased (since moves are now 128 dims instead of 156 dims).
3. **Deep Trace:** Ensure battle state logging still works as expected.
