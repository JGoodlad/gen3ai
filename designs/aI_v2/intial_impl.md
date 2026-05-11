# Goal Description
We are implementing **Step 1: Shared Move Processor** from the architectural roadmap. This refactor will update `Gen3FeaturesExtractor` to process all 48 moves (12 Pokémon * 4 slots) through a single, shared embedding network. 

This change forces the model to learn a generalized understanding of moves (e.g., "Earthquake is always Earthquake") regardless of which Pokémon or slot holds the move.

## User Review Required
- **Dimension Reduction:** By projecting each move down to 32 dimensions, we are reducing the "Move Section" of the Pokémon vector from 156 dimensions down to 128 dimensions. This makes the model more parameter-efficient.
- **Activation:** We are using ReLU inside the `MoveNetwork` to allow for non-linear feature extraction of move properties (Power/Type/Category).

## Proposed Changes

### `src/agents/model/`
---

#### [MODIFY] [features_extractor.py](file:///home/goodlad/dev/gen3ai/src/agents/model/features_extractor.py)
**1. Initialize `MoveNetwork` in `__init__`:**
We will define a new small MLP that will be applied to every move in parallel.
```python
self.move_network = nn.Sequential(
    nn.Linear(move_feature_dim, 64),
    nn.ReLU(),
    nn.Linear(64, 32)
)
```

**2. Refactor `forward_internal` - Shared Move Processing:**
- Extract all raw move features (Embeddings + Power/Acc/Flags).
- Reshape the tensor to `[Batch * 48, move_feature_dim]`.
- Pass through `self.move_network`.
- Reshape back to `[Batch, 12, 128]` (stitching 4 moves * 32 dims).

**3. Stitching:**
- Replace the raw move concatenations in `pokemon_enriched` with the output of the `MoveNetwork`.

## Verification Plan

### Automated Tests
- Run `python3 src/main/train_rl_agent.py --debug --steps 100` to verify tensor shapes.
- Check that the total parameter count in the summary has decreased slightly.

### Manual Verification
- Observe the training curve. We expect a steeper initial learning rate as the model generalizes move knowledge across the entire team instantly.
