# Implementation Plan: Step 4 - Full Team Cross-Attention

This plan focuses on implementing **Full Team Cross-Attention**, enabling the agent to consider counters on the bench and predict opponent switches.

## Goal
Pass all 12 Pokémon tokens through a Transformer Encoder to allow universal relational reasoning across both teams.

## Proposed Changes

### `src/agents/model/features_extractor.py`

#### 1. Initialize Transformer in `__init__`
```python
encoder_layer = torch.nn.TransformerEncoderLayer(
    d_model=128, 
    nhead=8, 
    batch_first=True
)
self.team_transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)

# Learned positional embeddings for the 12 slots
self.pos_embedding = torch.nn.Parameter(torch.randn(1, 12, 128))
```

#### 2. Refactor `forward_internal`
- **Apply Transformer:**
    - Add `pos_embedding` to the 12 Role Tokens.
    - Pass the sequence `[Batch, 12, 128]` through `self.team_transformer`.
- **Global Pooling:** Instead of just flattening, we can now take the Transformer output for the active Pokémon or use global average pooling to summarize the state of the whole battlefield.

## Verification Plan
1. **Switch Logic:** Monitor if the agent starts making "safe switches" into counters rather than staying in against bad matchups.
2. **Convergence:** Transformer training can be slower; monitor `entropy_loss` to ensure the agent is still converging.
