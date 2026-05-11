# Implementation Plan: Step 3 - Active Matchup Attention

This plan focuses on implementing **Active Matchup Attention**, allowing the agent to explicitly reason about the 1v1 battle between the two active Pokémon.

## Goal
Implement a `MultiheadAttention` layer that allows the allied active Pokémon to "query" the opponent's active Pokémon to calculate matchup advantages, directly utilizing the Role Tokens generated in Step 2.

## Proposed Changes

### `src/agents/model/features_extractor.py`

#### 1. Initialize Attention in `__init__`
```python
self.matchup_attention = torch.nn.MultiheadAttention(
    embed_dim=128, # Assuming Role Tokens are 128-dim
    num_heads=4, 
    batch_first=True
)
```

#### 2. Refactor `forward_internal`
- **Dynamic Indexing (The Static Ordering Fix):**
    - Because Step 2 compresses the entire Pokémon vector (including the active flag) into a dense Role Token, we cannot extract the active flag from the Role Token itself.
    - Instead, use the *original* `hp_and_active` tensor that was extracted from `pokemon_part` before the Role Encoder.
    - The `active_flag` is the last feature of the 133-dim Pokémon array: `active_flags = pokemon_part[:, :, -1]`
    - Find the active indices dynamically:
      ```python
      our_active_idx = torch.argmax(active_flags[:, 0:6], dim=1) # [Batch]
      opp_active_idx = torch.argmax(active_flags[:, 6:12], dim=1) + 6 # [Batch], offset by 6
      ```
- **Extract Active Tokens:**
    - Use `our_active_idx` and `opp_active_idx` to gather the specific Active Role Tokens from the `[Batch, 12, 128]` tensor outputted by Step 2.
    - `our_active_token = role_tokens[torch.arange(batch_size), our_active_idx].unsqueeze(1) # [Batch, 1, 128]`
    - `opp_active_token = role_tokens[torch.arange(batch_size), opp_active_idx].unsqueeze(1) # [Batch, 1, 128]`
- **Query / Key / Value:**
    - Allied Active Token is the **Query**.
    - Opponent Active Token is the **Key** and **Value**.
    - `matchup_context, _ = self.matchup_attention(our_active_token, opp_active_token, opp_active_token)`
- **Context Fusion:** Concatenate the flattened `matchup_context` (size 128) with the flattened team vector (`12 * 128 = 1536`) and global context before the final projection.

## Verification Plan
1. **Masking Check:** Ensure that if a Pokémon is fainted, it cannot be "queried" (though Step 3 only cares about Active).
2. **Win Rate vs Heuristic:** This is the upgrade expected to break the -15 reward plateau by explicitly teaching the agent 1v1 threat assessment.
