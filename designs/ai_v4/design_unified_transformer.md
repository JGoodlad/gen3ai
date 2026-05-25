# ai_v4: Unified Transformer Feature Extractor

## Motivation

The current `Gen3FeaturesExtractor` (ai_v3) is built from three completely isolated sub-networks:

```
Role encoder × 12 slots
  → 5 hand-crafted attention paths (pressure, safety, synergy, threat, opp-synergy)
  → team pool queries
  → [128, 128, 128]                 ┐
                                    ├─→ cat → LN → Linear → 512
Turn delta encoder                  │
  → self-attention over N turns     │
  → take last position [99]        ┘
  + active ctx, global scalars
```

The three towers only communicate through a single linear projection. This means:

- **Team role tokens never see turn history.** When the model asks "should I switch Gengar?", it has no direct access to "Gengar's item was consumed on turn 4."
- **History tokens never see team state.** The history sub-network can't weight recent turns by how relevant they are to the current matchup — it operates blind to what Pokémon are on the field.
- **Five attention paths are hand-engineered.** Pressure, safety, synergy, threat, and opp-synergy reflect human intuitions about what interactions matter. A learned transformer will discover better patterns.

This design replaces all of that with a single unified transformer where every token attends to every other token in each layer.

### Connection to v5+

The v5 team-completion model uses role tokens produced by the PPO feature extractor as its input. Under ai_v3, those role tokens are history-blind. Under ai_v4, each role token is produced after attending over the full turn history — it encodes not just "what is this Pokémon's kit" but "what has this Pokémon done and experienced." The team-completion model downstream gets richer input with no changes to its own design.

The same applies to the spread-inference auxiliary heads (v5) and the MCTS value function (v5): both benefit from feature extractor outputs that are history-informed at the per-slot level.

---

## Architecture

### Token sequence

All tokens are projected to a common `d_model = 128`.

| Group | Count | Source dim | Projection |
|-------|-------|-----------|------------|
| Our team slots | 6 | 128 (role encoder output) | None — already `d_model` |
| Their team slots | 6 | 128 (role encoder output) | None |
| Turn history | `N_HISTORY_TURNS` (10) | 99 (`_td_embed_dim`) | `Linear(99, 128)` |
| Global context | 1 | varies (scalar block) | `Linear(G, 128)` |
| **Total** | **23** | | |

Role encoder and move processor are **unchanged** from ai_v3 — they still compute per-slot token vectors as before. The unification happens at the integration layer.

### Token type embeddings

A learned embedding table with 4 entries (dim=128) is added to every token before the transformer:

```
TokenType: OUR_TEAM=0, THEIR_TEAM=1, HISTORY=2, GLOBAL=3
```

This lets the attention heads learn patterns conditioned on token role (e.g. "history tokens of type 2 are relevant to their_team tokens of type 1").

Turn history tokens additionally receive **positional encodings** (learned, `N_HISTORY_TURNS × 128` table) so the model can distinguish "2 turns ago" from "8 turns ago."

### Transformer layers

**L = 2 layers**, each identical:

```
x = LayerNorm(x + MHA(x, x, x, key_padding_mask))   # self-attention + residual
x = LayerNorm(x + FFN(x))                            # FFN + residual

FFN: Linear(128, 256) → ReLU → Linear(256, 128)
MHA: embed_dim=128, num_heads=4, batch_first=True
```

`L = 2` matches the team-completion transformer already designed in v5 and is sufficient for a 23-token sequence. The FFN 2× expansion (256) is standard.

### Key padding mask

A boolean mask `[B, 23]` is built before the transformer:

- **Fainted team slots** → masked as keys (same logic as current `fainted_mask_ours/opp`)
- **Empty history turns** (early game, all-zero after embedding) → masked as keys; detected by checking `history_slots.abs().sum(dim=-1) == 0` on the raw `[B, N, 39]` block
- **Global context token** → never masked

All-zero masking is robust: a real turn can never be all-zero because the `log_turn` scalar is always positive.

### Output extraction

Two learned CLS parameter vectors cross-attend over their respective team tokens after the transformer:

```python
our_cls   = Parameter[1, 1, 128]   # cross-attends over our 6 team output tokens
their_cls = Parameter[1, 1, 128]   # cross-attends over their 6 team output tokens

our_pooled   = LayerNorm(CLS_MHA(our_cls,   our_out,   our_out,   key=fainted_mask_ours))   # [B, 128]
their_pooled = LayerNorm(CLS_MHA(their_cls, their_out, their_out, key=fainted_mask_opp))    # [B, 128]
```

This is the same pattern used in ai_v3 for team pooling — preserved because it gives permutation-equivariant team summaries.

### Projection

The final concatenation and projection are **unchanged**:

```
combined = cat([our_pooled(128), their_pooled(128), our_active_out(128),
                our_ctx_enc(32), opp_ctx_enc(32), non_matchup_rest])
→ LayerNorm → Linear(combined_dim, 512) → ReLU → [B, 512]
```

`combined_dim` is auto-discovered via a dummy forward pass in `__init__` — no manual calculation needed.

---

## What Is Removed vs. Kept

### Removed from `features_extractor.py`

| Component | Replaced by |
|-----------|-------------|
| `pressure_attn` / `safety_attn` / `synergy_attn` / `threat_attn` / `opp_synergy_attn` | Unified transformer self-attention |
| `our_pool_query`, `our_pool_attn`, `norm_pool_our` | `our_cls` + `CLS_MHA` |
| `their_pool_query`, `their_pool_attn`, `norm_pool_their` | `their_cls` + `CLS_MHA` |
| `active_ctx_to_role` (Linear 23→64→128 injection) | Handled naturally via attention |
| `td_conditioner` (Linear 10→64→128 injection) | Handled naturally via attention |
| `turn_history_attn`, `turn_history_pos_emb`, `turn_history_norm` | Unified transformer handles history |
| `turn_history_pool_query/attn/norm` (if present) | Replaced entirely |

### Kept unchanged

| Component | Notes |
|-----------|-------|
| All embedding tables (species, move, item, ability, type) | Unchanged |
| Move processor (Linear 58→64→32 per slot) | Unchanged |
| Within-Pokémon move self-attention (MHA 32, 2 heads) | Unchanged |
| Role encoder (Linear 263→256→128) | Unchanged |
| Active context encoder (`active_ctx_to_role` → `Linear(active_ctx_dim, 32)`) | Kept for `combined` input |
| Turn delta `_embed_delta_slot` | Unchanged |
| `pre_proj_norm` + `projection` + `activation` | Unchanged |

---

## Key Design Decisions

**d_model = 128**: Matches `ROLE_TOKEN_SIZE`. Team role tokens are already 128D and need no projection, saving ~200K parameters.

**L = 2 layers**: Two transformer layers is consistent with the v5 team-completion design and is empirically sufficient for sequences of ~20 tokens. Adding a third layer is a one-line change if needed.

**4 heads, FFN = 256**: 128/4 = 32 dims per head. Standard 2× FFN expansion. Both match the team-completion model for architectural consistency.

**Token type + positional embeddings**: Type embeddings are critical — without them, the transformer can't learn role-specific patterns. Positional embeddings on history tokens only (team slots are permutation-equivariant by design; no position needed there).

**N_HISTORY_TURNS = 10**: Bumped from 5. Gen3 games run 20–40 turns; sleep tracks up to 4 turns, Toxic stacks accumulate, weather runs 5 turns — 10-turn context covers most strategically relevant history. Obs vector: `base + 11 + 10×39 ≈ 1504 dims`.

**Empty-turn masking**: Masking zero-padded early-game slots ensures attention scores over real turns don't get diluted by meaningless padding. Without masking, on turn 1 the model would attend equally to 9 zero vectors — this would likely train away, but explicit masking is cleaner.

---

## Model Versioning

| Field | Old | New |
|-------|-----|-----|
| `ARCH_SIGNATURE` | `"gen3_td_cond_v1"` | `"gen3_unified_v1"` |
| `MODEL_CONFIG_VERSION` | `2` | `2` (unchanged) |
| `N_HISTORY_TURNS` | `5` | `10` |

`ARCH_SIGNATURE` change ensures all prior checkpoints fail `check_compatible()` immediately with a clear architecture-family error. No migration logic needed — this is a clean break.

---

## Implementation Checklist

### `src/agents/model/features_extractor.py`

**New components (add to `__init__`):**
- [ ] `self.token_type_emb`: `Embedding(4, 128)` — type ids: our_team=0, their_team=1, history=2, global=3
- [ ] `self.turn_history_pos_emb`: `Embedding(N_HISTORY_TURNS, 128)` — positional encoding for history tokens
- [ ] `self.history_proj`: `Linear(_td_embed_dim, 128)` — project 99→128
- [ ] `self.global_proj`: `Linear(G, 128)` — project global scalar block (dim from layout); G to be read from `get_layout()` at init time
- [ ] `self.transformer_layers`: `ModuleList` of L=2 transformer encoder layers (`TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, batch_first=True, norm_first=False)`)
- [ ] `self.our_cls` / `self.their_cls`: `Parameter(torch.randn(1, 1, 128) * 0.02)`
- [ ] `self.our_cls_attn` / `self.their_cls_attn`: `MultiheadAttention(128, 4, batch_first=True)`
- [ ] `self.norm_pool_our` / `self.norm_pool_their`: `LayerNorm(128)`

**Update `N_HISTORY_TURNS`:**
- [ ] Line ~27: `N_HISTORY_TURNS = 10`

**Rewrite `forward_internal` integration section:**
- [ ] Build `history_slots [B, N, 39]` and detect empty turns → `history_key_mask [B, N]`
- [ ] Embed history via `_embed_delta_slot` → `[B, N, 99]` → `history_proj` → `[B, N, 128]`
- [ ] Add positional encodings to history tokens
- [ ] Project global scalar block → `[B, 1, 128]`
- [ ] Add token type embeddings to all groups
- [ ] Concatenate all tokens: `[B, 23, 128]`
- [ ] Build full `key_padding_mask [B, 23]` (fainted slots + empty history + never global)
- [ ] Run through `transformer_layers` (passing `src_key_padding_mask`)
- [ ] Extract team output tokens `[B, 6, 128]` per side; history and global outputs discarded
- [ ] CLS cross-attention → `our_pooled [B, 128]`, `their_pooled [B, 128]`
- [ ] Feed `our_active_out` (our team's active slot from transformer output) into existing active-context path

**Remove:**
- [ ] `pressure_attn`, `safety_attn`, `synergy_attn`, `threat_attn`, `opp_synergy_attn` (init + forward)
- [ ] `our_pool_query/attn/norm`, `their_pool_query/attn/norm` (init + forward)
- [ ] `active_ctx_to_role` (init + forward injection)
- [ ] `td_conditioner` (init + forward injection)
- [ ] Old `turn_history_attn/pos_emb/norm` block (init + forward)

**Verify:**
- [ ] `projection_input_dim` still auto-discovered via dummy forward pass in `__init__`

### `src/agents/model/model_version.py`
- [ ] `ARCH_SIGNATURE = "gen3_unified_v1"`

### Tests
- [ ] `features_extractor_test.py`: update expected shapes for new forward pass
- [ ] Run: `pytest src/ -m "not integration and not e2e" -q`

### Verification
```bash
# Dimension sanity check
export PYTHONPATH=$PYTHONPATH:src
python3 -c "
from agents.observation.state_encoder import Gen3ObservationEncoder
from utils.mappings import load_mappings
enc = Gen3ObservationEncoder(load_mappings())
print('dimension:', enc.dimension)           # expected: ~1504
layout = enc.get_layout()
print('n_history_turns:', layout['n_history_turns'])    # expected: 10
print('turn_history_dim:', layout['turn_history_dim'])  # expected: 390
"

# Smoke test (full pipeline, ~1 min)
export PYTHONPATH=$PYTHONPATH:src
python3 src/main/train_rl_agent.py --debug --steps 10000
# Look for: [ModelVersion] Round-trip smoke test PASSED
```
