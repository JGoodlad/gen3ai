# Implementation: Step 3 — Unified Transformer Feature Extractor

This step replaces the v3 feature extractor's three isolated sub-networks
(role-token + 5 hand-engineered attention paths, single-TurnDelta history
attention, and two pre-attention injection MLPs) with a single L=2 transformer
that attends over the entire game state in one pass. Per-Pokémon role tokens
are now history-informed at the per-slot level — a strict requirement for the
v5 team-completion and MCTS value heads which consume these tokens directly.

Primary themes: collapsing five hand-engineered attention paths into one
unified self-attention, packing all per-turn context (team slots, raw turn
history, global scalars) into a single 23-token sequence, replacing slot-order
flatten with permutation-equivariant CLS pooling, and a clean architecture
break via `ARCH_SIGNATURE`.

---

## Motivation

### The gap

The v3 extractor produces three classes of tokens that only meet at the final
projection layer:

```
Role tokens     ─→ 5 attention paths ─→ slot-flatten ─┐
                                                      ├─→ cat → LN → Linear → 512
Turn-delta block ─→ history self-attn ─→ last slot ──┘
Active context  ─→ Linear(32) ───────────────────────┘
```

Three immediate consequences:

1. **Team role tokens never see turn history.** "Should I switch Gengar?"
   never has direct access to "Gengar's Salac Berry was consumed on turn 4."
2. **History tokens never see team state.** The history sub-network cannot
   weight recent turns by how relevant they are to the current matchup.
3. **Five attention paths are hand-engineered.** Pressure, Safety, Synergy,
   Threat, and Opp Synergy reflect human intuitions about what interactions
   matter — a learned transformer should discover better patterns.

### Connection to v5+

The v5 team-completion model and the MCTS value function both consume role
tokens produced by this extractor. Under v3, those role tokens are
history-blind. Under v4 unified, each role token has already attended over the
full turn history and the opposing team — encoding not just "what is this
Pokémon's kit" but "what has this Pokémon done and experienced." Downstream
heads get richer input with no changes to their own designs.

---

## What Changed

### Architecture — 23 tokens, L=2 transformer

All tokens are projected to a common `d_model = 128`.

| Group | Count | Source dim | Projection |
|-------|-------|-----------|------------|
| Our team slots | 6 | 128 (role encoder output) | None — already `d_model` |
| Their team slots | 6 | 128 (role encoder output) | None |
| Turn history | `N_HISTORY_TURNS` = 10 | 99 (`_td_embed_dim`) | `Linear(99, 128)` |
| Global context | 1 | 71 (active ctx ×2 + non-matchup scalars) | `Linear(71, 128)` |
| **Total tokens** | **23** | | |

Each transformer layer:

```
x = LayerNorm(x + MHA(x, x, x, key_padding_mask))   # self-attention + residual
x = LayerNorm(x + FFN(x))                            # FFN + residual

FFN:    Linear(128, 256) → ReLU → Linear(256, 128)
MHA:    embed_dim=128, num_heads=4, batch_first=True
dropout: 0.0  (the rest of the network is dropout-free)
```

Two layers (L=2) is sufficient for a 23-token sequence and matches the v5
team-completion transformer's depth for architectural consistency.

### New module-level constants

| Constant | Value | Purpose |
|----------|-------|---------|
| `D_MODEL` | `128` (= `ROLE_TOKEN_SIZE`) | Transformer d_model; team tokens enter unprojected |
| `TRANSFORMER_N_LAYERS` | `2` | Encoder layer count |
| `TRANSFORMER_N_HEADS` | `4` | 128 / 4 = 32 dims per head |
| `TRANSFORMER_FFN_DIM` | `256` | Standard 2× FFN expansion |
| `N_HISTORY_TURNS` | `10` (was `5`) | History length covers Gen3 sleep/Toxic/weather windows |
| `NUM_TOKEN_TYPES` | `4` | our_team=0, their_team=1, history=2, global=3 |
| `TOKEN_TYPE_OUR_TEAM` / `_THEIR_TEAM` / `_HISTORY` / `_GLOBAL` | `0..3` | Named indices used at forward time |

### Observation vector dimensions

`N_HISTORY_TURNS` doubled, so the obs grows by 5 × 39 = 195 dims:

| Block | Step 2 | Step 3 | Notes |
|-------|--------|--------|-------|
| Base (teams + active ctx + global + reactive) | 1523 | 1523 | Unchanged |
| Prev-turn action mask | 11 | 11 | Unchanged |
| Turn history (N × 39) | 195 | 390 | `N_HISTORY_TURNS` 5 → 10 |
| **Total** | **1729** | **1924** | |

### Architecture version

`ARCH_SIGNATURE` changed from `"gen3_hp_v1"` to `"gen3_unified_v1"`. The
total_dim mismatch alone would catch old checkpoints via `check_compatible()`,
but the signature bump produces an explicit "architecture family mismatch"
error rather than a generic dim-list diff. `MODEL_CONFIG_VERSION` is
unchanged (no schema additions — every weight-shape difference is covered by
existing fields plus the signature bump).

---

## Implementation Details

### Constants and removed components

In `src/agents/model/features_extractor.py`:

**Added:**
- `D_MODEL`, `TRANSFORMER_N_LAYERS`, `TRANSFORMER_N_HEADS`, `TRANSFORMER_FFN_DIM`,
  `TOKEN_TYPE_*`, `NUM_TOKEN_TYPES`
- `N_HISTORY_TURNS` bumped 5 → 10

**Removed (init + forward):**
- `pressure_attn`, `safety_attn`, `synergy_attn`, `threat_attn`, `opp_synergy_attn` and their `norm1..norm4` + `norm_opp_synergy`
- `our_pool_query`, `their_pool_query`, `our_pool_attn`, `their_pool_attn` (replaced by CLS parameters)
- `status_embedding` (our_active / our_bench / their_active / their_bench bias) — replaced by `token_type_emb`'s 4-way group identity; active-vs-bench is already encoded in the role token's input via the active flag
- `active_ctx_to_role` (Linear 23→64→128 pre-attention injection)
- `td_conditioner` (Linear 10→64→128 pre-attention injection)
- The standalone `turn_history_pos_emb` (99-dim), `turn_history_attn` (3-head 99-dim MHA), and `turn_history_norm` block

**Kept unchanged:**
- All five embedding tables (species, move, item, ability, type)
- Shared move processor (`move_network`)
- Within-Pokémon move self-attention (`move_self_attn` / `move_self_norm`)
- Role encoder (`role_encoder`, dim auto-computed from layout)
- Active context encoder (`active_ctx_encoder`) — still feeds the projection
- `_embed_delta_slot` static helper
- `pre_proj_norm` + `projection` + ReLU activation; projection input dim still
  auto-discovered via dummy forward pass in `__init__`

### Unified transformer (init)

```python
# Token group identity (4: our_team / their_team / history / global)
self.token_type_emb = torch.nn.Embedding(NUM_TOKEN_TYPES, D_MODEL)

# History projection 99 → 128 plus a learned positional encoding per slot
self.history_proj = torch.nn.Linear(self._td_embed_dim, D_MODEL)
self.turn_history_pos_emb = torch.nn.Embedding(N_HISTORY_TURNS, D_MODEL)

# Global token input: our_ctx + opp_ctx + non_matchup_rest (= 71 dims)
self._non_matchup_rest_dim = GLOBAL_ENV_DIM + matchup_offset_in_reactive
self._global_token_input_dim = 2 * active_ctx_dim + self._non_matchup_rest_dim
self.global_proj = torch.nn.Linear(self._global_token_input_dim, D_MODEL)

# L=2 stack — dropout=0 to keep forward_internal deterministic outside train mode
self.transformer_layers = torch.nn.ModuleList([
    torch.nn.TransformerEncoderLayer(
        d_model=D_MODEL, nhead=TRANSFORMER_N_HEADS,
        dim_feedforward=TRANSFORMER_FFN_DIM, dropout=0.0,
        activation="relu", batch_first=True, norm_first=False,
    )
    for _ in range(TRANSFORMER_N_LAYERS)
])

# Permutation-equivariant team pooling via CLS cross-attention
self.our_cls = torch.nn.Parameter(torch.randn(1, 1, D_MODEL) * 0.02)
self.their_cls = torch.nn.Parameter(torch.randn(1, 1, D_MODEL) * 0.02)
self.our_cls_attn   = torch.nn.MultiheadAttention(D_MODEL, TRANSFORMER_N_HEADS, batch_first=True)
self.their_cls_attn = torch.nn.MultiheadAttention(D_MODEL, TRANSFORMER_N_HEADS, batch_first=True)
self.norm_pool_our   = torch.nn.LayerNorm(D_MODEL)
self.norm_pool_their = torch.nn.LayerNorm(D_MODEL)
```

The role encoder, move processor, and within-Pokémon move attention all stay
exactly as in v3, so their weight shapes remain compatible across the upgrade
modulo the signature break.

### Forward pass integration

After role tokens are built (steps 1–6 of `forward_internal` — unchanged), the
new integration replaces all of v3's attention-path logic with:

1. **History tokens.** Embed each of the 10 raw TurnDelta vectors through the
   shared move/type tables (4 raw IDs → 64-dim embedding) + 35 scalars =
   99-dim. Project to 128. Add learned positional encodings indexed 0..9.
2. **Global token.** Concatenate `our_ctx_raw + opp_ctx_raw + non_matchup_rest`
   (23 + 23 + 25 = 71 dims) and project to 128. Unsqueeze to `[B, 1, 128]`.
3. **Token type embedding.** Add the per-group embedding to each group before
   concatenation. Broadcasting handles the `[1, D_MODEL] → [B, N, D_MODEL]`
   expansion.
4. **Build the 23-token sequence.** Concatenate `our_team + their_team +
   history + global` along dim 1.
5. **Key padding mask.** True at padding positions:
   - Fainted team slots (HP == 0, with the active slot always unmasked so
     attention has at least one live key)
   - Empty history slots (detected as `history_slots.abs().sum(-1) == 0` —
     robust because real turns always have non-zero scalars like log_turn)
   - The global token is never masked
6. **Transformer stack.** Pass through both encoder layers with
   `src_key_padding_mask`.
7. **Slice team outputs.** Take the first 6 + next 6 transformer outputs as
   `our_team_out` / `their_team_out`. Discard history and global outputs —
   their information has already flowed into team tokens via attention.
8. **CLS pooling.** Each side's learned CLS query cross-attends over its 6
   team output tokens (fainted slots key-masked). Post-LayerNorm gives one
   128-dim pooled vector per side.
9. **Extract our_active_refined** from the appropriate slot of
   `our_team_out`.
10. **Projection input.** Concatenate `our_pooled + their_pooled +
    our_active_refined + our_ctx_enc + opp_ctx_enc + non_matchup_rest`
    (active context still passes through the separate `active_ctx_encoder`
    for the projection path). Auto-discovered combined dim is 473; projects
    to `features_dim = 512`.

### `dropout=0.0` is intentional

`nn.TransformerEncoderLayer` defaults to `dropout=0.1`. With non-zero dropout,
`forward_internal` is non-deterministic outside `model.eval()` — which broke
the snapshot save/load round-trip test (which doesn't toggle eval mode). The
rest of the network is dropout-free, so setting `dropout=0` here keeps the
forward pass deterministic and matches the surrounding code style.

### Auto-derived dims

Three dims that depend on layout fields are derived in `__init__` rather than
hardcoded, so they update automatically when the obs layout changes:

| Name | How |
|------|-----|
| `_td_embed_dim` | `2 * move_emb + 2 * type_emb + (TURN_DELTA_DIM - 4)` = 99 |
| `_non_matchup_rest_dim` | `GLOBAL_ENV_DIM + reactive_layout['our_matchups']['offset']` = 25 |
| `projection_input_dim` | Dummy forward pass through `forward_internal` |

### Slicing helpers stored at init

Token-group slices are precomputed at init for clarity in `forward_internal`:

```python
self._our_token_slice    = slice(0, TEAM_SIZE)
self._their_token_slice  = slice(TEAM_SIZE, 2 * TEAM_SIZE)
self._history_token_slice = slice(2 * TEAM_SIZE, 2 * TEAM_SIZE + N_HISTORY_TURNS)
self._global_token_index = 2 * TEAM_SIZE + N_HISTORY_TURNS
self._total_tokens       = 2 * TEAM_SIZE + N_HISTORY_TURNS + 1  # = 23
```

### Architecture version (`model_version.py`)

```python
ARCH_SIGNATURE = "gen3_unified_v1"   # was "gen3_hp_v1"
```

`MODEL_CONFIG_VERSION` is unchanged at `2`. Every weight-shape-relevant field
of `ModelVersion` is already covered by `_WEIGHT_FIELDS` in
`check_compatible()`, and the structural break is signalled cleanly by the
signature bump — no migration is needed.

---

## Edge Cases

### Empty history (early-game turns)

On turn 1 of an episode, the env writes 10 all-zero TurnDelta vectors into
the obs tail. After embedding through `_embed_delta_slot`, the move/type
embedding for ID 0 is non-zero, so the embedded slot is not zero — but the
RAW input is. `empty_history` is computed from the raw block before
projection, so the mask correctly marks all 10 slots as padding on turn 1.
Attention scores over real turns are never diluted by padding.

### All-zero observation in dummy forward

`__init__` runs a dummy forward over zeros to discover `projection_input_dim`.
Every history slot is empty (masked); every opponent slot has HP=0 and is
masked except for the unmasked-active-fallback. The transformer always has at
least one live key per side (the active slot), so softmax never sees a
fully-masked row and no NaN propagates.

### CLS pooling with mostly-fainted team

In a near-loss state, only one team slot may have HP > 0. The CLS query
attends over the 6 team tokens with the live slot as the only unmasked key —
attention reduces to "use the active token's value." This is the desired
behaviour; the CLS output is well-defined as long as one key is live, which
is guaranteed by the active-slot-unmask invariant.

### Token-type embedding broadcasting

`token_type_emb(torch.full((1,), TOKEN_TYPE_X))` returns a `[1, D_MODEL]`
tensor. Adding to `[B, n_tokens, D_MODEL]` broadcasts as `[1, 1, D_MODEL] →
[B, n_tokens, D_MODEL]` — the same group embedding is added to every token in
that group across the batch.

### Dropout, eval mode, and the snapshot round-trip

The snapshot round-trip test does not call `.eval()` before comparing
features before vs. after save/load. v3 worked because no submodule had
dropout. v4's `TransformerEncoderLayer` defaults to `dropout=0.1`, which
would randomise the forward pass each call. `dropout=0.0` keeps determinism
without requiring the test to toggle modes.

---

## Test Suite

`features_extractor_test.py` was rewritten. Old tests for `active_ctx_to_role`,
`td_conditioner`, and the standalone `turn_history_attn` are gone — those
modules no longer exist. New tests cover the unified-transformer path.

### Module-presence and shape checks

| Test | What it validates |
|------|-------------------|
| `test_d_model_matches_role_token_size` | `D_MODEL == ROLE_TOKEN_SIZE` (team tokens unprojected) |
| `test_td_strategic_constants_consistent` | `TD_STRATEGIC_DIM`/`OFFSET` still align with TurnDelta layout |
| `test_unified_transformer_modules_exist` | All 11 new components are present |
| `test_removed_modules_absent` | All 13 v3 modules (5 attention paths + 4 pool/inject + status_embedding + history attn block) are gone |
| `test_token_type_embedding_shape` | `(4, 128)` |
| `test_turn_history_pos_emb_shape` | `(10, 128)` |
| `test_history_proj_shape` | `99 → 128` |
| `test_global_proj_shape` | `71 → 128` (dynamically computed from layout) |
| `test_transformer_stack_shape` | 2 layers, 4 heads each, FFN out = 256 |
| `test_cls_parameters_shape` | `(1, 1, 128)` for both CLS queries |
| `test_token_count_matches_design` | `_total_tokens == 23` |

### Functional / wiring tests

| Test | What it validates |
|------|-------------------|
| `test_forward_output_shape` | Forward returns `(B, features_dim)` |
| `test_active_context_changes_output` | Non-zero our active context propagates through the global token |
| `test_opp_active_context_changes_output` | Same for opp side |
| `test_history_most_recent_slot_changes_output` | Last history slot affects output |
| `test_history_oldest_slot_changes_output` | Oldest history slot also affects output via attention |
| `test_history_two_distinct_slots_produce_distinct_outputs` | Position matters (proves positional encoding contributes) |
| `test_history_pos_embedding_wired_in` | Zeroing `turn_history_pos_emb` changes output |
| `test_history_proj_wired_in` | Zeroing `history_proj` changes output |
| `test_history_empty_slot_masking` | Filling an all-zero slot with non-zero scalars changes output (the empty-mask is dynamic, not hardcoded) |
| `test_transformer_layer_wired_in` | Zeroing the first layer's params changes output |
| `test_cls_pooling_wired_in` | Zeroing both CLS modules + their query parameters changes output |

### Dimension-only updates elsewhere

| File | Change |
|------|--------|
| `state_encoder_test.py` | `EXPECTED_OBS_DIM` 1729 → 1924 |
| `snapshot_test.py` | No code change; passes because of `dropout=0.0` (forward determinism preserved) |

---

## Verification

### Architecture summary (from dummy forward)

```
obs dimension : 1924
  base_dim    : 1523
  history_dim : 390 (= 10 × 39)

Unified transformer:
  d_model     : 128
  layers      : 2
  heads       : 4
  ffn dim     : 256
  history len : 10

feature extractor params: 819_186
projection input dim    : 473
features_dim (out)      : 512
total tokens            : 23
```

### Smoke test

`[ModelVersion] Round-trip smoke test PASSED (output shape: torch.Size([1, 512]))`
fires at startup, confirming `save_model_snapshot` + `load_model_snapshot` +
forward produce identical outputs.

### Unit-test snapshot

All 701 unit tests pass (`pytest -m "not integration and not e2e"`). The
launcher UI tests that fail on this branch also fail on a clean main checkout
and are unrelated.

---

## Files Changed

| File | Change |
|------|--------|
| `src/agents/model/features_extractor.py` | Unified L=2 transformer over 23 tokens replaces 5 attention paths + 2 injection MLPs + standalone history attention + slot-status embedding + pool queries; new constants `D_MODEL`, `TRANSFORMER_*`, `TOKEN_TYPE_*`, `NUM_TOKEN_TYPES`; `N_HISTORY_TURNS` 5 → 10; `history_proj`, `global_proj`, `token_type_emb`, CLS queries + cross-attn; `dropout=0.0`; auto-derived `_td_embed_dim`, `_non_matchup_rest_dim`, projection_input_dim |
| `src/agents/model/model_version.py` | `ARCH_SIGNATURE` `"gen3_hp_v1"` → `"gen3_unified_v1"` |
| `src/agents/model/features_extractor_test.py` | Rewritten — removed obsolete `active_ctx_to_role` / `td_conditioner` / `turn_history_attn` tests; added 22 tests covering unified-transformer module presence, shape sanity, wiring, history masking, positional encoding, CLS pooling, removal of v3 modules |
| `src/agents/observation/state_encoder_test.py` | `EXPECTED_OBS_DIM` 1729 → 1924 |
