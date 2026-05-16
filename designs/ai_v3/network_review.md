# Gen3FeaturesExtractor — Architectural Review

## 1. Strengths

**Dynamic layout with no magic constants.** The `get_layout()` / dummy-forward pattern means adding observation dimensions never silently breaks the projection layer. This is rare in RL codebases and already prevented at least one class of bug (the `remaining_part` slice issue documented in the step 2 design note).

**Semantically-routed prev_mask.** Routing `move_mask` bits to the move processor and `switch_mask` bits to the role encoder — rather than dumping all 11 bits at the projection layer — is the correct design. The signal is available where the gradient is most useful.

**Shared move processor with per-slot context.** Processing all 48 move slots (12 Pokémon × 4 moves) through a single MLP avoids learning 48 independent weight sets. Broadcasting `hp`, `weather`, `turn`, `fainted`, and `spikes` into every move slot gives the move processor the minimal context it needs to re-weight a move's value (e.g., Fire-type moves under sun).

**Matchup matrix piped directly into the move processor.** Each move slot receives 6 type-effectiveness scalars against all 6 opponents at [B, 12, 4, 6]. This lets the move network directly learn "don't pick this move against this team composition" without relying on attention to propagate it.

**Three semantically distinct attention paths.** Pressure (our active vs their team), Safety (our bench vs their active), and Synergy (our team self-attention) decompose the decision space cleanly. Each path has a LayerNorm residual, which is the correct post-attention formulation for stability.

**Status embedding as a positional bias.** Using a 4-class learned embedding (our active / our bench / their active / their bench) to bias role tokens before attention is elegant — it injects role information without polluting the token content, and is numerically stable because it adds to the token rather than concatenating.

**Integrity check with a raise on critical failure.** `integrity_check()` raising `ValueError` on a critical mismatch (multiple actives, fainted count disagreement) makes encoding bugs loud during debug/trace runs rather than silently corrupting training.

---

## 2. Issues / Weaknesses

### 2.1 Their team gets zero attention refinement

`their_team` tokens are produced by `role_encoder` and `status_embedding`, then concatenated into `their_team_flat` immediately — they never appear as a query in any attention path. Only our team queries theirs (Pressure/Safety), and our team queries itself (Synergy). There is no "opponent introspection" path where their bench considers our active or our team.

Concretely: the value head's estimate of "how threatening is their bench?" uses raw role tokens with no cross-team signal. A Pokémon that hard-counters our active is encoded identically whether it's on the bench or not from the value head's perspective.

### 2.2 The projection aggregation includes the raw matchup matrix

`remaining_part` (dims `ctx['start']` to `base_dim`, currently 361 dims) is concatenated directly into the projection input. This includes the full matchup matrix (288 dims at `reactive_start + 16` onward), which is **already processed** by the move network. Feeding it again raw to the projection means:

1. The projection layer must learn to ignore or re-weight 288 already-processed dims.
2. The projection input inflates unnecessarily, adding ~147K parameters to just the projection weight matrix with no structural gain.

The `active_context` (44 dims: boosts, volatiles) is a separate concern: it goes in raw and is never touched by any learned submodule. Boosts are critical for decision-making (a +2 Swords Dance Salamence is a completely different threat), but the model can only access them through the projection MLP with no structural bias for their non-linear effects.

### 2.3 `our_active_refined` is the pre-Safety/Synergy state

After the Pressure attention path, `our_active_refined` is appended as a separate 128-dim token. But `our_team` already contains the active slot — and after Safety and Synergy update `our_team`, the active slot in `our_team_flat` is already updated. `our_active_refined` is the post-Pressure, pre-Safety/Synergy state. The projection therefore sees the active mon in two contradictory states: once updated by all three paths (within `our_team_flat`), and once updated only by Pressure (as `our_active_refined`).

### 2.4 The move processor loses per-slot identity

After `move_network()`, the output is reshaped to `[B, 12, 4 * 32 = 128]` — the 4 move slots are concatenated in fixed order into a single 128-dim block. The role encoder only sees "a 128-dim block of move features" with no per-slot addressability. This will become a bottleneck if turn-history features are added (Step 3), where the model needs to reason about "did the move in slot 2 become unavailable between turns?"

### 2.5 Hardcoded magic offsets in `forward_internal`

Despite the dynamic layout philosophy, the reactive block is accessed by hardcoded arithmetic offsets:
- `reactive_start + 8` for fainted
- `reactive_start + 15` for forced_struggle
- `reactive_start + 16` for matchup start

These offsets will silently break if `ReactiveEncoder` changes its layout. `ReactiveEncoder.get_layout()` already returns a dictionary with these offsets; `forward_internal` should read from it.

### 2.6 No Dropout anywhere in the network

The model has ~2-3M parameters with no regularization. The move network and role encoder are bottleneck layers that could benefit from light Dropout (0.1) during training to prevent co-adaptation between attention keys/queries.

### 2.7 Type embedding uses summation rather than concatenation

`embedded_pk_types = embedded_t1 + embedded_t2` collapses a dual-type into 16 dims. The single-type case is represented as `E_type + E_none`, which forces `E_none` to be a learned additive identity — wasting embedding capacity and creating a semantic collision if type 0 is a real type placeholder. Concatenating E1 and E2 (32 dims instead of 16) preserves the full signal and lets the role encoder learn type-pair interactions.

---

## 3. Prioritised Suggestions

### P1 — Strip matchup matrix from projection input (HIGH IMPACT, ~3 lines)

The matchup matrix starting at `reactive_start + 16` (288 dims) is the single largest waste in the projection input. The move processor already encodes it into the 128-dim role tokens.

```python
# In forward_internal, replace:
combined = torch.cat([our_team_flat, their_team_flat, our_active_refined, remaining_part], dim=1)

# With:
non_matchup_context = remaining_part[:, : reactive_start + 16]
combined = torch.cat([our_team_flat, their_team_flat, our_active_refined, non_matchup_context], dim=1)
```

Removes ~147K projection params, loses no signal. **Does not break checkpoints.**

### P2 — Add opponent introspection attention path (HIGH IMPACT, ~7 lines)

Add a fourth path: `their_team` queries `our_active` — "given what we have out, which of their bench mons is most dangerous?" This is the highest-leverage missing structural element.

```python
# __init__:
self.threat_attn = torch.nn.MultiheadAttention(embed_dim=self.role_token_size, num_heads=4, batch_first=True)
self.norm4 = torch.nn.LayerNorm(self.role_token_size)

# forward_internal, after existing paths:
threat_delta, _ = self.threat_attn(their_team, our_active, our_active)
their_team = self.norm4(their_team + threat_delta)
```

Adds ~262K parameters (same as one existing attention head). **Does not break checkpoints.**

### P3 — Learned encoder for active_context boosts (MEDIUM IMPACT, MEDIUM EFFORT)

Boosts and volatiles (44 dims of `active_context`) are fed raw to the projection. A small MLP (`Linear 22→64→32` per side) would give the network a place to learn non-linear boost interactions (e.g., +2 attack is not "twice as good" — it changes the entire decision surface). Build this before Step 3 (turn history), which will likely attach historical boost deltas.

### P4 — Fix `our_active_refined` to post-all-paths state (MEDIUM IMPACT, ~3 lines)

Extract the active slot from `our_team` **after** Safety and Synergy have updated it, rather than using the Pressure-only refined token. This eliminates the contradictory dual-representation of the active mon at the projection layer.

### P5 — Replace hardcoded reactive offsets with layout lookups (LOW IMPACT, ~5 lines)

Add `reactive_layout` to `get_layout()` from `ReactiveEncoder.get_layout()`, and replace `reactive_start + 8`, `reactive_start + 15`, `reactive_start + 16` with named reads. Prevents silent breakage on any future reactive block change.

### P6 — Replace type summation with concatenation (LOW-MEDIUM IMPACT, ~2 lines, BREAKS CHECKPOINTS)

Change `embedded_pk_types = embedded_t1 + embedded_t2` to `torch.cat([embedded_t1, embedded_t2], dim=-1)`. Increases `role_input_dim` from 240 to 256. Best done before the next long training run.

---

## 4. Quick Wins

| # | Change | Lines | Impact | Checkpoint safe? |
|---|---|---|---|---|
| QW1 | Strip matchup dims from projection concat | 3 | High | Yes |
| QW2 | Add `Dropout(0.1)` in `move_network` and `role_encoder` | 2 | Low-Med | Yes |
| QW3 | Add opponent introspection attention path | 7 | High | Yes |
| QW4 | Named global offsets instead of arithmetic | 5 | Low | Yes |
| QW5 | `.float()` on `move_validity` after expand | 1 | Low | Yes |

**Best effort-to-impact ratio: QW1/P1 and QW3/P2.** Both are non-breaking, address concrete structural gaps, and can be applied independently before or after a training run.
