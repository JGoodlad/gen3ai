# Implementation: Step 2 — Action Mask as Observation Feature

This step adds the previous turn's action mask to the observation vector and routes each mask
component to the network submodule where it is semantically meaningful.

## Motivation

MaskablePPO already uses the *current* turn's action mask to zero out illegal logits at
sampling time. That prevents the model from choosing illegal actions, but it provides no signal
during the forward pass itself — the policy and value networks still see a flat observation with
no information about what was available *last turn*.

Adding the previous turn's mask as an input feature gives the model:

- **Move availability history** — was this move disabled last turn (Choice lock, PP depletion)?
  That predicts whether it will still be available now.
- **Switch availability context** — which bench slots were healthy enough to switch to last turn?
  Useful for tracking passive damage accumulation.
- **Struggle context** — was the active mon in Struggle last turn? Strongly predicts it needs to
  come out.

The *previous* turn's mask (not the current one) is used because the current mask is already
handled by MaskablePPO's logit masking; adding it redundantly as an input would provide no new
gradient signal. The previous turn's mask is genuinely new information.

---

## Observation Vector Change

`Gen3ObservationEncoder` now distinguishes two dimensions:

| Property | Value | Meaning |
|---|---|---|
| `base_dimension` | 1053 | Raw encoder output from `encode()` (see Phase 3 for evolution) |
| `dimension` | 1064 | Full obs fed to the network: base + 11-dim prev_mask |

The 11-dim prev_mask layout mirrors the action space:

| Indices | Meaning |
|---|---|
| 0–5 | Switch validity (slots 0-5 — switch to bench mon i) |
| 6–9 | Move validity (move slots 0-3) |
| 10 | Struggle validity |

### Where prev_mask comes from

**Training** (`Gen3Env.embed_battle()`):
```python
def embed_battle(self, battle):
    obs = self.observation_encoder.encode(battle)
    if battle is self.battle1 and not battle.finished:
        mask = Gen3ActionMasker.get_mask(battle).astype(np.int8)
        if mask.sum() > 0:
            self._tracker.record(battle, mask, obs)
    prev_mask = self._tracker.prev_mask if battle is self.battle1 else np.ones(11, dtype=np.float32)
    return np.concatenate([obs, prev_mask])
```

`EpisodeTracker.prev_mask` returns `history[-2].mask` if at least two turns have been recorded,
otherwise all-ones (turn 1 has no prior turn; all-ones is a neutral prior).

**Inference** (`Gen3Player.embed_battle()`):
```python
result = self.observation_encoder.get_observation(battle)
result["observation"] = np.concatenate([result["observation"], np.ones(11, dtype=np.float32)])
return result
```

Inference players have no episode history, so all-ones is the correct fallback.

---

## Network Changes

### Routing Philosophy

Each mask bit routes to the network component that most directly uses that information:

| Mask component | Routed to | Reason |
|---|---|---|
| `move_mask[0:4]` | Move processor input | Validity modulates how useful that move's features are |
| `switch_mask[0:6]` | Role encoder input (per-slot) | Validity modulates whether that mon is a live switch option |
| `struggle_mask` | Role encoder input (all slots) | "Struggle last turn" is a global context that affects switch decisions |

### Move Processor (`move_input_dim`: 55 → 56)

Each of the 4 move slots gets 1 validity bit from `prev_mask[6:10]`.

For our team (6 Pokémon): the move validity comes from the previous turn's mask.  
For the opponent's team (6 Pokémon): all-ones (we have no knowledge of their PP state).

```python
move_validity_ours = move_mask.unsqueeze(1).unsqueeze(3).expand(-1, TEAM_SIZE, -1, -1)  # [B, 6, 4, 1]
move_validity_opp  = torch.ones(batch_size, TEAM_SIZE, 4, 1, device=x.device)           # [B, 6, 4, 1]
move_validity      = torch.cat([move_validity_ours, move_validity_opp], dim=1)          # [B, 12, 4, 1]
```

### Role Encoder (`role_input_dim`: 238 → 240)

Two additional dims appended per Pokémon token before the role encoder:

1. **`switch_valid` (1 dim)** — was this specific slot a valid switch target last turn?  
   Our team: from `prev_mask[0:6]`. Opponent team: all-ones.

2. **`struggle_from_prev` (1 dim)** — was struggle the only option last turn?  
   Broadcast identically to all 12 tokens (it's a global signal about the active mon).

```python
switch_validity_ours = switch_mask.unsqueeze(2)                                        # [B, 6, 1]
switch_validity_opp  = torch.ones(batch_size, TEAM_SIZE, 1, device=x.device)          # [B, 6, 1]
switch_validity      = torch.cat([switch_validity_ours, switch_validity_opp], dim=1)  # [B, 12, 1]

struggle_from_prev   = struggle_mask.unsqueeze(1).expand(-1, 2 * TEAM_SIZE, -1)       # [B, 12, 1]

pokemon_enriched_with_context = torch.cat([
    pokemon_enriched, context_broadcasted, switch_validity, struggle_from_prev
], dim=2)  # [B, 12, 240]
```

Note: `struggle_from_prev` is distinct from the `is_forced_struggle` flag already in the
reactive observation block. The reactive flag reflects the *current* PP state; `struggle_from_prev`
reflects what the mask showed *last turn*. Both are useful and non-redundant.

### `remaining_part` slice fix

`forward_internal()` previously sliced `remaining_part = x[:, ctx['start']:]`, which
unintentionally included the 11-dim prev_mask tail when obs was 1032 dims. This would have caused
the final aggregation (`torch.cat([..., remaining_part], dim=1)`) to feed raw mask bits into the
projection layer without any routing logic.

Fixed to: `remaining_part = x[:, ctx['start']:base_dim]` — stops before the prev_mask tail.

### Trace logging fix

`forward()` strips the prev_mask tail before passing to `describe_vector()`, which expects the
base 1021-dim encoder output:

```python
x_base = x[:, :self.layout['base_dim']]
self._print_deep_trace(x_base, pokemon_part, species_ids)
```

---

## Files Changed (prev_mask feature)

| File | Change |
|---|---|
| `src/agents/observation/state_encoder.py` | Added `base_dimension` property (1021); `dimension` now 1032; `encode()` allocates only `base_dimension`; `get_layout()` exposes `base_dim`, `prev_mask_dim: 11`, `active_context_dim: 22`, `reactive_layout` |
| `src/agents/training/episode_tracker.py` | Added `prev_mask` property (returns `history[-2].mask` or all-ones) |
| `src/agents/training/gen3_env.py` | `embed_battle()` appends `_tracker.prev_mask` |
| `src/agents/inference/player.py` | `embed_battle()` appends all-ones prev_mask for inference players |
| `src/agents/model/features_extractor.py` | `move_input_dim` 55→56; `role_input_dim` 238→256; prev_mask slicing and routing in `forward_internal()`; `remaining_part` slice fixed; trace strips mask tail |
| `src/agents/observation/state_encoder_test.py` | `EXPECTED_OBS_DIM` 1021→1032; added `EXPECTED_BASE_DIM = 1021`; `encode()` tested against `base_dimension` |

---

## Phase 2: Network Architecture Fixes

Following a structured review of `features_extractor.py`, six architectural issues were fixed in
the same step. All changes are non-checkpoint-safe (rapid iteration phase — no trained weights to
preserve).

### P1 — Strip matchup matrix from projection input

The 288-dim matchup matrix (at `reactive_start + 16`) was being concatenated into the projection
input even though the move processor had already encoded it into role tokens. This added ~147K
projection parameters for zero structural gain.

**Fix:** Slice `remaining_part[:, 2*active_ctx_dim : reactive_start + matchup_offset]` for the
projection — global env scalars and reactive scalars only, no matchup block.

### P2 — Threat attention path (opponent introspection)

The three original attention paths (Pressure, Safety, Synergy) all update `our_team`. The
opponent bench tokens (`their_team`) received zero cross-team attention — they were raw role
encoder outputs concatenated directly into the projection. The value head had no structured way to
reason about which opponent bench mon is most dangerous given our current active.

**Fix:** Added a fourth path — **Threat**: `their_team` queries `our_active` (post-Pressure).
Adds `threat_attn` (MultiheadAttention, 4 heads) and `norm4` (LayerNorm).

```
Path 4: Threat — their_team ← our_active_post_pressure
```

### P3 — Learned encoder for active context (boosts and volatiles)

The 44 dims of `active_context` (22 each for our active and opponent active — boosts, stat stages,
volatiles, confusion) were fed raw to the projection MLP. No learned transformation meant the
network had no structural bias to learn non-linear boost interactions (e.g. +2 Atk is not "twice
as good" — it changes the entire decision surface).

**Fix:** Added `active_ctx_encoder`: `Linear(22→64)→ReLU→Linear(64→32)`, shared weights for both
sides. The raw 44-dim block is replaced in the projection by two 32-dim encoded tokens (64 dims
total).

### P4 — Fix `our_active_refined` to post-all-paths state

The original code extracted `our_active_refined` after the Pressure path only, before Safety and
Synergy updated `our_team`. The projection saw the active mon twice in contradictory
post-attention states.

**Fix:** Write the Pressure-refined active slot back into `our_team` before Safety/Synergy run,
so all three paths compose correctly:

```python
our_team[torch.arange(batch_size), our_active_idx] = our_active_post_pressure.squeeze(1)
# Safety and Synergy now see the Pressure-updated active slot
our_active_refined = our_team[torch.arange(batch_size), our_active_idx]  # after all paths
```

### P5 — Layout-driven reactive offsets

Hardcoded arithmetic (`reactive_start + 8`, `+15`, `+16`) in `forward_internal` was replaced with
lookups from `self.layout['reactive_layout']`, which is now populated from
`ReactiveEncoder.get_layout()` via `get_layout()` in `state_encoder.py`.

### P6 — Type embedding concatenation instead of summation

Pokémon dual-type was encoded as `E_type1 + E_type2` (16 dims). This is destructive: the
single-type case forces `E_none` to act as an additive identity, wasting embedding capacity.

**Fix:** `torch.cat([embedded_t1, embedded_t2], dim=-1)` — 32 dims, preserving the full type-pair
signal. This cascades: `pokemon_enriched` 226→242, `role_input_dim` 240→256.

### QW5 — `.float()` on move validity tensor

Added `.float()` to `move_validity_ours` after `.expand()` to be explicit about dtype regardless
of how `prev_mask` arrives.

---

## Phase 3: Hardening Pass

A third batch of fixes was applied after Phase 2. All are non-checkpoint-safe (rapid iteration
phase). Combined with an independent remote change (PP tracking) that was merged in.

### PP tracking in move slots (remote)

`MOVE_SLOT_DIM` 8→9: added `current_pp` (normalised by `MAX_PP=64`) and `max_pp` to each move
slot. These flow to the move processor as remnants — `move_remnant_dim` 4→6. Enables the network
to detect moves about to run out of PP.

`POKEMON_FULL_DIM` grows from 55→58 (condition 7 + moves 4×9=36 + HP + active).

### 1.5 — Dedup HP and spikes from reactive block

The reactive block previously encoded `our_hp` (offset 10), `opp_hp` (11), `our_spikes` (12),
`opp_spikes` (13) — all already present in the per-Pokémon vectors and global env block. Removed.

`REACTIVE_DIM` 304→300; matchup cursor shifts 16→12. `get_layout()` and all hardcoded offset
references updated.

### QW2 — base_power normalisation

`move.base_power / 100.0` → `/ 200.0`. Max base power in Gen 3 is ~150 (Eruption/Water Spout at
full HP); `/ 100` was producing values > 1. Fixed to keep the reactive move-power feature in [0, 1].

### QW3 — argmax guard on active-flag lookup

`torch.argmax(flags, dim=1)` returns 0 silently when all flags are zero (opponent not revealed, or
dummy forward pass in `__init__`). Replaced with `torch.where(flags.any(...), argmax(...), zeros)`.
Prevents silent index corruption on sparse observations.

### QW1 — fainted-key padding mask on synergy attention

Fainted Pokémon (HP=0) attending as *keys* in the synergy path let dead slots pollute the key
space. Added `key_padding_mask` to `synergy_attn` — fainted slots are masked out.

**NaN guard:** always unmask the active slot regardless of HP. An episode start where all HP fields
are zero (dummy forward in `__init__`) would mask every key → softmax of all -inf → NaN logits.
Forcing the active slot open prevents this.

### S3 — Screens routed to global context

Reflect and Light Screen (both sides, 4 dims total) were already encoded in the global env block
but excluded from `global_context`. Critical for switch-safety decisions — switching into a screen
turn dramatically changes the calculus.

`global_context` 12→16 dims; `role_input_dim` 256→259. Screen offsets are extracted from
`global_start + 9 : global_start + 13` (fixed within the 13-dim global block).

### QW5 (condition slot) — Remove unused 8th condition dim

`CONDITION_DIM` 8→7. The 8th slot (index 7) was reserved as a placeholder and never written.
Removing it drops one dim from each of the 12 Pokémon vectors → `pokemon_enriched` 242→241.

---

## Final Dimensions After All Fixes

| Component | Before step 2 | After phase 2 | After phase 3 |
|---|---|---|---|
| Per-Pokémon dims | — | 55 | 58 (condition 7, moves 4×9) |
| Observation (base) | 1021 | 1021 | 1053 |
| Observation (full) | 1021 | 1032 (+ 11 prev_mask) | 1064 |
| Move remnant dim | — | 4 | 6 (+ cur_pp, max_pp) |
| Move processor input | 55 | 56 | 58 |
| Pokémon type embedding | 16 (sum) | 32 (concat) | 32 |
| `pokemon_enriched` | 226 | 242 | 241 (condition −1) |
| `global_context` | — | 12 | 16 (+ screens) |
| Role encoder input | 238 | 256 | 259 |
| Attention paths | 3 | 4 (+ Threat) | 4 |
| Projection input | ~1961 (with matchup raw) | ~989 | N (auto-discovered) |
| Projection output | 512 | 512 | 512 |

---

## Tests

129 unit tests pass. The `test_encoder_and_features_extractor_are_compatible` integration test
exercises the full encoder → features_extractor forward pass with the 1064-dim observation.
`model_embedding_test.py` verifies forensic offsets for move known flag, matchup routing, PP
values, item known flag, and ability known flag.

Smoke test (`--debug --steps 10000`) passes end-to-end with no NaN logits.

---

## Final State

The observation carries 11 dims of prev_mask alongside the 1053 base features. Each mask
component flows to the most semantically relevant network component. The network architecture
is structurally sound: four attention paths, encoded boosts, no raw matchup data in the
projection, coherent active-slot composition across attention paths, and no duplicate features
in the reactive block.

**Ready for Step 3: Turn History**

- `EpisodeTracker._history` is a plain list — cap to `deque(maxlen=N)` for N-frame history
- Each `BattleContext` carries `obs` and `mask` — no new fields needed
- The 11-dim prev_mask slot could expand to N × 11 for concatenated history, or the
  `active_ctx_encoder` pattern could extend to encode per-turn snapshots via an LSTM/GRU
