# Gen3FeaturesExtractor — Second-Pass Architectural Review

All previously-suggested items (P1–P6, QW1–QW5 from `network_review.md`) are confirmed implemented. The review below covers **only new issues** not present in the prior review.

> **Note (post-review pruning):** the original second-pass listed two items
> that turned out to be wrong or not meaningful — the "alphabetical type
> sort" fix and the `is_ours` move-processor flag — and both have been
> removed from this document.

---

## 1. New Issues / Weaknesses

### 1.1 No `species_known` flag — model cannot distinguish unrevealed opponent slots from empty ones

The per-Pokémon vector for an unrevealed opponent slot is all zeros. The model cannot distinguish "slot 0 is an unknown Pokémon" from "there is no Pokémon here." The species embedding at ID 0 must simultaneously mean both "no Pokémon" and "unknown Pokémon," which are very different things strategically.

**Fix:** Add a `species_known` flag (1-dim per slot) at the observation level: 1.0 if the slot is populated (known or unknown), 0.0 if the slot truly doesn't exist. Costs +12 dims total (1 per Pokémon). **Breaks checkpoints.**

---

### 1.2 Fainted Pokémon tokens participate as keys in Synergy and Safety attention

```python
synergy_delta, _ = self.synergy_attn(our_team, our_team, our_team)
```

`our_team` is `[B, 6, 128]` — all six slots, including fainted Pokémon. Fainted tokens will still participate as keys. The attention will eventually learn to down-weight them via the HP signal, but they add noise. Late-game (2–4 live Pokémon), most of the key space is fainted.

**Fix:** Compute `fainted_mask = (hp_and_active[:, 0:TEAM_SIZE, 0] == 0)` and pass as `key_padding_mask` to `synergy_attn` (and optionally `safety_attn` for our query tokens). ~5 lines. **Checkpoint-safe** (no weight changes).

---

### 1.3 Active context (boosts) uses split positive/negative encoding — 14 dims for 7 stats

The `active_context` block encodes 7 stat stages as 2 dims each (positive half, negative half), totalling 14 of the 22 dims. A single signed value per stat normalised by ±6 would be 7 dims, freeing 7 dims for additional volatile features (Encore count, Substitute HP fraction, etc.) without growing `ACTIVE_CONTEXT_DIM`.

**Fix:** Change boost encoding to 1 signed dim per stat (`stage / 6.0`), saving 7 dims. **Breaks checkpoints** and changes observation layout. Low priority unless you want to encode more volatile features.

---

### 1.4 Reactive block contains duplicate HP and spikes already present elsewhere

`reactive.py` encodes:
- `vec[10]`: active mon HP fraction — duplicate of `hp_and_active` in the per-Pokémon vector
- `vec[11]`: opponent active HP fraction — same
- `vec[12]`: our spikes — duplicate of `GlobalEnvEncoder` `vec[6]`
- `vec[13]`: opp spikes — duplicate of `GlobalEnvEncoder` `vec[7]`

These 4 dims go into `non_matchup_rest` at the projection layer as redundant floats.

**Fix:** Remove these 4 dims from the reactive block, shifting `status` and `forced_struggle` to offsets 8 and 9. Update `get_layout()` and all downstream offset reads. **Breaks checkpoints**, medium effort.

---

### 1.5 Screen conditions (Reflect/Light Screen) not visible to attention paths

The global env block encodes `our_reflect`, `our_light_screen`, `opp_reflect`, `opp_light_screen` (4 bits at offsets 9–12). These currently go into `non_matchup_rest` at the projection only — they are not part of `global_context` broadcast to the role encoder.

This means the Safety attention path evaluates switch safety without knowing whether Reflect halves incoming physical damage. In Gen 3, Reflect is one of the most switch-relevant conditions: it directly affects which physical attacker can safely come in.

**Fix:** Add the 4 screen dims to `global_context` (16 dims instead of 12), broadcast to all role tokens, update `role_input_dim` from 256 to 260. **Breaks checkpoints.**

---

### 1.6 `argmax` on active flags silently returns slot 0 when no Pokémon is active

```python
our_active_idx = torch.argmax(active_flags[:, 0:TEAM_SIZE], dim=1)
```

If `active_flags` is all zeros (opponent not yet revealed, or during the all-zero dummy forward pass), `argmax` returns 0 and silently treats slot 0 as active. The dummy forward is currently unaffected since projection dims don't depend on active_idx values, but it is a latent correctness bug.

**Fix:** Guard with `active_flags.any(dim=1)` and clamp or handle the no-active case. ~4 lines. **Checkpoint-safe.**

---

### 1.7 `CONDITION_DIM = 8` but only 7 status classes are used — slot 7 is permanently zero

7 statuses: None, BRN, PAR, SLP, FRZ, PSN, TOX. The 8th dim is always 0.0, wasting 256 params in the role encoder's first Linear layer.

**Fix:** Reduce `CONDITION_DIM` to 7. Update `POKEMON_HP_OFFSET`, `POKEMON_VECTOR_DIM`, `POKEMON_FULL_DIM`, and all derived offsets. **Breaks checkpoints.** Pair with other breaking changes.

---

### 1.8 Reactive base_power normalised by /100, per-Pokémon move power normalised by /200

- `reactive.py`: `move.base_power / 100.0` → can exceed 1.0 for 120+ BP moves
- `moves.py`: `power / 200.0` → capped at 1.0

The model sees two different power scales for the same concept at different layers.

**Fix:** Change `reactive.py` line to `/ 200.0`. 1 line. **Does not break checkpoints** (value range shifts, behavior changes slightly but parameters preserved).

---

## 2. Prioritised Suggestions

### S1 — Fainted-key mask for Synergy self-attention (HIGH IMPACT, ~5 lines, checkpoint-safe)

```python
fainted_mask_ours = (hp_and_active[:, 0:TEAM_SIZE, 0] == 0)  # [B, 6], True = fainted
synergy_delta, _ = self.synergy_attn(our_team, our_team, our_team,
                                      key_padding_mask=fainted_mask_ours)
```

Zero parameter cost. Fainted tokens stop polluting the key space in self-attention. Most impactful in late-game scenarios where 3–4 slots are fainted.

### S2 — Route screen conditions into role encoder global context (MEDIUM IMPACT, ~4 lines, breaks checkpoints)

Reflect/Light Screen are among the most switch-relevant global conditions in Gen 3. The Safety attention path should see them. `global_context` 12→16 dims, `role_input_dim` 256→260.

### S3 — Add `species_known` flag to per-Pokémon vector (MEDIUM IMPACT, ~10 lines, breaks checkpoints)

Lets the role encoder cleanly distinguish revealed/unrevealed/absent opponent slots. Currently the species embedding at ID 0 must encode two contradictory concepts.

### S4 — Normalize reactive base_power to /200 (LOW IMPACT, 1 line, checkpoint-safe)

Eliminates >1.0 power values and aligns reactive and per-Pokémon representations.

---

## 3. Quick Wins

| # | What | Lines | Impact | Checkpoint-safe? |
|---|---|---|---|---|
| QW1 | Fainted-key mask in `synergy_attn` | 4 | High | Yes |
| QW2 | Fix reactive base_power `/100` → `/200` | 1 | Low | Yes |
| QW3 | Guard `argmax` with all-zero active_flags fallback | 4 | Low (robustness) | Yes |
| QW4 | Remove unused `CONDITION_DIM` slot 7 | 5 | Low | No |

---

## Recommended Batching Strategy

**Immediate (checkpoint-safe):**
- QW1 — fainted-key mask
- QW2 — base power normalization
- QW3 — argmax guard

**Next breaking-change batch** (group to invalidate checkpoints only once):
- S2 — screens into global_context
- QW4 — remove unused condition slot

**Later / optional:**
- S3 — species_known flag (highest value for opponent modeling, most observation-level effort)
- 1.3 — signed boost encoding (only if you want to pack more volatile features into active_context)
- 1.4 — dedup HP/spikes from reactive (correctness, medium effort)
