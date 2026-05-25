# Design: Spread Inference — IV/EV/Nature/HP-Type Signals in the PPO Backbone

Make the PPO model's per-slot representations encode enough information about each
Pokémon's EV/IV spread that the team completion model (ai_v6 Steps 3–4) can predict
**EV tiers, nature, speed tier, and Hidden Power type** from frozen role tokens — not
just species/moves/items.

Three complementary additions work together: raw IV/EV values in the observation for
own-team slots (exact ground truth), accumulated battle statistics per slot (battle-derived
constraints), and auxiliary prediction losses during PPO training (explicit representational
shaping). All three are batched into the next training run.

---

## Why the Current Backbone Falls Short

The 128D role token per slot currently encodes a static snapshot: species, base stats,
known moves, known item, current HP fraction, status, boost stages. It has no access to:

- **Own team**: exact IV/EV/nature values known from teambuilder — but not yet in the obs
- **Opponents**: how much total damage a Pokémon took from physical vs. special moves,
  whether passive healing was observed (Leftovers = 1/16 max HP/turn), speed tier
  relative to our own Pokémon, what Hidden Power type was revealed

These signals are either free (own team) or accumulate over turns. None reach the role
encoder in the current architecture. The PPO reward signal never directly teaches the
network to represent them — so even if they were present in the inputs, the network
wouldn't bother encoding them into role tokens unless forced to.

---

## Signal 1 — Raw IV/EV Encoding for Own Team

### The change

Add 13 dims to every Pokémon slot in `PokemonEncoder.encode()`:

| Field | Dims | Normalization | Value for own team | Value for opponent |
|-------|------|---------------|--------------------|--------------------|
| IVs (HP, Atk, Def, SpA, SpD, Spe) | 6 | `/31 → [0,1]` | actual from teambuilder | `0.0` |
| EVs (HP, Atk, Def, SpA, SpD, Spe) | 6 | `/252 → [0,1]` | actual from teambuilder | `0.0` |
| `hidden_stats_known` | 1 | 1.0 / 0.0 | `1.0` | `0.0` |

Zero is a valid IV value, so the flag is required to avoid ambiguity. The encoder derives
it naturally: `mon.ivs is not None → 1.0`. No parameter threading — poke-env only
populates `ivs`/`evs` for our own team.

```python
# In PokemonEncoder.encode(), appended after existing 61 dims:
ivs = mon.ivs   # list[int] | None — order: HP, Atk, Def, SpA, SpD, Spe
evs = mon.evs   # list[int] | None
if ivs is not None:
    vec[61:67] = [v / 31.0  for v in ivs]
    vec[67:73] = [v / 252.0 for v in evs]
    vec[73]    = 1.0   # hidden_stats_known
# opponent slots: np.zeros init leaves dims 61-73 as 0.0
```

### What this enables

The model now has precise own-team stats rather than relying on species embeddings to
implicitly encode typical spreads. The role encoder learns to interpret "our Blissey has
252 SpD EVs" explicitly. This is also the **ground truth signal** for the auxiliary
prediction losses (Signal 3) — no side-buffer or label pipeline needed for our own team.

---

## Signal 2 — Per-Slot Accumulated Battle Statistics

### The gap

`TurnDelta.opp_hp_delta` is a per-slot array but the encoder collapses it to a single
scalar. There is no memory of "how much damage did the opponent's Salamence take from my
Suicune Ice Beam last time it was active." Bench Pokémon damage history is invisible.

### 9 new dims per slot, both sides

Add running accumulators to `BattleContext` that persist across switch-outs:

```python
# Arrays shaped (6,) — our side tracks damage taken from opponent; opp side tracks
# damage taken from us. Both sides are symmetric and equally useful to the model.
slot_cumulative_dmg_taken   # total fraction of max HP lost this battle
slot_n_physical_hits        # count of physical moves that connected against this slot
slot_n_special_hits         # count of special moves that connected against this slot
slot_min_hp_pct             # lowest HP% ever observed (sharpest defensive constraint)
slot_cumulative_healed      # total passive recovery observed (Leftovers, Ingrain, etc.)
slot_n_turns_active         # total turns active this battle
slot_speed_win_frac         # fraction of non-priority turns this slot moved first
slot_speed_loss_frac        # fraction of non-priority turns this slot moved second
slot_hp_type                # observed Hidden Power type index (0=unknown, 1-17=type)
```

`slot_hp_type` is inferred when the Pokémon uses Hidden Power:

```python
# In EpisodeTracker.update(), after each TurnDelta:
if opp_move == "hiddenpower" and our_effectiveness is not None:
    hp_type = infer_hp_type(our_active_types, our_effectiveness)
    if hp_type is not None:
        opp_slot_hp_type[opp_active_slot] = hp_type
```

For own team, `slot_hp_type` is set once at episode start from known IVs using the Gen 3
formula:
```
hp_type_index = floor(15 * (iv_hp%2 + 2*iv_atk%2 + 4*iv_def%2 +
                             8*iv_spe%2 + 16*iv_spa%2 + 32*iv_spd%2) / 63)
```

`slot_hp_type` is fed into the role encoder as a lookup into the shared `type_embedding`
table (16D, zero new parameters). The other 8 fields are plain scalars.

### Why these signals matter

| Accumulator | What it constrains |
|------------|-------------------|
| `slot_cumulative_dmg_taken` | Combined `1 / (def × max_HP)` — tightest single constraint |
| `slot_n_physical_hits` / `slot_n_special_hits` | Which defensive stat is constrained |
| `slot_min_hp_pct` | Tightest observed HP% — bounds max HP given known damage |
| `slot_cumulative_healed` | Leftovers recovery % = `1/16 max_HP` → pins max HP |
| `slot_speed_win_frac` / `slot_speed_loss_frac` | Speed tier relative to our Pokémon |
| `slot_hp_type` | Parity of all 6 IVs simultaneously — sharpest IV signal in Gen 3 |

---

## Signal 3 — Auxiliary Spread Prediction Losses

### Why the RL signal alone is insufficient

Even with Signals 1 and 2 in the inputs, PPO won't reliably learn to encode spread
structure in the 128D role tokens unless the win-rate signal happens to correlate with it.
Knowing whether Salamence has 252 vs 200 Spe EVs rarely changes the optimal action.
Auxiliary supervised losses force the backbone to represent this structure explicitly.

### SB3 compatibility: sequential second pass

Subclass `MaskablePPO` with a thin override — no copy-paste of `train()`:

```python
class Gen3MaskablePPO(MaskablePPO):
    def train(self) -> None:
        super().train()          # SB3's full PPO update — completely untouched
        if self._aux_heads and self.aux_coef > 0.0:
            self._run_aux_pass()

    def _run_aux_pass(self) -> None:
        # rollout_buffer still populated: SB3 resets it at the START of the next
        # collect_rollouts(), not at the end of train()
        for rollout_data in self.rollout_buffer.get(self.batch_size):
            role_tokens = self.policy.features_extractor.extract_role_tokens(
                rollout_data.observations   # shape: (B, 12, 128)
            )
            aux_loss = self.aux_coef * self._aux_heads.compute_loss(
                role_tokens, rollout_data.observations
            )
            self._aux_optimizer.zero_grad()
            aux_loss.backward()   # gradients flow through feature extractor
            clip_grad_norm_(...)
            self._aux_optimizer.step()
```

### What to predict and where the ground truth comes from

**Own team — EV tier (ground truth is already in the observation):**

Signal 1 puts raw IVs and EVs directly in `rollout_data.observations`. No side-buffer
or label pipeline needed — the aux head reads the IV/EV dims from the obs and uses them
as supervision targets for the role token:

```python
EV tier head:   Linear(128 → 3) per stat × 6 stats
                classes: [0 EVs | 1–124 EVs | 125–252 EVs]
                ground truth: discretise obs dims 61–72 in-place
```

**Own team — nature (requires side-buffer):**

Nature is not in the obs. Pass via `info["our_natures"]` from `Gen3Env.reset()` and
accumulate in a thin callback synced to `rollout_buffer.pos`.

```python
Nature head:    Linear(128 → 25)
```

**Opponent team — spread archetype (ground truth from `--team-log`):**

When self-play logging is active both teams' full spreads are known. Cluster team log
records by species into k ≤ 8 groups using k-means on the 6-stat vector from
`compute_raw_stats()`. Store as `data/pokemon/gen3_spread_archetypes.json`.

```python
Spread archetype head:  Linear(128 → max_archetypes_per_species)
```

**Loss weights:**

| Head | Weight | Gate |
|------|--------|------|
| EV tier (own team, 6 stats × 6 slots) | 0.05 | always on |
| Nature (own team) | 0.02 | always on |
| Spread archetype (opp) | 0.02 | only when `--team-log` active |

Start with `aux_coef=0.05` on own-team heads only. If win rate vs. random drops >2%
vs. baseline at 500K steps, halve. Enable opp archetype head after convergence.

### `extract_role_tokens()` refactor

```python
def extract_role_tokens(self, obs: dict[str, Tensor]) -> Tensor:
    """Returns (B, 12, 128) role tokens after all attention paths, before projection."""
```

Pull token computation into `_compute_role_tokens()`, called by both `forward()` and
`extract_role_tokens()`. Clean internal refactor, no change to `forward()` outputs.

---

## Dimension Impact

| Constant | Before | After | Delta |
|----------|--------|-------|-------|
| `POKEMON_VECTOR_DIM` | 61 | 83 | +22 per slot |
| `POKEMON_FULL_DIM` | 62 | 84 | +22 per slot |
| Team blocks (12 × FULL_DIM) | 744 | 1008 | +264 |
| `base_dimension` | 1103 | 1367 | +264 |
| `dimension` (full obs) | 1309 | 1573 | +264 |

The +22 per slot: 13 (IVs + EVs + known flag) + 8 (accumulator scalars) + 1 (hp_type
raw index). `role_input_dim` and projection input dim both auto-update via the existing
dummy forward pass in `__init__`. `model_version.py` catches the shape change via
`total_dim` — old checkpoints are correctly rejected.

---

## Relationship to Other Design Docs

| Doc | Relationship |
|-----|-------------|
| `designs/ai_v4/design_unified_transformer.md` | The unified transformer benefits from richer per-slot tokens. Spread signals feed into the unified attention over the full token sequence — history tokens can attend to "this Pokémon's item was consumed turn 4 AND it has 252 SpD EVs." |
| `designs/ai_v6/design_ppo_embedding_improvements.md` Option B | This doc is the concrete implementation of that sketch; replaces the callback pattern with the cleaner `Gen3MaskablePPO` subclass. |
| `designs/ai_v6/impl_step4_team_completion_enrichment.md` | ai_v6 Step 4 adds HP type prediction to the completion model head. This doc adds HP type **observation** to the backbone input — complementary. |
| `designs/ai_v6/design_team_completion_detail.md` Step 6 | The `BattleConstraintTracker` (damage-calc item inference) can share the same per-slot accumulator fields — no duplication. |

---

## Files to Create / Modify

| File | Change |
|------|--------|
| `src/agents/observation/constants.py` | `POKEMON_VECTOR_DIM`: 61 → 83; `POKEMON_FULL_DIM`: 62 → 84 |
| `src/agents/observation/pokemon.py` | Append IV/EV/flag dims + accumulator scalars + hp_type index to `encode()`; update `get_layout()` |
| `src/agents/training/battle_context.py` | Add 9 accumulator fields to `BattleContext`; update `EpisodeTracker` to maintain them across switch-outs |
| `src/agents/model/features_extractor.py` | Refactor into `_compute_role_tokens()`; expose `extract_role_tokens()`; add hp_type embedding lookup |
| `src/agents/training/gen3_mpp.py` | **New**: `Gen3MaskablePPO` with `_run_aux_pass()` and `attach_aux_heads()` |
| `src/agents/training/aux_spread_heads.py` | **New**: `SpreadAuxHeads` module (EV tier + nature + archetype heads + `compute_loss()`) |
| `src/main/train_rl_agent.py` | Swap `MaskablePPO` → `Gen3MaskablePPO`; add `--aux-coef` flag |
| `data/pokemon/gen3_spread_archetypes.json` | **New**: precomputed k-means archetype clusters (offline script) |
| `CLAUDE.md` | Update per-Pokémon slot dims table (61 → 83) and obs offset table |

---

## Verification

1. **Dim smoke test** — `train_rl_agent.py --debug --steps 10000`: confirm
   `[ModelVersion] Round-trip smoke test PASSED`. Own-team slot dims = 84;
   `hidden_stats_known = 1.0`; opponent slots have zeros for dims 61–73.

2. **IV/EV unit test** — own-team slot has correct normalized IV/EV values; opponent
   slot has all zeros and `hidden_stats_known = 0.0`.

3. **Accumulator sanity** — `--debug --steps 5000`; a Pokémon that took 3 special hits
   has `slot_n_special_hits = 3.0` and `slot_cumulative_dmg_taken` equal to the sum of
   observed HP% drops.

4. **HP type inference** — integration test with opponent Zapdos (HP Ice): after first
   HP use against a Grass-type active, `slot_hp_type` = ice index (11).

5. **Aux loss convergence** — `--aux-coef 0.05 --steps 200K`: `aux/ev_tier_loss` falls
   from ~1.1 toward ~0.3 within 50K steps (EVs are in the obs, so the head just learns
   to read them through the role token).

6. **Win-rate stability** — aux loss must not degrade win rate vs. random by >2% at
   500K steps. If it does, halve `aux_coef`.

7. **Linear probe** — freeze backbone; train logistic regression on role tokens to
   predict EV tier. AUC should be meaningfully higher than a no-Signal-3 baseline.
