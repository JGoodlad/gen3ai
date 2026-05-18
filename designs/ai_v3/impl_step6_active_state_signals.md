# Implementation: Step 6 — Active State Signals

This step closed all known gaps in how the active Pokémon's state (boosts, volatile
effects, status duration) flows through the full pipeline — from observation encoding
to the attention heads.

Primary themes: fixing a silent bug that zeroed all volatile-effect bits, surfacing
status duration signals (sleep turns, toxic counter), consolidating Gen 3 mechanics,
and injecting active context into role tokens so every attention path can see boost and
volatile state before reasoning about team dynamics.

---

## Motivation

After Step 5's hyperparameter changes, profiling revealed three quality gaps:

1. **Volatile effects were silently zero every battle.** `active_context.py` read from
   `mon.volatiles`, which doesn't exist on poke-env's `Pokemon` class. The correct
   attribute is `mon.effects`. All 9 bits (Taunt, Confusion, Substitute, Encore, etc.)
   had been zero for the entire project lifetime.

2. **Status presence ≠ status duration.** The per-Pokémon condition one-hot told the
   network *"this mon is asleep"* but not *"this mon has slept 3 turns and will almost
   certainly wake next turn"*. The same gap applied to toxic: turn 1 vs. turn 7 of badly
   poisoned is completely different strategically.

3. **Mechanics were scattered.** Type effectiveness, boost helpers, notable-effect lists,
   and immunity logic lived in `type_utils.py`, inline in reward_manager, and duplicated
   across reactive/opponents. This made correctness hard to verify.

---

## What Was Changed

### 1. Gen 3 Mechanics Module (`src/agents/gen3_mechanics.py`)

New centralised module consolidating all Gen 3 mechanics helpers. Replaces the
now-deleted `src/agents/type_utils.py` and deduplicated inline logic across
`reward_manager.py`, `opponents.py`, and `reactive.py`.

Key exports:

| Symbol | Purpose |
|---|---|
| `effective_multiplier(move_type, target_types)` | Type matchup multiplier considering Gen 3 chart |
| `is_status_move_immune(move_id, mon)` | True if mon's types or existing status block the move |
| `boosts_array(mon) → np.ndarray (7,)` | Stat stages in `[atk, def, spa, spd, spe, acc, eva]` order |
| `boosts_str(mon) → str \| None` | Human-readable non-zero boosts e.g. `"atk:+2 spa:+1"` |
| `mon_status_str(mon) → str \| None` | Compact status + notable volatiles e.g. `"PAR, taunt"` |
| `NOTABLE_EFFECTS` | Tuple of `Effect` members to surface in logs |
| `STATUS_MOVE_IMMUNITY` | Dict mapping status move IDs to type-immunity frozensets |
| `PHAZING_MOVES`, `STATUS_MOVES`, `RECOVERY_MOVES`, `SETUP_MOVES` | Move classification sets |

`gen3_mechanics_test.py` covers all exports (280 lines, 293 tests pass).

### 2. Boost Tracking in `BattleContext` / `TurnDelta`

**`src/agents/training/battle_context.py`**

Two new fields added to `BattleContext`:

```python
our_boosts: np.ndarray   # shape (7,) int8 — stat stages at decision time
opp_boosts: np.ndarray   # shape (7,) int8 — opponent stat stages at decision time
```

Two new fields added to `TurnDelta`:

```python
our_boost_delta: np.ndarray   # shape (7,) int8 — stage change this turn
opp_boost_delta: np.ndarray   # shape (7,) int8 — opponent stage change this turn
```

`TurnDelta.build()` computes deltas as `curr.our_boosts - prev.our_boosts`. Boosts reset
on switch (poke-env handles this via Showdown messages), so the delta correctly captures
Intimidate drops, Calm Mind gains, and Roar-induced resets.

These fields are captured but not yet consumed by the observation encoder or reward
function — recorded in `todo.md` as a future item.

### 3. Silent Bug: Volatile Effects Always Zero

**`src/agents/observation/active_context.py`**

```python
# Before (wrong attribute — always returned {}):
volatiles = getattr(mon, "volatiles", {})

# After:
volatiles = getattr(mon, "effects", {})
```

poke-env's `Pokemon` class stores volatile effects under `mon.effects` (a
`Dict[Effect, int]`). The attribute `mon.volatiles` does not exist. All 9 volatile
bits (Confusion, Substitute, Taunt, Encore, Attract, Disable, Leech Seed, Focus Energy,
Destiny Bond) had been zero for the entire project. This is fixed.

`active_context_test.py` was updated: all `mon.volatiles` mocks → `mon.effects`.

### 4. Perish Song: Binary → Scalar in Active Context

**`src/agents/observation/active_context.py`**

```python
# Before: binary bit, dim unchanged
if any(e in volatiles for e in [Effect.PERISH0, Effect.PERISH1, Effect.PERISH2, Effect.PERISH3]):
    vec[cursor + 4] = 1.0

# After: scalar 0.0–1.0, same 1-dim slot
perish_turns = 0
for n, e in [(3, Effect.PERISH3), (2, Effect.PERISH2), (1, Effect.PERISH1)]:
    if e in volatiles:
        perish_turns = n
        break
vec[cursor + 4] = perish_turns / 3.0
```

`PERISH3=1.0`, `PERISH2=0.667`, `PERISH1=0.333`, absent or `PERISH0=0.0`.
`ACTIVE_CONTEXT_DIM` remains 23 — no architecture change.

Perish Song stays in active context only (not per-Pokémon) because in Gen 3, switching
out **clears** the Perish Song counter. It doesn't follow the mon to the bench.

### 5. Sleep and Toxic Counters in Per-Pokémon Vector

**`src/agents/observation/constants.py`**

```python
# Before
POKEMON_VECTOR_DIM = 58
POKEMON_FULL_DIM   = 59   # +1 active flag appended by state_encoder

# After
POKEMON_COUNTER_OFFSET = 58   # new: two counter scalars at the end
POKEMON_COUNTER_DIM    = 2
POKEMON_VECTOR_DIM     = 60
POKEMON_FULL_DIM       = 61
```

**`src/agents/observation/pokemon.py`** — two new dims appended to each Pokémon vector:

```python
ctr = getattr(mon, "status_counter", 0) or 0
vec[POKEMON_COUNTER_OFFSET]     = min(ctr, 4) / 4.0 if status == Status.SLP else 0.0
vec[POKEMON_COUNTER_OFFSET + 1] = min(ctr, 8) / 8.0 if status == Status.TOX else 0.0
```

| Dim | Signal | Encoding | Gen 3 Notes |
|---|---|---|---|
| `COUNTER_OFFSET+0` | Sleep duration | `min(ctr, 4) / 4` | Sleep lasts 1–4 turns. Sleep Talk/Snore increments the counter but the increment is discarded on switch-out (engine oversight). |
| `COUNTER_OFFSET+1` | Toxic severity | `min(ctr, 8) / 8` | Counter resets to 1 on switch-in. Practical max ~8 turns before fainting (even with Leftovers). |

These counters belong in the **per-Pokémon vector** (not active context) because sleep
and toxic persist on benched mons and their counters keep advancing off-screen.

Both dims flow through the role encoder and into all 5 attention paths, giving the
Safety, Pressure, and Synergy heads duration-aware information about bench mons.

**Dimension cascade:**

The role encoder takes `hp_and_active = pokemon_part[:, :, hp_offset:]` (open-ended
slice), so the new counter dims are automatically included without any explicit slice
change. The `role_input_dim` constant in `features_extractor.py` was updated:

```python
# Before
role_input_dim = 260   # pokemon_enriched (242) + global context (15) + validity (3)

# After
role_input_dim = 262   # pokemon_enriched (244) + global context (15) + validity (3)
```

Total observation dimension: **1107 → 1131** (12 mons × 2 new dims = +24 per base;
+7 for prev_mask tail = net +24 to `EXPECTED_BASE_DIM`, same +24 to `EXPECTED_OBS_DIM`).

### 6. Rich Status Display in `summary.json`

**`src/agents/training/battle_recorder.py`**

Replaced the thin `_mon_status_str()` delegate with two new static methods:

**`_mon_display_status(mon) → str | None`** — rich display string:

| Condition | Example output |
|---|---|
| Sleeping, turn 3 | `"SLP(3)"` |
| Badly poisoned, turn 5 | `"TOX(5)"` |
| Burned | `"BRN"` |
| Paralysed + Taunted | `"PAR\|TAUNT"` |
| Perish Song 2 turns left + Confused | `"PERISH(2)\|CONF"` |
| No condition | `None` (field omitted from JSON) |

**`_status_key(status_str) → str | None`** — normalised key for change-detection:

```python
# "SLP(3)" → "SLP";  "PERISH(2)|CONF" → "CONF|PERISH"
"|".join(sorted(p.split("(")[0] for p in status_str.split("|")))
```

This prevents spurious status-change events when only the counter changes (e.g., SLP(1)
→ SLP(2) does not emit an event; SLP → BRN does).

`_append_status_events` now uses `_status_key` for comparison and `_mon_display_status`
for the emitted event string, so logs read `"opp:toxicroak:TOX(3)"` instead of `"opp:toxicroak:TOX"`.

Bench summaries updated to include status for non-fainted mons:
```
"bench": "blissey(88%,TOX(2)), skarmory(100%,TAUNT), swampert(faint)"
```

### 7. Active Context Injection into Role Tokens (Pre-Attention)

**`src/agents/model/features_extractor.py`**

The active context (boosts + volatile effects, 23 dims per side) was previously fed
only into the final projection input — **after** all 5 attention paths had already run.
This meant Safety, Threat, Pressure, Synergy, and Opp Synergy were completely blind
to boost and volatile state.

Concrete examples of what was missing:
- **Safety** (our bench queries their active): couldn't see the opponent's active has
  +2 Atk from Swords Dance when deciding which bench mon to switch in.
- **Threat** (their bench queries our active): couldn't see our active has a Substitute
  up, which halves the threat value of their status-move bench users.
- **Pressure** (our active queries their bench): couldn't see our active is Taunted or
  Confused, changing how urgently we need to switch.

**Fix:** a new shared 2-layer MLP (`active_ctx_to_role: 23→64→128`) projects each
side's active context into role token space and adds it to the active slot's role token
**before** the 5 attention paths run. A 2-layer MLP (rather than a bare linear) captures
non-linear interactions between boost dimensions ("+2 SpA AND +2 Spe" is qualitatively
different from either alone).

The active context now flows two ways:

```
active_context (23 dims)
       │
       ├── active_ctx_to_role (23→64→128, MLP) → + injected into active slot's role token
       │                                                   │
       │                                         [5 attention paths run — boost/volatile aware]
       │
       └── active_ctx_encoder (23→64→32, MLP) → appended to final projection input
```

The second path (direct to projection) is preserved: it gives the policy/value MLP an
immediate read on current state for action selection.

Bench slots are not injected — in Gen 3, switching out resets all stat boosts and clears
all volatile effects, so bench slots have no boost/volatile signal.

`ARCH_SIGNATURE` bumped from `"gen3_attn_v1"` to `"gen3_attn_v2"`. The projection
input dimension is auto-discovered via dummy forward in `__init__`, so no constants
need updating.

New test file `src/agents/model/features_extractor_test.py` verifies:
- `active_ctx_to_role` output shape is correct
- Non-zero our-side active context changes the forward output
- Non-zero opp-side active context changes the forward output
- Zeroing `active_ctx_to_role` changes the output, proving the layer is wired in

---

## Reward Signals (unchanged from Step 5)

No reward constants were modified in this step.

---

## Files Changed

| File | Change |
|---|---|
| `src/agents/gen3_mechanics.py` | New — centralised Gen 3 mechanics module |
| `src/agents/gen3_mechanics_test.py` | New — 280-line test coverage |
| `src/agents/type_utils.py` | Deleted — replaced by `gen3_mechanics.py` |
| `src/agents/training/battle_context.py` | `our_boosts`, `opp_boosts` fields; boost delta in `TurnDelta` |
| `src/agents/training/battle_context_test.py` | Boost field tests |
| `src/agents/training/reward_manager.py` | Use `is_status_move_immune()`; import from gen3_mechanics |
| `src/agents/training/reward_manager_test.py` | Add `our_boost_delta`/`opp_boost_delta` to test fixtures |
| `src/agents/training/battle_recorder.py` | `_mon_display_status()`, `_status_key()`, rich bench summaries |
| `src/agents/training/battle_recorder_test.py` | Volatile/status fields on `_FakeMon` |
| `src/agents/observation/active_context.py` | `mon.volatiles` → `mon.effects`; Perish Song binary → scalar |
| `src/agents/observation/active_context_test.py` | Mock fix + perish scalar test |
| `src/agents/observation/constants.py` | `POKEMON_COUNTER_OFFSET/DIM`; `POKEMON_VECTOR_DIM` 58→60 |
| `src/agents/observation/pokemon.py` | Sleep + toxic counter dims; `get_layout()` entry |
| `src/agents/observation/pokemon_test.py` | 5 new counter tests; updated dimension assertions |
| `src/agents/observation/state_encoder_test.py` | `EXPECTED_BASE_DIM` 1067→1091; `EXPECTED_OBS_DIM` 1107→1131 |
| `src/agents/observation/turn_delta_encoder_test.py` | Boost delta fields in test fixture |
| `src/agents/observation/reactive.py` | Import from gen3_mechanics |
| `src/agents/opponents.py` | Import from gen3_mechanics |
| `src/agents/model/features_extractor.py` | `role_input_dim` 260→262; `active_ctx_to_role` MLP; pre-attention injection |
| `src/agents/model/features_extractor_test.py` | New — 4 injection tests |
| `src/agents/model/model_version.py` | `ARCH_SIGNATURE` `"gen3_attn_v1"` → `"gen3_attn_v2"` |
| `designs/ai_v3/README.md` | Architecture digraph + tables updated for injection path |
| `CLAUDE.md` | Obs dim 1107→1131; per-Pokémon slot 59→61 dims; role encoder input updated |
| `designs/ai_v3/todo.md` | §6 marked partial done; §9 added (boosts/volatiles blind to attention) |

---

## What's Next

See `designs/ai_v3/todo.md`. Priority items after this step:

1. **LR annealing** (§1) — implement Wang schedule `ℓ(x) = peak / (8x+1)^1.5` via SB3's
   callable `learning_rate` interface.
2. **Boost delta in observation / reward** — `our_boost_delta` / `opp_boost_delta` are
   captured in `TurnDelta` but not yet consumed. Reward signal for setup moves (Calm Mind,
   Dragon Dance) and penalty for opponent unchecked boosts.
3. **Turn-history memory** (§7) — sliding window of K `TurnDelta` blocks, then GRU.
