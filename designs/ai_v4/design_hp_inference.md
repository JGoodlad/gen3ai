# ai_v4: Opponent Hidden Power Type Inference

> Prior art: `designs/ai_v3/todo_hidden_power_inference.md` proposed a raw-events block
> (K=3 observations appended to the obs vector). This design supersedes it with a
> candidate-mask approach that lives inside the per-mon slot and feeds directly into the
> role encoder.

---

## Problem

Gen 3 Showdown transmits Hidden Power as `|move|...|Hidden Power|...` with no type
suffix. This is a protocol limitation — the type is not sent regardless of which side is
playing. The current encoder writes `type_id=0` (unknown sentinel) for every opponent
HP use:

- The **matchup matrix** (`reactive.py`) computes HP's effectiveness against our team
  using the Normal-type dummy in `gen3_moves.json`. HP cannot be Normal in Gen 3, so
  every matchup cell involving opponent HP is wrong.
- The **move slot** in the per-mon encoding carries `type_id=0`, so the model cannot
  reason about coverage (Fire vs Steel, Ice vs Dragon, etc.).

The model already sees HP's effectiveness in the TurnDelta history (last N turns), but
that signal is ephemeral — it falls off the history window — and it requires the model
to implicitly learn the type chart from RL signal rather than receiving it as structured
input.

---

## What We Can Infer

Every time the opponent uses HP against one of our Pokémon, two facts are available:

1. **Our Pokémon's types and ability** — always known; we built the team.
2. **The effectiveness tier** — immune (0×), resisted (0.5×), neutral (1×), or
   super-effective (2×), sourced from `battle.opp_last_effectiveness`.

The type chart is fully deterministic. For each of the 16 possible HP types, we check
whether it produces the observed effectiveness against our mon's types and ability. Any
type that does not match is **eliminated** from the candidate set.

`effective_multiplier(move_type, mon)` in `gen3_mechanics.py` already handles
ability-based immunities (Volt Absorb, Water Absorb, Flash Fire, Levitate). An apparent
"immune" result vs a Lanturn (Volt Absorb) correctly infers HP is Electric rather than
treating Electric as eliminated.

### Convergence examples

| HP hits | Remaining candidates |
|---------|---------------------|
| Blissey (Normal) at 2× | Fighting only — **certain on turn 1** |
| Skarmory (Steel/Flying) at 2× | Fire, Electric — 2 candidates |
| Above, then Raichu (Electric) at 0.5× | Fire resisted by Electric; Electric is not → **Electric certain** |
| Snorlax (Normal) at 0.5× | Rock and Steel resist Normal → 2 candidates |

Hitting Normal-types or mons with few shared SE types converges in one turn. Multi-typed
mons with overlapping weaknesses may take 2–3 observations.

---

## Scope

Two independent things could be inferred about opponent HP:

1. **Type identity** — which of the 16 HP types is it? This is what we encode here.
2. **Physical vs Special category** — Gen 3's split is type-based: {Fire, Water, Grass,
   Electric, Ice, Psychic, Dragon, Dark} are Special; all others Physical. This affects
   which defensive stat is relevant and the `category` field in the move slot is currently
   wrong for HP.

Physical/Special is derivable from the candidate distribution once it narrows (a mask
containing only Special types tells the model to respect Sp.Def). Explicit encoding of
this split is future work.

---

## Encoding

### Format: 17-dim block per mon slot

```
[hp_revealed (1 dim)]  [hp_type_probs (16 dims)]
```

**`hp_revealed`** — 1.0 if this mon has used HP at any point this battle; 0.0 otherwise.
Lets the model distinguish "HP not yet seen" (block all-zero) from "HP seen but type
ambiguous."

**`hp_type_probs`** — one float per HP type in fixed order:

```
idx:  0     1     2       3         4         5     6       7
type: Bug   Dark  Dragon  Electric  Fighting  Fire  Flying  Ghost

idx:  8     9      10    11      12       13    14     15
type: Grass Ground Ice   Poison  Psychic  Rock  Steel  Water
```

Each entry is the **prior probability** of that type from competitive usage data.
Eliminated types drop to 0.0; survivors retain their original prior weight without
renormalization.

### States

| State | hp_revealed | type prob entries |
|-------|-------------|-------------------|
| HP not yet used | 0.0 | all 0.0 |
| HP used, no constraints yet | 1.0 | species prior (e.g. ICE: 0.70, GRASS: 0.25) |
| HP used, partially constrained | 1.0 | non-zero only for surviving types at prior weight |
| HP used, type confirmed (1 survivor) | 1.0 | single non-zero entry at its prior weight |

### Why prior probabilities instead of binary flags

Binary flags (1 = possible, 0 = eliminated) treat all surviving types as equally likely.
Prior probabilities give the model the meta distribution before any observations: Jolteon
showing `ICE: 0.70, GRASS: 0.25` immediately signals Ice is the heavy favourite. After
Ice is eliminated, `ICE: 0.0, GRASS: 0.25` remains — Grass is confirmed but at its
minority-variant weight. This preserves the signal that an unusual HP choice was made.

### Why no renormalization after elimination

Renormalizing after each elimination creates discontinuities (GRASS jumps 0.25 → 1.0 the
moment Ice is ruled out). Keeping absolute prior weights produces a stable encoding the
model can learn to interpret consistently. When exactly one non-zero entry remains, the
matchup matrix uses that type directly — the magnitude doesn't matter for that inference.

### Why both sides get the block

The role encoder uses shared weights across our team and the opponent's team. Adding 17
dims to opponent slots only would require separate role encoders. Instead, all 12 slots
receive the 17-dim block. For our own mons, the block is always zeros (our HP type is
known at build time and is a separate fix — see Future Work). The shared role encoder
learns: "trailing HP dims are zero for my own team; they carry type information for the
opponent."

Per-mon slot expands from **62 → 79 dims** (both sides). The role encoder input expands
from 263 → 280 — dynamically computed in `__init__` from layout fields, no hardcoded
constant to update.

**OBS_DIM:** team blocks grow by 12 × 17 = 204 dims. OBS_DIM goes from **1309 → 1513**.
All downstream block offsets (active context, global env, reactive, prev mask, turn
history) shift automatically via `state_encoder.get_layout()`.

---

## Species Prior

Usage probabilities come from Smogon Gen3 OU stats and are hardcoded in
`SPECIES_HP_PRIOR` in `hidden_power_tracker.py`. Species not in the table receive a flat
`1/16 ≈ 0.063` per type.

Probabilities within a species sum to ≤ 1.0; the remainder is spread equally across the
other 14 types.

| Species | Type | Prior | Rationale |
|---------|------|-------|-----------|
| jolteon | ICE | ~0.70 | Dragon/Flying coverage |
| jolteon | GRASS | ~0.25 | Water coverage |
| celebi | FIRE | ~0.95 | Steel coverage (Skarmory, Metagross) |
| zapdos | ICE | ~0.65 | Dragon/Flying coverage |
| zapdos | GRASS | ~0.25 | Water/Ground coverage |
| starmie | FIRE | ~0.60 | Steel coverage |
| starmie | GRASS | ~0.30 | Water coverage |
| gengar | ICE | ~0.55 | Dragon/Flying coverage |
| gengar | FIRE | ~0.35 | Steel coverage |
| alakazam | FIRE | ~0.90 | Steel coverage |
| raikou | ICE | ~0.65 | Dragon/Flying coverage |
| raikou | GRASS | ~0.25 | Water coverage |
| tentacruel | ELECTRIC | ~0.80 | Ground/Rock coverage |
| cloyster | ELECTRIC | ~0.80 | Ground/Rock coverage |
| lanturn | ICE | ~0.75 | Grass coverage |

*Probabilities are approximate; update from current Smogon Gen3 OU usage data before
shipping.*

**Safety guard:** if an effectiveness observation would zero out all remaining non-zero
entries (prior was wrong, or a rare off-meta HP type is being run), reset to flat `1/16`
per type and re-filter. The candidate vector when `hp_revealed=1` must never be all-zero.

---

## Update Logic

### Where it lives

`HiddenPowerTracker` is a new stateful class instantiated on `EpisodeTracker`. It owns
the per-species probability vector and updates inside `EpisodeTracker.record()` after
each new `BattleContext` is built.

### Timing

At turn N's decision point, `BattleContext` captures:
- `opp_last_move_id` — the opponent's move from turn N-1 (via `opp_mon.last_move`)
- `opp_last_effectiveness` — effectiveness of that move from turn N-1

The tracker update happens inside `record()`, before the observation is encoded for turn
N. The observation at turn N therefore already incorporates evidence from all turns 1..N-1.

### Update pseudocode

```python
# Inside EpisodeTracker.record(), after ctx is built:
if (self._history
        and ctx.opp_last_move_id == "hiddenpower"
        and ctx.opp_last_effectiveness is not None):
    prev = self._history[-1]
    our_mon = battle.team.get(prev.our_active.lower())  # who was active when HP hit
    if our_mon:
        self._hp_tracker.observe(
            species=prev.opp_active,
            effectiveness=ctx.opp_last_effectiveness,
            target_mon=our_mon,          # provides type1, type2, ability for the filter
        )
```

`observe()` filters `species`'s probability vector: any type `t` where
`effective_multiplier(t, target_mon) != effectiveness` is zeroed out. If this would
produce an all-zero vector, the safety guard resets and re-filters.

### Matchup matrix override

`reactive.py`'s "their moves vs our mons" loop uses `move.type` for HP, which resolves
to `PokemonType.NORMAL` (wrong). The tracker exposes:

```python
def inferred_type(self, species: str) -> PokemonType | None:
    """Returns the confirmed HP type when exactly one non-zero entry remains; else None."""
```

When `inferred_type` is not None, `reactive.py` substitutes it into the
`effective_multiplier()` call for that move-slot/mon-slot pair.

---

## Relationship to ai_v4 Unified Transformer

The unified transformer (`design_unified_transformer.md`) replaces the five hand-crafted
attention paths with a learned 2-layer transformer over 23 tokens. The role encoder
is **unchanged** from ai_v3.

HP inference is strictly upstream: it enriches the per-mon observation slot that the role
encoder consumes. The role encoder output token is then one of the 23 tokens in the
unified transformer. The two designs are independent — HP inference ships before or after
the transformer rewrite without conflict.

The unified transformer bumps `N_HISTORY_TURNS` from 5 to 10. The candidate mask is
unaffected — it accumulates across the full battle regardless of the history window.

---

## Edge Cases

| Scenario | Handling |
|----------|----------|
| Prior was wrong (filter would zero all entries) | Reset to flat 1/16 per type and re-filter |
| Same observation twice | Idempotent — zeroing already-zero entries is a no-op |
| HP used on multiple of our mons | Each observation further narrows the same vector |
| HP-using mon switches out and back in | Candidate vector is keyed by species; persists all episode |
| Sleep Talk selects HP | poke-env may record "sleeptalk" not "hiddenpower" — guard skips; acceptable miss |
| HP used but our mon faints mid-turn | `opp_last_effectiveness` may be None; guard skips |

---

## Future Work

**Our own HP type.** Showdown strips the type suffix for both sides. Our HP type is
deterministic from IVs set at build time via `GEN3_HP_IVS`. Fix: read the HP type from
the team spec at env init and fill the correct one-hot into our own mon slots directly.
This is a different data flow from inference and should ship as a separate small change.

**Physical/Special category field.** The `category` dim in the move slot uses the Normal
dummy → Physical, which is wrong for any Special HP type. Once the confirmed type is
known, update `category` to match. Depends on this feature being implemented.

**Damage-based narrowing.** Exact damage values could further constrain type beyond the
tier (BP is fixed at 70, but EVs/stat stages are partially unknown). Too noisy for the
marginal gain.

---

## Model Versioning

Bump `MODEL_CONFIG_VERSION` in `model_version.py`. The observation dimension change
(1309 → 1513) means old checkpoints cannot load — weights in the projection layer have
the wrong shape. No migration; this is a clean break.

---

## Files

| File | Change |
|------|--------|
| `src/agents/training/hidden_power_tracker.py` | **New** — `HiddenPowerTracker`, `SPECIES_HP_PRIOR` |
| `src/agents/training/episode_tracker.py` | Add tracker instance; hook `record()`; expose 17-dim block per opp slot; `reset()` |
| `src/agents/training/gen3_env.py` | Pass HP probability blocks (6×17) to `observation_encoder.encode()` |
| `src/agents/observation/state_encoder.py` | Append 17 HP dims to every mon slot; zeros for our team, tracker output for opp |
| `src/agents/observation/moves.py` | Accept `hp_type_override: PokemonType \| None`; use when `move_id == "hiddenpower"` |
| `src/agents/observation/reactive.py` | Use `inferred_type(species)` override in matchup matrix for opponent HP moves |
| `src/agents/model/model_version.py` | Bump `MODEL_CONFIG_VERSION` |

---

## Verification

1. **Unit test `HiddenPowerTracker`** — assert: Blissey at 2× immediately isolates
   Fighting; prior initialisation sets correct weights; wrong-prior safety guard resets
   and re-filters; repeated identical observations are idempotent.
2. **OBS_DIM check** — `Gen3ObservationEncoder(load_mappings()).dimension == 1513`.
3. **Smoke test** — `train_rl_agent.py --debug --steps 10000`; look for
   `[ModelVersion] Round-trip smoke test PASSED`.
4. **Unit tests** — `pytest src/ -m "not integration and not e2e" -q`.
