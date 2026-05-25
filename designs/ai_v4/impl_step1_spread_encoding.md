# Implementation: Step 1 — IV/EV/Nature Spread Encoding

This step adds the 18-dim spread block to each per-Pokémon observation slot. Own-team
slots receive the actual IVs, EVs, and nature from the teambuilder. Opponent slots are
all zeros with `spread_known=0`, so the model can distinguish "opponent spread unknown"
from "own Pokémon with genuinely zero EVs." A poke-env bug that silently dropped IVs,
EVs, and nature for all-zero-EV Pokémon was found and fixed as part of this work.

Primary themes: extending the per-slot encoding without breaking existing code, the
`spread_known` flag design to disambiguate zero-EV own mons from unknown opponent
spreads, finding and fixing a latent poke-env correctness bug, and a clean architecture
break via `ARCH_SIGNATURE`.

---

## Motivation

### The gap

Before this step, the per-Pokémon slot had no IV, EV, or nature information. The role
encoder received 128D tokens with species stats, moves, items, HP, status — but no
spread. This created two problems:

1. **Our own team**: the exact IV/EV/nature for every Pokémon is known at team-build
   time and is never uncertain. The network had no access to it, so it could not reason
   about damage calculations, speed ties, or whether a Pokémon was holding a special HP
   Berry spread.

2. **Opponent team**: spread is genuinely unknown, but the network had no way to express
   "I don't know this Pokémon's spread" — it just saw zeros for fields that don't exist
   yet. Making the flag explicit lets the model learn that opponent tokens are always
   spread-unobserved.

### Why the `spread_known` flag matters

An all-zero-EV Pokémon on our own team (e.g. a Shedinja, or a team paste with no EV
line) would encode identically to an opponent slot if both just used `0.0` for all
spread dims. The `spread_known=1.0` flag for own-team slots and `spread_known=0.0` for
opponent slots resolves the ambiguity — the model always knows which cells are real data
and which are padding.

### Why this matters for Gen 3 OU

Nature and EVs determine damage outputs, speed tiers, and survival thresholds. Common
spreads like 252 Atk / 252 Spe Adamant vs. 4 HP / 252 SpA / 252 Spe Timid result in
very different match-up evaluations. Speed ties (e.g. Gengar vs. Alakazam at base 110)
resolve differently depending on EVs. Without spread, the role encoder cannot represent
these distinctions even if the architecture otherwise could.

---

## What Changed

### Per-Pokémon slot: 61 → 79 dims; POKEMON_FULL_DIM: 62 → 80

The 18-dim spread block appends after the existing 61 dims:

| Field | Dims | Normalization | Own slots | Opp slots |
|-------|------|---------------|-----------|-----------|
| IVs (HP, Atk, Def, SpA, SpD, Spe) | 6 | `/31 → [0, 1]` | actual from teambuilder | 0.0 |
| EVs (HP, Atk, Def, SpA, SpD, Spe) | 6 | `/252 → [0, 1]` | actual from teambuilder | 0.0 |
| `spread_known` | 1 | flag | 1.0 | 0.0 |
| Nature modifiers (Atk, Def, SpA, SpD, Spe) | 5 | raw float (0.9/1.0/1.1) | from `natures.json` | 0.0 |

Nature modifiers are the raw multiplier values from `natures.json` (e.g. Adamant →
`[1.1, 1.0, 0.9, 1.0, 1.0]`). HP is never nature-modified in any generation, so the
order is `[atk, def, spa, spd, spe]` — five stats, not six.

### New and changed constants

| Constant | Before | After | Notes |
|----------|--------|-------|-------|
| `POKEMON_SPREAD_OFFSET` | (new) | `61` | Appends after existing 61 dims |
| `POKEMON_SPREAD_DIM` | (new) | `18` | 6 IVs + 6 EVs + 1 flag + 5 nature |
| `POKEMON_VECTOR_DIM` | `61` | `79` | 61 + 18 |
| `POKEMON_FULL_DIM` | `62` | `80` | 79 + 1 (active flag) |

All top-level offsets derive from `POKEMON_FULL_DIM` and updated automatically:

| Constant | Before | After |
|----------|--------|-------|
| `OFFSET_OPP_TEAM` (= 6 × FULL_DIM) | 372 | 480 |
| `OFFSET_CONTEXT` (= 2 × OPP_TEAM) | 744 | 960 |
| `OFFSET_GLOBAL` (= CONTEXT + 2 × 23) | 790 | 1006 |
| `OFFSET_REACTIVE` (= GLOBAL + 13) | 803 | 1019 |

### Observation vector dimensions

12 Pokémon × 18 new dims = 216 additional dims in the base encoder output:

| Block | Before | After | Notes |
|-------|--------|-------|-------|
| Our team (6 × FULL_DIM) | 372 | 480 | +18 dims per slot × 6 |
| Opp team (6 × FULL_DIM) | 372 | 480 | +18 dims per slot × 6 |
| Active context ×2 | 46 | 46 | Unchanged |
| Global env | 13 | 13 | Unchanged |
| Reactive + matchups | 300 | 300 | Unchanged |
| Prev-turn action mask | 11 | 11 | Unchanged |
| Turn history (5 × 39) | 195 | 195 | Unchanged |
| **Base dim** | **1103** | **1319** | |
| **Total** | **1309** | **1525** | |

### Architecture version

`ARCH_SIGNATURE` changed from `"gen3_td_cond_v1"` to `"gen3_spread_v1"`. The obs
dimension change makes old checkpoints incompatible — the signature change guarantees
they fail at startup with a clear error rather than loading mismatched weights silently.

---

## Implementation Details

### Constants (`src/agents/observation/constants.py`)

```python
POKEMON_SPREAD_OFFSET = 61     # 59 + 2 (status counters)
POKEMON_SPREAD_DIM = 18        # 6 IVs + 6 EVs + 1 flag + 5 nature modifiers
POKEMON_VECTOR_DIM = 79        # 61 + 18 (spread block)
POKEMON_FULL_DIM = 80          # 79 + 1 (active flag appended by state_encoder)
```

The inline comments on top-level offsets were also corrected to reflect the new values.

### Nature loading (`src/agents/observation/state_encoder.py`)

`load_mappings()` was extended to load `natures.json` from the poke-env static data
directory:

```python
_natures_path = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..", "poke_env", "data", "static", "natures.json"
))
if not os.path.exists(_natures_path):
    raise FileNotFoundError(
        f"CRITICAL: natures.json missing at {_natures_path}. "
        "poke_env static data may be corrupted."
    )
with open(_natures_path, "r") as f:
    raw_natures = json.load(f)
mappings["natures"] = {
    name: {k: float(v) for k, v in entry.items() if k in ("atk", "def", "spa", "spd", "spe")}
    for name, entry in raw_natures.items()
}
```

The `"num"` key is stripped — only the five stat multipliers are kept. All 25 natures
are present in the file; neutral natures (e.g. Hardy) have all multipliers at `1.0`.

`Gen3ObservationEncoder.__init__` then passes `natures=mappings.get("natures", {})` to
`PokemonEncoder`.

The encode loop now passes `is_own=True` for our team and `is_own=False` for opponents:

```python
mon_vec = self.pokemon_encoder.encode(mon, battle, is_own=True)   # our team
mon_vec = self.pokemon_encoder.encode(mon, battle, is_own=False)  # opponent team
```

### PokemonEncoder (`src/agents/observation/pokemon.py`)

New class variable:
```python
_NATURE_STAT_ORDER = ("atk", "def", "spa", "spd", "spe")
```

Updated `__init__` to accept `natures: dict = None` and store as `self._natures`:
```python
def __init__(self, species_encoder, items_encoder, type_encoder,
             abilities_encoder, moves_encoder, natures: dict = None):
    ...
    self._natures = natures or {}
```

Updated `encode()` signature to `encode(self, mon, battle, is_own: bool = False)`.
The spread block is appended at the end of `encode()` for own-team slots:

```python
if is_own:
    ivs = mon.ivs  # list[int] | None: [HP, Atk, Def, SpA, SpD, Spe]
    evs = mon.evs  # list[int] | None: [HP, Atk, Def, SpA, SpD, Spe]
    off = POKEMON_SPREAD_OFFSET
    # IVs: fallback to all-31 (competitive standard) if None —
    # encoding 0.0 for unknown IVs would be silently wrong
    iv_vals = ivs if ivs is not None else [31] * 6
    for j, iv in enumerate(iv_vals):
        vec[off + j] = iv / 31.0
    if evs is not None:
        for j, ev in enumerate(evs):
            vec[off + 6 + j] = ev / 252.0
    vec[off + 12] = 1.0  # spread_known flag
    nature_name = mon.nature  # str | None; fallback to "serious" (all 1.0 modifiers)
    nature_mods = self._natures.get(nature_name or "serious", {})
    for j, stat in enumerate(self._NATURE_STAT_ORDER):
        vec[off + 13 + j] = float(nature_mods.get(stat, 1.0))
# Opponent slots: all 18 dims remain 0.0
```

`get_layout()` was extended with a `"spread"` entry and `"pokemon_vector_dim": POKEMON_VECTOR_DIM`.

### poke-env bug fix (`src/poke_env/battle/pokemon.py`)

`_update_from_teambuilder()` had a guard that prevented IVs, EVs, and nature from being
stored for any Pokémon with all-zero EVs:

```python
# BEFORE (buggy):
if not all(e == 0 for e in tb.evs):
    self._evs = tb.evs
    self._ivs = tb.ivs
    self._nature = tb.nature.lower() if tb.nature is not None else "serious"
```

This meant that a Shedinja, or any Pokémon from a team paste with no EV line, had
`_ivs = None`, `_evs = None`, and `_nature = None`. The encoder then:

- Fell back to `ivs = None if ivs is not None else [31] * 6` — but the guard was
  not yet present, so it encoded IVs as 0.0 instead of 1.0 (a lie to the network)
- Used `mon.nature = None → "serious"` fallback — silently wrong if the real nature
  had stat penalties (e.g. a Shedinja with Jolly was encoded as neutral)

The fix removes the guard entirely:

```python
# AFTER (correct):
# Always store IVs/EVs/nature from the teambuilder — all-zero EVs is a valid
# competitive spread (e.g. a Shedinja or a team paste with no EV line).
self._evs = tb.evs
self._ivs = tb.ivs
self._nature = tb.nature.lower() if tb.nature is not None else "serious"
```

`apply_teambuilder_team` is only ever called with `self.player_role` (i.e. our own team),
so this fix does not affect opponent Pokémon objects.

### Model versioning (`src/agents/model/model_version.py`)

```python
ARCH_SIGNATURE = "gen3_spread_v1"  # was "gen3_td_cond_v1"
```

`MODEL_CONFIG_VERSION` was not changed — no optional fields were added, only a
structural obs-shape change that the signature covers.

### Feature extractor (`src/agents/model/features_extractor.py`)

No functional changes. The `role_input_dim` is computed dynamically in `__init__` from
layout fields:

```python
_hp_and_active_dim = POKEMON_FULL_DIM - _pk_layout['hp']['offset']  # = 80 - 57 = 23
```

The spread dims sit between `spread_offset` (61) and `hp_offset` (57)... wait — spread
comes *after* `hp_offset`. The `pokemon_part[:, :, hp_offset:]` slice captures
`[HP, species_known, sleep_ctr, toxic_ctr, spread×18, active_flag]` = 23 dims,
unchanged from 62 → 80 because the active flag is still the last dim and the slice
anchor (hp_offset=57) is unchanged. The role input dim auto-updates via this dynamic
computation — no manual edit was needed.

A stale comment ("base 1021-dim obs") was corrected to "offsets read from layout, not hardcoded".

---

## Edge Cases

### None IVs — fallback must be 31, not 0

`mon.ivs` returns `None` for Pokémon where poke-env never populated the field (e.g.
a transient Pokemon object before teambuilder data arrives). The encoder falls back to
`[31] * 6`, not `[0] * 6`. All-31 IVs is the competitive standard default — encoding
all-0 IVs would tell the network "this Pokémon has minimum IVs in every stat," which
would be factually wrong and would mislead damage calculations.

### None nature — fallback is "serious"

`mon.nature` is `None` when poke-env has no nature data. The fallback is `"serious"`
(all multipliers 1.0), matching the poke-env TeambuilderPokemon default for a team
paste with no Nature line. The `"serious"` key is always present in `natures.json`.

### Unknown nature key — fallback to 1.0 per stat

If a nature name is present but not in `self._natures` (shouldn't happen with the
current data file), `nature_mods.get(stat, 1.0)` returns `1.0` for each stat, encoding
a neutral nature. This is conservative — no stat penalty is claimed — and the model
can detect it is not reliable via `spread_known`.

### Zero EVs on own team are distinguishable from opponent slots

A Shedinja (no EVs, always 1 HP) on our team encodes as:
- `spread_known = 1.0`
- All EV dims = 0.0

An opponent slot encodes as:
- `spread_known = 0.0`
- All EV dims = 0.0

These are distinct vectors. The `spread_known` flag is the decision boundary, not the EV
values themselves.

---

## Test Suite

Tests added to `src/agents/observation/pokemon_test.py`.

### Spread unit tests (10)

| Test | What it validates |
|------|-------------------|
| `test_pokemon_encoder_dimension` | `POKEMON_VECTOR_DIM` = 79 (updated from 61) |
| `test_pokemon_encoder_empty` | All-zeros for `None` mon with `is_own=False` |
| `test_spread_block_opponent_all_zeros` | All 18 spread dims = 0.0 for opponent slot |
| `test_spread_block_own_standard` | IVs all-31, EVs 252/252/4/0/0/0, Adamant nature |
| `test_spread_block_own_hp_iv_spread` | IVs [31,30,31,30,31,30] encodes correctly |
| `test_spread_block_own_zero_evs_distinguishable_from_opp` | `spread_known` 1.0 vs 0.0 |
| `test_spread_block_own_neutral_nature_all_ones` | Hardy → all five nature dims = 1.0 |
| `test_spread_block_is_own_default_false` | `is_own` defaults to False; spread all zeros |
| `test_spread_block_own_none_ivs_fallback` | None IVs → 1.0 not 0.0 |

### poke-env regression tests (3)

These tests construct real `TeambuilderPokemon` objects and verify that
`_update_from_teambuilder` stores the correct values even for all-zero-EV mons:

| Test | What it validates |
|------|-------------------|
| `test_poke_env_all_zero_evs_stores_ivs_and_nature` | Regression: all-zero EVs no longer blocks IV/nature storage |
| `test_poke_env_no_ev_line_stores_defaults` | No EV line in paste → `ivs=[31]*6`, `evs=[0]*6` |
| `test_spread_encoder_all_zero_evs_end_to_end` | Full pipeline: Jynx (Timid, [31,0,31,31,31,31]) encoded correctly |

### State encoder dimension test (updated)

`src/agents/observation/state_encoder_test.py`:

| Constant | Before | After |
|----------|--------|-------|
| `EXPECTED_BASE_DIM` | 1103 | 1319 |
| `EXPECTED_OBS_DIM` | 1309 | 1525 |

---

## Files Changed

| File | Change |
|------|--------|
| `src/agents/observation/constants.py` | `POKEMON_SPREAD_OFFSET = 61`, `POKEMON_SPREAD_DIM = 18`; `POKEMON_VECTOR_DIM` 61→79; `POKEMON_FULL_DIM` 62→80; inline offset comments corrected |
| `src/agents/observation/pokemon.py` | `_NATURE_STAT_ORDER` class var; `natures` param to `__init__`; `is_own` param to `encode()`; spread block at end of `encode()`; spread entry in `get_layout()`; docstring updated |
| `src/agents/observation/state_encoder.py` | `load_mappings()` loads `natures.json` with CRITICAL error; passes `natures=` to `PokemonEncoder`; encode loops pass `is_own=True/False` |
| `src/agents/observation/pokemon_test.py` | Dimension assertions updated; `_TEST_NATURES` + `_make_mon()` helpers; 9 new spread tests; 3 poke-env regression tests |
| `src/agents/observation/state_encoder_test.py` | `EXPECTED_BASE_DIM` 1103→1319; `EXPECTED_OBS_DIM` 1309→1525 |
| `src/agents/model/model_version.py` | `ARCH_SIGNATURE = "gen3_spread_v1"` |
| `src/agents/model/features_extractor.py` | Stale comment corrected; no functional changes |
| `src/poke_env/battle/pokemon.py` | Remove `if not all(e == 0 for e in tb.evs)` guard in `_update_from_teambuilder`; IVs/EVs/nature stored unconditionally |
| `CLAUDE.md` | Observation vector table updated (1525 dims, 6×80 per team, all offsets); per-slot layout updated to document spread block |
