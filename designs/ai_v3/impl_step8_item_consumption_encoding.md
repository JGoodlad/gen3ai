# Implementation: Step 8 — Item Consumption Encoding

This step fixed a silent information loss in the per-Pokémon item encoding: when a
Berry activated (or any item was consumed), the model could no longer distinguish
"item was spent" from "item is unknown." A third dimension — `consumed` — was added
to the item vector, and the poke-env `Pokemon` class was extended to track what item
was consumed so that identity is preserved after the item leaves.

Primary themes: detecting a latent encoding gap, extending poke-env state tracking,
cascading a one-dim change through all per-Pokémon offsets, and building a real E2E
fuzz test that validates the full pipeline using actual Showdown battle data.

---

## Motivation

### The gap

Before this step, the item block in each per-Pokémon slot was 2 dims: `[item_id, known]`.
The `known` flag indicated whether we had observed what item the mon was holding. When
an item was consumed — a Berry activating, Trick swapping it away, Knock Off removing
it — poke-env called `end_item()`, which set `mon.item = None`. The encoder's else
branch then produced `[0.0, 0.0]`.

This `[0.0, 0.0]` encoding is **identical** to three distinct states that require
different policy responses:

| State | Previous encoding | Semantically correct encoding |
|---|---|---|
| Opponent hasn't revealed their item | `[0, 0]` | `[0, 0, 0]` (no info) |
| No item held from the start | `[0, 0]` | `[0, 0, 0]` (no info) |
| Had a known Sitrus Berry — now consumed | `[0, 0]` | `[sitrus_id, 1, 1]` (spent) |

The third case is the problematic one. After an opponent's Sitrus Berry activates
to restore HP, the model loses all record that the recovery happened. On the very next
turn the encoding for that mon looks identical to an unrevealed-item slot. The model
cannot:

- Distinguish "they used their Berry last turn" from "we never knew their item"
- Know that the opponent's heal-on-low-HP safety net is gone
- Track our own Berry consumption to reason about our own recovery turns

### Why this matters for Gen 3 OU

HP Berries (Sitrus, Salac, Petaya, Liechi, Ganlon, Apicot) are common in Gen 3 OU.
Salac Berry (speed+1 at ≤25% HP) and Petaya Berry (SpA+1 at ≤25%) in particular
create high-risk plays — a mon near KO range is suddenly more dangerous. After the
Berry fires, that risk is gone. Without the consumed signal, the model treats a
Salamence that just activated its Salac as equivalent to one still sitting on it.

Lum Berry (status cure) is similarly important: once consumed, the mon is no longer
status-immune. A Thunder Wave that would have been wasted before is now fully effective.

---

## What Changed

### New item encoding: 2 → 3 dims

The item block for each Pokémon grew from `[id, known]` to `[id, known, consumed]`:

| Dim | Content | Values |
|---|---|---|
| 0 | item_id | Numeric ID from `gen3_items.json` (float), or 0.0 if unknown |
| 1 | known | 1.0 if we have observed the item (held or consumed); 0.0 if not |
| 2 | consumed | 1.0 if the item was spent this battle; 0.0 if still held |

The three reachable states:

| State | `[id, known, consumed]` |
|---|---|
| Unrevealed (opponent, unknown item) | `[0, 0, 0]` |
| Known and held | `[id, 1, 0]` |
| Known and consumed | `[id, 1, 1]` |

`id` is preserved in the consumed state — the model knows which item was spent, not
just that something was spent. A consumed Salac Berry is `[salac_id, 1, 1]` throughout
the rest of the battle. A consumed Lum Berry is `[lum_id, 1, 1]`.

**Held item takes priority.** If both `mon.item` (non-None, non-unknown) and
`mon.consumed_item` are set (which can happen transiently), the encoder uses the held
item path (`[id, 1, 0]`) and ignores `consumed_item`. This handles Recycle: once the
item is restored, it is shown as held again.

### Per-Pokémon vector: 60 → 61 dims; POKEMON_FULL_DIM: 61 → 62

The single new dimension cascades through all per-Pokémon offset constants:

| Constant | Before | After |
|---|---|---|
| `ITEM_CONSUMED_DIM` | (new) | `1` |
| `POKEMON_ITEMS_OFFSET` | 7 | 7 (unchanged — items still follow species) |
| `POKEMON_TYPES_OFFSET` | 9 | 10 |
| `POKEMON_ABILITIES_OFFSET` | 11 | 12 |
| `POKEMON_CONDITION_OFFSET` | 13 | 14 |
| `POKEMON_MOVES_OFFSET` | 20 | 21 |
| `POKEMON_HP_OFFSET` | 56 | 57 |
| `POKEMON_SPECIES_KNOWN_OFFSET` | 57 | 58 |
| `POKEMON_COUNTER_OFFSET` | 58 | 59 |
| `POKEMON_VECTOR_DIM` | 60 | 61 |
| `POKEMON_FULL_DIM` | 61 | 62 |

The projection input dimension in `Gen3FeaturesExtractor` is discovered automatically
by a dummy forward pass in `__init__`, so no architecture constant needed updating.

### Observation vector: 1141 → 1153 dims

12 Pokémon × 1 new dim = 12 additional dims in the base encoder output:

| Block | Before | After | Notes |
|---|---|---|---|
| Our team (6 × FULL_DIM) | 366 | 372 | +1 dim per slot × 6 |
| Opp team (6 × FULL_DIM) | 366 | 372 | +1 dim per slot × 6 |
| Active context ×2 | 46 | 46 | Unchanged |
| Global env | 13 | 13 | Unchanged |
| Reactive + matchups | 300 | 300 | Unchanged |
| Prev-turn action mask | 11 | 11 | Unchanged |
| TurnDelta block | 39 | 39 | Unchanged |
| **Total** | **1141** | **1153** | |

---

## Implementation Details

### poke-env: `Pokemon` class

**`__slots__` extension:**
```python
"_consumed_item",   # added before "_item"
```

Required because `Pokemon` uses `__slots__` for memory efficiency. Without adding it
here, `self._consumed_item = None` in `__init__` raises `AttributeError`.

**`__init__` addition:**
```python
self._item: Optional[str] = GenData.from_gen(gen).UNKNOWN_ITEM
self._consumed_item: Optional[str] = None
```

**`end_item()` — saves item before nulling:**
```python
def end_item(self, item: str):
    self._consumed_item = item   # NEW: persist identity before clearing
    self._item = None
    if item == "powerherb":
        ...                      # existing Power Herb logic unchanged
```

poke-env's `AbstractBattle` calls `end_item(item_name)` for every `-enditem` message.
The item name string from the Showdown protocol (e.g. `"Sitrus Berry"`) is now stored
verbatim in `_consumed_item`.

**`item.setter` — clears consumed on restoration:**
```python
@item.setter
def item(self, item: Optional[str]):
    self._item = to_id_str(item) if item is not None else None
    if self._item:
        self._consumed_item = None   # NEW: Recycle / Trick restores item
```

When a real item is assigned back (via `-item` Showdown message for Recycle, Trick,
or Frisk reveal), `consumed_item` is cleared so the encoding reverts to held.

**`update_from_request()` — same logic for our own team:**
```python
self._item = request_pokemon["item"]
if self._item:
    self._consumed_item = None   # NEW
```

`update_from_request()` is called for our Pokémon when the server sends a `|request|`
message. If Recycle restores our item, the next request reflects the restored item and
consumed is cleared.

**New property:**
```python
@property
def consumed_item(self) -> Optional[str]:
    return self._consumed_item
```

Returns the raw item name string as received from the Showdown `-enditem` message
(e.g. `"Sitrus Berry"`), or `None` if no item has been consumed this battle.

### `ItemsEncoder` (`src/agents/observation/items.py`)

**`dimension`:** `ITEM_ID_DIM + ITEM_KNOWN_DIM + ITEM_CONSUMED_DIM` = 3

**`encode()` — three-branch logic:**
```python
item = mon.item
consumed = getattr(mon, "consumed_item", None)

if item:
    item_key = item.lower().replace(" ", "").replace("_", "")
    if item_key == "unknownitem":
        return vec   # all zeros — not yet revealed
    entry = self.item_to_id[item_key]
    vec[0] = float(entry["num"])
    vec[ITEM_ID_DIM] = 1.0          # known
    # consumed stays 0 — still held
elif consumed:
    consumed_key = consumed.lower().replace(" ", "").replace("_", "")
    if consumed_key in self.item_to_id:
        vec[0] = float(self.item_to_id[consumed_key]["num"])
    vec[ITEM_ID_DIM] = 1.0                  # known (we observed the consumption)
    vec[ITEM_ID_DIM + ITEM_KNOWN_DIM] = 1.0  # consumed
# else: all zeros (no item / unrevealed)
```

`getattr(mon, "consumed_item", None)` is used rather than a direct attribute access so
that mock objects in unit tests that don't set `consumed_item` degrade gracefully to
the `[0, 0, 0]` path without AttributeError.

For consumed items not in `self.item_to_id` (items from outside our gen3 mapping),
`vec[0]` stays 0.0 but `known=1, consumed=1` are still set. The model learns "something
was consumed" even without a specific ID.

**`describe_vector()`:**
```python
known    = vector[ITEM_ID_DIM] >= 0.5
consumed = vector[ITEM_ID_DIM + ITEM_KNOWN_DIM] >= 0.5
if not known: return "ITM-UNKN"
name = reverse_mapping.get(int(vector[0]), f"Item({int(vector[0])})").upper()
return f"{name}(CONSUMED)" if consumed else name
```

### Constants (`src/agents/observation/constants.py`)

```python
ITEM_CONSUMED_DIM = 1  # 1.0 when item was consumed this battle (Berry, Trick, etc.)
```

Added alongside `ITEM_ID_DIM` and `ITEM_KNOWN_DIM`. All downstream offsets are
expressions that reference these constants and `POKEMON_FULL_DIM`, so they updated
automatically when the values were changed.

---

## Edge Cases

### Consumed item not in our mapping

An item that fires `-enditem` but is absent from `gen3_items.json` (e.g. a mon holding
an item our data file doesn't cover) sets `vec[0] = 0.0`, `known=1`, `consumed=1`.
The model gets a "something was consumed" signal without a specific identity. The
encoder does not raise on consumed items the way it does on unrecognized held items —
crashing mid-battle over a missing mapping entry would be worse than losing the ID.

### Recycle / Trick restoration

When `item.setter` is called with a real item (Recycle restoring our berry, Trick
giving us a new item), `_consumed_item` is cleared. The encoding reverts to held
(`[new_id, 1, 0]`). If both `mon.item` and `mon.consumed_item` are non-None
simultaneously (e.g. a partial-update race), the held-item branch takes priority in
the encoder regardless.

### Trick/Switcheroo — `-enditem` + `-item` same turn

When Trick fires, Showdown sends both `-enditem` (item leaving) and `-item` (new item
arriving) in the same turn. The `item.setter` call for the arriving item clears
`_consumed_item`. So at the end of the turn, the mon shows its new item as held, not
the old one as consumed. This is correct — the mon still has an item after Trick.

The E2E fuzz test skips consumed validation for idents that also appear in the
same-turn `item_gained` set, preserving this invariant.

### `unknown_item` placeholder vs `None`

poke-env initialises every Pokémon's item to `GenData.UNKNOWN_ITEM` (the string
`"unknown_item"`). This is the "we haven't seen their item yet" state. The encoder
handles both `item == "unknownitem"` and `item is None` as the "no data" path (all
zeros). `consumed_item` is only ever set by an explicit `end_item()` call — it never
starts as "unknown_item". So the three-state logic is clean.

---

## Test Suite

### Unit tests (`src/agents/observation/items_test.py`)

Full rewrite: 23 tests across two groups.

**Basic state tests (11):** one test per encoding state and edge case — held item,
no item, unknown_item placeholder, consumed known item, consumed unknown item,
None mon, dimension check, held-overrides-consumed, and all three `describe_vector`
paths.

**State-transition sequence tests (12):** use a `FuzzMon` helper that mirrors how
poke-env drives the state machine (`reveal()`, `consume()`, `restore()`), then
verifies encoder output after each transition:

| Test | Sequence | Validates |
|---|---|---|
| `test_fuzz_unrevealed_stays_unknown` | (start) | `[0,0,0]` from turn 0 |
| `test_fuzz_reveal_then_held` | reveal | `[id,1,0]` |
| `test_fuzz_reveal_then_consume` | reveal → consume | held then `[id,1,1]` |
| `test_fuzz_consume_without_prior_reveal` | consume | `[id,1,1]` from first sight |
| `test_fuzz_consume_then_restore_via_recycle` | consume → restore | consumed then back to held |
| `test_fuzz_trick_gives_new_item` | reveal → consume → restore(new) | swap ends in held |
| `test_fuzz_no_item_throughout` | (no item) | `[0,0,0]` throughout |
| `test_fuzz_repeated_consume_idempotent` | consume × 2 | output stable |
| `test_fuzz_item_id_zero_for_unrecognized_consumed` | consume(unknown) | `[0,1,1]` |
| `test_fuzz_all_known_items_roundtrip` (×3 items) | reveal → consume | parametrized over all mapping entries |

### E2E fuzz test (`src/agents/training/poke_env_gaps/item_consumption_fuzz_e2e_test.py`)

Three scenarios, 20 battles each (run at 50 for full validation):

| Scenario | Teams | Target state |
|---|---|---|
| A — Lum Berry | Both sides carry Lum Berry; heavy status-inflicting moves (Thunder Wave, Spore, Hypnosis, Will-O-Wisp) | Status-cure Berry consumption — both sides, every few turns |
| B — HP Berries | Salac Berry (≤25% Spe), Petaya Berry (≤25% SpA), Sitrus Berry (≤50% HP); offensive hard-hitting teams | HP-threshold Berry consumption |
| C — Held Items | Leftovers + Choice Band only; no consumable items | Held encoding stays clean (`consumed=0`) throughout |

**Two validation layers per consumption event:**

```
Raw -enditem message (intercepted in _handle_battle_message)
  [ident, item_raw] archived per turn, validated at next choose_move()
    ↓ Layer 1: poke-env state correctness
  mon.consumed_item == item_raw (exact string match)
    ↓ Layer 2: ItemsEncoder output correctness
  vec[consumed]=1, vec[known]=1, vec[id]=expected_num
```

**Per-turn held-item sweep:** at every `choose_move()`, all observable mons with a
known held item are encoded and checked for `known=1, consumed=0, id=correct`. This
catches spurious consumed-bit flips on mons that never had an item consumed.

**Run results (20 battles per scenario):**

```
Scenario A — Lum Berry
  Total turns:             1506
  Enditem events:           130  (66 our side, 64 opp)
  Held-item checks:        5512
  Layer 1 mismatches:         0
  Layer 2 mismatches:         0

Scenario B — HP Berries
  Total turns:              863
  Enditem events:           167  (87 our side, 80 opp)
  Held-item checks:        3729
  Layer 1 mismatches:         0
  Layer 2 mismatches:         0

Scenario C — Held Items
  Total turns:             4177
  Enditem events:             0  (expected — no consumable items)
  Held-item checks:       38128
  Layer 1 mismatches:         0
  Layer 2 mismatches:         0

Total consumption events validated:  297
Total held-item checks:           47369
```

---

## Files Changed

| File | Change |
|---|---|
| `src/poke_env/battle/pokemon.py` | `_consumed_item` slot + field; `end_item()` saves item; `item.setter` clears on restore; `update_from_request()` clears on restore; `consumed_item` property |
| `src/agents/observation/constants.py` | `ITEM_CONSUMED_DIM = 1`; all per-mon offsets +1 from `POKEMON_TYPES_OFFSET` onward; `POKEMON_VECTOR_DIM` 60→61; `POKEMON_FULL_DIM` 61→62 |
| `src/agents/observation/items.py` | `dimension` 2→3; three-branch `encode()`; `get_layout()` adds consumed entry; `describe_vector()` shows `(CONSUMED)` suffix |
| `src/agents/observation/items_test.py` | Full rewrite: 23 tests (basic states + state-transition fuzz sequence tests) |
| `src/agents/observation/pokemon_test.py` | `dimension` assertion 60→61 |
| `src/agents/observation/state_encoder_test.py` | `EXPECTED_BASE_DIM` 1091→1103; `EXPECTED_OBS_DIM` 1141→1153 |
| `src/agents/training/poke_env_gaps/item_consumption_fuzz_e2e_test.py` | New E2E fuzz test (3 scenarios, 2 validation layers, coverage checks) |
| `CLAUDE.md` | Observation vector table updated (1153 dims, 6×62 per team, all offsets); per-slot description updated to document 3-dim item block |
